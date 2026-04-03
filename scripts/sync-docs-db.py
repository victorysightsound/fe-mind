#!/usr/bin/env python3
"""Sync current FeMind markdown specs into the canonical .docs database."""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / ".docs" / "femind_spec.db"
DEFAULT_SOURCES = [
    ROOT / "specs" / "PRODUCTION_HARDENING.md",
    ROOT / "specs" / "PRODUCTION_ROADMAP.md",
]


@dataclass
class Section:
    section_id: str
    title: str
    parent_id: str | None
    level: int
    sort_order: int
    content: str
    word_count: int


def parse_markdown_sections(content: str) -> list[Section]:
    sections: list[Section] = []
    lines = content.splitlines()
    current: dict[str, object] | None = None
    current_content: list[str] = []
    counters = [0, 0, 0, 0, 0, 0]
    seen_ids: set[str] = set()
    in_code_fence = False

    for line in lines:
        if re.match(r"^```|^~~~", line):
            in_code_fence = not in_code_fence
            current_content.append(line)
            continue

        if in_code_fence:
            current_content.append(line)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            if current:
                current["content"] = "\n".join(current_content).strip()
                current["word_count"] = len(str(current["content"]).split())
                sections.append(Section(**current))

            level = len(heading.group(1))
            title = heading.group(2).strip()

            counters[level - 1] += 1
            for index in range(level, len(counters)):
                counters[index] = 0

            base_id = ".".join(str(x) for x in counters[:level] if x > 0)
            section_id = base_id
            suffix = 1
            while section_id in seen_ids:
                suffix += 1
                section_id = f"{base_id}_{suffix}"
            seen_ids.add(section_id)

            parent_id = None
            if level > 1:
                parent_id = ".".join(str(x) for x in counters[: level - 1] if x > 0)
                if parent_id not in seen_ids:
                    parent_id = None

            current = {
                "section_id": section_id,
                "title": title,
                "parent_id": parent_id,
                "level": level,
                "sort_order": len(sections),
            }
            current_content = []
            continue

        current_content.append(line)

    if current:
        current["content"] = "\n".join(current_content).strip()
        current["word_count"] = len(str(current["content"]).split())
        sections.append(Section(**current))

    return sections


def root_number(section_id: str) -> int:
    match = re.match(r"^(\d+)", section_id)
    if not match:
        raise ValueError(f"section id does not start with a numeric root: {section_id}")
    return int(match.group(1))


def source_key(source: Path) -> str:
    return str(source.relative_to(ROOT))


def shift_section_id(section_id: str, base_root: int) -> str:
    parts = section_id.split(".")
    parts[0] = str(base_root)
    return ".".join(parts)


def section_root_id(section_id: str) -> str:
    return section_id.split(".")[0]


def existing_root_for_source(conn: sqlite3.Connection, source_path: str) -> int | None:
    row = conn.execute(
        """
        SELECT section_id
        FROM sections
        WHERE source_path = ?
        ORDER BY level ASC, sort_order ASC
        LIMIT 1
        """,
        (source_path,),
    ).fetchone()
    if row is None:
        return None
    return root_number(row["section_id"])


def next_available_root(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT section_id
        FROM sections
        WHERE parent_id IS NULL OR parent_id = ''
        ORDER BY CAST(substr(section_id, 1, instr(section_id, '.') - 1) AS INTEGER)
        """
    ).fetchall()
    roots = [root_number(row["section_id"]) for row in rows if re.match(r"^\d+", row["section_id"])]
    return (max(roots) + 1) if roots else 1


def delete_source_sections(conn: sqlite3.Connection, source_path: str) -> None:
    conn.execute("DELETE FROM sections WHERE source_path = ?", (source_path,))


def sync_source(conn: sqlite3.Connection, source: Path, base_root: int) -> int:
    sections = parse_markdown_sections(source.read_text())
    source_ref = source_key(source)
    delete_source_sections(conn, source_ref)

    for section in sections:
        section_id = shift_section_id(section.section_id, base_root)
        parent_id = shift_section_id(section.parent_id, base_root) if section.parent_id else None
        conn.execute(
            """
            INSERT INTO sections (
                section_id, title, parent_id, level, sort_order, content,
                status, review_status, source_path, word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                section_id,
                section.title,
                parent_id,
                section.level,
                section.sort_order,
                section.content,
                "review",
                "unreviewed",
                source_ref,
                section.word_count,
            ),
        )

    return len(sections)


def ensure_decision(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO decisions (
            decision_id, tool_name, section_id, title, decision, rationale,
            status, supersedes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "FEMIND-DOCS-003",
            "femind",
            "21",
            "Canonical docs database stays in .docs/femind_spec.db",
            "Keep .docs/femind_spec.db as the canonical project source of truth. Sync the current markdown specs into it from the repository sources after spec edits so the database stays aligned with the live tree.",
            "The DB is already the long-lived project record. A repeatable sync step keeps the canonical store current without moving it.",
            "accepted",
            None,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("sources", nargs="*", type=Path, default=DEFAULT_SOURCES)
    args = parser.parse_args()

    db_path: Path = args.db
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")

    sources: Iterable[Path] = args.sources
    for source in sources:
        if not source.exists():
            raise SystemExit(f"source file not found: {source}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    conn.execute("PRAGMA foreign_keys = OFF")
    total_sections = 0
    for source in sources:
        existing_root = existing_root_for_source(conn, str(source))
        base_root = existing_root if existing_root is not None else next_available_root(conn)
        total_sections += sync_source(conn, source, base_root)

    ensure_decision(conn)
    conn.execute(
        "UPDATE documents SET updated_at = ? WHERE doc_id = ?",
        (datetime.now().isoformat(), "femind_spec"),
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    print(f"synced {len(list(sources))} source files into {db_path}")
    print(f"updated {total_sections} section rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
