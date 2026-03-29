# Project: FeMind

## On Entry (MANDATORY)

Run immediately when entering this project:
```bash
session-context
```

---

## Project Overview

**FeMind** is the branded display name for this project. Use `FeMind` in prose and human-facing documentation, and keep `fe-mind` / `femind` for repo, path, package, and code contexts.

**FeMind** is a standalone Rust crate providing a pluggable, feature-gated memory engine for AI agent applications.

**Status:** local repo/crate rename to `fe-mind` / `femind` is complete. External publication work is still pending.

---

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | Full crate structure and API design |
| `RESEARCH.md` | Landscape analysis, academic foundations, specification |
| `DECISIONS.md` | Architectural decisions log |
| `PRACTICAL_EVAL.md` | Real-world validation strategy and release criteria |
| `.docs/femind_spec.db` | Authoritative repo-local architecture and implementation database |
| `research/` | Competitive landscape research |
| `eval/practical/` | Curated real-world validation scenarios |
| `eval/live-library/` | Larger real-world follow-up validation library |
| `eval/memloft-slice/` | Memloft-derived technical real-data validation slice |

---

## Documentation Database

Primary local source of truth:
```bash
sqlite3 .docs/femind_spec.db "SELECT section_id, title FROM sections ORDER BY sort_order;"
```

The Markdown docs in this repo should stay aligned with `.docs/femind_spec.db`.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust |
| Storage | SQLite + rusqlite + FTS5 |
| Embeddings | all-MiniLM-L6-v2 via Candle BERT (native + WASM) or API (DeepInfra), feature-gated |
| Search | FTS5 keyword + vector similarity + RRF hybrid |
| Decay Model | ACT-R activation-based |

---

## Design Principles

1. **Library, not framework** — projects call into FeMind
2. **Feature-gated everything** — heavy deps behind compile-time flags
3. **Local-first** — SQLite-backed, no cloud dependency
4. **Pure Rust where possible** — candle over ort
5. **Proven patterns only** — every component backed by research or established practice

---

## Memory Commands

**Log decisions/notes:**
```bash
memory-log decision "topic" "what was decided and why"
memory-log note "topic" "content"
memory-log blocker "topic" "what is blocking"
```

**Manage tasks:**
```bash
task add "description" [priority]
task list
task done <id>
```

---

## Development Workflow

- Work from the current specs and task list; keep changes scoped and validate them before committing.
- Prefer local compile, lint, and non-network test paths first.
- Do not run real CLI/API LLM validation or benchmark paths without explicit user approval.
- Treat `eval/practical/` as the primary live-validation target, `eval/live-library/` as the larger synthetic real-world layer, and `eval/memloft-slice/` as the real technical corpus layer before benchmark work.

---

## External-Facing Writing

- Keep README files, architecture docs, changelogs, commit messages, PR text, and code comments in normal developer voice.
- Do not describe implementation work in process language that reads like internal automation or prompt transcripts.
- Mention AI, LLMs, embeddings, or memory orchestration only when they are part of the actual FeMind product surface being documented.
