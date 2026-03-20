# Project: MindCore

## On Entry (MANDATORY)

Run immediately when entering this project:
```bash
session-context
```

---

## Project Overview

**MindCore** is a standalone Rust crate providing a pluggable, feature-gated memory engine for AI agent applications.

**Status:** Architecture and research phase. Not yet in development.

**Consumers:** PIRDLY, Dial, Memloft — all will depend on MindCore for persistent memory.

---

## Key Files

| File | Purpose |
|------|---------|
| `MINDCORE_ARCHITECTURE.md` | Full crate structure and API design |
| `MINDCORE_RESEARCH.md` | Landscape analysis, source project analysis, specification |
| `DECISIONS.md` | Architectural decisions log |
| `research/` | Competitive landscape research |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust |
| Storage | SQLite + rusqlite + FTS5 |
| Embeddings | Candle (granite-small-r2 native, bge-small-en-v1.5 WASM), feature-gated |
| Search | FTS5 keyword + vector similarity + RRF hybrid |
| Decay Model | ACT-R activation-based |

---

## Design Principles

1. **Library, not framework** — projects call into MindCore
2. **Feature-gated everything** — heavy deps behind compile-time flags
3. **Local-first** — SQLite-backed, no cloud dependency
4. **Pure Rust where possible** — candle over ort
5. **Proven patterns only** — every component battle-tested in Memloft, Dial, or PIRDLY

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
