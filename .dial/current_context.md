# Task: Implement mindcore_meta table and migration framework

## ⚠️ SIGNS (Critical Rules)


- **ONE TASK ONLY: Complete exactly this task. No scope creep.**

- **SEARCH BEFORE CREATE: Always search for existing files/functions before creating new ones.**

- **NO PLACEHOLDERS: Every implementation must be complete. No TODO, FIXME, or stub code.**

- **VALIDATE BEFORE DONE: Run `dial validate` after implementing. Don't mark complete without testing.**

- **RECORD LEARNINGS: After success, capture what you learned with `dial learn "..." -c category`.**

- **FAIL FAST: If blocked or confused, stop and ask rather than guessing.**



## Related Specifications


### MindCore — Product Requirements Document > 3. Phases > Phase 1: Foundation (Storage + FTS5 + CRUD)
The core that everything else builds on. After this phase, MindCore is a functional keyword-search memory engine.

**Deliverables:**
- `Cargo.toml` with crate metadata, `default = ["fts5"]`
- `MemoryRecord` trait and `MemoryMeta` struct
- `MemoryType` enum (Episodic, Semantic, Procedural)
- `MindCoreError` enum with `Database`, `Serialization`, `Migration` variants
- SQLite storage engine with WAL, mmap, pragmas
- Core schema: `memories` table, `memories_fts` virtual table, FTS5 triggers
- `mind