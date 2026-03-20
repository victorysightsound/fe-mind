# Task: Write tests for scoring strategies and context assembly

## ⚠️ SIGNS (Critical Rules)


- **ONE TASK ONLY: Complete exactly this task. No scope creep.**

- **SEARCH BEFORE CREATE: Always search for existing files/functions before creating new ones.**

- **NO PLACEHOLDERS: Every implementation must be complete. No TODO, FIXME, or stub code.**

- **VALIDATE BEFORE DONE: Run `dial validate` after implementing. Don't mark complete without testing.**

- **RECORD LEARNINGS: After success, capture what you learned with `dial learn "..." -c category`.**

- **FAIL FAST: If blocked or confused, stop and ask rather than guessing.**



## Related Specifications


### MindCore — Product Requirements Document > 3. Phases > Phase 2: Scoring + Context Assembly
Post-search scoring and token-budget context assembly. After this phase, MindCore can rank results intelligently and produce LLM-ready context.

**Deliverables:**
- `ScoringStrategy` trait (takes `&MemoryMeta`)
- Shipped scorers: `RecencyScorer`, `ImportanceScorer`, `CategoryScorer`, `MemoryTypeScorer`
- `CompositeScorer` for combining strategies multiplicatively
- `ScoredResult<T>` struct with score breakdown
- `ContextBudget` struct and priority constants
- `ContextItem` and `ContextAssembly` 

## Project Learnings (apply these patterns)


- [gotcha] Module visibility: when engine.rs references types from other modules, those modules must be pub mod not mod. Fixed store and builder visibility.

- [pattern] Mutex<Vec<Connection>> makes Database auto-Sync without unsafe impl. Connection is Send, Mutex provides Sync. No need for unsafe.