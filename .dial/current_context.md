# Task: Implement EmbeddingBackend trait

## ⚠️ SIGNS (Critical Rules)


- **ONE TASK ONLY: Complete exactly this task. No scope creep.**

- **SEARCH BEFORE CREATE: Always search for existing files/functions before creating new ones.**

- **NO PLACEHOLDERS: Every implementation must be complete. No TODO, FIXME, or stub code.**

- **VALIDATE BEFORE DONE: Run `dial validate` after implementing. Don't mark complete without testing.**

- **RECORD LEARNINGS: After success, capture what you learned with `dial learn "..." -c category`.**

- **FAIL FAST: If blocked or confused, stop and ask rather than guessing.**



## Related Specifications


### MindCore — Product Requirements Document > 3. Phases > Phase 5: Vector Search + Hybrid RRF
Candle embedding backend, brute-force vector search, and RRF hybrid merge. After this phase, MindCore has semantic search.

**Deliverables:**
- `EmbeddingBackend` trait (async fn embed, embed_batch, dimensions, model_name)
- `CandleNativeBackend` — granite-small-r2 via ModernBERT (~100-130 lines)
- Mean pooling and L2 normalization (shared `pooling.rs`)
- `NoopBackend` for testing
- `FallbackBackend` wrapping `Option<Box<dyn EmbeddingBackend>>`
- Model auto-download via hf-hub, cached at `~/.c

## Project Learnings (apply these patterns)


- [gotcha] Module visibility: when engine.rs references types from other modules, those modules must be pub mod not mod. Fixed store and builder visibility.

- [pattern] Mutex<Vec<Connection>> makes Database auto-Sync without unsafe impl. Connection is Send, Mutex provides Sync. No need for unsafe.

- [gotcha] Tier filtering: default SearchDepth must be Deep (include tier 0) until consolidation promotes memories to higher tiers. Standard (tiers 1+2 only) breaks all tests when memories default to tier 0.

- [gotcha] ACT-R activation: t.max(1.0) gives ln(1.0)=0 for recent accesses. Use t.max(0.1) so sub-second accesses still contribute positively.