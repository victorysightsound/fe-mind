# FeMind — Product Requirements Document

**Version:** 1.0
**Date:** 2026-03-19
**Status:** v0.2.0 published

---

## 1. Vision

### 1.1 Problem

AI agent applications in Rust need persistent memory — SQLite + FTS5 + scoring + optional vector search. Each project that builds this reimplements the same primitives. The Rust ecosystem has no standalone, feature-gated agent memory crate.

### 1.2 Solution

FeMind is a standalone Rust library crate providing a pluggable, feature-gated memory engine. Consumers implement `MemoryRecord` for their types and get storage, search, decay, consolidation, and context assembly — from a 2MB FTS5-only build to a 40MB full-featured engine.

### 1.3 Success Criteria

- All tests pass with `cargo test` and `cargo test --features full`
- Any Rust project can implement `MemoryRecord` for their types and use the engine
- FTS5 search returns results in <5ms for 10K memories
- Vector search returns results in <50ms for 10K memories
- Binary size: ~2MB default, ~35-40MB full
- Zero `unsafe` outside of the candle mmap call (which is upstream)
- All public types documented with rustdoc

### 1.4 Non-Goals (v1)

- GUI or TUI interface
- Cloud sync or multi-machine replication
- Python/JS bindings (deferred)
- Belief memory type (deferred to post-v1)
- Native graph database (SQLite CTEs sufficient for v1 scale)
- LongMemEval benchmark harness (separate workspace member, post-v1)

---

## 2. Architecture Reference

Full architecture, traits, schemas, and search pipeline are defined in `ARCHITECTURE.md`. This PRD references it rather than duplicating. Key design documents:

| Document | Contents |
|----------|----------|
| `ARCHITECTURE.md` | Crate structure, traits, schema, search pipeline, API |
| `DECISIONS.md` | 22 architectural decisions with rationale |
| `RESEARCH.md` | Landscape analysis, academic foundations |

---

## 3. Phases

### Phase 1: Foundation (Storage + FTS5 + CRUD)

The core that everything else builds on. After this phase, FeMind is a functional keyword-search memory engine.

**Deliverables:**
- `Cargo.toml` with crate metadata, `default = ["fts5"]`
- `MemoryRecord` trait and `MemoryMeta` struct
- `MemoryType` enum (Episodic, Semantic, Procedural)
- `FemindError` enum with `Database`, `Serialization`, `Migration` variants
- SQLite storage engine with WAL, mmap, pragmas
- Core schema: `memories` table, `memories_fts` virtual table, FTS5 triggers
- `femind_meta` table for schema versioning
- Migration framework (version check + sequential migrations)
- CRUD: `store()`, `get()`, `update()`, `delete()`
- SHA-256 content hashing for dedup
- FTS5 keyword search with Porter stemming and BM25
- `SearchMode::Keyword`
- `SearchBuilder` fluent API with `.mode()`, `.limit()`, `.category()`, `.memory_type()`, `.execute()`
- `MemoryEngine<T>` struct with builder pattern
- `MemoryEngineBuilder` with `.database()`, `.build()`
- Thread safety: `Mutex<Connection>` for writer, connection pool for readers
- Integration tests: store/get/update/delete, FTS5 search, dedup, concurrent access

**Feature flags active:** `fts5` (default)

**Tasks (Phase 1):**
1. Create Cargo.toml with crate metadata and dependency skeleton
2. Implement `MemoryRecord` trait, `MemoryType` enum, `MemoryMeta` struct
3. Implement `FemindError` enum and `Result<T>` type alias
4. Implement SQLite storage engine (connection, WAL, pragmas, mmap)
5. Implement core schema creation (memories table, FTS5, triggers)
6. Implement `femind_meta` table and migration framework
7. Implement CRUD operations (store, get, update, delete) with SHA-256 dedup
8. Implement FTS5 keyword search with Porter stemming
9. Implement `SearchBuilder` fluent API
10. Implement `MemoryEngine<T>` and `MemoryEngineBuilder`
11. Implement thread safety (Mutex writer, connection pool readers)
12. Write integration tests for all Phase 1 functionality

---

### Phase 2: Scoring + Context Assembly

Post-search scoring and token-budget context assembly. After this phase, FeMind can rank results intelligently and produce LLM-ready context.

**Deliverables:**
- `ScoringStrategy` trait (takes `&MemoryMeta`)
- Shipped scorers: `RecencyScorer`, `ImportanceScorer`, `CategoryScorer`, `MemoryTypeScorer`
- `CompositeScorer` for combining strategies multiplicatively
- `ScoredResult<T>` struct with score breakdown
- `ContextBudget` struct and priority constants
- `ContextItem` and `ContextAssembly` structs
- `engine.assemble_context()` method
- Builder integration: `.scoring()` on `MemoryEngineBuilder`
- Tests for each scorer, composite scoring, context assembly budget limits

**Feature flags active:** `fts5` (default)

**Tasks (Phase 2):**
13. Implement `ScoringStrategy` trait and `ScoredResult<T>` struct
14. Implement `RecencyScorer` with configurable half-life
15. Implement `ImportanceScorer` and `CategoryScorer`
16. Implement `MemoryTypeScorer` with per-type weights
17. Implement `CompositeScorer` for strategy composition
18. Implement `ContextBudget`, `ContextItem`, `ContextAssembly`
19. Implement `engine.assemble_context()` with priority-ranked selection
20. Integrate scoring into `SearchBuilder.execute()` pipeline
21. Write tests for scoring strategies and context assembly

---

### Phase 3: Consolidation + Hierarchy

Hash-based dedup on store, three-tier memory hierarchy, and basic consolidation. After this phase, FeMind prevents duplicates and supports tiered memory.

**Deliverables:**
- `ConsolidationStrategy` trait (takes `&MemoryMeta`)
- `ConsolidationAction` enum (Add, Update, Delete, Noop)
- `HashDedup` implementation (SHA-256, zero cost)
- `StoreResult` enum reporting what action was taken
- `store()` runs consolidation before insert
- `tier` column (0-2) on memories table
- `source_ids` column for provenance
- `SearchDepth` enum (Standard, Deep, Forensic)
- `.depth()` on `SearchBuilder`
- `.tier()` filter on `SearchBuilder`
- Builder integration: `.consolidation()` on `MemoryEngineBuilder`
- Tests for hash dedup, tier filtering, search depth

**Feature flags active:** `fts5` + `consolidation` (new)

**Tasks (Phase 3):**
22. Implement `ConsolidationStrategy` trait, `ConsolidationAction`, `StoreResult`
23. Implement `HashDedup` (SHA-256 exact match)
24. Integrate consolidation into `store()` pipeline
25. Add `tier` and `source_ids` columns to schema (migration v1→v2)
26. Implement `SearchDepth` enum and tier-aware search filtering
27. Write tests for consolidation and tier-aware search

---

### Phase 4: Activation Model

ACT-R cognitive decay model. After this phase, memories naturally fade or strengthen based on access patterns.

**Deliverables:**
- `memory_access_log` table with index
- `activation_cache` and `activation_updated` columns on memories
- `record_access()` — logs retrieval to access log
- `compute_activation()` — ACT-R formula with per-type decay rates
- `ActivationScorer` implementing `ScoringStrategy`
- Access log compaction (90-day aggregation)
- Activation cache refresh (on access + periodic)
- `engine.search().execute()` automatically records access for returned results
- Tests for activation computation, decay rates, compaction, cache

**Feature flags active:** `fts5` + `activation-model` (new)

**Tasks (Phase 4):**
28. Add `memory_access_log` table and `activation_cache` columns (migration v2→v3)
29. Implement `record_access()` and `compute_activation()` with ACT-R formula
30. Implement `ActivationScorer` with per-type decay rates
31. Integrate access recording into search result pipeline
32. Implement access log compaction (90-day aggregation)
33. Implement activation cache with incremental refresh
34. Write tests for activation model, decay, compaction

---

### Phase 5: Vector Search + Hybrid RRF

Candle embedding backend, brute-force vector search, and RRF hybrid merge. After this phase, FeMind has semantic search.

**Deliverables:**
- `EmbeddingBackend` trait (async fn embed, embed_batch, dimensions, model_name)
- `CandleNativeBackend` — all-MiniLM-L6-v2 via BERT (~100-130 lines)
- Mean pooling and L2 normalization (shared `pooling.rs`)
- `NoopBackend` for testing
- `FallbackBackend` wrapping `Option<Box<dyn EmbeddingBackend>>`
- Model auto-download via hf-hub, cached at `~/.cache/femind/models/`
- `from_path()` constructor for offline/bundled models
- `memory_vectors` table (embedding BLOB, model_name, dimensions, content_hash)
- Brute-force dot product vector scan
- Vector search filtered by `model_name` match (Decision 020)
- `SearchMode::Vector`, `SearchMode::Hybrid`, `SearchMode::Auto`
- RRF merge with dynamic k-values (quoted → keyword, questions → semantic)
- `EmbeddingIndexer` for background batch embedding
- Content-hash skip for unchanged content
- Background indexer on dedicated OS thread
- Matryoshka dimension override support
- Builder integration: `.embedding_backend()` on builder
- Integration tests: embed, search, RRF merge, model mismatch fallback, background indexing

**Feature flags active:** `fts5` + `vector-search` (new, pulls tokio)

**Tasks (Phase 5):**
35. Implement `EmbeddingBackend` trait
36. Implement `pooling.rs` (mean pooling + L2 normalization)
37. Implement `CandleNativeBackend` with all-MiniLM-L6-v2 via BERT
38. Implement model auto-download via hf-hub with `from_path()` fallback
39. Implement `NoopBackend` and `FallbackBackend`
40. Add `memory_vectors` table (migration v3→v4)
41. Implement brute-force vector search with model_name filtering
42. Implement RRF merge with dynamic k-values
43. Implement `SearchMode::Vector`, `Hybrid`, `Auto` in SearchBuilder
44. Implement `EmbeddingIndexer` with batch processing and content-hash skip
45. Implement background indexer on dedicated OS thread
46. Implement Matryoshka dimension override
47. Write integration tests for vector search, RRF, fallback, background indexing

---

### Phase 6: Graph Memory

SQLite relationship tables with recursive CTE traversal. After this phase, FeMind supports memory relationships and multi-hop queries.

**Deliverables:**
- `memory_relations` table with indexes
- `RelationType` enum (CausedBy, SolvedBy, DependsOn, SupersededBy, RelatedTo, PartOf, ConflictsWith, ValidatedBy, Custom)
- `engine.relate()` — create relationships
- `engine.related()` — graph traversal via recursive CTEs
- Cycle prevention in CTE queries
- Depth-based scoring boost for connected memories
- Temporal validity on relationships (valid_from/valid_until)
- Graph traversal integrated into search pipeline (boost connected results)
- Tests for relationships, traversal, cycle prevention, depth scoring

**Feature flags active:** `fts5` + `graph-memory` (new)

**Tasks (Phase 6):**
48. Add `memory_relations` table with indexes (migration v4→v5)
49. Implement `RelationType` enum
50. Implement `engine.relate()` for creating relationships
51. Implement recursive CTE traversal with cycle prevention and depth limits
52. Implement `engine.related()` method
53. Integrate graph traversal boost into search pipeline
54. Write tests for graph relationships, traversal, cycles

---

### Phase 7: Advanced Features

Temporal validity, time-aware query expansion, two-tier memory, similarity dedup, cross-encoder reranking, and exhaustive search.

**Deliverables:**
- Temporal validity: `valid_from`/`valid_until` on MemoryRecord, `.valid_at()` on SearchBuilder
- Time-aware query expansion (`query_expand.rs`) — temporal expressions to date filters
- Two-tier memory: global (`~/.femind/global.db`) + project (`./.femind/memory.db`)
- Global/project merge with project scoring boost
- Promotion logic (N projects → global)
- `SimilarityDedup` consolidation strategy (requires vector-search)
- `SearchMode::Exhaustive { min_score }` for aggregation queries
- `CandleReranker` cross-encoder (ms-marco-MiniLM-L-6-v2)
- `RerankerBackend` trait
- Reranking step integrated into search pipeline
- Tests for all features

**Feature flags active:** `fts5` + `temporal` + `two-tier` + `reranking` (new flags)

**Tasks (Phase 7):**
55. Implement temporal validity fields and `.valid_at()` search filter
56. Implement time-aware query expansion (regex-based temporal pattern extraction)
57. Implement two-tier memory manager (global + project databases)
58. Implement global/project merge with project scoring boost
59. Implement promotion logic (cross-project access tracking)
60. Implement `SimilarityDedup` consolidation (vector similarity threshold)
61. Implement `SearchMode::Exhaustive` (bypass top-k limits)
62. Implement `CandleReranker` cross-encoder via candle BERT
63. Implement `RerankerBackend` trait and integrate into search pipeline
64. Write tests for temporal, two-tier, similarity dedup, reranking, exhaustive

---

### Phase 8: LLM-Assisted Features

LlmCallback trait and all features that depend on it. After this phase, consumers can plug in LLMs for intelligent consolidation, fact extraction, evolution, and reflection.

**Deliverables:**
- `LlmCallback` trait (async complete)
- `IngestStrategy` trait with `PassthroughIngest` and `LlmIngest`
- `LLMConsolidation` strategy
- `EvolutionStrategy` trait and `EvolutionAction` enum
- Post-write evolution pipeline (store → retrieve similar → evaluate → update)
- `engine.consolidate()` — episodic-to-semantic consolidation with LLM
- `engine.prune()` — activation-based pruning with `PruningPolicy`
- `engine.maintain()` — consolidation + pruning convenience method
- `engine.reflect()` — synthesize insights from memory clusters (Decision 018)
- All LLM-dependent features degrade gracefully when no callback provided
- Tests with mock LlmCallback

**Feature flags active:** All features

**Tasks (Phase 8):**
65. Implement `LlmCallback` trait
66. Implement `IngestStrategy` trait, `PassthroughIngest`, and `LlmIngest`
67. Implement `LLMConsolidation` strategy
68. Implement `EvolutionStrategy` trait and post-write evolution pipeline
69. Implement `engine.consolidate()` with LLM-assisted tier promotion
70. Implement `engine.prune()` with `PruningPolicy`
71. Implement `engine.maintain()` convenience method
72. Implement `engine.reflect()` for insight synthesis
73. Write tests for all LLM-assisted features with mock callback

---

### Phase 9: Encryption + MCP + Polish

Optional encryption, MCP server interface, and production hardening.

**Deliverables:**
- `EncryptionKey` enum (Passphrase, RawKey)
- SQLCipher integration via `bundled-sqlcipher` feature
- `PRAGMA key` first statement on encrypted connections
- Optional `keychain` module via `keyring` crate
- MCP server interface (feature-gated behind `mcp-server`)
- MCP tools: search, store, get, delete, relate, context
- `full` feature flag verified: all features compile together
- Rustdoc for all public types and methods
- README.md with usage examples, feature flag guide, performance targets
- WASM conditional compilation stubs (`cfg(target_family = "wasm")`)
- Clippy clean (`cargo clippy --features full -- -D warnings`)
- All features tested in isolation AND combined
- Final `cargo publish` to crates.io as v0.2.0

**Feature flags active:** All features including `encryption`, `keychain`, `mcp-server`

**Tasks (Phase 9):**
74. Implement `EncryptionKey` and SQLCipher integration
75. Implement `keychain` module via `keyring` crate
76. Implement MCP server interface with axum
77. Implement MCP tools (search, store, get, delete, relate, context)
78. Add WASM conditional compilation stubs
79. Write rustdoc for all public types and methods
80. Write README.md with usage examples and feature flag guide
81. Run `cargo clippy --features full`, fix all warnings
82. Run full test suite with `cargo test --features full`
83. Cross-feature integration tests (encryption + vector + graph + activation)
84. Publish v0.2.0 to crates.io

---

## 4. Task Dependencies

```
Phase 1 (tasks 1-12):  Foundation — no dependencies
Phase 2 (tasks 13-21): Scoring — depends on Phase 1
Phase 3 (tasks 22-27): Consolidation — depends on Phase 1
Phase 4 (tasks 28-34): Activation — depends on Phase 2 (needs ScoringStrategy)
Phase 5 (tasks 35-47): Vector — depends on Phase 1
Phase 6 (tasks 48-54): Graph — depends on Phase 1
Phase 7 (tasks 55-64): Advanced — depends on Phases 2, 3, 5, 6
Phase 8 (tasks 65-73): LLM — depends on Phases 3, 5, 6, 7
Phase 9 (tasks 74-84): Polish — depends on all previous phases
```

Phases 3, 4, 5, 6 can run in parallel after Phase 2 completes (they're independent feature flags). Phase 7 unifies them. Phase 8 adds LLM features on top. Phase 9 polishes and ships.

---

## 5. Quality Gates

Each phase must pass before the next begins:

| Gate | Criteria |
|------|----------|
| **Build** | `cargo build --features <phase-flags>` succeeds with zero warnings |
| **Test** | `cargo test --features <phase-flags>` — all tests pass |
| **Clippy** | `cargo clippy --features <phase-flags> -- -D warnings` — zero warnings |
| **Doc** | `cargo doc --features <phase-flags> --no-deps` — no broken links |

---

## 6. Constraints

- **No `unsafe`** except for candle's `VarBuilder::from_mmaped_safetensors` (upstream requirement)
- **No `unwrap()`** in library code — all errors propagated via `FemindError`
- **No `anyhow`** — structured errors only (`thiserror`)
- **No `println!`** — all logging via `tracing`
- **No panics** — library must never panic on valid input
- **Minimum Rust 1.85** — edition 2024, native async traits
- **Feature isolation** — each feature flag must compile independently
- **Zero cost when unused** — disabled features add zero binary size and zero runtime overhead

---

## 7. Testing Strategy

| Level | Scope | Location |
|-------|-------|----------|
| Unit | Individual functions (scoring formulas, RRF merge, activation computation) | Inline `#[cfg(test)]` modules |
| Integration | Multi-component workflows (store → search → score → assemble) | `tests/` directory |
| Feature isolation | Each feature flag compiles and tests independently | CI matrix |
| Cross-feature | All features enabled simultaneously | `cargo test --features full` |
| Performance | Latency targets met (FTS5 <5ms, vector <50ms) | `benches/` with criterion |

---

## 8. File Reference

| Spec Section | Architecture Reference |
|-------------|----------------------|
| Phase 1: Storage | `ARCHITECTURE.md` → Storage Layer, Core Schema |
| Phase 1: CRUD | `ARCHITECTURE.md` → Public API, MemoryEngine |
| Phase 2: Scoring | `ARCHITECTURE.md` → ScoringStrategy, Context Assembly |
| Phase 3: Consolidation | `ARCHITECTURE.md` → Consolidation Pipeline |
| Phase 4: Activation | `ARCHITECTURE.md` → Activation Model |
| Phase 5: Vector | `ARCHITECTURE.md` → Embedding Module, Search Pipeline |
| Phase 6: Graph | `ARCHITECTURE.md` → Graph Traversal |
| Phase 7: Advanced | `ARCHITECTURE.md` → Two-Tier, Temporal, Reranking |
| Phase 8: LLM | `ARCHITECTURE.md` → LlmCallback, IngestStrategy, Evolution |
| Phase 9: Encryption | `ARCHITECTURE.md` → Encryption Configuration |
