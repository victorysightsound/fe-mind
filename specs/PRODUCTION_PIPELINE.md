# MindCore Production Pipeline — Complete Build Spec

## Goal

Build the complete production-ready memory engine with every feature
independently toggleable. Build first, test later. No live LLM testing
until the build is complete and approved.

## Current State (2026-03-25)

### Built ✓
- LlmCallback trait (src/traits/llm.rs)
- ApiLlmCallback — OpenAI-compatible /v1/chat/completions via ureq
- CliLlmCallback — Claude/ChatGPT/Gemini via CLI
- LlmIngest extraction — prompts + parser (src/ingest/llm_extract.rs)
- Fact extraction — regex parser for structured facts (src/ingest/fact_extraction.rs)
- store_with_extraction() — extracts facts, stores, creates graph edges
- EngineConfig struct — has fields but NOT wired into operations
- AssemblyConfig — diversification, recency, graph_depth, search_limit
- Graph infrastructure — memory_relations table, GraphMemory CRUD, traversal
- Hybrid search — FTS5 OR-mode + vector + RRF + multi-query
- Embedding backends — CandleNativeBackend, ApiBackend, FallbackBackend
- Retrieval test harness — zero-cost search quality measurement
- Extraction test harness — LLM extraction quality measurement
- Pipeline test harness — structure only, no CLI subcommand
- Session cache — raw dataset storage
- Embedding cache — per chunk size

### NOT Built ✗
1. EngineConfig toggles not wired into store() — embedding always runs
2. EngineConfig toggles not wired into store_with_extraction() — graph/dedup always run
3. EngineConfig toggles not wired into search — graph filtering always checks depth from AssemblyConfig, not EngineConfig.graph_enabled
4. cli-llm feature flag missing from Cargo.toml
5. ANN indexing — not started (vector_search_mode field exists but does nothing)
6. Pipeline test harness — no CLI subcommand, no actual implementation
7. Extraction model not configurable — hardcoded in main.rs
8. store_with_extraction() ExtractionResult doesn't match spec (returns llm_extract::ExtractionResult, not the spec's struct)
9. Architecture documentation not updated

## Build Tasks

### Phase A: Wire Feature Toggles (no LLM, no API, pure code)

**A1.** Wire `config.embedding_enabled` into `store()`:
- If false, skip embedding computation after storing memory
- FTS5 trigger still fires (text search always works)

**A2.** Wire `config.embedding_enabled` into `store_batch()`:
- Same as A1 but for the batch path

**A3.** Wire `config.embedding_enabled` into `store_with_extraction()`:
- If false, skip embedding for extracted facts

**A4.** Wire `config.graph_enabled` into `store_with_extraction()`:
- If false, skip graph edge creation after extraction

**A5.** Wire `config.dedup_enabled` into `store()` and `store_with_extraction()`:
- If false, skip content hash dedup check (allow duplicates)

**A6.** Wire `config.graph_enabled` into `multi_query_search()`:
- If false, skip graph filtering step even if AssemblyConfig.graph_depth > 0
- EngineConfig.graph_enabled is the master switch

**A7.** Wire `config.vector_search_mode` into hybrid search:
- "off" → skip vector search entirely, FTS5 only
- "exact" → current brute-force (default for now)
- "ann" → placeholder that falls back to exact until ANN is built

**A8.** Add `cli-llm` feature flag to Cargo.toml

**A9.** Make extraction LLM configurable:
- `mindcore-extract` system in recallbench should accept --extract-model flag
- Remove hardcoded Llama model
- Default to Haiku via CLI when no API model specified

### Phase B: Fix API Contracts

**B1.** Align store_with_extraction() return type with spec:
- Return ExtractionResult with: facts_extracted, memories_stored,
  duplicates_skipped, graph_edges_created, superseded_count

**B2.** Add store_with_extraction() to handle large text splitting internally:
- Already partially done, verify and add unit test

**B3.** Consolidate AssemblyConfig fields into EngineConfig:
- EngineConfig should hold recency_weight and diversification_limit
  directly (not nested in assembly sub-struct)
- Or keep AssemblyConfig but derive it from EngineConfig at search time

### Phase C: ANN Indexing

**C1.** Research sqlite-vec Rust bindings availability and maturity
**C2.** Implement ANN search mode with fallback to exact
**C3.** Add `ann` feature flag to Cargo.toml
**C4.** Unit tests comparing ANN vs exact results

### Phase D: Pipeline Test Harness Completion

**D1.** Add `pipeline-test` CLI subcommand to recallbench
**D2.** Implement modular pipeline: each step toggleable
**D3.** Result saving with config snapshot (like retrieval test)

### Phase E: Documentation

**E1.** Update MINDCORE_ARCHITECTURE.md with complete pipeline
**E2.** Update STATUS doc with final build state
**E3.** Document all feature flags and their interactions

---

## GATE: Live LLM Testing

After all build phases complete, STOP and get user approval before:
- Running extraction on MAB with any LLM model
- Running extraction on LongMemEval with any LLM model
- Running any benchmark with LLM generation/judging

The model choice for extraction must be approved by the user.

---

## Feature Toggles Reference

### EngineConfig (runtime, per-engine)
```rust
pub struct EngineConfig {
    pub embedding_enabled: bool,     // A1-A3: skip embedding when false
    pub graph_enabled: bool,         // A4, A6: skip graph create/query when false
    pub dedup_enabled: bool,         // A5: skip dedup check when false
    pub vector_search_mode: String,  // A7: "ann", "exact", "off"
    pub assembly: AssemblyConfig,    // search-time config
}
```

### AssemblyConfig (search-time, per-query)
```rust
pub struct AssemblyConfig {
    pub max_per_session: usize,  // diversification limit
    pub recency_boost: f32,      // 0.0 = off, 0.3 = moderate
    pub search_limit: usize,     // multi-query result limit
    pub graph_depth: u32,        // graph traversal hops (0 = off)
}
```

### Cargo Feature Flags (compile-time)
```toml
local-embeddings    # CandleNativeBackend (MiniLM local)
api-embeddings      # ApiBackend (DeepInfra embedding API)
api-llm             # ApiLlmCallback (DeepInfra/OpenAI chat API)
cli-llm             # CliLlmCallback (Claude/ChatGPT CLI) — MISSING, add
llm-ingest          # LlmIngest extraction strategy
vector-search       # local-embeddings + tokio
graph-memory        # graph tables (always created, this gates logic)
ann                 # ANN vector indexing — NOT BUILT YET
```
