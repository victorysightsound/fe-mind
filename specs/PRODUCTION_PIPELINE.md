# femind Production Pipeline — Complete Build Spec

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
- EngineConfig toggles wired into store, search, and extraction paths
- AssemblyConfig — diversification, recency, graph_depth, search_limit
- Graph infrastructure — memory_relations table, GraphMemory CRUD, traversal
- Hybrid search — FTS5 OR-mode + vector + RRF + multi-query
- Embedding backends — CandleNativeBackend, ApiBackend, FallbackBackend
- Retrieval test harness — zero-cost search quality measurement
- Extraction test harness — LLM extraction quality measurement
- ANN runtime modes — exact, ann, and off now select real runtime behavior
- Pipeline test harness — structure only, no CLI subcommand
- Session cache — raw dataset storage
- Embedding cache — per chunk size

### Remaining Work
1. Live CLI/API LLM validation — approval required before running it
2. Rename and integrate the crate/repo as `femind`
3. Optional pipeline harness completion in downstream tooling

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
- "exact" → brute-force vector search
- "ann" → use the shared ANN index, rebuilding it when vectors change and falling back to exact if needed

**A8.** Add `cli-llm` feature flag to Cargo.toml

**A9.** Make extraction LLM configurable:
- `femind-extract` system in recallbench should accept --extract-model flag
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
**C2.** Implement ANN search mode with shared index lifecycle and fallback to exact
**C3.** Add `ann` feature flag to Cargo.toml
**C4.** Unit tests comparing ANN vs exact results

### Phase D: Pipeline Test Harness Completion

**D1.** Add `pipeline-test` CLI subcommand to recallbench
**D2.** Implement modular pipeline: each step toggleable
**D3.** Result saving with config snapshot (like retrieval test)

### Phase E: Documentation

**E1.** Update ARCHITECTURE.md with complete pipeline
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
    pub vector_search_mode: VectorSearchMode,  // exact, ann, off
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
cli-llm             # CliLlmCallback (Claude/ChatGPT CLI)
llm-ingest          # LlmIngest extraction strategy
vector-search       # local-embeddings + tokio
graph-memory        # graph tables (always created, this gates logic)
ann                 # ANN vector indexing via instant-distance HNSW
```
