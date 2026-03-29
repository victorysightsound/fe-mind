# Vector Search Production Readiness — Spec

## Goal

Make FeMind's vector search and embedding pipeline production-ready: correct,
efficient, well-tested, and performant enough for real workloads (LongMemEval-S:
~800 memories per question, 10 questions = 8000 embeddings + searches).

## Current State

- CandleNativeBackend exists (all-MiniLM-L6-v2, 384-dim, BERT)
- Vector storage/search is brute-force O(n) — fine for <100K
- Hybrid search (FTS5 + vector + RRF) is implemented
- FTS5 sanitizer has a bug with hyphens (fix in progress)
- Engine.store() now computes embeddings inline (just added, untested)
- No batch embedding optimization — each store() calls embed() individually
- Debug-mode Candle inference is ~100x slower than release — unusable for benchmarks
- No integration test covers the full store→embed→hybrid-search pipeline
- embedding_status column in memories table is never updated

## Success Criteria

1. `cargo test --features vector-search` passes all tests (existing + new)
2. Full pipeline integration test: ingest 50+ memories → hybrid search → retrieves correct results
3. Batch embedding via `embed_batch()` for throughput (≥10x faster than sequential for bulk ingest)
4. Dev profile builds fast enough for iteration (opt-level override for candle deps)
5. FTS5 sanitizer handles all special characters without errors
6. embedding_status tracked correctly (pending → success/failed)
7. Performance: embed + store 100 memories in <5s (release mode)
8. All edge cases handled: empty text, very long text, duplicate content

---

## Phase 1: Performance Foundations

### 1.1 — Dev profile opt-level override for candle deps

Add `[profile.dev.package."*"]` or targeted overrides in workspace Cargo.toml
so that candle-core, candle-nn, candle-transformers, and tokenizers compile
with `opt-level = 2` even in dev builds. This makes embedding inference
usable during development (~10-50x speedup).

### 1.2 — Batch embedding API on MemoryEngine

Add `store_batch(&[T]) -> Result<Vec<StoreResult>>` to MemoryEngine that:
1. Stores all records via SQL (existing store logic)
2. Collects texts from Added results
3. Calls `embed_batch()` once
4. Stores all vectors in a single transaction

This is the critical performance optimization — batching amortizes model
overhead across many inputs.

### 1.3 — Dedup check before embedding

Before computing an embedding, check `VectorSearch::vector_exists(content_hash)`.
Skip embedding if the vector already exists. Prevents wasted inference on
duplicate content.

---

## Phase 2: Correctness & Hardening

### 2.1 — FTS5 sanitizer: replace all hyphens with spaces

Already implemented — needs a test specifically for hyphenated queries like
"faith-related", "well-known", "self-driving".

### 2.2 — Update embedding_status in memories table

After embedding succeeds: `UPDATE memories SET embedding_status = 'success' WHERE id = ?`
After embedding fails: `UPDATE memories SET embedding_status = 'failed' WHERE id = ?`
This enables consumers to audit coverage.

### 2.3 — Edge case handling

- Empty or whitespace-only text: skip embedding (log warning)
- Very long text (>8192 tokens): truncate to model's context window
- Zero-length embedding result: treat as failure

---

## Phase 3: Integration Tests

### 3.1 — Full pipeline integration test

Test using NoopBackend (no model download needed):
1. Create engine with NoopBackend
2. Store 20+ memories across different categories
3. Search with Auto mode → verify hybrid path taken
4. Verify results contain expected memories

### 3.2 — Full pipeline test with CandleNativeBackend

Feature-gated test (`#[cfg(feature = "local-embeddings")]`):
1. Create engine with CandleNativeBackend
2. Store 10 memories with distinct topics
3. Hybrid search → verify semantic similarity ranking
4. Verify a query semantically similar (but keyword-different) finds the right memory

### 3.3 — Batch store integration test

Test store_batch with 50+ records, verify:
- All records stored
- All embeddings computed
- Hybrid search finds results
- Performance assertion (< N seconds in release)

### 3.4 — Distractor scenario test

Simulate LongMemEval-S:
1. Store 40 sessions of conversation (mix of relevant + distractors)
2. Query for specific information buried in one session
3. Verify hybrid search surfaces the relevant memory in top-5

---

## Phase 4: Polish & Instrumentation

### 4.1 — Embedding latency tracing

Add `tracing::debug!` spans for:
- Individual embed() calls with duration
- Batch embed_batch() with count + duration
- Vector store_vector() calls

### 4.2 — Coverage query helper

Add `MemoryEngine::embedding_coverage() -> Result<(u64, u64)>` returning
(memories_with_embeddings, total_memories). Useful for diagnostics.

### 4.3 — Ensure all existing tests pass

Run full test suite with `--features vector-search` and fix any regressions.
