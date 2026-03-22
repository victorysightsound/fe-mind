# MiniLM Migration + API Embedding Backend

## Goal

Switch from granite-small-r2 (ModernBERT, native-only) to all-MiniLM-L6-v2
(BERT, works everywhere). Add an OpenAI-compatible API embedding backend.
Build a fallback wrapper that tries API first, falls back to local.

One model everywhere: native, WASM, API. Same vectors, interchangeable.

## Changes Required

### 1. Switch CandleNativeBackend to MiniLM

- MODEL_REPO: "sentence-transformers/all-MiniLM-L6-v2"
- MODEL_NAME: "all-MiniLM-L6-v2"
- DIMENSIONS: 384 (same)
- Architecture: BERT (not ModernBERT) — change candle import from
  modernbert to bert
- Model loading: BERT uses different config/weight structure than ModernBERT
- Mean pooling + L2 normalization stays the same
- The tensor key renaming may differ (check sentence-transformers naming)

### 2. Add ApiBackend

New file: embeddings/api.rs
- Takes base_url, api_key, model_name
- Implements EmbeddingBackend trait
- POST to `{base_url}/embeddings` with OpenAI-compatible format:
  `{"model": "...", "input": ["text1", "text2"]}`
- Parse response: `{"data": [{"embedding": [...]}]}`
- Uses reqwest (blocking client for sync trait)
- Feature-gated behind `api-embeddings` feature flag
- api_key can be provided directly or via a command (like recallbench's
  api_key_cmd pattern)

### 3. Update FallbackBackend

Extend to support API-first-then-local pattern:
- `FallbackBackend::api_with_local_fallback(api, local)` constructor
- Try API embed() first; if it fails, try local
- Log when falling back

### 4. Add reqwest dependency

Optional, feature-gated behind `api-embeddings`:
- reqwest = { version = "0.12", features = ["json", "blocking"], optional = true }

### 5. Update Cargo.toml features

- Rename or update `local-embeddings` to use BERT instead of ModernBERT
  (candle deps stay the same, just different model import)
- Add `api-embeddings = ["dep:reqwest"]`
- Update `vector-search` to include both options
- Update `full` feature

### 6. Update tests

- candle_e2e.rs: change model_name assertion from granite to MiniLM
- All tests should pass with the new model
- Add API backend test (mock or integration)

### 7. Update documentation

- README.md: update model references
- specs: update references to granite
- Cargo.toml description if needed

### 8. Update recallbench adapter

- Remove granite-specific references
- Support configuring API vs local backend

## Success Criteria

1. `cargo test --features vector-search` passes all tests
2. CandleNativeBackend loads and embeds with all-MiniLM-L6-v2
3. ApiBackend works with DeepInfra endpoint
4. FallbackBackend tries API, falls back to local
5. All existing search/retrieval tests pass
6. Distractor scenario test still passes
7. WASM compatibility restored (standard BERT, no ModernBERT)
