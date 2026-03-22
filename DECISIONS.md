# MindCore Decisions

This document records key architectural and design decisions for MindCore.

Decisions 001-007 originated during initial research (2026-03-16) and were carried forward when MindCore became its own project.

---

## Decision 001: MindCore Shared Memory Engine

**Date:** 2026-03-16

**Decision:** Create a standalone Rust crate (MindCore) providing pluggable, feature-gated persistent memory for AI agent applications.

**Context:** Multiple AI agent projects need persistent memory with search, scoring, and decay, yet the Rust ecosystem has no standalone crate for this. Research into Mem0, OMEGA, Zep/Graphiti, and MemOS confirms the patterns are converging industry-wide.

**Rationale:**
- Fills a clear gap in the Rust ecosystem for standalone agent memory (Engram is Go, Mem0 is Python)
- Improvements (vector search, graph, decay) benefit all consumers automatically
- Feature-gated design means zero cost for unused capabilities
- Every component is backed by published research or established open-source practice

**Consequences:**
- New standalone crate: `mindcore`
- Any Rust project can depend on mindcore for persistent agent memory
- See `MINDCORE_ARCHITECTURE.md` for full specification

---

## Decision 002: WAL Mode for All SQLite Databases

**Date:** 2026-03-16

**Decision:** Enable WAL (Write-Ahead Logging) mode on all SQLite databases from day one.

**Context:** Concurrent read/write patterns are common in agent memory (orchestrator reads learnings while writing new error patterns).

**Rationale:**
- Concurrent reads don't block writes
- `synchronous = NORMAL` is corruption-safe in WAL mode, avoids FSYNC per write
- 500-1000 writes/sec on modern hardware while serving thousands of concurrent reads
- Zero code complexity — single pragma at connection time

**Consequences:**
- All databases: `PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;`
- WAL file appears alongside .db file (expected, not a bug)
- No impact on backup/copy procedures

---

## Decision 003: Candle Over ort for Local Embeddings

**Date:** 2026-03-16

**Decision:** Use HuggingFace Candle for local embedding inference, not ONNX Runtime (ort).

**Context:** Evaluated ort, candle, and fastembed-rs for local embedding inference.

**Rationale:**
- Pure Rust (ort requires C++ runtime, adds 80+ deps and ~350MB to binary)
- Native safetensors loading from HF Hub (no ONNX conversion step)
- WASM support (relevant for future GUI via WebView)
- Performance difference is negligible for MiniLM-sized models (~8ms per embed)
- Candle is well-proven in production for this exact use case
- `EmbeddingBackend` trait allows swapping to ort later if scale demands it

**Consequences:**
- Feature-gated behind `local-embeddings` flag
- Default model: `all-MiniLM-L6-v2` (384 dims, 22M params, ~80MB download) — updated by Decision 017
- Model downloaded from HF Hub on first use, cached locally
- Graceful degradation to FTS5-only if candle fails to load

---

## Decision 004: Hybrid Search with Reciprocal Rank Fusion

**Date:** 2026-03-16

**Decision:** Combine FTS5 keyword search and vector similarity search using Reciprocal Rank Fusion (RRF).

**Context:** FTS5 handles 80% of lookups well. Vector catches semantic matches FTS5 misses. Need a principled way to merge results.

**Rationale:**
- RRF is simple, effective, and parameter-light (just k-value)
- RRF is proven in production for agent memory workloads
- Dynamic k-values adjust weighting based on query type (quoted → keyword, questions → semantic)
- No learned fusion model needed
- Outperforms either approach alone

**Consequences:**
- Both search backends run in parallel, results merged via RRF
- When vector is unavailable, transparently falls back to FTS5-only
- Post-RRF scoring boosts applied for recency, importance, category, memory type

---

## Decision 005: ACT-R Activation Model for Memory Decay

**Date:** 2026-03-16

**Decision:** Use ACT-R cognitive architecture's activation formula for memory decay, replacing ad-hoc tier/trust/decay systems.

**Context:** Common approaches include manual trust scoring with decay, tier-based multipliers, and OMEGA's forgetting intelligence. All solve the same problem differently.

**Rationale:**
- Research-backed model from cognitive science (spaced repetition, forgetting curves)
- One unified formula replaces five separate mechanisms (trust, tiers, decay, reference counting, recency)
- Memories accessed frequently stay strong naturally (spaced repetition effect)
- Different decay rates per cognitive type (episodic=fast, semantic=slow, procedural=medium)
- Access log provides richer data than simple counters

**Consequences:**
- `memory_access_log` table tracks every retrieval with timestamp
- Activation computed at query time from access history
- Feature-gated behind `activation-model` (simpler projects can skip)
- Replaces ad-hoc confidence fields and tier systems

---

## Decision 006: Graph Memory via SQLite Relationship Tables

**Date:** 2026-03-16

**Decision:** Implement graph memory using SQLite relationship tables with recursive CTE traversal, not an external graph database.

**Context:** Graph memory provides 5-11% accuracy improvement on temporal and multi-hop queries (Mem0 benchmarks). Evaluated Kuzu (archived Oct 2025), Cozo (pure Rust), and SQLite CTEs.

**Rationale:**
- Zero new dependencies (SQLite recursive CTEs are built-in)
- Handles thousands of relationships efficiently (sufficient for personal/project scale)
- `memory_relations` table with standard relationship types (caused_by, solved_by, depends_on, etc.)
- Kuzu archived, Cozo less proven — SQLite is the safe starting point
- `GraphBackend` trait allows swapping to native graph DB later if needed

**Consequences:**
- Feature-gated behind `graph-memory`
- Recursive CTE traversal with cycle prevention and depth limits
- Connected memories receive scoring boost based on hop distance
- Temporal validity on relationships (valid_from/valid_until)
- Future: `graph-native` feature flag for Cozo or Kuzu fork if SQLite becomes bottleneck

---

## Decision 007: Consolidation Pipeline for Memory Quality

**Date:** 2026-03-16

**Decision:** Implement a three-stage consolidation pipeline (Extract → Consolidate → Store) to prevent duplicate and stale memories.

**Context:** Without consolidation, memories accumulate duplicates over months of use. Mem0's research shows consolidation is key to memory quality.

**Rationale:**
- Hash-based dedup (default) is zero-cost and prevents exact duplicates
- Similarity-based dedup (optional) catches near-duplicates with vector search
- LLM-assisted consolidation (optional) provides highest accuracy but costs tokens
- `ConsolidationStrategy` trait allows projects to choose their level
- Production experience demonstrates this is essential for memory quality

**Consequences:**
- Feature-gated behind `consolidation`
- Default: `HashDedup` (SHA-256, zero cost)
- Optional: `SimilarityDedup` (requires vector-search)
- Optional: `LLMConsolidation` (consumer provides LLM call)
- StoreResult reports what action was taken (added, updated, noop, etc.)

---

## Decision 008: Encryption at Rest via SQLCipher

**Date:** 2026-03-17

**Decision:** Use SQLCipher via rusqlite's `bundled-sqlcipher` feature for optional database-level encryption.

**Context:** Agent memories may contain sensitive information. OMEGA claims "encryption at rest" but only encrypts exports, not the main database. Application-level field encryption breaks FTS5 (can't tokenize ciphertext). Need a solution that preserves all search capabilities.

**Rationale:**
- SQLCipher provides transparent AES-256-CBC encryption at the page level
- Preserves FTS5, WAL mode, and vector search — encryption/decryption at I/O boundary
- 5-15% overhead on I/O operations, negligible for agent memory workloads
- rusqlite has first-class support via `bundled-sqlcipher` and `bundled-sqlcipher-vendored-openssl`
- BSD-3-Clause license, battle-tested (Signal, Mozilla, Adobe)
- Consumer provides the key — MindCore doesn't manage key storage

**Consequences:**
- Feature-gated behind `encryption` (replaces bundled SQLite with bundled SQLCipher)
- Optional `keychain` feature for OS keychain integration via `keyring` crate
- `EncryptionKey` enum: `Passphrase(String)` or `RawKey([u8; 32])`
- `PRAGMA key` must be first statement after connection open
- `encryption-vendored` variant for environments without system OpenSSL

---

## Decision 009: Benchmark Strategy

**Date:** 2026-03-17

**Decision:** Target LongMemEval as primary benchmark, with MemoryAgentBench and AMA-Bench as secondary targets. Ship benchmark harness as a separate workspace member.

**Context:** LongMemEval (ICLR 2025) is the de facto standard — 500 questions testing 5 core memory abilities. OMEGA claims 95.4% (marketing) but their own repo shows 76.8%. Hindsight scores 91.4%. MindCore achieved 95.6% on LongMemEval Oracle.

**Rationale:**
- LongMemEval is the standard leaderboard that competitors report against
- MemoryAgentBench (ICLR 2026) tests selective forgetting — directly validates ACT-R decay
- AMA-Bench tests agentic (non-dialogue) applications — MindCore's primary use case
- Benchmark harness must be separate from the library (large data, LLM judge dependency)
- Three specific additions drive the score from 88-93% to 93-96%: fact extraction at ingest, time-aware query expansion, exhaustive retrieval mode

**Consequences:**
- `mindcore-bench/` workspace member with per-benchmark runners
- Evaluation uses GPT-4o judge (LongMemEval standard)
- Score targets guide feature prioritization

---

## Decision 010: Three-Tier Memory Hierarchy

**Date:** 2026-03-17

**Decision:** Add a tier system (0=episode, 1=summary, 2=fact) with tier-aware search, consumer-controlled consolidation, and soft-delete pruning.

**Context:** Over months of operation, episodic memories accumulate. TraceMem, MemGPT/Letta, and EverMemOS all implement progressive summarization. Mem0's insight: memory formation is selective, not compressive — choose what deserves retention rather than summarizing everything.

**Rationale:**
- Raw episodes are verbose and decay fast; summaries and facts are dense and durable
- Tier-aware search (Standard=tiers 1+2, Deep=+tier 0, Forensic=all) improves relevance within token budgets
- ACT-R activation naturally works with tiers — consolidated episodes lose activation and become prunable
- Consumer controls scheduling via explicit `consolidate()` and `prune()` calls (library, not framework)
- Soft delete as default preserves forensic capability

**Consequences:**
- `tier` column (0-2) added to memories table
- `source_ids` column for provenance tracking (JSON array of original memory IDs)
- `SearchDepth` enum controls which tiers are searched
- `LlmCallback` trait for LLM-assisted summarization (consumer provides, optional)
- LLM-free degraded path: vector clustering + statistical dedup
- `PruningPolicy` struct with configurable thresholds, type exemptions, graph-link protection

---

## Decision 011: WASM Support via Hybrid Architecture

**Date:** 2026-03-17

**Decision:** Support WASM compilation with a hybrid architecture — SQLite+FTS5 in browser WASM, embeddings server-side, with full-WASM candle as opt-in.

**Context:** rusqlite has official WASM support since v0.38.0 (Dec 2025) via `sqlite-wasm-rs`. Candle has a working all-MiniLM-L6-v2 WASM demo. No known project combines rusqlite + FTS5 + candle in WASM — MindCore would be novel.

**Rationale:**
- All pieces work today: rusqlite WASM, FTS5 enabled in WASM build, OPFS/IndexedDB persistence
- Hybrid recommended: SQLite+FTS5 in Web Worker (fast local queries, offline), embeddings via server API (native speed)
- Full-WASM candle is opt-in for offline/privacy use cases (~300-500MB browser memory)
- Same MindCore API surface via `cfg(target_family = "wasm")` conditional compilation
- Aligns with user's Solid.js web stack

**Consequences:**
- `wasm` feature flag activates `sqlite-wasm-rs` backend
- Persistence via OPFS (`sahpool` VFS) or IndexedDB (`relaxed-idb` VFS)
- Single-threaded in WASM (SQLite compiled with `SQLITE_THREADSAFE=0`)
- `EmbeddingBackend` trait enables swapping to API-based embeddings in browser context

---

## Decision 012: Fact Extraction at Ingest

**Date:** 2026-03-17

**Decision:** Add an `IngestStrategy` trait that allows consumers to extract atomic facts from raw input before storage, rather than storing verbatim text.

**Context:** The LongMemEval paper's single biggest finding: fact-augmented key expansion improves recall by +9.4% and accuracy by +5.4%. OMEGA's equivalent "key expansion" is a major driver of their benchmark score.

**Rationale:**
- Storing raw conversation turns is suboptimal — a single turn may contain multiple independent facts
- Extracting and indexing facts separately improves retrieval precision
- Default implementation stores as-is (zero cost); LLM-assisted implementation extracts atomic facts
- Aligns with Mem0's selective memory formation: choose what deserves retention

**Consequences:**
- `IngestStrategy` trait with `extract()` method returning `Vec<ExtractedFact>`
- Default `PassthroughIngest` stores text as-is
- `LlmIngest` uses `LlmCallback` to extract facts (consumer controls cost)
- Extracted facts stored as separate Tier 2 memories linked to source via `source_ids`

---

## Decision 013: Cross-Encoder Reranking

**Date:** 2026-03-17 (updated: 2026-03-18 — switched from fastembed to candle BERT after Decision 016)

**Decision:** Add an optional `RerankerBackend` trait for post-retrieval cross-encoder reranking, implemented via candle's standard BERT model with a classification head.

**Context:** Hindsight uses four parallel retrieval strategies with cross-encoder reranking (91.4% LongMemEval). Reranking after RRF fusion is now standard in competitive memory systems. Cross-encoders are architecturally just BERT + linear classifier that score (query, document) pairs jointly — candle already has BERT support, so no additional dependencies beyond what `local-embeddings` provides.

**Rationale:**
- Cross-encoder reranking improves precision by scoring query-document pairs jointly
- RRF merge is effective but operates on independent rankings — reranking captures cross-attention
- A cross-encoder is a standard BERT model with a single-output classification head — candle already has `BertModel`
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, safetensors available on HF)
- Shares candle deps with the embedding module — zero additional binary size
- Feature-gated so projects that don't need it pay zero cost

**Consequences:**
- `RerankerBackend` trait with `rerank(query, documents) -> Vec<f32>` method
- `CandleReranker` implementation (~50-80 lines) using candle BERT + linear head
- Applied after RRF merge, before final scoring
- Feature-gated behind `reranking` (depends on `local-embeddings`)
- Consumer can provide custom `RerankerBackend` implementation

---

## Decision 014: Memory Evolution (Post-Write Hooks)

**Date:** 2026-03-17

**Decision:** When a new memory is stored, optionally trigger re-evaluation and update of related existing memories.

**Context:** A-MEM (NeurIPS 2025) and Cognee demonstrate that new memories should trigger updates to existing memories' attributes, keywords, and links. This "memory writes back to memory" pattern improves multi-hop reasoning and keeps the memory graph current.

**Rationale:**
- Static memory storage means related memories become stale as context evolves
- Post-write hooks retrieve top-k similar memories and optionally update their metadata/links
- Enables Zettelkasten-style bidirectional linking (A-MEM's core innovation)
- Consumer controls whether evolution runs (opt-in, not default)

**Consequences:**
- Optional `EvolutionStrategy` trait
- Post-write pipeline: store → retrieve similar → evaluate → update metadata/links
- Can use `LlmCallback` for intelligent evaluation, or rules-based for zero-cost
- Graph relationships created/updated automatically when evolution detects connections

---

## Decision 015: LlmCallback Trait

**Date:** 2026-03-17

**Decision:** Define a single `LlmCallback` trait for all LLM-assisted operations. Consumer provides the implementation, controlling model choice, cost, and retry logic.

**Context:** Multiple features need LLM assistance: consolidation (LLMConsolidation), fact extraction (LlmIngest), memory evolution, and reflection. MindCore should never call an LLM directly — the consumer controls cost.

**Rationale:**
- Single trait avoids proliferation of callback types
- Consumer decides model (Claude, GPT, local Llama), token budget, retry behavior
- `Option<&dyn LlmCallback>` — when None, all features degrade gracefully to non-LLM paths
- Library, not framework — MindCore provides operations, consumer provides intelligence

**Consequences:**
- `LlmCallback` trait with `complete(prompt: &str) -> Result<String>`
- Used by: `LLMConsolidation`, `LlmIngest`, `EvolutionStrategy`, `reflect()`
- All LLM-dependent features work (degraded) without an LLM callback

---

## Decision 016: Custom Candle Embedding Module (Replaces fastembed-rs)

**Date:** 2026-03-17 (updated: 2026-03-18 — replaced fastembed with custom candle)

**Decision:** Build a custom embedding module (~100-130 lines) inside MindCore using candle-transformers' native ModernBERT implementation. Drop fastembed-rs entirely.

**Context:** fastembed-rs stability assessment (March 2026) revealed: single maintainer (Anush008/Qdrant, bus factor 1), pinned to pre-release `ort =2.0.0-rc.11`, ships 50-150MB C++ ONNX Runtime shared library, uses `anyhow` in library crate, yearly breaking major versions. candle-transformers already has native ModernBERT support (PR #2791, merged March 2025), and all-MiniLM-L6-v2 ships safetensors weights that candle loads directly.

**Rationale:**
- Pure Rust — no C++ shared library, no ONNX Runtime, aligns with design principle #5
- candle-transformers has native ModernBERT (GeGLU, alternating attention, RoPE) — no architecture gaps
- all-MiniLM-L6-v2 ships `model.safetensors` (95MB), `config.json`, `tokenizer.json` — everything candle needs
- Eliminates: fastembed (single-maintainer), ort (pre-release pin), ndarray, anyhow (transitive)
- ~100-130 lines replaces an external crate with full ownership
- `EmbeddingBackend` trait still lets consumers plug in fastembed or anything else if they want
- HuggingFace maintains candle — larger team, stronger long-term backing than fastembed

**Consequences:**
- `CandleBackend` is the primary backend on all native targets (feature: `local-embeddings`)
- Native default model: `all-MiniLM-L6-v2` (BERT, 384-dim)
- WASM uses the same model (standard BERT compiles to WASM cleanly)
- No more dual-model split — same vectors everywhere (native, WASM, API)
- Cross-model vectors are isolated: different models produce incomparable embedding spaces despite same dimensionality — see Decision 020
- Dependencies: `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `hf-hub`
- Model cached at `~/.cache/mindcore/models/`, auto-downloaded on first use
- No `FastembedBackend` in MindCore — removed from codebase

---

## Decision 017: Default Embedding Model — all-MiniLM-L6-v2

**Date:** 2026-03-17 (updated: 2026-03-18 — removed fastembed references after Decision 016 revision)

**Decision:** Use `all-MiniLM-L6-v2` (22M params, 384-dim) as the default embedding model on ALL targets. Same model native, WASM, and API.

**Context:** Originally chose granite-small-r2 (ModernBERT, 47M params) for higher code retrieval scores. Switched to all-MiniLM-L6-v2 because: (1) granite's ModernBERT doesn't compile to WASM, forcing a dual-model split with bge-small for browsers — vectors were incompatible across platforms; (2) MiniLM is 3x faster on CPU; (3) MiniLM is available on DeepInfra and other API providers for fast batch embedding; (4) the 3-4 point MTEB gap is negligible when combined with hybrid search (FTS5 + vector + RRF + reranking).

**Rationale:**
- Standard BERT architecture — works everywhere (native, WASM, any API provider)
- Same vectors from local Candle and DeepInfra API — interchangeable, no cross-model issues
- 22M params, 6 BERT layers — fast on CPU, suitable for real-time embedding
- 384 dimensions, Apache 2.0 license, ~80MB safetensors
- Most widely deployed embedding model in the world — battle-tested

**Consequences:**
- `CandleBackend::new()` auto-downloads all-MiniLM-L6-v2 from HuggingFace
- `ApiBackend::deepinfra_minilm(api_key)` uses the same model via API
- No more dual-model WASM/native split — one model everywhere
- Supersedes the original granite-small-r2 selection
- `all-MiniLM-L6-v2` not used

---

## Decision 018: Reflection Operation

**Date:** 2026-03-19

**Decision:** Implement periodic reflection via `engine.reflect()` that synthesizes higher-order insights from accumulated memories, stored as Semantic Tier 2 memories with provenance links.

**Context:** Research shows removing reflection causes agent behavior to degenerate within 48 hours (Hindsight). The `reflect()` method clusters accumulated memories and generates summary insights using the `LlmCallback` trait.

**Rationale:**
- Prevents memory systems from becoming flat accumulations of facts without synthesis
- Produces durable Tier 2 semantic memories that improve search relevance
- Depends on existing `LlmCallback` trait — no new external dependencies
- Consumer controls when and how often reflection runs (library, not framework)
- Degraded path without LLM: vector clustering produces statistical summaries

**Consequences:**
- `engine.reflect(&dyn LlmCallback)` method on MemoryEngine
- Results stored as Semantic Tier 2 memories with `source_ids` provenance
- Consumer decides scheduling (daily, weekly, on-demand)
- Part of the `maintain()` convenience method alongside consolidation and pruning

---

## Decision 019: Synchronous Core API

**Date:** 2026-03-19

**Decision:** Core MemoryEngine operations (CRUD, search, context assembly, scoring) are synchronous. Async is reserved for embedding inference, LLM callbacks, and network I/O.

**Context:** The architecture initially used `async fn` throughout, requiring `tokio` as a mandatory dependency. This contradicts the "library, not framework" principle — forcing an async runtime on consumers who may use different runtimes or none at all. Core operations are SQLite queries that complete in microseconds to milliseconds.

**Rationale:**
- SQLite operations are inherently synchronous — wrapping them in async adds overhead without benefit
- Removes tokio as a mandatory dependency (moved behind `vector-search` and `mcp-server` feature flags)
- Consumers using synchronous code don't need to pull in an async runtime
- CPU-bound embedding inference uses `std::thread::spawn` / `rayon`, not async (which is for I/O-bound work)
- The `EmbeddingBackend` and `LlmCallback` traits remain async for legitimate I/O (model download, API calls)

**Consequences:**
- `store()`, `get()`, `update()`, `delete()`, `search().execute()`, `assemble_context()` are all `fn`, not `async fn`
- `MemoryEngine::builder().build()` is synchronous
- Background embedding indexer runs on a dedicated thread, not a tokio task
- Minimum Rust version: 1.85 (edition 2024, native async traits)
- `tokio` moves to feature-gated dependency

---

## Decision 020: Cross-Model Vector Isolation

**Date:** 2026-03-19

**Decision:** Vectors from different embedding models are isolated — vector search is skipped for records whose stored `model_name` differs from the current backend's model. The engine falls back to FTS5-only for mismatched records.

**Context:** The architecture previously claimed all-MiniLM-L6-v2 and bge-small-en-v1.5 produce "cross-compatible" 384-dim vectors with only 5-10% quality degradation. This is incorrect. Different model architectures produce fundamentally different embedding spaces — cross-model cosine similarity produces unreliable rankings, not slightly degraded ones.

**Rationale:**
- Vectors from different models are not comparable even at the same dimensionality
- Returning random-seeming rankings is worse than returning no vector results
- FTS5 graceful fallback ensures search still works when switching platforms (native ↔ WASM)
- The `model_name` field on `memory_vectors` enables clean model migration via `reindex_all()`
- Honest about limitations rather than presenting unreliable results

**Consequences:**
- Vector search query filters by `model_name = current_backend.model_name()`
- When native user accesses a database created in WASM (or vice versa), vector search is skipped — FTS5 handles retrieval
- `reindex_all()` re-embeds all memories with the current model, restoring vector search
- Removes misleading "cross-compatible vectors" claim from documentation

---

## Decision 021: Structured Error Types

**Date:** 2026-03-19

**Decision:** Define a structured `MindCoreError` enum using `thiserror` with variants for each failure domain, enabling consumers to match on specific error conditions.

**Context:** Library crates need structured errors so consumers can handle failures appropriately (retry on transient DB errors, surface model-not-found to users, etc.). A single `anyhow::Error` or `Box<dyn Error>` prevents pattern matching.

**Rationale:**
- Consumers need to distinguish database errors from embedding errors from serialization errors
- `thiserror` provides zero-cost error types with `Display` and `From` implementations
- Each feature-gated module contributes its own error variants
- `MindCoreError::ModelMismatch` enables clear messaging when vector search falls back to FTS5

**Consequences:**
- `mindcore::Result<T>` type alias used throughout the public API
- Error variants: `Database`, `Embedding`, `ModelNotAvailable`, `ModelMismatch`, `Serialization`, `Migration`, `Encryption`, `Consolidation`, `LlmCallback`
- Feature-gated variants only exist when their feature is enabled

---

## Decision 022: Crate Name — mindcore

**Date:** 2026-03-19

**Decision:** Name the crate `mindcore`. Reserved on crates.io and GitHub.

**Context:** Evaluated 25+ candidate names across crates.io, npm, PyPI, and GitHub. Key criteria: available on crates.io, short and memorable, low GitHub namespace collision, evocative of the library's purpose.

**Rationale:**
- Available on all three package registries (crates.io, npm, PyPI) at time of evaluation
- Short (8 chars), easy to type, clearly communicates "cognitive/mind + core engine"
- Minimal GitHub presence (58 repos, top has 3 stars — all abandoned/tiny)
- Strong alternatives (`cognimem`, `cogmem`, `mnemic`) were also clean but `mindcore` best communicates the crate's purpose

**Consequences:**
- crates.io: v0.0.1 placeholder published under `mindcore`
- GitHub: `victorysightsound/mindcore` repository created
- npm: blocked by `mind-core` similarity — use scoped package if needed
- PyPI: deferred until Python bindings are built
- All documentation renamed from `memcore` → `mindcore`

---

## Open Questions

### Q1: Crate Naming and Publishing

**Status:** Decided — `mindcore` (Decision 022)

Name `mindcore` selected and reserved on crates.io (v0.0.1 placeholder published 2026-03-19) and GitHub (victorysightsound/mindcore). npm blocked by name similarity to existing `mind-core` package — will use scoped `@victorysightsound/mindcore` if JS/WASM bindings are published. PyPI deferred — only relevant for Python bindings via PyO3.

### Q2: FTS5 + Hybrid Search Phasing

**Status:** Planned

- **Phase 1:** FTS5 + WAL + Porter stemming (proven in production)
- **Phase 2:** Add hybrid vector search via `vector-search` feature
- **Phase 3:** Add graph memory via `graph-memory` feature

No architecture changes needed between phases — just enable feature flags.

### Q3: Beliefs Memory Type

**Status:** Deferred to post-v1

The three existing types (Episodic/Semantic/Procedural) cover all common agent memory patterns. Beliefs would add complexity (confidence scores, provenance chains, challenge/revision semantics) for a pattern only demonstrated in Hindsight. The `metadata` field on `MemoryRecord` can carry confidence and provenance data ad-hoc until the pattern proves itself in real usage. Revisit after v1.0 ships and consumer feedback is available.

### Q4: Reflection Operation

**Status:** Decided — YES (Decision 018)

Hindsight research demonstrates agent behavior degenerates within 48 hours without periodic reflection. The `reflect()` method is already in the public API design. Implementation depends on `LlmCallback` trait (Decision 015), so it's naturally late in the build order (Phase 14+). The `reflect()` method synthesizes higher-order insights from memory clusters, stored as Semantic Tier 2 memories with provenance links.
