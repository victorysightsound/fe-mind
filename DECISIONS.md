# MemCore Decisions

This document records key architectural and design decisions for MemCore.

Decisions 001-007 originated during research in the PIRDLY project (2026-03-16) and were migrated here when MemCore became its own project.

---

## Decision 001: MemCore Shared Memory Engine

**Date:** 2026-03-16

**Decision:** Extract memory system into a standalone crate (MemCore) reusable across PIRDLY, Dial, and Memloft.

**Context:** Three projects (Dial, Memloft, PIRDLY) all need persistent memory with search, and each implements the same primitives differently. Research into Mem0, OMEGA, Zep/Graphiti, and MemOS confirms the patterns are converging industry-wide.

**Rationale:**
- Eliminates duplicate memory code across three projects
- Improvements (vector search, graph, decay) benefit all projects automatically
- Rust ecosystem lacks a standalone agent memory crate (Engram is Go, Mem0 is Python)
- Feature-gated design means zero cost for unused capabilities
- Every component is already proven in at least one existing project

**Consequences:**
- New standalone crate: `memcore`
- PIRDLY depends on memcore instead of implementing its own memory
- Dial and Memloft migrate to memcore over time
- See `MEMCORE_ARCHITECTURE.md` for full specification

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

**Context:** Evaluated ort, candle, and fastembed-rs. Studied Memloft's production use of candle.

**Rationale:**
- Pure Rust (ort requires C++ runtime, adds 80+ deps and ~350MB to binary)
- Native safetensors loading from HF Hub (no ONNX conversion step)
- WASM support (relevant for future GUI via WebView)
- Performance difference is negligible for MiniLM-sized models (~8ms per embed)
- Memloft proves candle works in production for this exact use case
- `EmbeddingBackend` trait allows swapping to ort later if scale demands it

**Consequences:**
- Feature-gated behind `local-embeddings` flag
- Model: `all-MiniLM-L6-v2` (384 dims, 22M params, ~80MB download)
- Model downloaded from HF Hub on first use, cached locally
- Graceful degradation to FTS5-only if candle fails to load

---

## Decision 004: Hybrid Search with Reciprocal Rank Fusion

**Date:** 2026-03-16

**Decision:** Combine FTS5 keyword search and vector similarity search using Reciprocal Rank Fusion (RRF).

**Context:** FTS5 handles 80% of lookups well. Vector catches semantic matches FTS5 misses. Need a principled way to merge results.

**Rationale:**
- RRF is simple, effective, and parameter-light (just k-value)
- Memloft proves RRF works in production for agent memory
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

**Context:** Dial uses trust scoring with manual decay. Memloft uses tier-based multipliers. OMEGA uses forgetting intelligence. All solve the same problem differently.

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
- Replaces Dial's `confidence` field and Memloft's tier system

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
- Mem0 demonstrates this is essential for production-quality memory

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
- Consumer provides the key — MemCore doesn't manage key storage

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

**Context:** LongMemEval (ICLR 2025) is the de facto standard — 500 questions testing 5 core memory abilities. OMEGA scores 95.4% (#1). Hindsight scores 91.4%. MemCore's architecture as-designed could hit 88-93%; with targeted additions, 93-96% is realistic.

**Rationale:**
- LongMemEval is the standard leaderboard that competitors report against
- MemoryAgentBench (ICLR 2026) tests selective forgetting — directly validates ACT-R decay
- AMA-Bench tests agentic (non-dialogue) applications — MemCore's primary use case
- Benchmark harness must be separate from the library (large data, LLM judge dependency)
- Three specific additions drive the score from 88-93% to 93-96%: fact extraction at ingest, time-aware query expansion, exhaustive retrieval mode

**Consequences:**
- `memcore-bench/` workspace member with per-benchmark runners
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

**Context:** rusqlite has official WASM support since v0.38.0 (Dec 2025) via `sqlite-wasm-rs`. Candle has a working all-MiniLM-L6-v2 WASM demo. No known project combines rusqlite + FTS5 + candle in WASM — MemCore would be novel.

**Rationale:**
- All pieces work today: rusqlite WASM, FTS5 enabled in WASM build, OPFS/IndexedDB persistence
- Hybrid recommended: SQLite+FTS5 in Web Worker (fast local queries, offline), embeddings via server API (native speed)
- Full-WASM candle is opt-in for offline/privacy use cases (~300-500MB browser memory)
- Same MemCore API surface via `cfg(target_family = "wasm")` conditional compilation
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

**Context:** The LongMemEval paper's single biggest finding: fact-augmented key expansion improves recall by +9.4% and accuracy by +5.4%. OMEGA's equivalent "key expansion" is a major driver of their 95.4% score.

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

**Date:** 2026-03-17

**Decision:** Add an optional `RerankerBackend` trait for post-retrieval cross-encoder reranking, with fastembed-rs as the recommended implementation.

**Context:** Hindsight uses four parallel retrieval strategies with cross-encoder reranking (91.4% LongMemEval). fastembed-rs v5.12.0 (March 2026) provides 3 reranking models in Rust. Reranking after RRF fusion is now standard in competitive memory systems.

**Rationale:**
- Cross-encoder reranking improves precision by scoring query-document pairs jointly
- RRF merge is effective but operates on independent rankings — reranking captures cross-attention
- fastembed-rs provides BGE-reranker-base, BGE-reranker-v2-m3, jina-reranker-v1-turbo-en
- Feature-gated so projects that don't need it pay zero cost

**Consequences:**
- `RerankerBackend` trait with `rerank()` method
- Applied after RRF merge, before final scoring
- Feature-gated behind `reranking`
- fastembed-rs recommended; consumer can provide custom implementation

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

**Context:** Multiple features need LLM assistance: consolidation (LLMConsolidation), fact extraction (LlmIngest), memory evolution, and reflection. MemCore should never call an LLM directly — the consumer controls cost.

**Rationale:**
- Single trait avoids proliferation of callback types
- Consumer decides model (Claude, GPT, local Llama), token budget, retry behavior
- `Option<&dyn LlmCallback>` — when None, all features degrade gracefully to non-LLM paths
- Library, not framework — MemCore provides operations, consumer provides intelligence

**Consequences:**
- `LlmCallback` trait with `complete(prompt: &str) -> Result<String>`
- Used by: `LLMConsolidation`, `LlmIngest`, `EvolutionStrategy`, `reflect()`
- All LLM-dependent features work (degraded) without an LLM callback

---

## Decision 016: fastembed-rs as Primary Embedding Backend

**Date:** 2026-03-17

**Decision:** Use fastembed-rs as the primary embedding backend on native targets. Retain candle as WASM-only fallback. Both backends share the same default model for vector compatibility.

**Context:** fastembed-rs v5.13.0 provides 44+ embedding models, 4 reranking models, SPLADE sparse embeddings, synchronous API, no Tokio dependency. Significantly more capable than raw candle for native use.

**Rationale:**
- 44+ models vs manually loading one — consumers choose the best model for their domain
- Built-in reranking (Decision 013) without additional integration work
- SPLADE sparse embeddings for better keyword-aware retrieval
- Simpler API: one crate vs candle-core + candle-nn + candle-transformers + tokenizers + hf-hub
- Synchronous — no Tokio dependency for embedding operations
- Custom model support via `try_new_from_user_defined()` for models not yet in fastembed

**Consequences:**
- `FastembedBackend` is the primary backend on native targets (feature: `fastembed`)
- `CandleBackend` is the WASM-only fallback (feature: `local-embeddings`)
- Both use the same default model (`bge-small-en-v1.5`, 384-dim) for vector cross-compatibility
- Conditional compilation via `cfg(target_family = "wasm")` selects the right backend
- Reranking available only on native (fastembed); WASM clients defer to server-side reranking

---

## Decision 017: Default Embedding Model — granite-small-r2

**Date:** 2026-03-17 (updated: granite promoted to default)

**Decision:** Use `granite-embedding-small-english-r2` (IBM, 47M params, 384-dim, 8K context) as the default embedding model on native targets. Use `bge-small-en-v1.5` (384-dim) as the WASM fallback and zero-config native fallback. Drop `all-MiniLM-L6-v2`.

**Context:** Benchmarked three 384-dim models for agent memory retrieval. granite-small-r2 scores 17% better than bge-small on code retrieval (CoIR 53.8 vs 45.8) and has 16x longer context (8K vs 512 tokens). It's not built into fastembed natively, but MemCore wraps the custom model loading behind a clean API with auto-download and caching — consumers never see the boilerplate.

**Rationale:**
- Code retrieval (CoIR): granite 53.8 vs bge-small 45.8 — 17% better on the workload that matters most for dev tool memory
- 8K token context captures full error traces, decision rationale, and code blocks without truncation
- Standard retrieval matches bge-small exactly (MTEB-v2: 53.9 vs 53.9)
- Same 384 dimensions — vectors cross-compatible with bge-small (WASM fallback)
- ModernBERT architecture with Flash Attention 2 keeps inference fast despite 47M params
- Apache 2.0 license, ONNX Q8 variant is ~52MB
- MemCore's `MemCoreModel` enum abstracts the custom loading — one line: `FastembedBackend::new()?`

**Consequences:**
- Default `FastembedBackend::new()` auto-downloads granite-small-r2 ONNX from HuggingFace, caches at `~/.cache/memcore/models/`
- `MemCoreModel::BgeSmallV15` available as zero-config fallback (uses fastembed built-in)
- `CandleBackend` (WASM) uses `bge-small-en-v1.5` (same 384-dim, compatible vectors)
- `all-MiniLM-L6-v2` available in fastembed but not recommended or referenced

---

## Open Questions

### Q1: Crate Naming and Publishing

**Status:** Open

Evaluate crate name availability on crates.io. Candidates: `memcore`, `agentmem`, `cognimem`. Publish after v0.1.0 is stable with Dial and Memloft migrations validated.

### Q2: FTS5 + Hybrid Search Phasing

**Status:** Planned

- **Phase 1:** FTS5 + WAL + Porter stemming (proven in Dial)
- **Phase 2:** Add hybrid vector search via `vector-search` feature
- **Phase 3:** Add graph memory via `graph-memory` feature

No architecture changes needed between phases — just enable feature flags.

### Q3: Beliefs Memory Type

**Status:** Under consideration

Hindsight separates world facts from agent beliefs (synthesized conclusions that can be revised). Consider adding `MemoryType::Belief` with confidence score and provenance chain. This would be the fourth cognitive type alongside Episodic/Semantic/Procedural.

### Q4: Reflection Operation

**Status:** Under consideration

Hindsight's `reflect` operation periodically clusters accumulated memories and synthesizes higher-order insights. Research shows removing reflection causes agent behavior to degenerate within 48 hours. Would depend on `LlmCallback` trait (Decision 015). Store as Semantic Tier 2 memories with provenance links.
