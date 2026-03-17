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
