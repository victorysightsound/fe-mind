# femind Gap Analysis — March 2026

> **Note:** This research predates the current femind naming and package cleanup. Many of the recommendations here were incorporated into the shipped architecture.

**Date:** 2026-03-17
**Status:** Research complete. Findings incorporated into the current v0.2.0 architecture.
**Context:** Targeted research across 5 areas to validate and strengthen femind's architecture against the cutting edge of agent memory systems.

---

## Executive Summary

femind's core architecture is sound and competitive. The fundamental decisions (SQLite, FTS5, candle, RRF, ACT-R, feature gates) are validated by the 2025-2026 landscape. However, five areas need attention to reach parity with or exceed the current state of the art, and several architectural additions would push femind ahead of the field.

**Confidence level:** femind as-designed could score 88-93% on LongMemEval. With the additions identified here, 93-96% is realistic (femind achieved 95.6%, surpassing OMEGA's verified 76.8%).

---

## 1. Encryption at Rest

### Finding

**SQLCipher via rusqlite's `bundled-sqlcipher` feature is the clear winner.** It provides transparent AES-256-CBC encryption of the entire database file at the page level, preserving FTS5, WAL mode, and vector search with 5-15% I/O overhead.

OMEGA's "encryption at rest" claim is narrower than it appears — they only encrypt exports and profile data via application-level Fernet encryption. The main `omega.db` is unencrypted. Application-level field encryption is a dead end for femind because it fundamentally breaks FTS5 (can't tokenize ciphertext).

### Recommendation

Add two feature flags:

| Feature | What it does | Dependency impact |
|---------|-------------|-------------------|
| `encryption` | Replaces bundled SQLite with bundled SQLCipher | ~500KB-1MB over plain SQLite |
| `keychain` | OS keychain integration for key storage | `keyring` crate, ~200-500KB |

**Key management:** Consumer provides the key. femind should offer:
- `EncryptionKey::Passphrase(String)` — SQLCipher derives via PBKDF2
- `EncryptionKey::RawKey([u8; 32])` — pre-derived 256-bit key
- Optional `keychain` helper: `femind::keychain::get_or_create_key()` using the `keyring` crate (macOS Keychain, Windows Credential Manager, Linux Secret Service)

**Cargo.toml pattern:**
```toml
[features]
encryption = ["rusqlite/bundled-sqlcipher"]
encryption-vendored = ["encryption", "rusqlite/bundled-sqlcipher-vendored-openssl"]
keychain = ["dep:keyring"]
```

### Decision Needed

Decision 008: Encryption at Rest via SQLCipher

---

## 2. Benchmarking (LongMemEval and Beyond)

### Finding

**LongMemEval** (ICLR 2025) is the de facto standard benchmark — 500 questions testing 5 core memory abilities across 7 question types. OMEGA claims 95.4% (marketing) but verified at 76.8% in their own repo. Hindsight scores 91.4%. femind achieved 95.6%.

Six additional benchmarks exist: LOCOMO (Snap Research), MemBench (ACL 2025), MemoryAgentBench (ICLR 2026), MemoryBench, AMA-Bench, and MemoryStress (OMEGA).

### femind's Realistic Score Trajectory

| Configuration | Estimated Score | Key capabilities |
|--------------|----------------|-----------------|
| FTS5 only | 65-72% | Keyword matching, Porter stemming |
| + Vector + RRF | 78-85% | Semantic search, hybrid fusion |
| + Temporal + Consolidation + ACT-R | 88-93% | Knowledge updates, decay, dedup |
| + Fact extraction + Query expansion + Exhaustive retrieval | 93-96% | Full pipeline optimizations |

### Three Additions to Reach 93-96%

**a) Fact extraction at ingest (`IngestStrategy` trait):**
The LongMemEval paper's biggest single finding (+5-9% accuracy). Instead of storing raw text, extract atomic facts and index them separately. This is OMEGA's "key expansion" equivalent.

```rust
#[async_trait]
pub trait IngestStrategy: Send + Sync {
    /// Extract indexable facts from raw input.
    /// Default: store as-is. LLM-assisted: extract atomic facts.
    async fn extract(&self, raw: &str) -> Result<Vec<ExtractedFact>>;
}
```

**b) Time-aware query expansion:**
Converting "last month", "before Christmas", "in 2024" to date-range filters. The paper reports +6.8-11.3% on temporal reasoning.

**c) Exhaustive retrieval mode:**
For multi-session aggregation queries ("how many times did X happen"), bypass top-k limits and return all matches above a threshold. OMEGA's weakest category (83.5%) is multi-session reasoning — this is where femind can differentiate.

```rust
pub enum SearchMode {
    Keyword, Vector, Hybrid, Auto,
    /// Return all matches above threshold (for aggregation queries)
    Exhaustive { min_score: f32 },
}
```

### Benchmark Harness

Should live in a separate benchmark application, not in the library crate. Benchmark data is large (115K-1.5M tokens) and evaluation requires an LLM judge (GPT-4o). RecallBench now fills that role for femind and other memory systems.

### Decision Needed

Decision 009: Benchmark Strategy — target LongMemEval first, then MemoryAgentBench (tests selective forgetting, maps to ACT-R).

---

## 3. Memory Compression and Hierarchical Memory

### Finding

The field has converged on three operations beyond dedup:

1. **Episodic-to-semantic consolidation** — cluster related episodic memories, extract the pattern, store as semantic
2. **Progressive summarization** — raw episode → session summary → topic summary → key fact (TraceMem's three-stage pipeline)
3. **Selective memory formation** — not "summarize everything" but "choose what deserves permanent retention" (Mem0's insight)

### Recommendation: Three-Tier Memory Hierarchy

```
Tier 0: Raw Episodes (full detail, fast decay)
    |
    v  [consolidate()]
Tier 1: Session/Topic Summaries (condensed, medium decay)
    |
    v  [consolidate()]
Tier 2: Key Facts (atomic knowledge, slow decay)
```

**Schema addition:**
```sql
ALTER TABLE memories ADD COLUMN tier INTEGER NOT NULL DEFAULT 0 CHECK(tier BETWEEN 0 AND 2);
ALTER TABLE memories ADD COLUMN source_ids TEXT;  -- JSON array of source memory IDs
```

**Tier-aware search:**
```rust
pub enum SearchDepth {
    Standard,  // Tiers 1+2 only (default, fastest)
    Deep,      // Also search Tier 0 if summary results are sparse
    Forensic,  // All tiers (slowest, most complete)
}
```

**LLM callback for consolidation:**
```rust
#[async_trait]
pub trait LlmCallback: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String>;
}
```

Consumer controls the LLM, model choice, and cost. `Option<&dyn LlmCallback>` — when `None`, consolidation uses degraded LLM-free path (vector clustering + statistical dedup).

**Pruning policy:**
```rust
pub struct PruningPolicy {
    pub min_age_days: u32,                    // default: 30
    pub max_activation: f32,                  // default: -2.0
    pub pruneable_types: Vec<MemoryType>,     // default: [Episodic]
    pub respect_graph_links: bool,            // default: true
    pub respect_hierarchy: bool,              // default: true
    pub soft_delete: bool,                    // default: true
}
```

Semantic and procedural memories are exempt from automatic pruning by default. Only episodic memories with low activation, no graph links, and no hierarchy references are prunable.

### Decision Needed

Decision 010: Three-Tier Memory Hierarchy with `LlmCallback` trait.

---

## 4. WASM Compilation

### Finding

**All pieces work today:**

| Component | WASM Status | Since |
|-----------|-------------|-------|
| rusqlite | Official support | v0.38.0 (Dec 2025) |
| FTS5 | Enabled in WASM build | v0.38.0 |
| candle + all-MiniLM-L6-v2 | Working demo | Existing |
| OPFS persistence | Production-ready | sqlite-wasm-vfs v0.2 |
| IndexedDB persistence | Production-ready | sqlite-wasm-vfs v0.2 |

**Performance in WASM:**
- SQLite queries: 1.5-3x slower than native
- Candle embedding: 3-10x slower (50-200ms per embed vs 10-30ms native)
- Memory: 50MB database + MiniLM-L6-v2 model = ~300-500MB WASM memory (fits in 2-4GB browser limit, tight on mobile)

**No known project combines all three (rusqlite + FTS5 + candle in WASM). femind would be novel.**

### Recommended Architecture: Hybrid

```
Browser (WASM Web Worker)          Server (Native Rust)
┌──────────────────────┐          ┌──────────────────┐
│ SQLite + FTS5        │          │ Candle embeddings│
│ Cached embeddings    │  ←API→  │ (native speed)   │
│ RRF search           │          │                  │
│ Context assembly     │          │                  │
└──────────────────────┘          └──────────────────┘
```

SQLite + FTS5 runs in a WASM Web Worker for fast local queries and offline capability. Embeddings computed server-side at native speed. Full-WASM candle available as opt-in offline mode for power users.

**Feature structure:**
```toml
[features]
default = ["native"]
native = ["rusqlite/bundled"]
wasm = []  # Activates sqlite-wasm-rs backend, OPFS/IndexedDB persistence
```

Conditional compilation via `cfg(target_family = "wasm")` — same API surface, different storage backend.

### Decision Needed

Decision 011: WASM Support via Hybrid Architecture

---

## 5. Latest Innovations and Architectural Gaps

### 5a. New Competitive Threats

| System | Score | Key Innovation |
|--------|-------|---------------|
| **Hindsight** (Vectorize) | 91.4% LongMemEval | Four parallel retrieval strategies + cross-encoder reranking |
| **Cognee** | N/A | Six-stage "cognify" pipeline + "memify" graph refinement |
| **A-MEM** | NeurIPS 2025 | Zettelkasten-style linked notes with memory evolution |
| **EverMemOS** | 92.3% LoCoMo | Self-organizing engram lifecycle |
| **ALMA** | Meta-learning | Discovers novel memory schemas via code search |

### 5b. Critical Ecosystem Development: fastembed-rs

**fastembed-rs v5.12.0** (March 2026) provides 25+ embedding models, 3 reranking models (BGE-reranker-base, jina-reranker-v1-turbo-en), and sparse embeddings (SPLADE) in one Rust crate. Dual ONNX/candle backends. Synchronous, no Tokio dependency.

**Recommendation:** Evaluate fastembed-rs as an alternative to raw candle for the embedding backend. It would give femind:
- 25+ models instead of just all-MiniLM-L6-v2
- Built-in cross-encoder reranking (a gap identified in this analysis)
- SPLADE sparse embeddings for better keyword-aware retrieval
- Simpler API (one crate vs candle-core + candle-nn + candle-transformers + tokenizers + hf-hub)

The `EmbeddingBackend` trait means this is a non-breaking addition — ship a `FastembedBackend` alongside `CandleBackend`.

### 5c. Six Architectural Gaps

**Gap 1: Memory Evolution**
New memories are stored statically. A-MEM and Cognee show that storing a new memory should trigger re-evaluation and optional update of related existing memories (their keywords, links, embeddings). This "memory writes back to memory" pattern improves multi-hop reasoning.

*Addition:* Post-write hook that retrieves top-k similar memories and optionally updates their metadata.

**Gap 2: Cross-Encoder Reranking**
femind has RRF fusion but no reranking stage. Hindsight's four-strategy parallel retrieval with cross-encoder reranking is now standard. fastembed-rs makes this trivial to add.

*Addition:* Optional `RerankerBackend` trait, applied after RRF merge and before final scoring.

```rust
#[async_trait]
pub trait RerankerBackend: Send + Sync {
    async fn rerank(&self, query: &str, candidates: Vec<ScoredResult>) -> Result<Vec<ScoredResult>>;
}
```

**Gap 3: Reflection Operation**
femind consolidates but doesn't synthesize higher-order insights. Research shows removing reflection causes agent behavior to degenerate within 48 hours. Hindsight's `reflect` operation clusters accumulated memories and generates summary insights.

*Addition:* `engine.reflect()` method that uses `LlmCallback` to synthesize insights from memory clusters, stored as semantic Tier 2 memories.

**Gap 4: Bi-Temporal Validity**
femind has `valid_from`/`valid_until` in the architecture doc but this needs to be a first-class concept, not just optional columns. Zep's bi-temporal model tracks both when an event occurred and when it was ingested — enabling "what did we know at time X?" queries.

*Status:* Already partially designed (temporal feature flag). Needs promotion to a core concept with query support: `engine.search("X").valid_at(timestamp)`.

**Gap 5: Beliefs / Evolving Conclusions**
femind has episodic/semantic/procedural types. Hindsight adds a fourth network: "beliefs" — agent-synthesized conclusions that can be revised. These differ from facts (which are ground truth) in that they have confidence scores and provenance chains.

*Addition:* Add `MemoryType::Belief` with confidence field and source memory references.

**Gap 6: MCP Server**
Every competitive memory system (OMEGA, Engram, Hindsight, Cognee) ships with an MCP interface. This is now table stakes.

*Status:* Already planned as feature `mcp-server` in Phase 14. Priority should be raised.

### 5d. Embedding Model Consideration

femind plans `all-MiniLM-L6-v2` (384-dim). OMEGA uses `bge-small-en-v1.5` (also 384-dim, slightly better on MTEB retrieval benchmarks). For maximum score, `bge-small-en-v1.5` or its successor `bge-m3` (1024-dim) would be worth evaluating. The `EmbeddingBackend` trait makes this a configuration choice, not an architectural change.

---

## Summary: Architecture Additions by Priority

### Must-Have (Competitive Parity)

| Addition | Impact | Complexity |
|----------|--------|-----------|
| Encryption via SQLCipher | Security table stakes | Low — feature flag swap |
| Fact extraction at ingest (`IngestStrategy`) | +5-9% on LongMemEval | Medium — new trait |
| Cross-encoder reranking (`RerankerBackend`) | Standard in competitors | Medium — new trait + fastembed |
| MCP server (raise priority) | Table stakes | Already planned |

### Should-Have (Competitive Advantage)

| Addition | Impact | Complexity |
|----------|--------|-----------|
| Three-tier memory hierarchy | Handles memory growth at scale | Medium — schema + search changes |
| Memory evolution (post-write hooks) | Better multi-hop reasoning | Medium — post-write pipeline |
| Time-aware query expansion | +6-11% temporal reasoning | Low — query pre-processor |
| Exhaustive retrieval mode | +5-10% multi-session queries | Low — new SearchMode variant |
| `LlmCallback` trait for consolidation | Enables LLM-assisted operations | Low — trait definition |

### Nice-to-Have (Cutting Edge)

| Addition | Impact | Complexity |
|----------|--------|-----------|
| Reflection operation | Synthesize higher-order insights | Medium — needs LlmCallback |
| Beliefs memory type | Richer cognitive model | Low — enum variant |
| WASM support | Browser deployment | Medium — conditional compilation |
| Benchmark harness (RecallBench) | Validation and marketing | Medium — separate application |
| fastembed-rs backend | 25+ models, reranking, SPLADE | Medium — new backend impl |

---

## What femind Already Does Better Than Anyone

1. **Rust performance** — No other memory library is in Rust
2. **ACT-R activation decay** — Research-backed model, not ad-hoc
3. **Three-tier consolidation** — Hash/similarity/LLM pipeline
4. **Cognitive memory types** — Episodic/semantic/procedural split
5. **Token-budget context assembly** — Ahead on context engineering
6. **Feature-gated everything** — 2MB to 45MB, compile what you need
7. **Library, not framework** — Embed anywhere via `MemoryRecord` trait

---

## Sources

### Papers
- LongMemEval (ICLR 2025): arxiv.org/abs/2410.10813
- A-MEM (NeurIPS 2025): arxiv.org/abs/2502.12110
- EverMemOS: arxiv.org/abs/2601.02163
- ALMA: arxiv.org/abs/2602.07755
- Hindsight: arxiv.org/abs/2512.12818
- Zep/Graphiti: arxiv.org/abs/2501.13956
- TraceMem: arxiv.org/abs/2602.09712
- Mem0: arxiv.org/abs/2504.19413
- MemGPT/Letta: arxiv.org/abs/2310.08560
- Memory in the Age of AI Agents (survey): arxiv.org/abs/2512.13564
- Memory for Autonomous LLM Agents (survey): arxiv.org/abs/2603.07670
- MemoryAgentBench (ICLR 2026): arxiv.org/abs/2507.05257
- AMA-Bench: arxiv.org/html/2602.22769v1

### Projects
- OMEGA Memory: github.com/omega-memory/core
- Hindsight: github.com/vectorize-io/hindsight
- Cognee: github.com/topoteretes/cognee
- Mem0: github.com/mem0ai/mem0
- MemOS: github.com/MemTensor/MemOS
- Engram: github.com/Gentleman-Programming/engram
- Letta/MemGPT: github.com/letta-ai/letta
- SQLite-Memory: github.com/sqliteai/sqlite-memory
- fastembed-rs: github.com/Anush008/fastembed-rs
- sqlite-vec: github.com/asg017/sqlite-vec
- sqlite-wasm-rs: github.com/nicholasgasior/sqlite-wasm-rs
- lucid-core: crates.io/crates/lucid-core

### Benchmarks
- LongMemEval leaderboard: xiaowu0162.github.io/long-mem-eval
- OMEGA 95.4% marketing claim: dev.to/singularityjason (verified score: 76.8% per their own repo)
- Emergence AI 86%: emergence.ai/blog/sota-on-longmemeval-with-rag
- Mastra OM 94.87%: mastra.ai/research/observational-memory
