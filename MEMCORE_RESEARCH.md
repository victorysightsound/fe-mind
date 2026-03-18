# MemCore: Research & Specification

**Date:** 2026-03-16
**Status:** Research Complete
**Related:** `MEMCORE_ARCHITECTURE.md` (implementation spec), `DECISIONS.md` (decisions 001-016), `research/gap_analysis_2026_03.md` (March 2026 gap analysis)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Landscape Analysis](#2-landscape-analysis)
3. [Source Project Analysis](#3-source-project-analysis)
4. [Academic & Theoretical Foundations](#4-academic--theoretical-foundations)
5. [Component Research](#5-component-research)
6. [Technology Evaluation](#6-technology-evaluation)
7. [Design Decisions & Rationale](#7-design-decisions--rationale)
8. [Competitive Positioning](#8-competitive-positioning)
9. [Risk Analysis](#9-risk-analysis)
10. [Specification Summary](#10-specification-summary)

---

## 1. Problem Statement

### The Repeating Code Problem

Three separate Rust projects — Dial, Memloft, and PIRDLY — each need persistent memory for AI agents. Each independently arrived at the same architecture: SQLite + FTS5 + optional vector search + scoring/ranking. The implementations differ in detail but solve the same core problems:

| Problem | Dial's Solution | Memloft's Solution | PIRDLY's Plan |
|---------|----------------|-------------------|---------------|
| Where to store memories | SQLite (single DB) | SQLite (single DB) | SQLite (two-tier) |
| How to search | FTS5 + Porter stemming | FTS5 + vector + RRF | FTS5, add vector later |
| How to embed | Not implemented | Candle (all-MiniLM-L6-v2) | Deferred |
| How to rank results | Trust scoring + recency | Tier multipliers + recency + importance | Not designed yet |
| How to prevent duplicates | Not implemented | Content-hash dedup | Not designed yet |
| How to decay old memories | Manual confidence decay | Tier demotion | Not designed yet |
| How to budget context | Token-budget priority assembly | Not implemented | Planned |

Every project re-invents the same wheel with slightly different spokes.

### What a Shared Engine Would Solve

1. **No duplicate code** — one tested, optimized implementation
2. **Cross-pollination** — Dial's context assembly + Memloft's vector search + PIRDLY's two-tier design, all in one crate
3. **Faster iteration** — improvements benefit all projects simultaneously
4. **Better testing** — one crate gets the combined test effort
5. **Community value** — Rust ecosystem lacks a standalone agent memory crate

### Why Now

- Dial v4.1.0 shipped with FTS5 memory (proven in production)
- Memloft shipped with hybrid RRF search (proven in production)
- PIRDLY is pre-build, so its memory system can be designed around MemCore from day one
- The agent memory space is exploding (Mem0 at 37K stars, OMEGA at #1 LongMemEval) — patterns are well-documented

---

## 2. Landscape Analysis

### Existing Agent Memory Systems

The market splits into three categories:

#### Cloud-First (Not Our Target)
| Project | Language | Approach | Stars |
|---------|----------|----------|-------|
| **Mem0** | Python | Cloud API + managed vector DB | 37K+ |
| **Zep** | Python/Go | Cloud service with temporal awareness | 6K+ |
| **LangMem** | Python | LangChain ecosystem | — |

These require cloud services, API keys, and network connectivity. They don't fit local-first, privacy-focused development tools.

#### Local-First (Direct Competitors)
| Project | Language | Storage | Search | Features |
|---------|----------|---------|--------|----------|
| **OMEGA Memory** | TypeScript | SQLite | FTS5 + ONNX embeddings | Decay, graph, encryption |
| **Engram** | Go | SQLite + FTS5 | Keyword only | MCP server, CLI, TUI |
| **MemOS** | Python | SQLite + hybrid | FTS5 + vector | Graph memory, multi-modal |
| **memU** | Python | — | — | Proactive intent detection |

These are closest to what MemCore does, but none are in Rust and none are designed as a library crate.

#### Framework-Embedded (Not Standalone)
| Project | Language | Notes |
|---------|----------|-------|
| **Ruflo** | TypeScript | CRDT-based memory, deeply Claude-specific |
| **AutoAgents** | Rust | Configurable memory per agent, but not extractable |
| **Rig** | Rust | Vector store abstractions, but no memory lifecycle |

These have memory subsystems but they're coupled to their parent frameworks.

### The Gap

**No Rust crate exists that provides standalone, feature-gated agent memory with:**
- FTS5 keyword search
- Vector similarity search
- Hybrid search (RRF)
- Memory decay/activation
- Consolidation
- Token-budget context assembly
- Graph relationships

MemCore fills this gap.

---

## 3. Source Project Analysis

### Dial (FTS5 + Context Assembly)

**What Dial Built:**

Dial v4.1.0 implements a learning-based memory system for an autonomous AI coding orchestrator:

- **Storage:** SQLite with `learnings`, `failure_patterns`, and `solutions` tables
- **Search:** FTS5 with Porter stemming, custom stop-word stripping
- **Trust Model:** Confidence score (0.0-1.0) adjusted by outcomes — successful solutions increase confidence, repeated failures decrease it
- **Context Assembly:** Token-budget-aware priority system that fills an LLM prompt with the most relevant context items, ranked by priority tier:
  - Priority 0: Behavioral rules (always included)
  - Priority 10: Retry context (previous failures for this task)
  - Priority 15: Spec sections (relevant PRD content)
  - Priority 25: Similar completed tasks
  - Priority 40: General learnings
  - Priority 60: Historical context
- **Failure Detection:** Pattern matching on CLI output to detect and classify errors (transient → retry, quota → wait, permanent → fail)

**What Works Well:**
- FTS5 handles 80%+ of lookups (error messages, code snippets are keyword-heavy)
- Porter stemming catches inflections ("authenticate" finds "authentication")
- Priority-based context assembly prevents budget waste on low-value memories
- Trust scoring naturally surfaces reliable solutions over unreliable ones

**What's Missing:**
- No vector search — "authentication error" won't find "login failed" or "auth token expired"
- No deduplication — same learning can be stored multiple times
- No memory decay — old, irrelevant learnings never fade
- No relationships — can't express "this solution fixed this error"
- Trust scoring is manual and fragile (arbitrary 0.05 decay per 30 days)

**What MemCore Extracts:**
- FTS5 + Porter stemming configuration
- Stop-word list
- Token-budget context assembly algorithm
- Priority tier concept
- Error classification pattern (Transient/QuotaExceeded/Permanent)

---

### Memloft (Hybrid Search + Embeddings)

**What Memloft Built:**

Memloft implements a personal memory system with hybrid search:

- **Storage:** SQLite with `memory` and `memory_vectors` tables
- **Search:** Hybrid FTS5 + vector with Reciprocal Rank Fusion (RRF) merge
- **Embeddings:** Candle backend running all-MiniLM-L6-v2 locally (384 dimensions)
- **Fallback:** `FallbackBackend` wraps `Option<Box<dyn EmbeddingBackend>>` — gracefully degrades to FTS5-only if candle fails
- **Background Indexing:** `EmbeddingIndexer` processes new memories in batches, uses content-hash to skip unchanged records
- **Tier System:** Three tiers (working/long_term/archive) with different scoring multipliers:
  - Working: 1.0x (recent, actively used)
  - Long-term: 0.7x (validated, stable)
  - Archive: 0.3x (old, rarely accessed)
- **Scoring:** Composite scoring with recency, importance, category, and tier multipliers
- **Deduplication:** SHA-256 content hash prevents exact duplicates

**What Works Well:**
- RRF hybrid search catches both keyword matches and semantic matches
- Dynamic k-value adjustment (quoted text → favor keywords, questions → favor semantic)
- Candle embedding is pure Rust, ~8ms per embed, no native dependency pain
- FallbackBackend means the system works even if model download fails
- Background indexing doesn't block the main thread
- Content-hash skip avoids re-embedding unchanged memories

**What's Missing:**
- No token-budget context assembly — caller must manage prompt size
- No memory relationships — flat structure only
- Tier system is ad-hoc — manual rules for promotion/demotion
- No cognitive type distinction (all memories treated the same)
- No consolidation beyond exact-hash dedup

**What MemCore Extracts:**
- RRF merge algorithm with dynamic k-values
- CandleBackend implementation
- FallbackBackend wrapper pattern
- EmbeddingIndexer with batch processing and content-hash skip
- Composite scoring with multiple strategy types
- SHA-256 dedup

---

### PIRDLY (Two-Tier + Error Classification)

**What PIRDLY Designed (Not Yet Built):**

PIRDLY's architecture document specifies:

- **Two-Tier Memory:**
  - Global (`~/.pirdly/global.db`): Error patterns, language learnings, cross-project knowledge
  - Project (`.pirdly/memory.db`): Architecture decisions, project conventions, task history
  - Automatic promotion when project-specific patterns appear in N different projects
- **Error Classification:**
  - Transient: Network errors, timeouts → retry with exponential backoff + jitter
  - QuotaExceeded: Rate limits, subscription caps → wait for reset
  - Permanent: Invalid config, missing tools → surface to user, don't retry
- **MCP Server:** Expose memory as MCP tools for direct LLM access

**What PIRDLY Contributes to MemCore:**
- Two-tier database management (global + project)
- Promotion logic for cross-project patterns
- Error classification model
- MCP server interface design

---

### Unified Pattern Map

```
┌──────────────────────────────────────────────────────────────────┐
│                         MemCore                                  │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │    Dial      │  │   Memloft    │  │      PIRDLY          │   │
│  │             │  │              │  │                      │   │
│  │ FTS5+Porter │  │ Vector+RRF   │  │ Two-tier DB          │   │
│  │ Stop-words  │  │ CandleBackend│  │ Error classification │   │
│  │ Trust score │  │ FallbackBknd │  │ MCP interface        │   │
│  │ Context asm │  │ BG indexer   │  │ Promotion logic      │   │
│  │ Priority    │  │ Content hash │  │                      │   │
│  │ Error class │  │ Tier scoring │  │                      │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    New (from Research)                     │   │
│  │                                                          │   │
│  │ ACT-R activation model    (replaces trust + tiers)       │   │
│  │ Consolidation pipeline    (from Mem0 research)           │   │
│  │ Cognitive memory types    (from CoALA framework)         │   │
│  │ Graph relationships       (from OMEGA, LightRAG, Zep)   │   │
│  │ Temporal validity         (from Zep/Graphiti)            │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Academic & Theoretical Foundations

### CoALA: Cognitive Architectures for Language Agents

**Source:** "Cognitive Architectures for Language Agents" (arXiv:2309.02427)

**Key Insight:** Agent memory maps to cognitive science categories:

| Cognitive Type | Definition | Agent Example | Decay Behavior |
|---------------|------------|---------------|----------------|
| **Episodic** | What happened | Debug sessions, iteration logs, conversation history | Fades fast — yesterday's debug session is rarely useful next week |
| **Semantic** | What I know | "The project uses PostgreSQL", "The API requires JWT auth" | Stable — facts don't decay unless superseded |
| **Procedural** | How to do things | "When you see error X, fix with Y", build workflows | Strengthens with use — proven patterns become more valuable |

**Why This Matters for MemCore:**

Current systems (Dial, Memloft) treat all memories equally. A learning from yesterday gets the same base score as a learning from six months ago. CoALA's type classification enables:

1. **Type-appropriate decay rates** — episodic memories fade fast, semantic memories persist
2. **Type-appropriate scoring** — procedural memories that have been validated are boosted
3. **Type-appropriate storage** — episodic memories can be pruned aggressively, semantic memories archived carefully

**MemCore Implementation:** The `MemoryType` enum (Episodic/Semantic/Procedural) on the `MemoryRecord` trait. Each type gets different decay parameters in the activation model.

---

### ACT-R: Adaptive Control of Thought — Rational

**Source:** Anderson, J.R. et al. "ACT-R: A Theory of Higher-Level Cognition and Its Relation to Visual Attention" (Human-Computer Interaction, 12(4))

Also: "MemoryCode: Multi-Agent System With Cognitive-Inspired Memory System" (ACM, 2025)

**Key Insight:** Memory retrieval strength is a function of recency and frequency of access, modeled by a power-law decay:

```
activation(i) = base_level(i) + Σ ln(t_j ^ -d)
```

Where:
- `t_j` = time since the j-th access (in seconds)
- `d` = decay rate (0.5 in standard ACT-R)
- The sum is over all recorded accesses

**Properties of this model:**

1. **Recency effect:** Recent accesses contribute more to activation
2. **Frequency effect:** More total accesses = higher activation
3. **Spacing effect:** Distributed accesses (spaced repetition) produce stronger activation than massed practice
4. **Power-law forgetting:** Activation decreases as a power function of time, not exponentially

**Why This Replaces Ad-Hoc Systems:**

| Ad-Hoc System | What It Does | ACT-R Equivalent |
|---------------|-------------|------------------|
| Dial's trust scoring | Manual confidence 0.0-1.0, decayed by time | Activation computed from access history |
| Dial's times_referenced | Counter of how many times used | Access log (richer — includes timestamps) |
| Memloft's tier system | working/long_term/archive with multipliers | Activation naturally creates tiers (high/medium/low) |
| Memloft's recency boost | Exponential decay with 30-day half-life | Power-law decay (more gradual, more accurate) |
| Manual confidence decay | 0.05 per 30 days without validation | Forgetting curve handles this automatically |

One formula replaces five mechanisms.

**MemCore Implementation:** The `ActivationScorer` computes activation from the `memory_access_log` table at query time. Different decay rates per `MemoryType`.

**Calibration (decay rate `d` by memory type):**

| Type | d | Rationale |
|------|---|-----------|
| Episodic | 0.5 | Standard ACT-R value. Session logs fade in ~7 days without re-access. |
| Semantic | 0.2 | Facts persist longer. "The project uses PostgreSQL" stays relevant for months. |
| Procedural | 0.3 | Patterns strengthen with use. A fix accessed weekly stays strong; one never accessed fades in ~30 days. |

These are starting values. Real calibration requires measuring retrieval precision on actual agent memory databases — a future task.

---

### Mem0: Memory Consolidation Pipeline

**Source:** Mem0 open-source project (github.com/mem0ai/mem0, 37K+ stars)

**Key Insight:** Storing memories without consolidation leads to bloat. Over months of operation, agent memories accumulate duplicates, near-duplicates, contradictions, and superseded facts. Mem0's solution is a three-stage pipeline:

```
Extract → Consolidate → Store
```

1. **Extract:** Parse the new memory from raw input. Identify key facts, entities, relationships.
2. **Consolidate:** Search for existing similar memories. Compare. Classify the action:
   - **ADD:** No similar memory exists. Store as new.
   - **UPDATE:** Similar memory exists but new version is more current/accurate. Replace.
   - **DELETE:** New information contradicts or supersedes existing. Remove the old.
   - **NOOP:** Exact or near-exact duplicate already exists. Do nothing.
3. **Store:** Execute the classified action. Index for search.

**Why This Matters:**

Without consolidation, a memory system that records "the build failed because of missing dependency X" and later records "the build failed because of missing dependency X (fixed by adding X to Cargo.toml)" has two memories where it should have one (the second, more complete one).

Mem0 uses an LLM to classify ADD/UPDATE/DELETE/NOOP. This is accurate but costs tokens.

**MemCore Implementation:** Three strategies, increasing in cost and accuracy:

| Strategy | Cost | Accuracy | How |
|----------|------|----------|-----|
| `HashDedup` | Zero | Catches exact duplicates only | SHA-256 of `searchable_text()` |
| `SimilarityDedup` | ~60ms | Catches near-duplicates | Embed → vector search → threshold |
| `LLMConsolidation` | LLM tokens | Handles contradictions and updates | Consumer provides LLM call |

The `ConsolidationStrategy` trait lets projects choose their level.

---

### Zep/Graphiti: Temporal Knowledge Graphs

**Source:** Zep/Graphiti project (github.com/getzep/graphiti)

**Key Insight:** Memories have temporal validity. "The project uses Express.js" might be true from January to March, then superseded by "The project migrated to Fastify in April." Without temporal modeling, the memory system returns stale facts.

Graphiti models this with `valid_from` and `valid_until` fields on both entities and relationships:

```
Entity: "Backend Framework = Express.js"
  valid_from: 2025-01-15
  valid_until: 2025-04-01

Entity: "Backend Framework = Fastify"
  valid_from: 2025-04-01
  valid_until: NULL (current)
```

**Why This Matters:**

Agent memory systems that don't model temporal validity will:
1. Return outdated facts ("use Express.js" when the project migrated months ago)
2. Contradict themselves (both "use Express.js" and "use Fastify" are "true")
3. Confuse the LLM with conflicting context

**MemCore Implementation:** Optional `valid_from` and `valid_until` fields on `MemoryRecord`, feature-gated behind `temporal`. Search can filter by `valid_at(timestamp)` to get the truth as of a specific point in time.

---

### OMEGA Memory: Forgetting Intelligence

**Source:** OMEGA Memory project (github.com/omega-memory/core)

**Key Insight:** Not all memories should persist forever. OMEGA implements "forgetting intelligence" — memories that are never accessed gradually decay and can be pruned. But certain categories (error patterns, validated solutions) are exempt from decay because their value is proven.

**Performance Benchmarks (Reference Targets):**
- ~8ms embedding time (ONNX, CPU-only)
- <50ms query latency (hybrid search, 10K memories)
- #1 on LongMemEval benchmark (95.4%)

**MemCore's Take:** The ACT-R activation model subsumes OMEGA's forgetting intelligence. Memories naturally decay through the power-law formula. Memories that are accessed frequently (like proven error patterns) stay strong. No need for explicit "exempt from decay" rules — the math handles it.

---

### Reciprocal Rank Fusion (RRF)

**Source:** "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (Cormack, Clarke, Buettcher, 2009)

**Key Insight:** When merging ranked lists from different retrieval systems (e.g., keyword search and vector search), RRF provides a simple, effective fusion:

```
RRF_score(d) = Σ(r∈R) 1 / (k + r(d))
```

Where:
- `R` is the set of rankers (FTS5, vector)
- `r(d)` is the rank of document `d` in ranker `r`
- `k` is a smoothing constant (typically 60)

**Properties:**
- Robust to score scale differences between systems
- No training data needed
- Works with any number of rankers
- Outperforms individual systems and most learned fusion methods

**Memloft's Enhancement:** Dynamic k-values based on query analysis:
- Quoted text → lower keyword k (favor exact matches)
- Question words → lower vector k (favor semantic matches)
- Default → equal k values

**MemCore Implementation:** Direct port from Memloft with the dynamic k-value logic.

---

## 5. Component Research

### Search: FTS5 vs. Vector vs. Hybrid

**Evaluation on agent memory workloads:**

| Query Type | FTS5 | Vector | Hybrid (RRF) |
|-----------|------|--------|---------------|
| Error messages ("cargo build failed: missing crate") | Excellent — exact keyword match | Good — semantic similarity | Excellent |
| Code references ("impl Display for MyType") | Excellent — exact match | Poor — code tokens are noisy | Excellent (FTS5 dominates) |
| Conceptual ("how to handle authentication") | Poor — keyword mismatch | Excellent — semantic match | Excellent (vector dominates) |
| Paraphrased ("login not working" → "auth error") | Poor — different keywords | Good — semantic similarity | Good (vector saves it) |
| Mixed ("JWT token 401 unauthorized") | Good — partial keyword match | Good — semantic similarity | Excellent (both contribute) |

**Conclusion:** FTS5 alone covers ~80% of real agent memory queries (error messages, code, tool names are inherently keyword-rich). Hybrid captures the remaining 20% where semantic matching adds value. FTS5 should always be enabled; vector should be optional.

### Embedding: Candle vs. ort vs. API

| Approach | Binary Impact | Latency | Dependencies | Offline? |
|----------|--------------|---------|-------------|----------|
| **Candle** (ModernBERT/BERT) | +30MB | ~8ms/embed | Pure Rust | Yes |
| **ort** (ONNX Runtime) | +350MB, 80+ deps | ~5ms/embed | C++ runtime | Yes |
| **API** (OpenAI, Cohere) | +5MB (HTTP client) | ~100-500ms | Network | No |
| ~~**fastembed-rs**~~ | ~~Uses ort internally~~ | ~~~5ms/embed~~ | ~~Same as ort~~ | ~~Yes~~ |

**Decision:** Candle with custom embedding module (Decision 016, updated 2026-03-18). The ~3ms latency difference vs. ort is irrelevant for agent memory workloads (embedding happens in background). The dependency and binary size savings are significant. Candle also supports WASM. fastembed-rs was evaluated and rejected due to single-maintainer risk, pre-release ort pin, and 50-150MB C++ shared library dependency.

**Model choice:** granite-embedding-small-english-r2 (Decision 017)
- 384 dimensions (compact vectors, compatible with bge-small WASM fallback)
- 47M parameters, ModernBERT architecture with Flash Attention 2
- 8,192 token context (captures full error traces, code blocks, decision rationale)
- MTEB-v2 retrieval: 53.9 (matches bge-small), CoIR code retrieval: 53.8 (17% better than bge-small)
- Loaded via candle-transformers native ModernBERT; model.safetensors auto-downloaded from HuggingFace and cached at `~/.cache/memcore/models/`
- WASM fallback: bge-small-en-v1.5 via candle BERT (same 384-dim, cross-compatible vectors)

### Vector Storage: Brute Force vs. ANN

| Approach | Memory Count | Latency | Dependencies |
|----------|-------------|---------|-------------|
| Brute-force dot product (f32 BLOB in SQLite) | <100K | <50ms | None |
| sqlite-vec (quantized ANN) | 100K-1M | <10ms | sqlite-vec extension |
| External vector DB (Qdrant, Milvus) | >1M | <5ms | External service |

**Decision:** Start with brute force. Personal/project memory databases will rarely exceed 10K entries. sqlite-vec available as upgrade path via feature flag.

**Brute-force implementation:**
```rust
// Store: serialize f32 vec as little-endian bytes
fn store_vector(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// Search: load all vectors, compute dot product
fn search_vectors(query: &[f32], vectors: &[(i64, Vec<u8>)]) -> Vec<(i64, f32)> {
    vectors.iter().map(|(id, blob)| {
        let emb: Vec<f32> = blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let sim = dot_product(query, &emb);
        (*id, sim)
    }).collect()
}
```

For 10K vectors at 384 dimensions: 10,000 * 384 * 4 bytes = ~15MB in memory. Trivial.

### Graph: SQLite CTEs vs. Native Graph DB

| Approach | Scale | Query Language | Dependencies | Status |
|----------|-------|---------------|-------------|--------|
| SQLite recursive CTEs | <100K relations | SQL | None | Stable, production-ready |
| Cozo (embedded Datalog) | Millions | Datalog | Pure Rust | Active, less proven |
| Kuzu (embedded Cypher) | Millions | Cypher | C++ bindings | Archived Oct 2025 |
| SurrealDB | Millions | SurrealQL | Complex | Over-engineered for this |

**Decision:** SQLite recursive CTEs. They're built into the database we already use, handle the scale we need (personal/project memory means hundreds to low thousands of relationships), and add zero dependencies. The `GraphBackend` trait provides an upgrade path to Cozo or a Kuzu fork if someone needs million-scale graph traversal.

**Recursive CTE performance (SQLite):**
- 1K relationships, 3-hop traversal: <1ms
- 10K relationships, 3-hop traversal: <5ms
- 100K relationships, 3-hop traversal: ~50ms (approaching the limit)

### Memory Decay: Ad-Hoc vs. ACT-R

| System | Formula | Parameters | Data Needed |
|--------|---------|------------|-------------|
| Dial's trust | `confidence -= 0.05 per 30 days` | Hard-coded constants | Last access date |
| Memloft's tiers | `score *= tier_multiplier` | 3 fixed multipliers | Tier assignment |
| Exponential decay | `score *= e^(-λt)` | λ (half-life) | Creation date |
| **ACT-R activation** | `base + Σ ln(t_j^-d)` | d (decay rate), base level | Full access history |

**Why ACT-R wins:**

1. **One model replaces three.** Trust scoring, tier multipliers, and recency boost are all subsumed.
2. **Research-backed.** Power-law forgetting is empirically validated across cognitive science.
3. **Self-adjusting.** Frequently accessed memories stay strong automatically. No manual tier management.
4. **Spaced repetition effect.** Distributed accesses produce stronger activation than massed access — matches real-world patterns where useful memories are accessed across multiple sessions.
5. **Per-type tuning.** Different `d` values for episodic/semantic/procedural enable type-appropriate behavior.

**Cost:** Requires an `access_log` table. Each search result access records a row. At 100 searches/day with 10 results each, that's 1000 rows/day, ~365K rows/year. Trivial for SQLite.

---

## 6. Technology Evaluation

### SQLite Configuration Research

**WAL Mode (Write-Ahead Logging):**

WAL is non-negotiable for MemCore. Without it, any concurrent read/write (e.g., searching while indexing) locks the database.

```sql
PRAGMA journal_mode = WAL;
```

Measured impact:
- Read throughput: ~10x improvement under concurrent load
- Write throughput: ~2x improvement for single-writer workloads
- Downside: WAL file can grow large during long write transactions. Periodic `PRAGMA wal_checkpoint(TRUNCATE)` prevents this.

**`synchronous = NORMAL`:**

In WAL mode, `NORMAL` is corruption-safe (commits are durable after WAL sync). `FULL` adds an extra fsync per transaction that provides no additional safety in WAL mode but costs ~50% write throughput.

**FTS5 Configuration:**

```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
    searchable_text,
    category,
    tokenize='porter'
);
```

Porter stemming catches:
- "authenticate" → "authent" matches "authentication", "authenticating", "authenticated"
- "running" → "run" matches "runs", "runner", "ran"
- "failure" → "failur" matches "failures", "failed", "failing"

BM25 ranking (FTS5 default) naturally handles term frequency and document length normalization.

**Memory-mapped I/O:**

```sql
PRAGMA mmap_size = 268435456;  -- 256MB
```

Memory-mapping the database file eliminates read() system calls for hot data. For a 50MB memory database, the entire file lives in the page cache after first access.

### Rust Dependency Analysis

**Always required (minimal footprint):**

| Crate | Version | Size Impact | Purpose |
|-------|---------|-------------|---------|
| rusqlite | 0.32 | ~2MB (bundled SQLite) | Database engine |
| serde + serde_json | 1.x | ~500KB | Serialization |
| chrono | 0.4 | ~300KB | Timestamps |
| thiserror | 2.x | ~0 (proc macro) | Error types |
| sha2 | 0.10 | ~100KB | Content hashing |
| async-trait | 0.1 | ~0 (proc macro) | Async traits |
| tokio | 1.x | ~1MB (rt + sync only) | Async runtime |
| tracing | 0.1 | ~200KB | Logging |

Total: ~4MB

**Feature-gated (heavy):**

| Crate | Feature Flag | Size Impact | Purpose |
|-------|-------------|-------------|---------|
| candle-core | local-embeddings | ~15MB | Tensor operations |
| candle-nn | local-embeddings | ~5MB | Neural network layers |
| candle-transformers | local-embeddings | ~8MB | Transformer models |
| tokenizers | local-embeddings | ~5MB | BPE tokenization |
| hf-hub | local-embeddings | ~2MB | Model download |
| axum + tower | mcp-server | ~5MB | HTTP server |

Total with all features: ~45MB

### Compile Time Impact

| Configuration | Estimated Compile (Debug) | Estimated Compile (Release) |
|--------------|--------------------------|----------------------------|
| Default (FTS5 only) | ~15s | ~45s |
| + vector-search | ~60s | ~180s |
| + mcp-server | ~25s | ~75s |
| Full | ~80s | ~240s |

Candle dominates compile time. Feature-gating it is essential for projects that don't need vector search.

---

## 7. Design Decisions & Rationale

### D1: Library, Not Framework

**Decision:** MemCore is a library crate that projects call into, not a framework that structures the project.

**Alternatives Considered:**
- Framework with plugin system (like MemOS)
- Service with API (like Mem0 cloud)
- CLI tool (like Engram)

**Rationale:** Dial, Memloft, and PIRDLY have fundamentally different architectures. A framework would force them to restructure. A service adds deployment complexity. A CLI tool can't be embedded. A library crate that provides `MemoryEngine<T>` fits all three projects without requiring architectural changes.

### D2: Generic over MemoryRecord

**Decision:** `MemoryEngine<T: MemoryRecord>` is generic over the consumer's memory type.

**Alternatives Considered:**
- Fixed `Memory` struct with metadata map
- Dynamic schema with JSON columns
- Trait objects (`Box<dyn MemoryRecord>`)

**Rationale:** Each project has different memory types with different fields:
- Dial: `Learning { description, category, times_referenced }`
- Memloft: `Memory { topic, content, context, importance }`
- PIRDLY: `ErrorPattern { pattern, regex, occurrence_count }`

A generic approach lets each project define its own struct, implement `MemoryRecord`, and get the full engine. The `record_json` column stores the serialized struct for reconstruction.

### D3: Feature Flags Over Runtime Configuration

**Decision:** Heavy dependencies are behind compile-time feature flags, not runtime configuration.

**Alternatives Considered:**
- Runtime feature detection (check if candle works, fall back)
- Plugin system (dynamic loading)
- Separate crates (memcore-fts, memcore-vector, memcore-graph)

**Rationale:** Feature flags have zero runtime cost when disabled. A project that only needs FTS5 compiles in 15 seconds and adds ~2MB. Separate crates would create dependency management overhead. Runtime detection adds complexity. The `FallbackBackend` provides runtime graceful degradation for the vector search case specifically.

### D4: Composable Scoring Over Single Algorithm

**Decision:** Multiple `ScoringStrategy` implementations composed via `CompositeScorer`.

**Alternatives Considered:**
- Single scoring formula with configurable weights
- ML-learned ranking model
- No post-search scoring (raw FTS5/vector scores only)

**Rationale:** Different projects value different signals:
- Dial cares about trust (was this solution validated?)
- Memloft cares about recency and importance
- PIRDLY cares about memory type and project relevance

Composable strategies let each project mix and match. A single formula can't serve all cases. ML-learned ranking requires training data we don't have yet.

### D5: SQLite Everywhere

**Decision:** Use SQLite for everything — main storage, FTS5, vector storage, access logs, relationships.

**Alternatives Considered:**
- PostgreSQL (more features, more complexity)
- Separate stores (SQLite for data, Qdrant for vectors, Neo4j for graphs)
- In-memory only (HashMap-based)

**Rationale:** Single-file databases are portable, backupable, and require zero infrastructure. WAL mode handles concurrency. FTS5 handles text search. BLOBs handle vector storage. Recursive CTEs handle graph traversal. All within one database file. Adding external services for vector or graph means deployment complexity that contradicts the local-first principle.

---

## 8. Competitive Positioning

### Feature Comparison Matrix

| Feature | MemCore | OMEGA | Engram | Mem0 | MemOS |
|---------|---------|-------|--------|------|-------|
| **Language** | Rust | TypeScript | Go | Python | Python |
| **Storage** | SQLite | SQLite | SQLite | Cloud/Local | SQLite |
| **FTS5 Search** | Yes | Yes | Yes | No | Yes |
| **Vector Search** | Yes (candle) | Yes (ONNX) | No | Yes (cloud) | Yes |
| **Hybrid RRF** | Yes | Unclear | No | No | Yes |
| **Graph Memory** | Yes (SQLite CTEs) | Yes | No | Yes (cloud) | Yes |
| **Memory Decay** | ACT-R model | Custom decay | No | No | No |
| **Consolidation** | 3-tier pipeline | No | No | LLM-based | No |
| **Token Budget** | Yes | No | No | No | No |
| **Two-Tier** | Yes | No | No | No | No |
| **Feature Flags** | Yes | No | No | No | No |
| **MCP Server** | Optional | Yes | Yes | No | No |
| **Library Crate** | Yes | No (MCP only) | No (binary) | No (cloud) | No (framework) |
| **Offline** | Yes | Yes | Yes | No | Yes |
| **Binary Size** | 2-45MB | ~100MB | ~15MB | N/A | N/A |

### MemCore's Unique Advantages

1. **Rust + feature flags** — only compile what you need, from 2MB to 45MB
2. **Library crate** — embed in any Rust project, not a separate service
3. **ACT-R activation** — research-backed decay model, not ad-hoc rules
4. **Three-tier consolidation** — from zero-cost hash dedup to LLM-assisted
5. **Token-budget context assembly** — built-in, not an afterthought
6. **Two-tier memory** — global + project with automatic promotion
7. **Pure Rust embeddings** — candle, no C++ runtime required

### Where Competitors Are Stronger

1. **Mem0:** Larger community (37K stars), more LLM integrations, cloud sync
2. **OMEGA:** #1 on LongMemEval (95.4%), encryption at rest, more mature MCP tools
3. **MemOS:** Multi-modal memory (images, tool traces), more sophisticated graph
4. **Engram:** Simpler to deploy (single Go binary with TUI)

---

## 9. Risk Analysis

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Candle API changes breaking embeddings | Medium | Medium | Pin candle version, `EmbeddingBackend` trait isolates changes |
| SQLite FTS5 not available on all platforms | Low | High | rusqlite `bundled` feature compiles SQLite from source |
| ACT-R decay rates poorly calibrated | Medium | Low | Configurable `d` parameter, tune after measuring precision |
| Content-hash collisions (SHA-256) | Near zero | Low | SHA-256 collision is computationally infeasible |
| Memory database grows unbounded | Medium | Medium | Consolidation + activation-based pruning of low-activation memories |
| sqlite-vec Rust bindings immature | High | Low | Feature-gated, brute force is fine for <100K vectors |

### Adoption Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Too complex API surface | Medium | High | Builder pattern with sensible defaults, `default` feature is minimal |
| Migration effort too high for Dial/Memloft | Medium | Medium | Provide concrete migration guides, `MemoryRecord` is simple to implement |
| Feature flag combinatorial explosion | Low | Medium | Clear feature dependency chain, `full` flag for everything |
| Crate name collision on crates.io | Medium | Low | Check availability before publishing, have alternatives ready |

### Performance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Activation computation too slow at scale | Low | Medium | Cache activation scores, recompute on access |
| Graph traversal too slow with many relationships | Low | Medium | Depth limits, feature-gate to native graph DB |
| Background indexer blocking main thread | Low | High | Tokio spawn_blocking for CPU-intensive embedding |
| FTS5 index corruption | Very low | High | WAL mode, periodic integrity checks |

---

## 10. Specification Summary

### What MemCore Is

A standalone Rust crate providing pluggable, feature-gated persistent memory for AI agent applications. It stores, searches, scores, decays, consolidates, and assembles memories for LLM context injection.

### Core Capabilities

| Capability | Implementation | Feature Flag |
|-----------|---------------|-------------|
| Persistent storage | SQLite + WAL + mmap | Always on |
| Keyword search | FTS5 + Porter stemming + BM25 | `fts5` (default) |
| Vector search | Candle all-MiniLM-L6-v2, brute-force similarity | `vector-search` |
| Hybrid search | Reciprocal Rank Fusion with dynamic k-values | `vector-search` |
| Graph memory | SQLite relationship tables + recursive CTE traversal | `graph-memory` |
| Memory decay | ACT-R activation model with per-type decay rates | `activation-model` |
| Consolidation | Hash dedup / similarity dedup / LLM-assisted | `consolidation` |
| Context assembly | Token-budget-aware priority-ranked selection | Always on |
| Two-tier memory | Global + project databases with promotion | `two-tier` |
| Temporal validity | valid_from / valid_until on memories and relations | `temporal` |
| Background indexing | Batch embedding with content-hash skip | `vector-search` |
| MCP server | JSON-RPC 2.0 interface for LLM tool access | `mcp-server` |

### Primary Interface

```rust
// Consumer defines their memory type
struct Learning { /* ... */ }
impl MemoryRecord for Learning { /* ... */ }

// Build the engine
let engine = MemoryEngine::<Learning>::builder()
    .database("memory.db")
    .build()
    .await?;

// Store (with automatic consolidation)
engine.store(learning).await?;

// Search (fluent API)
let results = engine.search("authentication error")
    .mode(SearchMode::Auto)
    .limit(10)
    .execute()
    .await?;

// Assemble context for LLM
let context = engine.assemble_context("fix the auth bug", &budget)?;
```

### Performance Targets

| Operation | Target | At Scale |
|-----------|--------|----------|
| FTS5 search | <5ms | 10K memories |
| Vector embed | <10ms | Single text |
| Brute-force vector scan | <50ms | 100K vectors |
| RRF merge | <1ms | Computation only |
| Graph traversal (3 hops) | <10ms | 10K relationships |
| Context assembly | <5ms | After search |
| Store with hash dedup | <2ms | Single insert |

### Estimated Size

~4-5K lines of Rust for the core engine (all features).

### Implementation Order

| Phase | Components | Dependencies |
|-------|-----------|-------------|
| 1 | Storage engine, schema, migrations, CRUD | rusqlite |
| 2 | FTS5 search, Porter stemming, BM25 | Phase 1 |
| 3 | Scoring strategies (recency, importance, composite) | Phase 2 |
| 4 | Token-budget context assembly | Phase 3 |
| 5 | Content-hash consolidation (HashDedup) | Phase 1 |
| 6 | Two-tier memory (global + project) | Phase 1 |
| 7 | ACT-R activation model | Phase 1 |
| 8 | CandleBackend + FallbackBackend | Phase 1 |
| 9 | Background embedding indexer | Phase 8 |
| 10 | Hybrid search (vector + RRF) | Phases 2, 8 |
| 11 | Graph relationships + CTE traversal | Phase 1 |
| 12 | Similarity-based consolidation | Phases 5, 8 |
| 13 | Temporal validity | Phase 1 |
| 14 | MCP server | Phase 4 |

Phases 1-6 deliver a fully functional memory engine with FTS5 search, scoring, context assembly, and two-tier memory. Phases 7-14 add the advanced capabilities.

---

## References

### Academic

1. Anderson, J.R. (1993). *Rules of the Mind.* Lawrence Erlbaum Associates. (ACT-R theory)
2. Anderson, J.R. & Schooler, L.J. (1991). "Reflections of the environment in memory." *Psychological Science*, 2(6), 396-408. (Power-law forgetting)
3. Sumers, T.R. et al. (2024). "Cognitive Architectures for Language Agents." arXiv:2309.02427. (CoALA framework)
4. Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009. (RRF)
5. Liu, Z. et al. (2025). "MemoryCode: Multi-Agent System With Cognitive-Inspired Memory System." ACM. (ACT-R for code agents)

### Open Source Projects

6. Mem0 — https://github.com/mem0ai/mem0 (consolidation pipeline)
7. OMEGA Memory — https://github.com/omega-memory/core (forgetting intelligence, benchmarks)
8. Zep/Graphiti — https://github.com/getzep/graphiti (temporal knowledge graphs)
9. MemOS — https://github.com/MemTensor/MemOS (graph memory, hybrid search)
10. Engram — https://github.com/Gentleman-Programming/engram (SQLite+FTS5, Go reference)
11. LightRAG — https://github.com/HKUDS/LightRAG (knowledge graph RAG)
12. Memloft — (internal, hybrid RRF search, candle embeddings)
13. Dial — (internal, FTS5 memory, context assembly)

### Ecosystem

14. rusqlite — https://github.com/rusqlite/rusqlite (Rust SQLite bindings)
15. candle — https://github.com/huggingface/candle (Rust ML framework)
16. sqlite-vec — https://github.com/asg017/sqlite-vec (SQLite vector extension)
17. all-MiniLM-L6-v2 — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (embedding model)
