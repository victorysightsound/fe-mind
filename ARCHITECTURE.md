# FeMind: Universal Memory Engine Architecture

**Version:** 0.2.0
**Updated:** 2026-03-25
**Status:** local repo/crate rename to `fe-mind` / `femind` is complete in this workspace. External publication work is still pending.

---

## Current Pipeline (2026-03-25)

### Ingest Modes

**1. LLM Extraction (production mode):**
```
Raw text → LlmCallback → extract facts → store each fact as individual memory
  → compute embedding → create graph edges (SupersededBy, RelatedTo)
  → deduplicate via content hash
```
Method: `engine.store_with_extraction(text, llm)`

**2. Chunk-based (benchmark/no-LLM mode):**
```
Session text → chunk at configurable size → store each chunk as memory
  → batch compute embeddings → diversify by session
```
Method: `engine.store(record)` / `engine.store_batch(records)`

### Search Pipeline
```
Query → multi-query (original + key-phrases)
  → Hybrid: FTS5 OR-mode + vector similarity + RRF fusion
  → Graph filtering: demote superseded results (if graph_enabled)
  → Recency weighting (configurable)
  → Session diversification (configurable limit)
  → Context assembly within token budget
```
Method: `engine.assemble_context(query, budget)` or `engine.assemble_context_with_config(query, budget, config)`

### Runtime Feature Toggles (EngineConfig)

| Toggle | Controls | Default |
|--------|----------|---------|
| `embedding_enabled` | Vector embedding at store time | true |
| `graph_enabled` | Graph edge creation + search filtering | true |
| `dedup_enabled` | Content hash deduplication | true |
| `vector_search_mode` | "exact" / "ann" / "off" | "exact" |

### Search Tuning (AssemblyConfig)

| Setting | Controls | Default |
|---------|----------|---------|
| `max_per_session` | Diversification (0=unlimited, 1=max diversity) | 1 |
| `recency_boost` | Score boost for newer content (0.0-1.0) | 0.0 |
| `search_limit` | Max results from multi-query search | 200 |
| `graph_depth` | Graph traversal hops (0=off) | 0 |

### Presets
- `AssemblyConfig::multi_session()` — max 1/session, no recency (LongMemEval)
- `AssemblyConfig::single_document()` — unlimited, 0.3 recency (MAB)

### Embedding Backends

| Backend | Feature | Usage |
|---------|---------|-------|
| `CandleNativeBackend` | `local-embeddings` | all-MiniLM-L6-v2 via Candle BERT, local CPU |
| `ApiBackend` | `api-embeddings` | DeepInfra/OpenAI-compatible /v1/embeddings |
| `FallbackBackend` | — | API-first with local fallback |
| `NoopBackend` | — | Zero vectors for testing |

### LLM Backends

| Backend | Feature | Usage |
|---------|---------|-------|
| `ApiLlmCallback` | `api-llm` | DeepInfra/OpenAI-compatible /v1/chat/completions |
| `CliLlmCallback` | `cli-llm` | Claude/ChatGPT/Gemini via CLI |

### ANN Vector Search

| Mode | Feature | Implementation |
|------|---------|---------------|
| `"exact"` | — | Brute-force cosine similarity (default) |
| `"ann"` | `ann` | HNSW via instant-distance with a shared in-memory index that rebuilds when stored vectors change |
| `"off"` | — | FTS5 keyword search only |

Current stabilization note: non-LLM verification now confirms `vector_search_mode` selects real runtime behavior for `exact`, `ann`, and `off`. The remaining approval-gated work is live CLI/API LLM validation, not search-mode implementation.

### Compile-Time Feature Flags

```toml
local-embeddings   # CandleNativeBackend (MiniLM local)
api-embeddings     # ApiBackend (DeepInfra embedding API)
api-llm            # ApiLlmCallback (chat completions API)
cli-llm            # CliLlmCallback (CLI tools)
llm-ingest         # LlmIngest extraction strategy
vector-search      # local-embeddings + tokio
graph-memory       # Graph relationship tables
ann                # HNSW approximate nearest neighbor
reranking          # Cross-encoder reranking (candle BERT)
temporal           # Temporal validity fields
consolidation      # Consolidation pipeline
activation-model   # ACT-R cognitive decay
two-tier           # Global + project databases
encryption         # SQLCipher encryption at rest
keychain           # OS keychain for encryption keys
mcp-server         # MCP server interface
full               # Everything except encryption/mcp-server
```

### Key Files

| File | Purpose |
|------|---------|
| `src/engine.rs` | MemoryEngine, EngineConfig, store/search/extract |
| `src/context/budget.rs` | AssemblyConfig, ContextBudget, ContextAssembly |
| `src/traits/llm.rs` | LlmCallback trait |
| `src/llm/api.rs` | ApiLlmCallback (OpenAI-compatible) |
| `src/llm/cli.rs` | CliLlmCallback (CLI tools) |
| `src/ingest/llm_extract.rs` | LLM fact extraction prompts + parser |
| `src/ingest/fact_extraction.rs` | Regex fact parser (structured data) |
| `src/ingest/chunking.rs` | Session chunking with overlap |
| `src/search/builder.rs` | SearchBuilder, hybrid search, RRF |
| `src/search/vector.rs` | Brute-force vector search + vector-count diagnostics |
| `src/search/ann.rs` | HNSW approximate nearest neighbor with invalidation/rebuild support |
| `src/search/fts5.rs` | FTS5 keyword search, stop words, sanitizer |
| `src/search/hybrid.rs` | RRF merge with dynamic k-values |
| `src/memory/relations.rs` | GraphMemory, RelationType, traversal |
| `src/embeddings/candle_native.rs` | all-MiniLM-L6-v2 local backend |
| `src/embeddings/api.rs` | OpenAI-compatible embedding API |

---

## Original Architecture (historical notes below this line may still reflect the prior project name)

## Overview

**FeMind** is a standalone Rust crate providing a pluggable, feature-gated memory engine for AI agent applications. It handles persistent storage, keyword search (FTS5), vector search (candle), hybrid retrieval (RRF), graph relationships, memory consolidation, cognitive decay modeling, and token-budget-aware context assembly.

### Design Principles

1. **Library, not framework** — projects call into FeMind, they are not structured around it
2. **Feature-gated everything** — heavy dependencies behind compile-time flags, zero cost when unused
3. **Opinionated about mechanics, unopinionated about schema** — FeMind handles how to store/search/decay; the consumer defines what a "memory" is
4. **Local-first** — SQLite-backed, single-file databases, no cloud dependency
5. **Pure Rust where possible** — candle over ort, SQLite over Postgres
6. **Proven patterns only** — every component is backed by published research or established open-source practice

### Informed By

- [Mem0](https://github.com/mem0ai/mem0) — consolidation pipeline (extract → consolidate → store)
- [Zep/Graphiti](https://github.com/getzep/graphiti) — temporal validity modeling
- [OMEGA Memory](https://github.com/omega-memory/core) — forgetting intelligence, performance benchmarks
- [CoALA framework](https://arxiv.org/pdf/2309.02427) — cognitive memory types (episodic/semantic/procedural)
- [ACT-R research](https://dl.acm.org/doi/10.1145/3765766.3765803) — activation-based decay modeling

---

## Crate Structure

```
femind/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Public API re-exports
│   ├── engine.rs              # MemoryEngine<T> — primary interface
│   ├── error.rs               # FemindError enum, Result type
│   │
│   ├── traits/
│   │   ├── mod.rs
│   │   ├── record.rs          # MemoryRecord trait (consumer implements)
│   │   ├── scoring.rs         # ScoringStrategy trait
│   │   ├── consolidation.rs   # ConsolidationStrategy trait
│   │   ├── evolution.rs       # EvolutionStrategy trait
│   │   └── reranker.rs        # RerankerBackend trait
│   │
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── engine.rs          # SQLite connection, WAL, pragmas
│   │   ├── schema.rs          # Schema creation, table definitions, index management
│   │   ├── migrations.rs      # Versioned schema migrations
│   │   ├── two_tier.rs        # Global + project database management
│   │   └── encryption.rs      # SQLCipher encryption support
│   │
│   ├── search/
│   │   ├── mod.rs
│   │   ├── builder.rs         # Fluent SearchBuilder API
│   │   ├── fts5.rs            # FTS5 keyword search, Porter stemming, stop-words
│   │   ├── vector.rs          # Brute-force cosine similarity search
│   │   ├── hybrid.rs          # Reciprocal Rank Fusion merge
│   │   └── query_expand.rs    # Time-aware query expansion (temporal expressions → date ranges)
│   │
│   ├── scoring/
│   │   ├── mod.rs
│   │   ├── activation.rs      # ACT-R activation-based scoring
│   │   ├── category.rs        # Category match boost
│   │   ├── composite.rs       # CompositeScorer (combines strategies)
│   │   ├── importance.rs      # Importance-based scoring
│   │   ├── memory_type.rs     # Cognitive type weighting
│   │   └── recency.rs         # Recency decay scoring
│   │
│   ├── embeddings/
│   │   ├── mod.rs
│   │   ├── backend.rs         # EmbeddingBackend trait
│   │   ├── candle_native.rs   # CandleNativeBackend: all-MiniLM-L6-v2 via BERT
│   │   ├── pooling.rs         # Mean pooling + L2 normalization
│   │   ├── noop.rs            # NoopBackend: zero vectors (testing)
│   │   └── fallback.rs        # FallbackBackend: graceful degradation
│   │
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── store.rs           # CRUD operations, deduplication
│   │   ├── hash_dedup.rs      # SHA-256 content hash deduplication
│   │   ├── similarity_dedup.rs # Vector similarity-based near-duplicate detection
│   │   ├── activation.rs      # ACT-R forgetting curve + access tracking
│   │   ├── relations.rs       # Graph relationships (memory_relations table)
│   │   └── pruning.rs         # Activation-based pruning with policy controls
│   │
│   ├── context/
│   │   ├── mod.rs
│   │   └── budget.rs          # Token-budget-aware context assembly
│   │
│   ├── ingest/
│   │   ├── mod.rs
│   │   └── passthrough.rs     # Default: store text as-is
│   │
│   └── callbacks/
│       ├── mod.rs
│       └── llm.rs             # LlmCallback trait (consumer-provided LLM access)
│
└── tests/                     # Integration tests
```

---

## Feature Flags

```toml
[features]
default = ["fts5"]

# Core search (always available)
fts5 = []                       # FTS5 + Porter stemming + BM25

# Vector search — custom candle embedding module (pure Rust, no ONNX Runtime)
# all-MiniLM-L6-v2 via BERT (native + WASM)
local-embeddings = [
    "dep:candle-core",
    "dep:candle-nn",
    "dep:candle-transformers",
    "dep:tokenizers",
    "dep:hf-hub"
]
vector-search = ["local-embeddings", "dep:tokio"]

# ANN indexing for >100K scale
ann = ["dep:instant-distance"]

# Cross-encoder reranking (post-RRF refinement, uses candle BERT)
reranking = ["local-embeddings"]

# Graph memory (SQLite relationship tables)
graph-memory = []

# Native graph DB for scale (alternative to SQLite graph)
# graph-native = ["dep:kuzu"]  # Future: when stable fork available

# Temporal validity fields
temporal = []

# Consolidation pipeline (extract → consolidate → store)
consolidation = []

# ACT-R activation-based decay model
activation-model = []

# MCP server interface
mcp-server = ["dep:axum", "dep:tower"]

# Two-tier memory (global + project databases)
two-tier = []

# Encryption at rest via SQLCipher
encryption = ["rusqlite/bundled-sqlcipher"]
encryption-vendored = ["encryption", "rusqlite/bundled-sqlcipher-vendored-openssl"]

# OS keychain integration for encryption key storage
keychain = ["dep:keyring"]

# Everything (native)
full = [
    "vector-search",
    "reranking",
    "graph-memory",
    "temporal",
    "consolidation",
    "activation-model",
    "two-tier"
]
```

### Feature Dependency Chain

```
fts5 (default, always on)
  └── vector-search
       ├── local-embeddings (candle — pure Rust, native + WASM)
       ├── vector-indexed (optional ANN / HNSW path for >100K)
       └── reranking (cross-encoder via candle BERT, post-RRF)

graph-memory (independent, SQLite-only)
temporal (independent, schema addition)
consolidation (independent, logic only)
activation-model (independent, logic only)
two-tier (independent, multi-db management)
mcp-server (independent, network stack)
encryption (swaps bundled SQLite for bundled SQLCipher)
  └── keychain (OS keychain key storage)
```

### Binary Size Impact

**Minimum Rust version: 1.85** (Rust edition 2024).

| Features Enabled | Approximate Binary Impact |
|-----------------|--------------------------|
| `default` (FTS5 only) | ~2MB (just rusqlite, no async runtime) |
| `+ graph-memory + temporal + consolidation + activation-model` | ~2.5MB (pure logic, no new deps) |
| `+ vector-search` (candle, pure Rust, pulls in tokio) | ~30-35MB |
| `+ reranking` (candle BERT cross-encoder) | +~0 (shares candle deps) |
| `+ vector-indexed` (sqlite-vec) | +~5MB |
| `+ mcp-server` (axum) | +~5MB |
| `+ encryption` (SQLCipher) | +~500KB-1MB over plain SQLite |
| `+ keychain` (keyring) | +~200-500KB |
| `full` (everything) | ~35-40MB |

---

## Core Traits

### MemoryRecord (Consumer Implements)

The central trait that defines what a "memory" is for a given project.

```rust
use chrono::{DateTime, Utc};
use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashMap;

/// Cognitive memory type classification (CoALA framework + Hindsight)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// What happened — events, sessions, iteration logs
    /// Decays fastest. Most useful when recent.
    Episodic,
    /// What I know — facts, preferences, project context
    /// Stable over time. Core knowledge.
    Semantic,
    /// How to do things — workflows, error patterns, solutions
    /// Strengthens with validation. Most valuable when proven.
    Procedural,
    /// What I conclude — agent-synthesized conclusions that can be revised.
    /// Has confidence score and provenance chain. Challengeable.
    /// (Deferred to post-v1 — see Decision Q3)
    // Belief,
}

/// Consumers implement this for their memory types.
/// femind handles storage, indexing, search, and decay.
/// The shared FeSuite harness at ../fe-harness/ uses this same surface for
/// resettable memory-backed fixture scenarios.
pub trait MemoryRecord: Send + Sync + Serialize + DeserializeOwned + 'static {
    /// Unique identifier
    fn id(&self) -> Option<i64>;

    /// Text to embed (vector search) and index (FTS5)
    fn searchable_text(&self) -> String;

    /// Cognitive memory type — determines decay rate and scoring
    fn memory_type(&self) -> MemoryType;

    /// Importance score (1-10, default 5). Affects scoring.
    fn importance(&self) -> u8 { 5 }

    /// When this memory was created
    fn created_at(&self) -> DateTime<Utc>;

    /// Optional category for boost matching (e.g., "error", "decision", "pattern")
    fn category(&self) -> Option<&str> { None }

    /// Optional metadata for filtering (key-value pairs)
    fn metadata(&self) -> HashMap<String, String> { HashMap::new() }

    /// Optional temporal validity — when this fact became true
    #[cfg(feature = "temporal")]
    fn valid_from(&self) -> Option<DateTime<Utc>> { None }

    /// Optional temporal validity — when this fact stopped being true
    #[cfg(feature = "temporal")]
    fn valid_until(&self) -> Option<DateTime<Utc>> { None }
}
```

**Example implementations:**

```rust
// A learning record for a coding agent
struct Learning {
    id: Option<i64>,
    description: String,
    category: String,        // "build", "test", "gotcha", etc.
    times_referenced: u32,
    created_at: DateTime<Utc>,
}

impl MemoryRecord for Learning {
    fn id(&self) -> Option<i64> { self.id }
    fn searchable_text(&self) -> String { self.description.clone() }
    fn memory_type(&self) -> MemoryType { MemoryType::Semantic }
    fn importance(&self) -> u8 { (self.times_referenced.min(10) as u8).max(3) }
    fn created_at(&self) -> DateTime<Utc> { self.created_at }
    fn category(&self) -> Option<&str> { Some(&self.category) }
}

// A failure pattern record
struct FailurePattern {
    id: Option<i64>,
    pattern: String,
    regex: String,
    occurrence_count: u32,
    created_at: DateTime<Utc>,
}

impl MemoryRecord for FailurePattern {
    fn id(&self) -> Option<i64> { self.id }
    fn searchable_text(&self) -> String { self.pattern.clone() }
    fn memory_type(&self) -> MemoryType { MemoryType::Procedural }
    fn importance(&self) -> u8 { (self.occurrence_count.min(10) as u8).max(5) }
    fn created_at(&self) -> DateTime<Utc> { self.created_at }
    fn category(&self) -> Option<&str> { Some("error") }
}

// A personal memory record
struct Memory {
    id: Option<i64>,
    topic: String,
    content: String,
    context: Option<String>,
    importance: u8,
    category: String,
    created_at: DateTime<Utc>,
}

impl MemoryRecord for Memory {
    fn id(&self) -> Option<i64> { self.id }
    fn searchable_text(&self) -> String {
        format!("{} {}", self.topic, self.content)
    }
    fn memory_type(&self) -> MemoryType {
        match self.category.as_str() {
            "decision" | "fact" | "preference" => MemoryType::Semantic,
            "pattern" | "lesson" | "workflow" => MemoryType::Procedural,
            _ => MemoryType::Episodic,
        }
    }
    fn importance(&self) -> u8 { self.importance }
    fn created_at(&self) -> DateTime<Utc> { self.created_at }
    fn category(&self) -> Option<&str> { Some(&self.category) }
}
```

### MemoryMeta

```rust
/// Extracted metadata from a MemoryRecord for use in scoring, consolidation,
/// and evolution traits. These traits cannot use `dyn MemoryRecord` because
/// `MemoryRecord` requires `Serialize + DeserializeOwned` (not object-safe).
/// The engine extracts `MemoryMeta` from `T: MemoryRecord` before passing
/// to scoring/consolidation/evolution strategies.
pub struct MemoryMeta {
    pub id: Option<i64>,
    pub searchable_text: String,
    pub memory_type: MemoryType,
    pub importance: u8,
    pub category: Option<String>,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}
```

### EmbeddingBackend

```rust
/// Requires Rust 1.85+ (edition 2024, native async fn in traits).
pub trait EmbeddingBackend: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Default: sequential. Implementations can optimize.
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Number of dimensions in output vectors
    fn dimensions(&self) -> usize;

    /// Whether the backend is ready to serve requests
    fn is_available(&self) -> bool;

    /// Model identifier for tracking which model produced a vector
    fn model_name(&self) -> &str;
}
```

**Shipped implementations:**

| Backend | Feature Flag | Default Model | Target | Use Case |
|---------|-------------|---------------|--------|----------|
| `CandleNativeBackend` | `local-embeddings` | `all-MiniLM-L6-v2` (BERT, 384-dim, 8K ctx) | Native | **Primary.** Pure Rust. Auto-downloads safetensors on first use. |
| `RemoteEmbeddingBackend` | `remote-embeddings` | `all-MiniLM-L6-v2` (BERT, 384-dim) | Native | **Remote execution.** Local-network embedding service with profile verification and optional local fallback. |
| `CandleWasmBackend` | `local-embeddings` | `all-MiniLM-L6-v2` (BERT, 384-dim, 512 ctx) | WASM | **Browser.** Standard BERT, smaller model, proven in candle WASM demos. |
| `NoopBackend` | (always available) | N/A | Any | Testing, returns zero vectors |
| `FallbackBackend` | (always available) | N/A | Any | Wraps `Option<Box<dyn EmbeddingBackend>>`, degrades gracefully to FTS5-only |

**Embedding model: `all-MiniLM-L6-v2`**

| Property | Value |
|----------|-------|
| Architecture | BERT (6 layers) |
| Parameters | 22M |
| Dimensions | 384 |
| Max Tokens | 256 |
| MTEB Retrieval | ~49 |
| License | Apache 2.0 |
| Download | ~80MB safetensors |

**One model everywhere:** all-MiniLM-L6-v2 uses standard BERT architecture which compiles to native and WASM. The same model is available on DeepInfra and other API providers. Vectors from local Candle and API are identical — fully interchangeable with zero quality degradation.

**Embedding identity:**
- canonical model label: `local-minilm`
- strict compatibility key: embedding profile (`provider|repo|dimensions|preprocessing|truncation`)

**Backend and runtime options:**
- `CandleNativeBackend` — local CPU inference (offline, free)
- `ApiBackend` — OpenAI-compatible API endpoint like DeepInfra (fast, cheap)
- `RemoteEmbeddingBackend` — local-network MiniLM execution with strict profile verification
- `FallbackBackend::api_with_local_fallback(api, local)` — tries API first, falls back to local
- target runtime modes: `local-cpu`, `local-gpu`, `remote-cpu`, `remote-gpu`, `off`
- `femind-embed-service` selects `auto`, `cpu`, or `cuda` on the host and reports the
  resolved execution mode via `/status`
- `femind-embed-service` also exposes a FeMind-native operator surface:
  - `serve` for the host process
  - `status` for resolved embedding runtime/config status
  - `verify-remote` for profile/auth verification against a configured remote host
  - when `reranking` is enabled, `/rerank/status` and `/rerank/rerank` plus
    `verify-remote-reranker` for the cross-encoder role
- WSL/Windows host lifecycle is handled by:
  - `scripts/remote/install-femind-embed-systemd.sh`
  - `scripts/remote/configure-windows-wsl-autostart.ps1`
  with `off`, `status`, `logon`, and `startup` modes for autostart behavior

**Model isolation (Decision 020):** When the `model_name` stored alongside a vector differs from the current backend's `model_name()`, vector search is skipped and the engine falls back to FTS5-only. This prevents cross-model comparison issues if the model is ever changed.

---

### Embedding Module — Implementation Reference

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (BERT, 22M params, 384-dim)

Auto-downloaded from HuggingFace on first use (~80MB), cached at `~/.cache/huggingface/hub/`.

**Candle API mapping:**

| Operation | Candle API | Module |
|-----------|-----------|--------|
| Load config | `serde_json::from_str::<bert::Config>(&json)` | `candle_transformers::models::bert` |
| Load weights | `VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)` | `candle_nn` |
| Build model | `BertModel::load(vb, &config)` | `candle_transformers::models::bert` |
| Tokenize | `Tokenizer::from_file(path)` + `encode_batch()` | `tokenizers` crate |
| Forward pass | `model.forward(&token_ids, &token_type_ids, Some(&attention_mask))` | BERT requires token_type_ids (zeros) |
| Mean pooling | Mask-weighted sum / mask count | Custom (~5 lines) |
| L2 normalize | `v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)` | Custom (1 line) |
| Download model | `Api::new()?.repo(Repo::with_revision(...)).get("model.safetensors")` | `hf_hub` crate |

**Usage examples:**

```rust
// Local backend (all-MiniLM-L6-v2 via Candle BERT)
let local = CandleNativeBackend::new()?;

// API backend (DeepInfra or any OpenAI-compatible endpoint)
let api = ApiBackend::deepinfra_minilm(api_key);

// Fallback: tries API first, falls back to local
let fallback = FallbackBackend::api_with_local_fallback(
    Box::new(api),
    Box::new(local),
);

// Wire into engine
let engine = MemoryEngine::<MyRecord>::builder()
    .embedding_backend(fallback)
    .build()?;
```

Same model everywhere — native, WASM, and API produce identical vectors.

**User-provided implementations (not shipped):**

| Backend | Use Case |
|---------|----------|
| `ApiBackend` | Remote embedding API (OpenAI, Cohere) — useful for WASM hybrid |
| Custom | Any embedding source via `EmbeddingBackend` trait |

---

### Cross-Encoder Reranking — Implementation Reference

Cross-encoder reranking uses a BERT model with a classification head that scores (query, document) pairs jointly. This is architecturally different from bi-encoder embeddings — the cross-encoder sees both texts simultaneously, capturing cross-attention patterns.

**How it works:**
1. Concatenate query + document as a single input: `[CLS] query [SEP] document [SEP]`
2. Run through BERT → take CLS token output
3. Pass through linear classification head → single score (sigmoid)
4. Higher score = more relevant

**Implementation (~50-80 lines):** Uses candle's standard BERT model (same crate already in deps) with a cross-encoder model from HuggingFace:

```rust
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

pub struct CandleReranker {
    model: BertModel,
    classifier: candle_nn::Linear,  // Single output logit
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleReranker {
    pub fn new() -> Result<Self> {
        // Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params, safetensors available)
        // Alternative: BAAI/bge-reranker-base (110M params, more accurate)
        // ...load model, tokenizer, classifier head from safetensors...
    }
}

impl RerankerBackend for CandleReranker {
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        // For each document: tokenize (query, document) pair, forward, sigmoid
        let mut scores = Vec::with_capacity(documents.len());
        for doc in documents {
            let encoding = self.tokenizer.encode((query, *doc), true)?;
            let token_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
            let token_type_ids = Tensor::new(encoding.get_type_ids(), &self.device)?.unsqueeze(0)?;
            let output = self.model.forward(&token_ids, &token_type_ids, None)?;
            let cls = output.i((.., 0, ..))?.contiguous()?;  // CLS token
            let logit = cls.apply(&self.classifier)?;
            let score = sigmoid(logit.to_scalar::<f32>()?);
            scores.push(score);
        }
        Ok(scores)
    }
}
```

**Reranking is optional and applied after RRF merge, before final scoring.** For FeMind's scale (<100K memories, search returns top 20-50 candidates), reranking adds ~50-100ms per query. Whether this is worthwhile depends on the consumer — it's behind the `reranking` feature flag.

The current implementation ships these reranker runtime options:

| Backend | Feature Flag | Model / Contract |
|---------|-------------|------------------|
| `CandleReranker` | `reranking` | `cross-encoder/ms-marco-MiniLM-L6-v2` via candle |
| `RemoteRerankerBackend` | `remote-reranking` | FeMind `/rerank` service with optional local fallback |
| `ApiRerankerBackend` | `api-reranking` | Generic HTTPS `/rerank` endpoint |
| `FallbackRerankerBackend` | (always available) | Remote/API primary with local fallback |

### ScoringStrategy

```rust
/// Post-search scoring adjustments.
/// Applied after FTS5/vector/RRF merge, before final ranking.
pub trait ScoringStrategy: Send + Sync {
    /// Adjust the score of a search result.
    /// Returns a multiplier (1.0 = no change).
    fn score_multiplier(
        &self,
        record: &MemoryMeta,
        query: &SearchQuery,
        base_score: f32,
    ) -> f32;
}
```

**Shipped strategies (composable):**

| Strategy | Description |
|----------|-------------|
| `RecencyScorer` | Exponential decay with configurable half-life |
| `ImportanceScorer` | Linear scale from importance 1-10 |
| `CategoryScorer` | Boost when query implies a matching category |
| `MemoryTypeScorer` | Different base weights per cognitive type (CoALA-derived) |
| `ActivationScorer` | ACT-R forgetting curve based on access history |
| `TrustScorer` | Confidence adjusted by success/failure outcomes |
| `CompositeScorer` | Combines multiple strategies multiplicatively |

### ConsolidationStrategy

```rust
/// Determines what happens when a new memory is stored.
/// Prevents duplicates, updates existing, or merges.
pub trait ConsolidationStrategy: Send + Sync {
    /// Given a new memory and existing similar memories,
    /// decide what operations to perform.
    fn consolidate(
        &self,
        new: &MemoryMeta,
        existing: &[ScoredResult],
    ) -> Vec<ConsolidationAction>;
}

pub enum ConsolidationAction {
    /// Store as a new memory
    Add,
    /// Update an existing memory (replace content)
    Update { target_id: i64 },
    /// Delete an existing memory (superseded or contradicted)
    Delete { target_id: i64 },
    /// Do nothing (duplicate or irrelevant)
    Noop,
    /// Link new memory to existing via relationship
    #[cfg(feature = "graph-memory")]
    Relate { target_id: i64, relation: String },
}
```

**Shipped strategies:**

| Strategy | Complexity | Description |
|----------|-----------|-------------|
| `HashDedup` | O(1) | SHA-256 content hash, exact duplicate prevention |
| `SimilarityDedup` | O(n) | Vector similarity threshold, near-duplicate detection |
| `LLMConsolidation` | External | Uses LLM to classify ADD/UPDATE/DELETE/NOOP (Mem0 pattern) |

### LlmCallback (Decision 015)

```rust
/// Consumer-provided LLM access for all LLM-assisted operations.
/// femind never calls an LLM directly — the consumer controls model,
/// cost, and retry behavior.
pub trait LlmCallback: Send + Sync {
    /// Given a prompt, return the LLM's response.
    async fn complete(&self, prompt: &str) -> Result<String>;
}
```

Used by: `LLMConsolidation`, `LlmIngest`, `EvolutionStrategy`, `reflect()`. All LLM-dependent features work (degraded) when no callback is provided.

### IngestStrategy (Decision 012)

```rust
/// Controls how raw input is processed before storage.
/// Default: store as-is. LLM-assisted: extract atomic facts.
pub trait IngestStrategy: Send + Sync {
    async fn extract(&self, raw: &str) -> Result<Vec<ExtractedFact>>;
}

pub struct ExtractedFact {
    pub text: String,
    pub category: Option<String>,
    pub memory_type: MemoryType,
    pub importance: u8,
}
```

**Shipped implementations:**

| Strategy | Cost | Description |
|----------|------|-------------|
| `PassthroughIngest` | Zero | Store text as-is (default) |
| `LlmIngest` | LLM tokens | Extract atomic facts via LlmCallback |

### RerankerBackend (Decision 013)

```rust
/// Cross-encoder reranking applied after RRF merge, before final scoring.
pub trait RerankerBackend: Send + Sync {
    /// Rerank candidates by query-document relevance.
    async fn rerank(
        &self,
        query: &str,
        candidates: Vec<ScoredResult>,
    ) -> Result<Vec<ScoredResult>>;
}
```

**Shipped implementations:**

| Backend | Feature Flag | Models |
|---------|-------------|--------|
| `CandleReranker` | `reranking` | `cross-encoder/ms-marco-MiniLM-L-6-v2` (default, 22M params), `BAAI/bge-reranker-base` (optional, 110M params) |

### EvolutionStrategy (Decision 014)

```rust
/// Post-write hook: when a new memory is stored, optionally update
/// related existing memories (their metadata, keywords, links).
pub trait EvolutionStrategy: Send + Sync {
    /// Given a newly stored memory and its top-k similar existing memories,
    /// return updates to apply to existing memories.
    async fn evolve(
        &self,
        new_memory: &MemoryMeta,
        similar: &[ScoredResult],
    ) -> Result<Vec<EvolutionAction>>;
}

pub enum EvolutionAction {
    /// Update metadata on an existing memory
    UpdateMetadata { target_id: i64, metadata: HashMap<String, String> },
    /// Create a relationship between memories
    Relate { source_id: i64, target_id: i64, relation: RelationType },
    /// No changes needed
    Noop,
}
```

### PruningPolicy (Decision 010)

```rust
pub struct PruningPolicy {
    /// Minimum age before eligible for pruning
    pub min_age_days: u32,              // default: 30
    /// Activation must be below this threshold
    pub max_activation: f32,            // default: -2.0
    /// Only prune these memory types (default: [Episodic])
    pub pruneable_types: Vec<MemoryType>,
    /// Never prune memories with graph relationships
    pub respect_graph_links: bool,      // default: true
    /// Never prune memories referenced by higher-tier summaries
    pub respect_hierarchy: bool,        // default: true
    /// Never prune memories with importance >= this value
    pub min_importance_exempt: u8,      // default: 8
    /// Soft delete (archive) vs hard delete
    pub soft_delete: bool,              // default: true
}
```

### Error Types

All fallible FeMind operations return `femind::Result<T>`, backed by a unified error enum:

```rust
#[derive(Debug, thiserror::Error)]
pub enum FemindError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("model not available: {0}")]
    ModelNotAvailable(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("model mismatch: stored with '{stored}', current backend is '{current}'")]
    ModelMismatch { stored: String, current: String },
    #[error("migration error: {0}")]
    Migration(String),
    #[error("encryption error: {0}")]
    Encryption(String),
    #[error("consolidation error: {0}")]
    Consolidation(String),
    #[error("llm callback error: {0}")]
    LlmCallback(String),
}

pub type Result<T> = std::result::Result<T, FemindError>;
```

`ModelMismatch` is returned when an operation explicitly requires vector similarity but the stored vectors were produced by a different model than the current backend. In normal search flow this does not surface as an error — the engine silently falls back to FTS5. It is only raised when the caller explicitly requests `SearchMode::Vector` and no compatible vectors exist.

### Thread Safety

`MemoryEngine` is `Send + Sync` and safe to share across threads:

- **Writer:** A single `Mutex<Connection>` serializes all write operations (store, update, delete, schema migrations). SQLite only supports one writer at a time; the mutex ensures this without runtime errors.
- **Readers:** A connection pool (sized to `std::thread::available_parallelism()`, minimum 2) provides concurrent read access. WAL mode allows readers to proceed without blocking on the writer.
- **Shared ownership:** For use across threads or async tasks, wrap in `Arc<MemoryEngine<T>>`. The engine itself holds no thread-local state.
- **Background indexer:** The embedding indexer runs on a dedicated OS thread via `std::thread::spawn`, not on a tokio runtime. It pulls pending items from the database, calls `EmbeddingBackend::embed_batch` (using a thread-local tokio `Runtime::block_on` for the async trait methods), and writes vectors back. This keeps the async runtime optional for consumers who only use synchronous operations.

---

## Storage Layer

### SQLite Configuration

Every database connection is configured with:

```sql
-- Encryption (feature: encryption) — MUST be first statement
PRAGMA key = '<consumer-provided key>';

PRAGMA journal_mode = WAL;          -- Concurrent reads during writes
PRAGMA synchronous = NORMAL;        -- Safe in WAL, avoids FSYNC per write
PRAGMA temp_store = MEMORY;         -- Temp tables in RAM
PRAGMA mmap_size = 268435456;       -- 256MB memory-mapped I/O
PRAGMA cache_size = -64000;         -- 64MB page cache
PRAGMA foreign_keys = ON;           -- Enforce relationships
```

### Encryption Configuration (Decision 008)

```rust
#[cfg(feature = "encryption")]
pub enum EncryptionKey {
    /// Raw passphrase — SQLCipher derives the key via PBKDF2 (256K iterations)
    Passphrase(String),
    /// Pre-derived raw key bytes (256-bit AES key)
    RawKey([u8; 32]),
}

/// Optional OS keychain helper (feature: keychain)
#[cfg(feature = "keychain")]
pub mod keychain {
    /// Retrieve or generate an encryption key from the OS keychain.
    /// macOS Keychain, Windows Credential Manager, Linux Secret Service.
    pub fn get_or_create_key(database_name: &str) -> Result<EncryptionKey>;
}
```

### Core Schema

```sql
-- Main memory table (columns adapt to MemoryRecord impl)
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    searchable_text TEXT NOT NULL,
    memory_type     TEXT NOT NULL CHECK(memory_type IN ('episodic','semantic','procedural')),
    importance      INTEGER NOT NULL DEFAULT 5 CHECK(importance BETWEEN 1 AND 10),
    category        TEXT,
    metadata_json   TEXT,             -- JSON serialized HashMap
    content_hash    TEXT NOT NULL,     -- SHA-256 for dedup
    embedding_status TEXT DEFAULT 'pending' CHECK(embedding_status IN ('pending','success','failed')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),

    -- Cached activation score (feature: activation-model)
    activation_cache REAL,                -- Last-computed activation score
    activation_updated TEXT,              -- When the cache was last refreshed

    -- Memory hierarchy (Decision 010)
    tier            INTEGER NOT NULL DEFAULT 0 CHECK(tier BETWEEN 0 AND 2),
    source_ids      TEXT,              -- JSON array of source memory IDs (provenance)

    -- Temporal validity (feature: temporal)
    valid_from      TEXT,
    valid_until     TEXT,

    -- Custom data (serialized MemoryRecord)
    record_json     TEXT NOT NULL
);

-- FTS5 full-text search index
-- Note: This creates an independent content copy. The searchable_text is stored in
-- both the `memories` table and the FTS5 shadow tables, roughly doubling text storage.
-- Future optimization: use `content=memories, content_rowid=id` to make FTS5 a
-- content-less index that reads from the memories table directly. This eliminates
-- duplication at the cost of more complex update triggers (FTS5 content-sync must
-- handle DELETE-before-UPDATE pattern). For databases under 100K memories the
-- duplication is negligible; optimize when storage becomes a concern.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    searchable_text,
    category,
    tokenize='porter'
);

-- FTS5 sync triggers
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, searchable_text, category)
    VALUES (new.id, new.searchable_text, new.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
    UPDATE memories_fts SET searchable_text = new.searchable_text,
                            category = new.category
    WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE rowid = old.id;
END;

-- Vector storage (feature: vector-search)
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id   INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,        -- f32 little-endian bytes
    model_name  TEXT NOT NULL,
    dimensions  INTEGER NOT NULL,
    content_hash TEXT NOT NULL,        -- Skip re-embedding unchanged content
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Access log for activation model (feature: activation-model)
CREATE TABLE IF NOT EXISTS memory_access_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
    query_text  TEXT                   -- What query retrieved this memory
);

CREATE INDEX IF NOT EXISTS idx_access_log_memory ON memory_access_log(memory_id);

-- Graph relationships (feature: graph-memory)
CREATE TABLE IF NOT EXISTS memory_relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,          -- "caused", "solved_by", "depends_on", etc.
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  TEXT,
    valid_until TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON memory_relations(relation);
```

### Two-Tier Memory (feature: `two-tier`)

```
~/.femind/global.db     — Cross-project memories (error patterns, language learnings)
./.femind/memory.db     — Project-specific memories (architecture decisions, conventions)
```

Both databases share the same schema. The engine queries both and merges results, with project memories receiving a scoring boost over global memories.

**Promotion logic:** When a project-specific memory is accessed across N different projects (configurable, default 3), it is promoted to global memory automatically.

### Schema Migrations

FeMind uses a simple version-based migration system. The database stores its schema version in a `femind_meta` table, while retaining compatibility reads from older metadata tables:

```sql
CREATE TABLE IF NOT EXISTS femind_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Initial: INSERT INTO femind_meta VALUES ('schema_version', '1');
```

On connection open, the engine checks the stored version against the compiled version and runs any pending migrations sequentially:

```rust
const CURRENT_SCHEMA_VERSION: u32 = 1;

/// Each migration is a function that takes a connection and upgrades from version N to N+1.
type Migration = fn(&Connection) -> Result<()>;

const MIGRATIONS: &[Migration] = &[
    // v1 → v2: example future migration
    // |conn| { conn.execute_batch("ALTER TABLE ..."); Ok(()) },
];
```

Migrations run inside a transaction — if any step fails, the database rolls back to its previous state. The engine refuses to open a database with a schema version newer than `CURRENT_SCHEMA_VERSION` (prevents downgrade corruption).

---

## Search Pipeline

### Search Modes

```rust
pub enum SearchMode {
    /// FTS5 keyword search only (always available)
    Keyword,
    /// Vector similarity search only (requires vector-search feature)
    Vector,
    /// Hybrid: FTS5 + Vector merged via RRF (requires vector-search feature)
    Hybrid,
    /// Auto-detect: Hybrid if vector available, Keyword otherwise
    Auto,
    /// Return all matches above threshold (for aggregation queries).
    /// Bypasses top-k limits. Critical for multi-session reasoning.
    Exhaustive { min_score: f32 },
}

/// Controls which memory tiers are searched (Decision 010)
pub enum SearchDepth {
    /// Search summaries and facts only — tiers 1+2 (default, fastest)
    Standard,
    /// Also search raw episodes if summary results are sparse
    Deep,
    /// Search all tiers (slowest, most complete, for forensic/audit)
    Forensic,
}
```

### Search Flow

```
                          ┌─────────────────────────────────────────┐
                          │              SearchQuery                 │
                          │  text: "authentication error JWT"       │
                          │  mode: Auto                             │
                          │  limit: 10                              │
                          └──────────────┬──────────────────────────┘
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                    ┌───────────────┐         ┌───────────────┐
                    │   FTS5 Search │         │ Vector Search │
                    │               │         │  (if enabled) │
                    │ Strip stops   │         │               │
                    │ Porter stem   │         │ embed(query)  │
                    │ BM25 rank     │         │ dot product   │
                    │               │         │ scan          │
                    │ limit * 3     │         │ limit * 3     │
                    └───────┬───────┘         └───────┬───────┘
                            │                         │
                            └────────────┬────────────┘
                                         ▼
                              ┌─────────────────────┐
                              │    RRF Merge         │
                              │                     │
                              │ score = Σ 1/(k+rank+1) │
                              │                     │
                              │ Dynamic k-values:   │
                              │ • quoted → keyword  │
                              │ • question → vector │
                              │ • default → equal   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────────────────┐
                              │ Cross-Encoder         │
                              │ Reranking             │
                              │ (if reranking)        │
                              │                       │
                              │ Score (query,doc)     │
                              │ pairs jointly         │
                              │ Re-sort by score      │
                              └──────────┬────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │ Graph Traversal     │
                              │ (if graph-memory)   │
                              │                     │
                              │ Find related via    │
                              │ recursive CTEs      │
                              │ Boost connected     │
                              │ memories            │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Post-Search        │
                              │  Scoring            │
                              │                     │
                              │ • Activation score  │
                              │ • Recency boost     │
                              │ • Importance boost  │
                              │ • Memory type boost │
                              │ • Category boost    │
                              │ • Tier boost        │
                              │   (global vs proj)  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Final Results      │
                              │                     │
                              │  Top N scored,      │
                              │  ranked memories    │
                              │  with metadata      │
                              └─────────────────────┘
```

### Reciprocal Rank Fusion (RRF)

RRF merge with dynamic k-value adjustment:

```rust
fn rrf_merge(
    keyword_results: &[(i64, f32)],   // (memory_id, bm25_score)
    vector_results: &[(i64, f32)],    // (memory_id, cosine_similarity)
    query: &str,
    limit: usize,
) -> Vec<(i64, f32)> {
    let (keyword_k, vector_k) = analyze_query(query);

    let mut scores: HashMap<i64, f32> = HashMap::new();

    for (rank, (id, _)) in keyword_results.iter().enumerate() {
        *scores.entry(*id).or_default() += 1.0 / (keyword_k as f32 + rank as f32 + 1.0);
    }

    for (rank, (id, _)) in vector_results.iter().enumerate() {
        *scores.entry(*id).or_default() += 1.0 / (vector_k as f32 + rank as f32 + 1.0);
    }

    let mut merged: Vec<_> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    merged.truncate(limit);
    merged
}

fn analyze_query(query: &str) -> (u32, u32) {
    if query.contains('"') {
        (40, 60) // Quoted text → favor keyword
    } else if query.split_whitespace().any(|w|
        ["what", "how", "why", "when", "where", "explain", "describe"]
            .contains(&w.to_lowercase().as_str())
    ) {
        (60, 40) // Question → favor semantic
    } else {
        (60, 60) // Default → equal weight
    }
}
```

### Graph Traversal (feature: `graph-memory`)

Multi-hop relationship traversal via recursive CTEs:

```sql
-- Find all memories connected to a given memory within N hops
WITH RECURSIVE chain(id, relation, depth, path) AS (
    -- Start from search results
    SELECT target_id, relation, 1, source_id || '→' || target_id
    FROM memory_relations
    WHERE source_id IN (/* search result IDs */)
      AND (valid_until IS NULL OR valid_until > datetime('now'))

    UNION ALL

    -- Traverse outward
    SELECT r.target_id, r.relation, c.depth + 1,
           c.path || '→' || r.target_id
    FROM memory_relations r
    JOIN chain c ON r.source_id = c.id
    WHERE c.depth < ?max_depth
      AND c.path NOT LIKE '%' || r.target_id || '%'  -- cycle prevention
      AND (r.valid_until IS NULL OR r.valid_until > datetime('now'))
)
SELECT DISTINCT m.*, c.relation, c.depth
FROM chain c
JOIN memories m ON m.id = c.id
ORDER BY c.depth ASC;
```

Connected memories receive a scoring boost: `1.0 / (depth + 1)` — directly connected memories get 0.5x boost, two hops get 0.33x, etc.

### Standard Relationship Types

```rust
pub enum RelationType {
    CausedBy,       // "X caused Y" (error → root cause)
    SolvedBy,       // "X was solved by Y" (error → fix)
    DependsOn,      // "X depends on Y" (task → prerequisite)
    SupersededBy,   // "X was replaced by Y" (old → new approach)
    RelatedTo,      // Generic association
    PartOf,         // "X is part of Y" (subtask → parent)
    ConflictsWith,  // "X contradicts Y" (opposing learnings)
    ValidatedBy,    // "X was confirmed by Y" (learning → evidence)
    Custom(String), // User-defined relationship type
}
```

### Time-Aware Query Expansion (search/query_expand.rs)

Converts natural language temporal expressions in queries to date-range filters before search. This pre-processing step improves temporal reasoning accuracy by +6-11% (LongMemEval findings).

**Supported patterns:**

| Expression | Expansion |
|-----------|-----------|
| "last week" | `created_at > datetime('now', '-7 days')` |
| "last month" | `created_at > datetime('now', '-30 days')` |
| "yesterday" | `created_at BETWEEN datetime('now', '-1 day') AND datetime('now')` |
| "in January" / "in 2025" | `created_at BETWEEN '2025-01-01' AND '2025-01-31'` |
| "before Christmas" | `created_at < '2025-12-25'` |
| "3 days ago" | `created_at BETWEEN datetime('now', '-4 days') AND datetime('now', '-2 days')` |

**Implementation:**

```rust
pub struct ExpandedQuery {
    /// The query text with temporal expressions removed
    pub cleaned_text: String,
    /// SQL date-range filters extracted from temporal expressions
    pub date_filters: Vec<DateFilter>,
}

pub struct DateFilter {
    pub column: String,        // "created_at" or "valid_from"/"valid_until"
    pub operator: FilterOp,    // Before, After, Between
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

pub fn expand_query(query: &str, now: DateTime<Utc>) -> ExpandedQuery {
    // Regex-based extraction of temporal patterns
    // Strips matched expressions from query text
    // Converts relative dates to absolute using `now`
    todo!()
}
```

The expanded date filters are applied as SQL WHERE clauses during both FTS5 and vector search, narrowing the candidate set before RRF merge.

---

## Activation Model (feature: `activation-model`)

Based on ACT-R cognitive architecture. Replaces ad-hoc decay/tier systems with a unified, research-backed model.

### How It Works

Every memory has an **activation level** that determines how easily it is retrieved. Activation is computed from the access history:

```
activation(i) = base_level(i) + Σ ln(time_since_access_j ^ -d)
```

Where:
- `base_level(i)` is derived from memory type and importance
- `time_since_access_j` is seconds since the j-th retrieval
- `d` is the decay rate (varies by memory type)
- The sum is over all recorded accesses

### Decay Rates by Memory Type

| Memory Type | Decay Rate (d) | Half-Life | Rationale |
|-------------|---------------|-----------|-----------|
| Episodic | 0.5 | ~7 days | Yesterday's debug session fades fast |
| Semantic | 0.2 | ~90 days | "The project uses PostgreSQL" stays relevant |
| Procedural | 0.3 | ~30 days | Patterns strengthen with use, fade without |

### Reinforcement (Spaced Repetition Effect)

Each time a memory is retrieved (accessed by search), its activation increases:

```rust
fn record_access(&self, memory_id: i64, query: &str) -> Result<()> {
    self.db.execute(
        "INSERT INTO memory_access_log (memory_id, query_text) VALUES (?1, ?2)",
        params![memory_id, query],
    )?;
    Ok(())
}

fn compute_activation(&self, memory_id: i64) -> Result<f32> {
    let accesses: Vec<f64> = self.db.prepare(
        "SELECT (julianday('now') - julianday(accessed_at)) * 86400.0
         FROM memory_access_log WHERE memory_id = ?1"
    )?.query_map(params![memory_id], |row| row.get(0))?
      .collect::<Result<_, _>>()?;

    let memory_type = self.get_memory_type(memory_id)?;
    let decay = match memory_type {
        MemoryType::Episodic => 0.5,
        MemoryType::Semantic => 0.2,
        MemoryType::Procedural => 0.3,
    };

    let base = self.get_importance(memory_id)? as f64 / 10.0;

    let activation = base + accesses.iter()
        .map(|t| (t.max(1.0)).powf(-decay).ln())
        .sum::<f64>();

    Ok(activation as f32)
}
```

### What This Replaces

The ACT-R activation model replaces several ad-hoc mechanisms commonly found in agent memory systems:

| Previous Approach | FeMind Equivalent |
|-------------------|-------------------|
| Manual trust/confidence scores | Activation + `ValidatedBy` relations |
| Tier-based multipliers (working/long-term/archive) | Activation naturally creates tiers |
| Fixed confidence decay rates | Forgetting curve handles this automatically |
| Reference counters | Access log (richer data with timestamps) |
| Exponential recency boost | Part of activation formula |

One model replaces five separate mechanisms.

### Access Log Maintenance

The `memory_access_log` table grows with every search retrieval. Without maintenance, it becomes an unbounded append-only log. FeMind uses two strategies to keep it manageable:

**Compaction:** Accesses older than 90 days are aggregated into summary records. A periodic maintenance pass replaces individual access rows with a single summary row per memory:

```sql
-- Aggregate old accesses into summary records
INSERT INTO memory_access_log (memory_id, accessed_at, query_text)
SELECT memory_id, MAX(accessed_at), '[compacted: ' || COUNT(*) || ' accesses]'
FROM memory_access_log
WHERE accessed_at < datetime('now', '-90 days')
GROUP BY memory_id;

-- Remove the individual old rows (replaced by summaries above)
DELETE FROM memory_access_log
WHERE accessed_at < datetime('now', '-90 days')
  AND query_text NOT LIKE '[compacted:%';
```

The activation formula uses the compacted count to approximate the contribution of old accesses without storing each one individually.

**Cached activation:** Recomputing activation from the full access log on every search is wasteful. Instead, a `memory_activation_cache` column on the `memories` table stores the last-computed activation score. This cache is refreshed when a memory is accessed (incremental update) or periodically during maintenance. Search scoring reads from the cache rather than recomputing from the log.

---

## Consolidation Pipeline (feature: `consolidation`)

Based on Mem0's three-stage pipeline, adapted for local-first use.

### Flow

```
New Memory
    │
    ▼
┌───────────────────┐
│ 1. EXTRACT        │  Hash content (SHA-256)
│                   │  Classify memory type
│                   │  Extract category
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 2. CONSOLIDATE    │  Search for similar existing memories
│                   │  Compare via content hash (cheap)
│                   │  Compare via vector similarity (if enabled)
│                   │  Decide: ADD / UPDATE / DELETE / NOOP / RELATE
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 3. STORE          │  Execute consolidation actions
│                   │  Index in FTS5
│                   │  Queue for embedding (background)
│                   │  Create relationships (if graph-memory)
└───────────────────┘
```

### Consolidation Strategies

**HashDedup (default, zero cost):**
- SHA-256 hash of `searchable_text()`
- If hash exists → NOOP (exact duplicate)
- Otherwise → ADD

**SimilarityDedup (requires vector-search):**
- Embed the new memory
- Search for top-5 similar existing memories
- If similarity > 0.95 → NOOP (near-duplicate)
- If similarity > 0.85 → UPDATE (refine existing)
- If similarity > 0.70 and contradicts → flag for review
- Otherwise → ADD

**LLMConsolidation (external, user provides LLM call):**
- Passes new memory + similar existing memories to an LLM
- LLM classifies: ADD / UPDATE / DELETE / NOOP
- Most accurate but requires LLM tokens
- Follows Mem0 pattern

---

## Context Assembly

Token-budget-aware context selection for injecting memories into LLM prompts.

```rust
pub struct ContextBudget {
    /// Maximum tokens to spend on context
    pub max_tokens: usize,
    /// Approximate tokens per character (default: 0.25 for English)
    pub tokens_per_char: f32,
}

pub struct ContextItem {
    pub memory_id: i64,
    pub content: String,
    pub priority: u8,          // Lower = higher priority
    pub estimated_tokens: usize,
    pub memory_type: MemoryType,
}

/// Priority levels (lower number = included first)
pub const PRIORITY_BEHAVIORAL: u8 = 0;    // System-critical context
pub const PRIORITY_RETRY: u8 = 10;        // Previous failure context
pub const PRIORITY_SPEC: u8 = 15;         // Specification sections
pub const PRIORITY_SIMILAR: u8 = 25;      // Similar completed tasks
pub const PRIORITY_LEARNING: u8 = 40;     // General learnings
pub const PRIORITY_HISTORICAL: u8 = 60;   // Older episodic context
```

Assembly algorithm:
1. Gather all candidate context items from search results
2. Sort by priority (ascending), then by activation score (descending) within each priority
3. Add items until token budget is exhausted
4. Skip items that would exceed remaining budget
5. Return assembled context string with section headers

---

## Embedding Indexer (Background)

Embeddings are generated asynchronously in the background, not at query time.

```rust
pub struct EmbeddingIndexer {
    backend: Arc<dyn EmbeddingBackend>,
    batch_size: usize,       // Default: 32
    retry_failed: bool,      // Re-attempt previously failed items
}

impl EmbeddingIndexer {
    /// Process all pending memories in batches
    pub async fn index_pending(&self, db: &Database) -> Result<IndexReport> {
        let pending = db.query(
            "SELECT id, searchable_text, content_hash
             FROM memories
             WHERE embedding_status IN ('pending', 'failed')
             ORDER BY created_at DESC"
        )?;

        let mut report = IndexReport::default();

        for batch in pending.chunks(self.batch_size) {
            // Skip if content_hash already exists in memory_vectors
            let new_items: Vec<_> = batch.iter()
                .filter(|m| !self.vector_exists(db, &m.content_hash))
                .collect();

            if new_items.is_empty() { continue; }

            let texts: Vec<&str> = new_items.iter()
                .map(|m| m.searchable_text.as_str())
                .collect();

            match self.backend.embed_batch(&texts).await {
                Ok(embeddings) => {
                    for (item, embedding) in new_items.iter().zip(embeddings) {
                        self.store_vector(db, item.id, &embedding, &item.content_hash)?;
                        self.update_status(db, item.id, "success")?;
                        report.succeeded += 1;
                    }
                }
                Err(e) => {
                    for item in &new_items {
                        self.update_status(db, item.id, "failed")?;
                        report.failed += 1;
                    }
                    report.errors.push(e.to_string());
                }
            }
        }

        Ok(report)
    }

    /// Re-embed all memories (e.g., after model change)
    pub async fn reindex_all(&self, db: &Database) -> Result<IndexReport> {
        db.execute("UPDATE memories SET embedding_status = 'pending'", [])?;
        db.execute("DELETE FROM memory_vectors", [])?;
        self.index_pending(db).await
    }
}
```

---

## Public API

### MemoryEngine

The primary interface. Generic over the consumer's memory type.

```rust
pub struct MemoryEngine<T: MemoryRecord> {
    db: Database,
    #[cfg(feature = "two-tier")]
    global_db: Option<Database>,
    #[cfg(feature = "vector-search")]
    embedding_backend: Arc<dyn EmbeddingBackend>,
    #[cfg(feature = "vector-search")]
    indexer: EmbeddingIndexer,
    scoring: Arc<dyn ScoringStrategy>,
    #[cfg(feature = "consolidation")]
    consolidation: Arc<dyn ConsolidationStrategy>,
    _phantom: PhantomData<T>,
}

impl<T: MemoryRecord> MemoryEngine<T> {
    /// Create a new engine with builder pattern
    pub fn builder() -> MemoryEngineBuilder<T>;

    // --- CRUD (all synchronous — these are SQLite queries) ---

    /// Store a new memory (runs consolidation if enabled).
    /// Embedding is queued for background indexing, not done inline.
    pub fn store(&self, record: T) -> Result<StoreResult>;

    /// Retrieve a memory by ID
    pub fn get(&self, id: i64) -> Result<Option<T>>;

    /// Update an existing memory
    pub fn update(&self, id: i64, record: T) -> Result<()>;

    /// Delete a memory
    pub fn delete(&self, id: i64) -> Result<()>;

    // --- Search ---

    /// Search memories with fluent API
    pub fn search(&self, query: &str) -> SearchBuilder<T>;

    /// Find memories related to a given memory (graph traversal)
    #[cfg(feature = "graph-memory")]
    pub fn related(&self, memory_id: i64, max_depth: u32) -> Result<Vec<ScoredResult<T>>>;

    // --- Relationships ---

    /// Create a relationship between two memories
    #[cfg(feature = "graph-memory")]
    pub fn relate(&self, source: i64, target: i64, relation: RelationType) -> Result<()>;

    // --- Context ---

    /// Assemble context for an LLM prompt within a token budget
    pub fn assemble_context(
        &self,
        query: &str,
        budget: &ContextBudget,
    ) -> Result<ContextAssembly>;

    // --- Maintenance ---

    /// Run background embedding indexer
    #[cfg(feature = "vector-search")]
    pub async fn index_pending(&self) -> Result<IndexReport>;

    /// Promote frequently-seen project memories to global
    #[cfg(feature = "two-tier")]
    pub fn promote_to_global(&self, memory_id: i64) -> Result<()>;

    // --- Consolidation & Pruning (Decision 010) ---

    /// Consolidate old episodic memories into summaries/facts.
    /// LLM callback optional — degraded mode uses vector clustering.
    #[cfg(feature = "consolidation")]
    pub async fn consolidate(
        &self,
        policy: &ConsolidationPolicy,
        llm: Option<&dyn LlmCallback>,
    ) -> Result<ConsolidationReport>;

    /// Prune memories meeting all policy criteria (activation, age, type, links).
    #[cfg(feature = "consolidation")]
    pub fn prune(&self, policy: &PruningPolicy) -> Result<PruneReport>;

    /// Run consolidation + pruning as a single maintenance pass.
    #[cfg(feature = "consolidation")]
    pub async fn maintain(
        &self,
        consolidation: &ConsolidationPolicy,
        pruning: &PruningPolicy,
        llm: Option<&dyn LlmCallback>,
    ) -> Result<MaintenanceReport>;

    /// Synthesize higher-order insights from accumulated memories.
    /// Stores results as Semantic Tier 2 memories with provenance links.
    #[cfg(feature = "consolidation")]
    pub async fn reflect(
        &self,
        llm: &dyn LlmCallback,
    ) -> Result<ReflectionReport>;
}

// Fluent search API
impl<T: MemoryRecord> SearchBuilder<T> {
    pub fn mode(self, mode: SearchMode) -> Self;
    pub fn depth(self, depth: SearchDepth) -> Self;  // Decision 010
    pub fn limit(self, n: usize) -> Self;
    pub fn category(self, cat: &str) -> Self;
    pub fn memory_type(self, t: MemoryType) -> Self;
    pub fn tier(self, tier: u8) -> Self;              // Filter by tier
    pub fn min_score(self, score: f32) -> Self;
    #[cfg(feature = "temporal")]
    pub fn valid_at(self, time: DateTime<Utc>) -> Self;
    /// Execute the search. Synchronous — vector similarity uses pre-computed
    /// embeddings from the background indexer, not inline inference.
    pub fn execute(self) -> Result<Vec<ScoredResult<T>>>;
}
```

> **Sync vs. async boundary:** Core operations (`store`, `get`, `update`, `delete`, `search().execute()`, `build()`) are synchronous — they only touch SQLite. Async is reserved for operations involving inference or network I/O: `EmbeddingBackend::embed/embed_batch`, `IngestStrategy::extract`, `RerankerBackend::rerank`, `EvolutionStrategy::evolve`, `LlmCallback::complete`, and the background embedding indexer.

### Builder Pattern

```rust
let engine = MemoryEngine::<Learning>::builder()
    .database("path/to/memory.db")
    .global_database("~/.femind/global.db")          // optional, two-tier
    .embedding_backend(CandleBackend::new()?)          // optional, vector-search
    .scoring(CompositeScorer::new(vec![
        Box::new(RecencyScorer::new(Duration::days(30))),
        Box::new(ImportanceScorer),
        Box::new(ActivationScorer::default()),
    ]))
    .consolidation(SimilarityDedup::new(0.90))         // optional
    .build()?;
```

---

## Performance Targets

Based on OMEGA Memory benchmarks and production experience:

| Operation | Target | Measurement |
|-----------|--------|-------------|
| FTS5 keyword search | <5ms | 10K memories |
| Vector embedding (single) | <10ms | all-MiniLM-L6-v2 on CPU |
| Brute-force vector scan | <50ms | 100K vectors, 384 dims |
| RRF hybrid merge | <1ms | Pure computation |
| Graph traversal (3 hops) | <10ms | 10K relationships |
| Context assembly | <5ms | After search completes |
| Memory store (with hash dedup) | <2ms | Single insert |
| Memory store (with similarity dedup) | <60ms | Includes embed + search |
| Activation score computation | <1ms | Per memory |
| Background index (batch of 32) | <300ms | 32 embeddings |

### Scaling Thresholds

| Memory Count | Vector Approach | Expected Search Latency |
|-------------|----------------|------------------------|
| <10K | Brute force | <10ms |
| 10K-100K | Brute force | <50ms |
| 100K-1M | sqlite-vec with quantization | <10ms |
| >1M | External vector DB or ANN | <5ms |

For personal/project use, you will likely never exceed 10K memories per database.

---

## Future Considerations

### Graph-Native Backend (v2)

When SQLite recursive CTEs become a bottleneck (>100K relationships), add a `graph-native` feature flag with an embedded graph database:

| Candidate | Language | Query Language | Status |
|-----------|----------|---------------|--------|
| Cozo | Pure Rust | Datalog | Active, best Rust fit |
| Kuzu | C++ (Rust bindings) | Cypher | Archived Oct 2025, forks exist |

The `GraphBackend` trait would abstract over SQLite CTEs and native graph DBs:

```rust
pub trait GraphBackend: Send + Sync {
    fn add_relation(&self, source: i64, target: i64, relation: &str) -> Result<()>;
    fn traverse(&self, start: i64, max_depth: u32) -> Result<Vec<GraphNode>>;
    fn find_path(&self, from: i64, to: i64) -> Result<Option<Vec<GraphEdge>>>;
}
```

### Embedding Model Upgrades

The `EmbeddingBackend` trait + `model_name` field on stored vectors enables model migration:

1. Ship new model version
2. Run `reindex_all()` to re-embed with new model
3. Old vectors overwritten with new dimensions/quality

### Multi-Agent Shared Memory

For team-scale use with multiple agents accessing the same database:
- WAL mode already supports concurrent readers
- Single writer is fine for most cases
- If needed, add connection pooling with `r2d2` or similar

---

## Dependencies

### Always Required

```toml
[dependencies]
rusqlite = { version = "0.32", features = ["bundled", "functions"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
thiserror = "2"
sha2 = "0.10"         # Content hashing
tracing = "0.1"       # Structured logging
```

**Note:** FeMind requires **Rust 1.85+** (edition 2024).

### Feature-Gated

```toml
[dependencies]
# async runtime (only needed for vector-search background indexer and embedding inference)
tokio = { version = "1", features = ["rt", "sync"], optional = true }

# local-embeddings (candle)
candle-core = { version = "0.9", optional = true }
candle-nn = { version = "0.9", optional = true }
candle-transformers = { version = "0.9", optional = true }
tokenizers = { version = "0.22", optional = true }
hf-hub = { version = "0.4", optional = true }

# vector-indexed
# sqlite-vec = { version = "...", optional = true }  # TBD: Rust bindings maturity

# mcp-server
axum = { version = "0.7", optional = true }
tower = { version = "0.5", optional = true }

# keychain (encryption key storage)
keyring = { version = "3", optional = true }
```

---

## Summary

FeMind is a **composable, feature-gated memory engine** that unifies proven patterns from published research and the 2025-2026 agent memory landscape into a single Rust crate. It provides:

**Core (always on):**
- **FTS5 keyword search** with Porter stemming and BM25 ranking
- **Token-budget context assembly** for LLM prompt injection
- **Three-tier memory hierarchy** (episodes → summaries → facts) with tier-aware search
- **Cognitive memory types** (Episodic/Semantic/Procedural) with type-appropriate behavior

**Feature-gated:**
- **Vector search** via custom candle module (pure Rust) with hybrid RRF merge
- **Cross-encoder reranking** via candle BERT (post-RRF refinement)
- **Graph relationships** via SQLite recursive CTE traversal
- **ACT-R activation model** — research-backed decay replacing ad-hoc systems
- **Consolidation pipeline** — hash dedup / similarity dedup / LLM-assisted
- **Memory evolution** — post-write hooks that update related existing memories
- **Fact extraction at ingest** — `IngestStrategy` trait for atomic fact extraction
- **Two-tier memory** (global + project) with automatic promotion
- **Temporal validity** — bi-temporal tracking of when facts were true
- **Encryption at rest** via SQLCipher (preserves FTS5/WAL/vector)
- **MCP server** interface for direct LLM tool access
- **Background embedding indexer** with content-hash skip
- **Graceful degradation** — always falls back to FTS5 if vector unavailable

**Design principles:**
- `MemoryRecord` trait lets any project define its own memory types
- `LlmCallback` trait lets consumers control all LLM operations (model, cost, retries)
- Feature flags ensure zero cost for unused capabilities (2MB to ~45MB)
- Library, not framework — consumer controls scheduling and lifecycle
- WASM-compatible for browser deployment (SQLite+FTS5 via `sqlite-wasm-rs`)

**LongMemEval achieved: 95.6%** (surpasses OMEGA's verified 76.8% and their marketing claim of 95.4%)

**Target: ~6-8K lines of Rust for the full engine.**
