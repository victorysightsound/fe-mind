# MemCore: Universal Memory Engine Architecture

**Version:** 0.1.0 (Design)
**Date:** 2026-03-16
**Status:** Architecture Complete, Ready for Review

---

## Overview

**MemCore** is a standalone Rust crate providing a pluggable, feature-gated memory engine for AI agent applications. It handles persistent storage, keyword search (FTS5), vector search (candle), hybrid retrieval (RRF), graph relationships, memory consolidation, cognitive decay modeling, and token-budget-aware context assembly.

### Design Principles

1. **Library, not framework** — projects call into MemCore, they are not structured around it
2. **Feature-gated everything** — heavy dependencies behind compile-time flags, zero cost when unused
3. **Opinionated about mechanics, unopinionated about schema** — MemCore handles how to store/search/decay; the consumer defines what a "memory" is
4. **Local-first** — SQLite-backed, single-file databases, no cloud dependency
5. **Pure Rust where possible** — candle over ort, SQLite over Postgres
6. **Proven patterns only** — every component is battle-tested in Memloft, Dial, or published research

### Origin

MemCore extracts and unifies patterns from three projects:

| Source | Contribution |
|--------|-------------|
| **Memloft** | Hybrid search (RRF), candle embeddings, FallbackBackend, background indexing, tier-based scoring |
| **Dial** | FTS5 + Porter stemming, trust scoring, token-budget context assembly, failure pattern detection |
| **PIRDLY** | Two-tier memory (global + project), error classification, MCP server interface |

Additionally informed by research into:
- [Mem0](https://github.com/mem0ai/mem0) — consolidation pipeline (extract → consolidate → store)
- [Zep/Graphiti](https://github.com/getzep/graphiti) — temporal validity modeling
- [OMEGA Memory](https://github.com/omega-memory/core) — forgetting intelligence, performance benchmarks
- [CoALA framework](https://arxiv.org/pdf/2309.02427) — cognitive memory types (episodic/semantic/procedural)
- [ACT-R research](https://dl.acm.org/doi/10.1145/3765766.3765803) — activation-based decay modeling

---

## Crate Structure

```
memcore/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Public API: MemoryEngine<T>
│   │
│   ├── traits/
│   │   ├── mod.rs
│   │   ├── record.rs          # MemoryRecord trait (consumer implements)
│   │   ├── embedding.rs       # EmbeddingBackend trait
│   │   ├── scoring.rs         # ScoringStrategy trait
│   │   └── consolidation.rs   # ConsolidationStrategy trait
│   │
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── engine.rs          # SQLite connection, WAL, pragmas
│   │   ├── schema.rs          # Dynamic schema generation from MemoryRecord
│   │   ├── migrations.rs      # Versioned schema migrations
│   │   └── two_tier.rs        # Global + project database management
│   │
│   ├── search/
│   │   ├── mod.rs
│   │   ├── fts5.rs            # FTS5 keyword search, Porter stemming, stop-words
│   │   ├── vector.rs          # Brute-force dot product (+ optional sqlite-vec ANN)
│   │   ├── hybrid.rs          # Reciprocal Rank Fusion merge
│   │   ├── graph.rs           # Relationship traversal via recursive CTEs
│   │   ├── reranker.rs        # Cross-encoder reranking (post-RRF, feature-gated)
│   │   ├── query_expand.rs    # Time-aware query expansion (temporal expressions → date ranges)
│   │   └── scoring.rs         # Post-search scoring (activation, recency, type boosts)
│   │
│   ├── embeddings/
│   │   ├── mod.rs
│   │   ├── candle.rs          # CandleBackend: bge-small-en-v1.5 for WASM (feature-gated)
│   │   ├── fastembed.rs       # FastembedBackend: granite-small-r2 default (feature-gated)
│   │   ├── models.rs          # Model auto-download, caching, and custom model helpers
│   │   ├── noop.rs            # NoopBackend: zero vectors (testing)
│   │   ├── fallback.rs        # FallbackBackend: graceful degradation
│   │   └── indexer.rs         # Background batch indexer, content-hash skip
│   │
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── store.rs           # CRUD operations, deduplication
│   │   ├── consolidation.rs   # Extract → Consolidate → Store pipeline
│   │   ├── activation.rs      # ACT-R forgetting curve + access tracking
│   │   ├── relations.rs       # Graph relationships (memory_relations table)
│   │   ├── temporal.rs        # valid_from / valid_until support
│   │   ├── hierarchy.rs       # Three-tier memory (episode → summary → fact)
│   │   ├── evolution.rs       # Post-write hooks, memory-writes-back-to-memory
│   │   └── pruning.rs         # Activation-based pruning with policy controls
│   │
│   ├── context/
│   │   ├── mod.rs
│   │   ├── budget.rs          # Token-budget-aware context assembly
│   │   └── priority.rs        # Priority-ranked selection by memory type
│   │
│   ├── ingest/
│   │   ├── mod.rs
│   │   ├── passthrough.rs     # Default: store text as-is
│   │   └── fact_extract.rs    # LLM-assisted atomic fact extraction
│   │
│   ├── callbacks/
│   │   ├── mod.rs
│   │   └── llm.rs             # LlmCallback trait (consumer-provided LLM access)
│   │
│   └── interface/
│       ├── mod.rs
│       └── mcp.rs             # MCP server (feature-gated)
│
└── models/                    # Bundled or downloaded embedding models
    └── README.md              # Instructions for model acquisition
```

---

## Feature Flags

```toml
[features]
default = ["fts5"]

# Core search (always available)
fts5 = []                       # FTS5 + Porter stemming + BM25

# Vector search (adds candle dependency)
local-embeddings = [
    "dep:candle-core",
    "dep:candle-nn",
    "dep:candle-transformers",
    "dep:tokenizers",
    "dep:hf-hub"
]
vector-search = ["local-embeddings"]

# ANN indexing for >100K scale
vector-indexed = ["vector-search", "dep:sqlite-vec"]

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
mcp-server = ["dep:axum", "dep:tower", "dep:serde_json"]

# Two-tier memory (global + project databases)
two-tier = []

# fastembed-rs backend (25+ models, reranking, SPLADE sparse embeddings)
fastembed = ["dep:fastembed"]

# Cross-encoder reranking (post-RRF refinement)
reranking = ["fastembed"]

# Encryption at rest via SQLCipher
encryption = ["rusqlite/bundled-sqlcipher"]
encryption-vendored = ["encryption", "rusqlite/bundled-sqlcipher-vendored-openssl"]

# OS keychain integration for encryption key storage
keychain = ["dep:keyring"]

# Everything (native)
full = [
    "vector-search",
    "graph-memory",
    "temporal",
    "consolidation",
    "activation-model",
    "two-tier",
    "reranking"
]
```

### Feature Dependency Chain

```
fts5 (default, always on)
  └── vector-search
       ├── local-embeddings (candle)
       ├── vector-indexed (sqlite-vec, optional)
       └── fastembed (alternative to candle, adds reranking + SPLADE)
            └── reranking (cross-encoder post-RRF)

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

| Features Enabled | Approximate Binary Impact |
|-----------------|--------------------------|
| `default` (FTS5 only) | ~2MB (just rusqlite) |
| `+ graph-memory + temporal + consolidation + activation-model` | ~2.5MB (pure logic, no new deps) |
| `+ vector-search` (candle) | ~30-35MB |
| `+ fastembed` (ONNX + reranking) | ~35-40MB |
| `+ vector-indexed` (sqlite-vec) | +~5MB |
| `+ mcp-server` (axum) | +~5MB |
| `+ encryption` (SQLCipher) | +~500KB-1MB over plain SQLite |
| `+ keychain` (keyring) | +~200-500KB |
| `full` (everything) | ~40-45MB |

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
    /// (Under consideration — see Decision Q3)
    // Belief,
}

/// Consumers implement this for their memory types.
/// MemCore handles storage, indexing, search, and decay.
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
// Dial's learning type
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

// Dial's failure pattern
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

// Memloft's memory
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

### EmbeddingBackend

```rust
#[async_trait]
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

| Backend | Feature Flag | Default Model | Use Case |
|---------|-------------|---------------|----------|
| `FastembedBackend` | `fastembed` | `granite-small-r2` (384-dim, 8K context) | **Primary.** Native targets. Auto-downloads model on first use. |
| `CandleBackend` | `local-embeddings` | `bge-small-en-v1.5` (384-dim) | **WASM fallback.** Pure Rust, compiles to wasm32. |
| `NoopBackend` | (always available) | N/A | Testing, returns zero vectors |
| `FallbackBackend` | (always available) | N/A | Wraps `Option<Box<dyn EmbeddingBackend>>`, degrades gracefully |

**Model management:**

MemCore wraps fastembed's custom model loading behind a clean API. Models are auto-downloaded from HuggingFace on first use and cached in `~/.cache/memcore/models/`. The consumer never deals with ONNX files directly.

```rust
/// Built-in model presets — MemCore handles download, caching, and loading.
pub enum MemCoreModel {
    /// IBM granite-embedding-small-english-r2 (47M params, 384-dim, 8K context)
    /// Best code retrieval. Default.
    GraniteSmallR2,
    /// BAAI bge-small-en-v1.5 (33M params, 384-dim, 512 context)
    /// Built into fastembed natively. Zero-config fallback.
    BgeSmallV15,
    /// Custom ONNX model from a local path or HuggingFace repo.
    Custom { repo_id: String },
}

impl FastembedBackend {
    /// Create with default model (granite-small-r2).
    /// Auto-downloads on first use, cached for subsequent calls.
    pub fn new() -> Result<Self> {
        Self::with_model(MemCoreModel::GraniteSmallR2)
    }

    /// Create with a specific model preset.
    pub fn with_model(model: MemCoreModel) -> Result<Self> {
        match model {
            MemCoreModel::GraniteSmallR2 => {
                // Download from onnx-community/granite-embedding-small-english-r2-ONNX
                // Load via fastembed try_new_from_user_defined()
                // Cache at ~/.cache/memcore/models/granite-small-r2/
            }
            MemCoreModel::BgeSmallV15 => {
                // Use fastembed's built-in BGESmallENV15
                // Zero setup, fastembed handles everything
            }
            MemCoreModel::Custom { repo_id } => {
                // Download from HuggingFace repo, load as user-defined model
            }
        }
    }
}
```

**Dual-backend pattern (native + WASM):**

Both backends produce 384-dimensional vectors, so embeddings stored by one can be searched by the other. Conditional compilation selects the right backend:

```rust
#[cfg(all(feature = "fastembed", not(target_family = "wasm")))]
pub fn default_backend() -> Result<Box<dyn EmbeddingBackend>> {
    // granite-small-r2: best code retrieval, 8K context
    Ok(Box::new(FastembedBackend::new()?))
}

#[cfg(all(feature = "local-embeddings", target_family = "wasm"))]
pub fn default_backend() -> Result<Box<dyn EmbeddingBackend>> {
    // bge-small-en-v1.5: same 384-dim, compatible vectors
    Ok(Box::new(CandleBackend::new("bge-small-en-v1.5")?))
}
```

**Model comparison (384-dim class):**

| Model | Params | Retrieval (MTEB) | Code Retrieval (CoIR) | Max Tokens | Status |
|-------|--------|-----------------|----------------------|------------|--------|
| `granite-embedding-small-english-r2` | 47M | 53.9 | **53.8** | **8,192** | **Default** (native) |
| `bge-small-en-v1.5` | 33M | 53.9 | 45.8 | 512 | **Default** (WASM), fallback (native) |
| `all-MiniLM-L6-v2` | 22M | ~41.9 | — | 256 | Available but not recommended |

**User-provided implementations (not shipped):**

| Backend | Use Case |
|---------|----------|
| `ApiBackend` | Remote embedding API (OpenAI, Cohere) — useful for WASM hybrid |
| Custom | Any embedding source via `EmbeddingBackend` trait |

### ScoringStrategy

```rust
/// Post-search scoring adjustments.
/// Applied after FTS5/vector/RRF merge, before final ranking.
pub trait ScoringStrategy: Send + Sync {
    /// Adjust the score of a search result.
    /// Returns a multiplier (1.0 = no change).
    fn score_multiplier(
        &self,
        record: &dyn MemoryRecord,
        query: &SearchQuery,
        base_score: f32,
    ) -> f32;
}
```

**Shipped strategies (composable):**

| Strategy | Description | Source |
|----------|-------------|--------|
| `RecencyScorer` | Exponential decay with configurable half-life | Memloft |
| `ImportanceScorer` | Linear scale from importance 1-10 | Memloft |
| `CategoryScorer` | Boost when query implies a matching category | Memloft |
| `MemoryTypeScorer` | Different base weights per cognitive type | CoALA |
| `ActivationScorer` | ACT-R forgetting curve based on access history | ACT-R research |
| `TrustScorer` | Confidence adjusted by success/failure outcomes | Dial |
| `CompositeScorer` | Combines multiple strategies multiplicatively | New |

### ConsolidationStrategy

```rust
/// Determines what happens when a new memory is stored.
/// Prevents duplicates, updates existing, or merges.
pub trait ConsolidationStrategy: Send + Sync {
    /// Given a new memory and existing similar memories,
    /// decide what operations to perform.
    fn consolidate(
        &self,
        new: &dyn MemoryRecord,
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
/// MemCore never calls an LLM directly — the consumer controls model,
/// cost, and retry behavior.
#[async_trait]
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
#[async_trait]
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
#[async_trait]
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
| `FastembedReranker` | `reranking` | BGE-reranker-base, BGE-reranker-v2-m3, jina-reranker-v1-turbo-en |

### EvolutionStrategy (Decision 014)

```rust
/// Post-write hook: when a new memory is stored, optionally update
/// related existing memories (their metadata, keywords, links).
#[async_trait]
pub trait EvolutionStrategy: Send + Sync {
    /// Given a newly stored memory and its top-k similar existing memories,
    /// return updates to apply to existing memories.
    async fn evolve(
        &self,
        new_memory: &dyn MemoryRecord,
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
~/.memcore/global.db     — Cross-project memories (error patterns, language learnings)
./.memcore/memory.db     — Project-specific memories (architecture decisions, conventions)
```

Both databases share the same schema. The engine queries both and merges results, with project memories receiving a scoring boost over global memories.

**Promotion logic:** When a project-specific memory is accessed across N different projects (configurable, default 3), it is promoted to global memory automatically.

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

From Memloft, with dynamic k-value adjustment:

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

| Previous System | In Dial | In Memloft | MemCore Equivalent |
|----------------|---------|------------|-------------------|
| Trust scoring | `confidence` 0.0-1.0, adjusted by success/failure | N/A | Activation + `ValidatedBy` relations |
| Tier system | N/A | working/long_term/archive with multipliers | Activation naturally creates tiers |
| Confidence decay | 0.05 per 30 days without validation | N/A | Forgetting curve handles this |
| Times referenced | Counter on learnings | N/A | Access log (richer data) |
| Recency boost | N/A | Exponential decay, 30-day half-life | Part of activation formula |

One model replaces five separate mechanisms.

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

## Context Assembly (from Dial)

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

From Memloft's pattern — embeddings are generated asynchronously, not at query time.

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

    // --- CRUD ---

    /// Store a new memory (runs consolidation if enabled)
    pub async fn store(&self, record: T) -> Result<StoreResult>;

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
    pub async fn execute(self) -> Result<Vec<ScoredResult<T>>>;
}
```

### Builder Pattern

```rust
let engine = MemoryEngine::<Learning>::builder()
    .database("path/to/memory.db")
    .global_database("~/.memcore/global.db")          // optional, two-tier
    .embedding_backend(CandleBackend::new()?)          // optional, vector-search
    .scoring(CompositeScorer::new(vec![
        Box::new(RecencyScorer::new(Duration::days(30))),
        Box::new(ImportanceScorer),
        Box::new(ActivationScorer::default()),
    ]))
    .consolidation(SimilarityDedup::new(0.90))         // optional
    .build()
    .await?;
```

---

## Performance Targets

Based on OMEGA Memory benchmarks and Memloft's production performance:

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

## Migration Path for Existing Projects

### Dial → MemCore

| Dial Component | MemCore Equivalent | Migration |
|---------------|-------------------|-----------|
| `learnings` table + FTS5 | `MemoryEngine<Learning>` | Implement `MemoryRecord` for `Learning` |
| `failure_patterns` table | `MemoryEngine<FailurePattern>` | Implement `MemoryRecord` for `FailurePattern` |
| `solutions` table + trust | `MemoryEngine<Solution>` with `TrustScorer` | Implement `MemoryRecord`, use `ValidatedBy` relations |
| `find_similar_completed_tasks()` | `engine.search().mode(Auto)` | Direct replacement, gains vector search |
| `assemble_context()` | `engine.assemble_context()` | Direct replacement |
| Stop-word stripping | Built into MemCore FTS5 | Automatic |
| Porter stemming | Built into MemCore FTS5 | Automatic |

**What Dial gains:** Vector search for task similarity, graph relationships between errors/solutions/learnings, activation-based decay replacing manual trust scoring, consolidation preventing duplicate learnings.

### Memloft → MemCore

| Memloft Component | MemCore Equivalent | Migration |
|-------------------|-------------------|-----------|
| `memory` table + FTS5 | `MemoryEngine<Memory>` | Implement `MemoryRecord` for `Memory` |
| `memory_vectors` table | Built into MemCore | Automatic |
| `HybridSearcher` with RRF | Built into MemCore | Automatic |
| `LocalBackend` (candle) | `CandleBackend` | Direct port |
| `FallbackBackend` | `FallbackBackend` | Direct port |
| `EmbeddingIndexer` | Built into MemCore | Automatic |
| Tier system (working/long_term/archive) | `ActivationScorer` + `MemoryTypeScorer` | Activation model replaces tiers |
| Content-hash dedup | `HashDedup` consolidation | Direct port |

**What Memloft gains:** Cognitive memory types, activation-based decay, graph relationships, token-budget context assembly, two-tier global/project memory, consolidation pipeline.

### PIRDLY → MemCore

PIRDLY uses MemCore as its memory system from the start:

```toml
# PIRDLY's Cargo.toml
[dependencies]
memcore = { version = "0.1", features = ["full"] }
```

PIRDLY implements `MemoryRecord` for its types (learnings, error patterns, project context) and gets the full memory engine with no custom memory code needed.

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
async-trait = "0.1"
tokio = { version = "1", features = ["rt", "sync"] }
tracing = "0.1"       # Structured logging
```

### Feature-Gated

```toml
[dependencies]
# local-embeddings (candle)
candle-core = { version = "0.9", optional = true }
candle-nn = { version = "0.9", optional = true }
candle-transformers = { version = "0.9", optional = true }
tokenizers = { version = "0.22", optional = true }
hf-hub = { version = "0.4", optional = true }

# fastembed (alternative embedding + reranking)
fastembed = { version = "5.12", optional = true }

# vector-indexed
# sqlite-vec = { version = "...", optional = true }  # TBD: Rust bindings maturity

# mcp-server
axum = { version = "0.7", optional = true }
tower = { version = "0.4", optional = true }

# keychain (encryption key storage)
keyring = { version = "3.6", optional = true }
```

---

## Summary

MemCore is a **composable, feature-gated memory engine** that unifies proven patterns from Memloft, Dial, published research, and the 2025-2026 agent memory landscape into a single Rust crate. It provides:

**Core (always on):**
- **FTS5 keyword search** with Porter stemming and BM25 ranking
- **Token-budget context assembly** for LLM prompt injection
- **Three-tier memory hierarchy** (episodes → summaries → facts) with tier-aware search
- **Cognitive memory types** (Episodic/Semantic/Procedural) with type-appropriate behavior

**Feature-gated:**
- **Vector search** via candle or fastembed with hybrid RRF merge
- **Cross-encoder reranking** via fastembed (post-RRF refinement)
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

**LongMemEval target: 93-96%** (competitive with OMEGA's #1 ranking of 95.4%)

**Target: ~6-8K lines of Rust for the full engine.**
