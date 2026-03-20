//! # MindCore
//!
//! A pluggable, feature-gated memory engine for AI agent applications.
//!
//! MindCore provides persistent storage, keyword search (FTS5), vector search,
//! hybrid retrieval (RRF), graph relationships, memory consolidation, cognitive
//! decay modeling, and token-budget-aware context assembly.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mindcore::prelude::*;
//!
//! // Define your memory type
//! // (implement MemoryRecord for your struct)
//!
//! // Build the engine
//! // let engine = MemoryEngine::<MyMemory>::builder()
//! //     .database("memory.db")
//! //     .build()?;
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `fts5` (default) | FTS5 keyword search with Porter stemming |
//! | `vector-search` | Candle embeddings + hybrid RRF search |
//! | `reranking` | Cross-encoder reranking (post-RRF) |
//! | `graph-memory` | SQLite relationship tables + CTE traversal |
//! | `temporal` | Bi-temporal validity tracking |
//! | `consolidation` | Hash/similarity/LLM-assisted dedup pipeline |
//! | `activation-model` | ACT-R cognitive decay model |
//! | `two-tier` | Global + project memory with promotion |
//! | `encryption` | SQLCipher encryption at rest |
//! | `mcp-server` | MCP server interface |
//! | `full` | All features except encryption and mcp-server |

pub mod engine;
pub mod error;
pub mod memory;
pub mod scoring;
pub mod search;
pub mod storage;
pub mod traits;

/// Prelude module — common imports for consumers.
pub mod prelude {
    pub use crate::engine::MemoryEngine;
    pub use crate::error::{MindCoreError, Result};
    pub use crate::memory::store::StoreResult;
    pub use crate::search::{SearchBuilder, SearchDepth, SearchMode, SearchResult};
    pub use crate::traits::{MemoryMeta, MemoryRecord, MemoryType};
}
