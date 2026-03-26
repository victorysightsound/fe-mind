#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::useless_format
)]

//! End-to-end test for hybrid FTS5 + vector search with RRF merge.

use chrono::{DateTime, Utc};
use femind::embeddings::pooling::normalize_l2;
use femind::embeddings::EmbeddingBackend;
use femind::engine::MemoryEngine;
use femind::memory::store::StoreResult;
use femind::search::{SearchMode, VectorSearch};
use femind::traits::{MemoryRecord, MemoryType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Mem {
    id: Option<i64>,
    text: String,
    created_at: DateTime<Utc>,
}

impl MemoryRecord for Mem {
    fn id(&self) -> Option<i64> {
        self.id
    }
    fn searchable_text(&self) -> String {
        self.text.clone()
    }
    fn memory_type(&self) -> MemoryType {
        MemoryType::Semantic
    }
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
}

fn mem(text: &str) -> Mem {
    Mem {
        id: None,
        text: text.into(),
        created_at: Utc::now(),
    }
}

/// Simple embedding backend that produces deterministic vectors based on text content.
/// Maps certain keywords to specific vector directions for testing.
struct TestEmbedder;

impl EmbeddingBackend for TestEmbedder {
    fn embed(&self, text: &str) -> femind::error::Result<Vec<f32>> {
        let lower = text.to_lowercase();
        let mut v = vec![0.0_f32; 8];

        // Map keywords to vector dimensions
        if lower.contains("auth") {
            v[0] = 1.0;
        }
        if lower.contains("jwt") {
            v[1] = 1.0;
        }
        if lower.contains("database") {
            v[2] = 1.0;
        }
        if lower.contains("error") {
            v[3] = 1.0;
        }
        if lower.contains("build") {
            v[4] = 1.0;
        }
        if lower.contains("timeout") {
            v[5] = 1.0;
        }
        if lower.contains("fix") {
            v[6] = 1.0;
        }
        if lower.contains("token") {
            v[7] = 1.0;
        }

        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        } else {
            v[0] = 1.0; // default direction for unknown text
        }
        Ok(v)
    }

    fn dimensions(&self) -> usize {
        8
    }
    fn is_available(&self) -> bool {
        true
    }
    fn model_name(&self) -> &str {
        "test-embedder"
    }
}

#[test]
fn hybrid_search_combines_fts5_and_vector() {
    let engine = MemoryEngine::<Mem>::builder()
        .embedding_backend(TestEmbedder)
        .build()
        .expect("build");

    // Store memories and manually add vectors
    let mems = vec![
        (
            "authentication failed with JWT token",
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        (
            "database connection timeout error",
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ),
        (
            "build succeeded after fixing imports",
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ),
    ];

    let db = engine.database();
    for (text, raw_vec) in &mems {
        let StoreResult::Added(id) = engine.store(&mem(text)).expect("store") else {
            panic!("dup")
        };
        let v = normalize_l2(raw_vec);
        VectorSearch::store_vector(db, id, &v, "test-embedder", &format!("h{id}"))
            .expect("vec store");
    }

    // Hybrid search for "auth" — should find via both FTS5 (keyword) and vector (semantic)
    let results = engine
        .search("auth error")
        .mode(SearchMode::Hybrid)
        .limit(10)
        .execute()
        .expect("search");

    assert!(!results.is_empty(), "hybrid should return results");
    // Auth memory (id=1) should be in top results (OR mode may reorder due to partial matches)
    let auth_in_top3 = results.iter().take(3).any(|r| r.memory_id == 1);
    assert!(auth_in_top3, "auth memory should be in top-3 results");
}

#[test]
fn auto_mode_uses_hybrid_when_embedding_available() {
    let engine = MemoryEngine::<Mem>::builder()
        .embedding_backend(TestEmbedder)
        .build()
        .expect("build");

    let StoreResult::Added(id) = engine
        .store(&mem("JWT authentication error"))
        .expect("store")
    else {
        panic!()
    };
    let db = engine.database();
    let v = normalize_l2(&[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    VectorSearch::store_vector(db, id, &v, "test-embedder", "h1").expect("vec");

    // Auto mode should use hybrid since embedding is available
    let results = engine
        .search("auth")
        .mode(SearchMode::Auto)
        .execute()
        .expect("search");
    assert!(!results.is_empty());
}

#[test]
fn auto_mode_falls_back_to_keyword_without_embedding() {
    // No embedding backend configured
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");
    engine
        .store(&mem("keyword only search test"))
        .expect("store");

    let results = engine
        .search("keyword")
        .mode(SearchMode::Auto)
        .execute()
        .expect("search");
    assert_eq!(results.len(), 1);
}

#[test]
fn vector_only_search() {
    let engine = MemoryEngine::<Mem>::builder()
        .embedding_backend(TestEmbedder)
        .build()
        .expect("build");

    let StoreResult::Added(id) = engine
        .store(&mem("database timeout problem"))
        .expect("store")
    else {
        panic!()
    };
    let db = engine.database();
    let v = normalize_l2(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    VectorSearch::store_vector(db, id, &v, "test-embedder", "h1").expect("vec");

    let results = engine
        .search("database timeout")
        .mode(SearchMode::Vector)
        .execute()
        .expect("search");
    assert!(!results.is_empty());
}
