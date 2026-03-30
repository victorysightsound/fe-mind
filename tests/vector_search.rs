#![allow(
    clippy::approx_constant,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used
)]

//! Integration tests for Phase 5: Vector Search + Hybrid RRF
//!
//! Uses NoopBackend (zero vectors) to test the infrastructure
//! without requiring candle model download.

use femind::embeddings::pooling::{bytes_to_vec, cosine_similarity, normalize_l2, vec_to_bytes};
use femind::embeddings::{EmbeddingBackend, FallbackBackend, NoopBackend};
use femind::search::FtsResult;
use femind::search::{VectorSearch, rrf_merge};
use femind::storage::Database;
use femind::storage::migrations;

fn setup_db() -> Database {
    let db = Database::open_in_memory().expect("open");
    db.with_writer(|conn| {
        migrations::migrate(conn)?;
        Ok(())
    })
    .expect("migrate");
    db
}

fn insert_mem(db: &Database, id: i64, text: &str) {
    db.with_writer(|conn| {
        conn.execute(
            "INSERT INTO memories (id, searchable_text, memory_type, content_hash, record_json)
             VALUES (?1, ?2, 'semantic', ?3, '{}')",
            rusqlite::params![id, text, format!("hash_{id}")],
        )?;
        Ok(())
    })
    .expect("insert");
}

// --- Vector Storage ---

#[test]
fn store_and_retrieve_vectors() {
    let db = setup_db();
    insert_mem(&db, 1, "auth error");

    let v = normalize_l2(&[1.0, 0.0, 0.0]);
    VectorSearch::store_vector(&db, 1, &v, "test-model", "hash_1").expect("store");

    let query = normalize_l2(&[1.0, 0.0, 0.0]);
    let results =
        VectorSearch::search(&db, &query, &[String::from("test-model")], 10).expect("search");
    assert_eq!(results.len(), 1);
    assert!(
        (results[0].score - 1.0).abs() < 0.01,
        "identical vectors should have sim ~1.0"
    );
}

#[test]
fn vector_model_isolation() {
    let db = setup_db();
    insert_mem(&db, 1, "mem a");
    insert_mem(&db, 2, "mem b");

    let v = normalize_l2(&[1.0, 0.0, 0.0]);
    VectorSearch::store_vector(&db, 1, &v, "model-alpha", "h1").expect("store");
    VectorSearch::store_vector(&db, 2, &v, "model-beta", "h2").expect("store");

    // Only vectors from matching model should be returned
    let r = VectorSearch::search(&db, &v, &[String::from("model-alpha")], 10).expect("search");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].memory_id, 1);
}

// --- RRF Merge ---

#[test]
fn rrf_merges_results() {
    let kw = vec![
        FtsResult {
            memory_id: 1,
            score: 1.0,
        },
        FtsResult {
            memory_id: 2,
            score: 0.5,
        },
    ];
    let vec = vec![
        FtsResult {
            memory_id: 2,
            score: 0.9,
        },
        FtsResult {
            memory_id: 3,
            score: 0.3,
        },
    ];

    let merged = rrf_merge(&kw, &vec, "test query", 10);
    assert!(!merged.is_empty());
    // Memory 2 appears in both → should rank highest
    assert_eq!(merged[0].memory_id, 2);
}

#[test]
fn rrf_quoted_favors_keyword() {
    let kw = vec![FtsResult {
        memory_id: 1,
        score: 1.0,
    }];
    let vec_results = vec![FtsResult {
        memory_id: 2,
        score: 1.0,
    }];

    let merged = rrf_merge(&kw, &vec_results, "\"exact match\"", 10);
    // With quoted query, keyword result (id=1) should rank higher
    assert_eq!(merged[0].memory_id, 1);
}

#[test]
fn rrf_question_favors_vector() {
    let kw = vec![FtsResult {
        memory_id: 1,
        score: 1.0,
    }];
    let vec_results = vec![FtsResult {
        memory_id: 2,
        score: 1.0,
    }];

    let merged = rrf_merge(&kw, &vec_results, "how does authentication work", 10);
    // With question query, vector result (id=2) should rank higher
    assert_eq!(merged[0].memory_id, 2);
}

// --- Pooling Utilities ---

#[test]
fn vector_serialization_roundtrip() {
    let original = vec![0.1_f32, -0.5, 3.14, 0.0, 1.0];
    let bytes = vec_to_bytes(&original);
    let restored = bytes_to_vec(&bytes);
    assert_eq!(original.len(), restored.len());
    for (a, b) in original.iter().zip(restored.iter()) {
        assert!((a - b).abs() < f32::EPSILON);
    }
}

#[test]
fn similarity_ranking_correct() {
    let query = normalize_l2(&[1.0, 0.0, 0.0]);
    let exact = normalize_l2(&[1.0, 0.0, 0.0]);
    let similar = normalize_l2(&[0.9, 0.1, 0.0]);
    let orthogonal = normalize_l2(&[0.0, 1.0, 0.0]);

    let sim_exact = cosine_similarity(&query, &exact);
    let sim_similar = cosine_similarity(&query, &similar);
    let sim_orth = cosine_similarity(&query, &orthogonal);

    assert!(sim_exact > sim_similar, "exact > similar");
    assert!(sim_similar > sim_orth, "similar > orthogonal");
    assert!(sim_orth.abs() < 0.01, "orthogonal should be ~0");
}

// --- FallbackBackend ---

#[test]
fn fallback_with_backend() {
    let inner = Box::new(NoopBackend::new(384));
    let backend = FallbackBackend::new(inner);
    assert!(backend.is_available());

    let vec = backend.embed("test").expect("embed");
    assert_eq!(vec.len(), 384);
}

#[test]
fn fallback_without_backend() {
    let backend = FallbackBackend::none(384);
    assert!(!backend.is_available());

    let result = backend.embed("test");
    assert!(result.is_err());
}

// --- NoopBackend ---

#[test]
fn noop_produces_zero_vectors() {
    let backend = NoopBackend::new(128);
    let vec = backend.embed("anything").expect("embed");
    assert_eq!(vec.len(), 128);
    assert!(vec.iter().all(|&v| v == 0.0));
}

#[test]
fn noop_batch_consistent() {
    let backend = NoopBackend::new(64);
    let vecs = backend.embed_batch(&["a", "b", "c"]).expect("batch");
    assert_eq!(vecs.len(), 3);
    assert!(vecs.iter().all(|v| v.len() == 64));
}
