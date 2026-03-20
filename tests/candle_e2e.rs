//! End-to-end test for CandleNativeBackend with real model inference.
//!
//! Only runs with: cargo test --features local-embeddings --test candle_e2e
//! Requires ~95MB model download on first run.

#![cfg(feature = "local-embeddings")]

use mindcore::embeddings::{CandleNativeBackend, EmbeddingBackend};
use mindcore::embeddings::pooling::cosine_similarity;

#[test]
fn candle_backend_loads_and_embeds() {
    let backend = CandleNativeBackend::new().expect("failed to load granite-small-r2");

    assert_eq!(backend.dimensions(), 384);
    assert!(backend.is_available());
    assert_eq!(backend.model_name(), "granite-embedding-small-english-r2");

    let vec = backend.embed("authentication error with JWT token").expect("embed failed");
    assert_eq!(vec.len(), 384, "expected 384 dimensions");

    // Vector should be L2-normalized (magnitude ≈ 1.0)
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.01,
        "vector should be L2-normalized, got magnitude {magnitude}"
    );
}

#[test]
fn similar_texts_have_high_similarity() {
    let backend = CandleNativeBackend::new().expect("load");

    let v1 = backend.embed("authentication failed with invalid JWT token").expect("embed 1");
    let v2 = backend.embed("auth error: JWT token expired").expect("embed 2");
    let v3 = backend.embed("the weather is sunny today").expect("embed 3");

    let sim_related = cosine_similarity(&v1, &v2);
    let sim_unrelated = cosine_similarity(&v1, &v3);

    println!("Similar texts cosine similarity: {sim_related}");
    println!("Unrelated texts cosine similarity: {sim_unrelated}");

    assert!(
        sim_related > sim_unrelated,
        "related texts ({sim_related}) should have higher similarity than unrelated ({sim_unrelated})"
    );
    assert!(
        sim_related > 0.5,
        "related texts should have similarity > 0.5, got {sim_related}"
    );
}

#[test]
fn batch_embedding_consistent() {
    let backend = CandleNativeBackend::new().expect("load");

    let texts = &["hello world", "authentication error", "database timeout"];

    // Single embeddings
    let singles: Vec<Vec<f32>> = texts
        .iter()
        .map(|t| backend.embed(t).expect("single embed"))
        .collect();

    // Batch embedding
    let batch = backend.embed_batch(texts).expect("batch embed");

    assert_eq!(batch.len(), 3);

    // Each batch result should match the single result
    for (i, (single, batched)) in singles.iter().zip(batch.iter()).enumerate() {
        let sim = cosine_similarity(single, batched);
        assert!(
            sim > 0.999,
            "text {i}: batch vs single similarity should be ~1.0, got {sim}"
        );
    }
}

#[test]
fn embedding_deterministic() {
    let backend = CandleNativeBackend::new().expect("load");

    let v1 = backend.embed("deterministic test input").expect("embed 1");
    let v2 = backend.embed("deterministic test input").expect("embed 2");

    let sim = cosine_similarity(&v1, &v2);
    assert!(
        (sim - 1.0).abs() < 0.001,
        "same input should produce identical vectors, got similarity {sim}"
    );
}
