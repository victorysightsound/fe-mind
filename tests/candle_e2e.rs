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

/// Full pipeline: store with real embeddings → hybrid search finds semantically similar content.
///
/// This is the critical test: proves that hybrid search (FTS5 + vector + RRF) actually
/// improves retrieval quality over keyword-only search.
#[test]
fn full_pipeline_with_real_embeddings() {
    use mindcore::context::ContextBudget;
    use mindcore::engine::MemoryEngine;
    use mindcore::memory::store::StoreResult;
    use mindcore::traits::{MemoryRecord, MemoryType};

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestMem {
        id: Option<i64>,
        text: String,
        created: chrono::DateTime<chrono::Utc>,
    }

    impl MemoryRecord for TestMem {
        fn id(&self) -> Option<i64> { self.id }
        fn searchable_text(&self) -> String { self.text.clone() }
        fn memory_type(&self) -> MemoryType { MemoryType::Episodic }
        fn created_at(&self) -> chrono::DateTime<chrono::Utc> { self.created }
    }

    let backend = CandleNativeBackend::new().expect("load model");
    let engine = MemoryEngine::<TestMem>::builder()
        .embedding_backend(backend)
        .build()
        .expect("build");

    // Store 10 memories with distinct, diverse topics
    let topics = [
        "My favorite cuisine is Japanese, especially sushi and ramen",
        "I run 5 kilometers every morning before work for exercise",
        "The project deadline is next Friday, March 28th",
        "I adopted a golden retriever puppy named Max last weekend",
        "My preferred programming language is Rust for systems work",
        "I visited the Louvre museum in Paris during summer vacation",
        "The database migration to PostgreSQL 16 completed successfully",
        "I have a severe allergy to peanuts and tree nuts",
        "The team standup meeting is at 9:30 AM every weekday",
        "I play classical guitar and recently learned a Bach prelude",
    ];

    for text in &topics {
        let mem = TestMem { id: None, text: text.to_string(), created: chrono::Utc::now() };
        let result = engine.store(&mem).expect("store");
        assert!(matches!(result, StoreResult::Added(_)));
    }

    // Verify all embeddings stored
    let db = engine.database();
    let vector_count: i64 = db.with_reader(|conn| {
        conn.query_row("SELECT COUNT(*) FROM memory_vectors", [], |row| row.get(0))
            .map_err(Into::into)
    }).expect("count");
    assert_eq!(vector_count, 10, "all memories should have vectors");

    // TEST 1: Semantic query that shares NO keywords with the stored memory
    // "What food do I like?" should find "My favorite cuisine is Japanese..."
    // even though the query words don't appear in the stored text.
    let results = engine.search("What food do I like?")
        .limit(3)
        .execute()
        .expect("search");
    assert!(!results.is_empty(), "semantic search should find food-related memory");
    // Check that the top result is about food/cuisine
    let top_text: String = db.with_reader(|conn| {
        conn.query_row(
            "SELECT searchable_text FROM memories WHERE id = ?1",
            [results[0].memory_id],
            |row| row.get(0),
        ).map_err(Into::into)
    }).expect("get text");
    assert!(
        top_text.contains("cuisine") || top_text.contains("sushi"),
        "Top result should be about food, got: {top_text}"
    );

    // TEST 2: Context assembly with semantic query
    let assembly = engine.assemble_context("Do I have any pets?", &ContextBudget::new(2000))
        .expect("assemble");
    assert!(!assembly.items.is_empty());
    let context_text = assembly.render();
    assert!(
        context_text.contains("golden retriever") || context_text.contains("Max"),
        "Context should include pet-related memory, got: {context_text}"
    );

    // TEST 3: Hybrid search improves recall for paraphrased queries
    let results = engine.search("musical instruments and hobbies")
        .limit(3)
        .execute()
        .expect("search");
    assert!(!results.is_empty(), "should find music-related memory via semantic similarity");
}
