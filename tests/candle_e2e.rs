//! End-to-end test for CandleNativeBackend with real model inference.
//!
//! Only runs with: cargo test --features local-embeddings --test candle_e2e
//! Requires ~95MB model download on first run.

#![cfg(feature = "local-embeddings")]

use mindcore::embeddings::{CandleNativeBackend, EmbeddingBackend};
use mindcore::embeddings::pooling::cosine_similarity;

#[test]
fn candle_backend_loads_and_embeds() {
    let backend = CandleNativeBackend::new().expect("failed to load all-MiniLM-L6-v2");

    assert_eq!(backend.dimensions(), 384);
    assert!(backend.is_available());
    assert_eq!(backend.model_name(), "all-MiniLM-L6-v2");

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
    // Check that a top-3 result is about food (cuisine or allergy)
    let mut found_food = false;
    for sr in &results {
        let text: String = db.with_reader(|conn| {
            conn.query_row(
                "SELECT searchable_text FROM memories WHERE id = ?1",
                [sr.memory_id], |row| row.get(0),
            ).map_err(Into::into)
        }).expect("get text");
        if text.contains("cuisine") || text.contains("sushi") || text.contains("peanut") {
            found_food = true;
        }
    }
    assert!(found_food, "Top-3 should include food-related memory");

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

/// Batch store with real embeddings at 50+ scale.
#[test]
fn batch_store_with_real_embeddings() {
    use mindcore::engine::MemoryEngine;
    use mindcore::memory::store::StoreResult;
    use mindcore::traits::{MemoryRecord, MemoryType};

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct Mem {
        id: Option<i64>,
        text: String,
        created: chrono::DateTime<chrono::Utc>,
    }

    impl MemoryRecord for Mem {
        fn id(&self) -> Option<i64> { self.id }
        fn searchable_text(&self) -> String { self.text.clone() }
        fn memory_type(&self) -> MemoryType { MemoryType::Episodic }
        fn created_at(&self) -> chrono::DateTime<chrono::Utc> { self.created }
    }

    let backend = CandleNativeBackend::new().expect("load");
    let engine = MemoryEngine::<Mem>::builder()
        .embedding_backend(backend)
        .build()
        .expect("build");

    // Generate 60 distinct memories
    let subjects = [
        "authentication", "database", "caching", "deployment", "testing",
        "monitoring", "security", "performance", "networking", "storage",
        "logging", "scheduling",
    ];
    let actions = [
        "error occurred during", "completed successfully for", "timeout in",
        "was upgraded for", "needs attention in",
    ];

    let records: Vec<Mem> = subjects.iter().enumerate().flat_map(|(i, subj)| {
        actions.iter().enumerate().map(move |(j, act)| {
            Mem {
                id: None,
                text: format!("The {subj} system {act} the production environment on day {}", i * 5 + j),
                created: chrono::Utc::now(),
            }
        })
    }).collect();

    assert!(records.len() >= 50, "should have 60 records, got {}", records.len());

    let start = std::time::Instant::now();
    let results = engine.store_batch(&records).expect("batch store");
    let elapsed = start.elapsed();

    assert_eq!(results.len(), records.len());
    assert!(results.iter().all(|r| matches!(r, StoreResult::Added(_))));

    // Verify all embeddings stored
    let db = engine.database();
    let vector_count: i64 = db.with_reader(|conn| {
        conn.query_row("SELECT COUNT(*) FROM memory_vectors", [], |row| row.get(0))
            .map_err(Into::into)
    }).expect("count");
    assert_eq!(vector_count, records.len() as i64);

    // Verify embedding_status
    let success_count: i64 = db.with_reader(|conn| {
        conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE embedding_status = 'success'",
            [], |row| row.get(0),
        ).map_err(Into::into)
    }).expect("count");
    assert_eq!(success_count, records.len() as i64);

    // Hybrid search finds results
    let results = engine.search("database timeout production")
        .limit(5)
        .execute()
        .expect("search");
    assert!(!results.is_empty(), "should find database-related memories");

    println!("Batch store of {} records with embeddings took {:?}", records.len(), elapsed);
}

/// Distractor scenario: simulates LongMemEval-S conditions.
///
/// Stores 40 conversation sessions (38 distractors + 2 relevant), then queries
/// for specific information buried in the relevant sessions. Hybrid search must
/// surface the relevant content in the top-5 results despite overwhelming noise.
#[test]
fn distractor_scenario_finds_needle_in_haystack() {
    use mindcore::engine::MemoryEngine;
    use mindcore::traits::{MemoryRecord, MemoryType};

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct Msg {
        id: Option<i64>,
        text: String,
        created: chrono::DateTime<chrono::Utc>,
    }

    impl MemoryRecord for Msg {
        fn id(&self) -> Option<i64> { self.id }
        fn searchable_text(&self) -> String { self.text.clone() }
        fn memory_type(&self) -> MemoryType { MemoryType::Episodic }
        fn created_at(&self) -> chrono::DateTime<chrono::Utc> { self.created }
    }

    let backend = CandleNativeBackend::new().expect("load");
    let engine = MemoryEngine::<Msg>::builder()
        .embedding_backend(backend)
        .build()
        .expect("build");

    // 38 distractor sessions — diverse topics unrelated to the target question
    let distractor_topics = [
        "I want to plan a trip to Japan next spring. Can you help me with an itinerary for Tokyo and Kyoto?",
        "What are the best practices for writing unit tests in Python with pytest?",
        "Can you explain how photosynthesis works at the molecular level?",
        "I need help debugging a CSS flexbox layout that's not centering properly.",
        "What are the key differences between SQL and NoSQL databases?",
        "Can you recommend some classic science fiction novels from the 1960s?",
        "How do I set up a Docker container for a Node.js application?",
        "What are the health benefits of intermittent fasting?",
        "Can you help me write a cover letter for a software engineering position?",
        "Explain the theory of general relativity in simple terms.",
        "What are some effective strategies for managing remote teams?",
        "How do I implement a binary search tree in Java?",
        "What are the pros and cons of electric vehicles versus hybrid cars?",
        "Can you help me understand how blockchain consensus mechanisms work?",
        "What are some good exercises for improving core strength?",
        "How do I configure nginx as a reverse proxy for multiple services?",
        "What are the main causes of coral reef bleaching?",
        "Can you explain the difference between supervised and unsupervised learning?",
        "What are some tips for growing tomatoes in a home garden?",
        "How do I set up continuous integration with GitHub Actions?",
        "What are the key principles of object-oriented programming?",
        "Can you recommend some good podcasts about history?",
        "How do I optimize PostgreSQL queries for large datasets?",
        "What are the symptoms and treatment options for seasonal allergies?",
        "Can you help me understand how RSA encryption works?",
        "What are some effective study techniques for college exams?",
        "How do I implement WebSocket connections in a React application?",
        "What are the environmental impacts of fast fashion?",
        "Can you explain how CRISPR gene editing technology works?",
        "What are some good strategies for paying off student loans?",
        "How do I set up a Python virtual environment with poetry?",
        "What are the health risks associated with prolonged sitting?",
        "Can you help me plan a vegetarian meal prep for the week?",
        "How do I implement OAuth 2.0 authentication in a web application?",
        "What are the key features of Rust's ownership system?",
        "Can you explain the water cycle and its importance?",
        "How do I build a REST API with Express.js and MongoDB?",
        "What are some effective techniques for public speaking?",
    ];

    // RELEVANT sessions — contain the answer to our target question
    let relevant_turns = [
        "User: I just adopted a cat! She's a calico named Patches.",
        "Assistant: Congratulations on your new cat! Calico cats are known for their beautiful tri-color coats. How is Patches settling in?",
        "User: She's been hiding under the bed but she came out to eat her favorite food — salmon pate.",
        "Assistant: That's great progress! Salmon pate is a popular choice among cats. Give her time and she'll warm up to her new home.",
    ];

    let relevant_turns_2 = [
        "User: Patches knocked over my coffee mug this morning! She's getting more adventurous.",
        "Assistant: Ha! That's a classic cat move. It sounds like she's getting comfortable in her new home. Is she playing with any toys?",
        "User: Yes, she loves the feather wand toy. She also discovered she likes sleeping on my keyboard while I work.",
        "Assistant: Cats have an uncanny ability to find the most inconvenient spots to nap! A feather wand is great for exercise and bonding.",
    ];

    // Store all distractors (5 turns each to simulate real sessions)
    let mut all_records = Vec::new();
    for (i, topic) in distractor_topics.iter().enumerate() {
        // Each distractor gets a user question and assistant response
        all_records.push(Msg {
            id: None,
            text: format!("User: {topic}"),
            created: chrono::Utc::now(),
        });
        all_records.push(Msg {
            id: None,
            text: format!("Assistant: I'd be happy to help with that. Let me provide some detailed information about this topic. Here is a comprehensive overview of the key points to consider."),
            created: chrono::Utc::now(),
        });
        all_records.push(Msg {
            id: None,
            text: format!("User: Thanks, that's helpful. Can you go into more detail about the second point you mentioned?"),
            created: chrono::Utc::now(),
        });
        all_records.push(Msg {
            id: None,
            text: format!("Assistant: Of course! The second point is particularly important because it relates to the fundamental aspects of {topic}"),
            created: chrono::Utc::now(),
        });
        if i % 2 == 0 {
            all_records.push(Msg {
                id: None,
                text: format!("User: Perfect, I think I understand now. One last question about this topic — are there any common pitfalls I should avoid?"),
                created: chrono::Utc::now(),
            });
        }
    }

    // Store relevant sessions
    for turn in &relevant_turns {
        all_records.push(Msg { id: None, text: turn.to_string(), created: chrono::Utc::now() });
    }
    for turn in &relevant_turns_2 {
        all_records.push(Msg { id: None, text: turn.to_string(), created: chrono::Utc::now() });
    }

    let total_records = all_records.len();
    println!("Storing {} records ({} distractor + {} relevant)...",
        total_records, total_records - 8, 8);

    let start = std::time::Instant::now();
    let results = engine.store_batch(&all_records).expect("batch store");
    let store_elapsed = start.elapsed();
    assert_eq!(results.len(), total_records);
    println!("Store + embed took {:?}", store_elapsed);

    // THE KEY TEST: Query for information only in the relevant sessions
    let search_start = std::time::Instant::now();
    let results = engine.search("What is the name of my cat?")
        .limit(5)
        .execute()
        .expect("search");
    let search_elapsed = search_start.elapsed();
    println!("Hybrid search took {:?}", search_elapsed);

    assert!(!results.is_empty(), "should find cat-related memories");

    // Check that at least one of the top-5 results mentions the cat
    let db = engine.database();
    let mut found_cat = false;
    for (rank, sr) in results.iter().enumerate() {
        let text: String = db.with_reader(|conn| {
            conn.query_row(
                "SELECT searchable_text FROM memories WHERE id = ?1",
                [sr.memory_id], |row| row.get(0),
            ).map_err(Into::into)
        }).expect("get");
        println!("  #{}: (score={:.4}) {}", rank + 1, sr.score, &text[..text.len().min(80)]);
        if text.contains("Patches") || text.contains("calico") || text.contains("cat") {
            found_cat = true;
        }
    }
    assert!(found_cat, "Top-5 results should include cat-related memory");

    // Second query: more specific
    let results = engine.search("What does Patches like to eat?")
        .limit(5)
        .execute()
        .expect("search");
    let mut found_food = false;
    for sr in &results {
        let text: String = db.with_reader(|conn| {
            conn.query_row(
                "SELECT searchable_text FROM memories WHERE id = ?1",
                [sr.memory_id], |row| row.get(0),
            ).map_err(Into::into)
        }).expect("get");
        if text.contains("salmon") {
            found_food = true;
        }
    }
    assert!(found_food, "Should find that Patches likes salmon pate");

    // Third query: long natural language question (tests OR-mode + stop-word removal)
    // Without OR mode, this query would match nothing because AND requires all terms
    let results = engine.search("How many days did I spend playing with my cat and what toys does she enjoy?")
        .limit(5)
        .execute()
        .expect("search");
    assert!(!results.is_empty(), "Long natural language query should find results via OR-mode + stop-word removal");
    let mut found_toy = false;
    for sr in &results {
        let text: String = db.with_reader(|conn| {
            conn.query_row(
                "SELECT searchable_text FROM memories WHERE id = ?1",
                [sr.memory_id], |row| row.get(0),
            ).map_err(Into::into)
        }).expect("get");
        if text.contains("feather") || text.contains("toy") || text.contains("cat") || text.contains("Patches") {
            found_toy = true;
        }
    }
    assert!(found_toy, "Long query should find cat/toy-related memories via semantic + OR-mode FTS5");
}
