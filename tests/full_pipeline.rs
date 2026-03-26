#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::useless_format
)]

//! Full pipeline end-to-end tests.
//!
//! Exercises: store → embed → search hybrid → score → context assembly → graph → activation → prune

use chrono::{DateTime, Duration, Utc};
use femind::context::ContextBudget;
use femind::engine::MemoryEngine;
use femind::memory::activation;
use femind::memory::pruning::{self, PruningPolicy};
use femind::memory::store::StoreResult;
use femind::memory::{GraphMemory, RelationType};
use femind::scoring::{CompositeScorer, ImportanceScorer, RecencyScorer};
use femind::traits::{MemoryRecord, MemoryType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Mem {
    id: Option<i64>,
    text: String,
    importance: u8,
    mem_type: MemoryType,
    category: Option<String>,
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
        self.mem_type
    }
    fn importance(&self) -> u8 {
        self.importance
    }
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }
}

fn mem(text: &str, imp: u8, mt: MemoryType, cat: Option<&str>) -> Mem {
    Mem {
        id: None,
        text: text.into(),
        importance: imp,
        mem_type: mt,
        category: cat.map(String::from),
        created_at: Utc::now(),
    }
}

/// Test the full pipeline: store 100 memories, search, score, assemble context.
#[test]
fn full_pipeline_100_memories() {
    let scorer = CompositeScorer::new(vec![
        Box::new(RecencyScorer::default_half_life()),
        Box::new(ImportanceScorer::default()),
    ]);

    let engine = MemoryEngine::<Mem>::builder()
        .scoring(scorer)
        .build()
        .expect("build");

    // Store 100 diverse memories
    let categories = ["error", "decision", "pattern", "note"];
    let types = [
        MemoryType::Episodic,
        MemoryType::Semantic,
        MemoryType::Procedural,
    ];

    for i in 0..100 {
        let cat = categories[i % categories.len()];
        let mt = types[i % types.len()];
        let imp = ((i % 10) + 1) as u8;
        engine
            .store(&mem(
                &format!(
                    "memory {i}: authentication JWT error handling pattern in production system"
                ),
                imp,
                mt,
                Some(cat),
            ))
            .expect("store");
    }

    assert_eq!(engine.count().expect("count"), 100);

    // Search with scoring
    let results = engine
        .search("authentication JWT")
        .limit(10)
        .execute()
        .expect("search");
    assert_eq!(results.len(), 10);

    // Verify results are scored (not just raw FTS5)
    for r in &results {
        assert!(r.score > 0.0, "all results should have positive scores");
    }

    // Context assembly
    let assembly = engine
        .assemble_context("JWT error", &ContextBudget::new(2000))
        .expect("assemble");
    assert!(!assembly.items.is_empty());
    assert!(assembly.total_tokens <= 2000);

    let rendered = assembly.render();
    assert!(!rendered.is_empty());
}

/// Test graph relationships end-to-end via the engine.
#[test]
fn graph_relationships_via_engine() {
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");

    let StoreResult::Added(error_id) = engine
        .store(&mem(
            "error: JWT token expired during auth flow",
            7,
            MemoryType::Procedural,
            Some("error"),
        ))
        .expect("store")
    else {
        panic!()
    };

    let StoreResult::Added(fix_id) = engine
        .store(&mem(
            "fix: implement token refresh before expiry",
            8,
            MemoryType::Procedural,
            Some("fix"),
        ))
        .expect("store")
    else {
        panic!()
    };

    let StoreResult::Added(cause_id) = engine
        .store(&mem(
            "root cause: clock drift between auth server and app",
            9,
            MemoryType::Semantic,
            Some("cause"),
        ))
        .expect("store")
    else {
        panic!()
    };

    // Create relationship chain: error → solved_by → fix, error → caused_by → cause
    let db = engine.database();
    GraphMemory::relate(db, error_id, fix_id, &RelationType::SolvedBy).expect("relate");
    GraphMemory::relate(db, error_id, cause_id, &RelationType::CausedBy).expect("relate");

    // Traverse from the error
    let related = GraphMemory::traverse(db, error_id, 3).expect("traverse");
    assert_eq!(related.len(), 2, "should find fix and cause");

    // Direct relations
    let direct = GraphMemory::direct_relations(db, error_id).expect("direct");
    assert_eq!(direct.len(), 2);
}

/// Test activation model end-to-end.
#[test]
fn activation_model_via_engine() {
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");

    let StoreResult::Added(id) = engine
        .store(&mem(
            "activation test memory",
            5,
            MemoryType::Semantic,
            None,
        ))
        .expect("store")
    else {
        panic!()
    };

    let db = engine.database();

    // Initial activation (base only)
    let a0 = activation::compute_activation(db, id).expect("activation");

    // Record multiple accesses
    for i in 0..5 {
        activation::record_access(db, id, &format!("query {i}")).expect("access");
    }

    // Activation should increase
    let a5 = activation::compute_activation(db, id).expect("activation");
    assert!(
        a5 > a0,
        "activation should increase with accesses: {a0} → {a5}"
    );

    // Update cache
    let cached = activation::update_activation_cache(db, id).expect("cache");
    assert!((cached - a5).abs() < 0.01);
}

/// Test pruning removes only what it should.
#[test]
fn pruning_selective() {
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");
    let db = engine.database();

    // Old episodic (should be pruned)
    let old_ep = Mem {
        id: None,
        text: "old debug session log".into(),
        importance: 3,
        mem_type: MemoryType::Episodic,
        category: Some("log".into()),
        created_at: Utc::now() - Duration::days(60),
    };
    engine.store(&old_ep).expect("store");

    // Old semantic (should NOT be pruned — wrong type)
    let old_sem = Mem {
        id: None,
        text: "project uses PostgreSQL".into(),
        importance: 7,
        mem_type: MemoryType::Semantic,
        category: Some("fact".into()),
        created_at: Utc::now() - Duration::days(90),
    };
    engine.store(&old_sem).expect("store");

    // Recent episodic (should NOT be pruned — too new)
    engine
        .store(&mem(
            "recent debug session",
            3,
            MemoryType::Episodic,
            Some("log"),
        ))
        .expect("store");

    // Set low activation on old memories
    db.with_writer(|conn| {
        conn.execute("UPDATE memories SET activation_cache = -3.0 WHERE created_at < datetime('now', '-30 days')", [])?;
        Ok(())
    }).expect("set activation");

    let report = pruning::prune(db, &PruningPolicy::default()).expect("prune");
    assert_eq!(
        report.pruned, 1,
        "should prune only the old episodic memory"
    );
    assert_eq!(
        engine.count().expect("count"),
        2,
        "semantic and recent should survive"
    );
}

/// Test two-tier database via engine.
#[test]
fn two_tier_via_engine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let project_path = dir.path().join("project.db");
    let global_path = dir.path().join("global.db");

    let engine = MemoryEngine::<Mem>::builder()
        .database(project_path.to_string_lossy().to_string())
        .global_database(global_path.to_string_lossy().to_string())
        .build()
        .expect("build");

    // Store in project database
    engine
        .store(&mem("project-specific fact", 5, MemoryType::Semantic, None))
        .expect("store");

    // Store in global database directly
    let gdb = engine.global_database().expect("global db");
    gdb.with_writer(|conn| {
        conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
             VALUES ('global cross-project learning', 'semantic', 'ghash', '{}')",
            [],
        )?;
        Ok(())
    })
    .expect("global insert");

    // Both databases accessible
    assert_eq!(engine.count().expect("count"), 1); // project db only
}

/// Test deduplication at scale.
#[test]
fn dedup_at_scale() {
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");

    // Store 50 unique memories
    for i in 0..50 {
        engine
            .store(&mem(
                &format!("unique memory {i}"),
                5,
                MemoryType::Semantic,
                None,
            ))
            .expect("store");
    }

    // Try to store all 50 again — should all be duplicates
    for i in 0..50 {
        let result = engine
            .store(&mem(
                &format!("unique memory {i}"),
                5,
                MemoryType::Semantic,
                None,
            ))
            .expect("store");
        assert!(
            matches!(result, StoreResult::Duplicate(_)),
            "memory {i} should be a duplicate"
        );
    }

    assert_eq!(engine.count().expect("count"), 50);
}

/// Full pipeline with NoopBackend: store → embed → hybrid search → context assembly.
///
/// Uses NoopBackend (zero vectors), so vector search won't add signal, but it verifies
/// the entire embedding-at-store + hybrid-search path executes without errors.
#[test]
fn full_pipeline_with_noop_embedding() {
    use femind::embeddings::NoopBackend;

    let backend = NoopBackend::new(384);
    let engine = MemoryEngine::<Mem>::builder()
        .embedding_backend(backend)
        .build()
        .expect("build");

    // Store 25 diverse memories
    let topics = [
        "The authentication system uses JWT tokens with 15-minute expiry",
        "Database queries timeout after 30 seconds of inactivity",
        "The CI pipeline runs tests in parallel across 4 workers",
        "User preferences are stored in a PostgreSQL JSONB column",
        "The caching layer uses Redis with a 5-minute TTL",
        "Error monitoring is handled by Sentry with PII scrubbing",
        "The API rate limiter allows 100 requests per minute per user",
        "Deployments use blue-green strategy with health checks",
        "The search feature uses Elasticsearch with custom analyzers",
        "File uploads are stored in S3 with server-side encryption",
        "The notification system supports email SMS and push",
        "Background jobs are processed by Sidekiq with Redis backend",
        "The frontend uses React with server-side rendering",
        "GraphQL subscriptions use WebSocket connections",
        "The payment gateway integration handles Stripe webhooks",
        "Logging is structured JSON sent to CloudWatch",
        "The auth middleware validates tokens on every request",
        "Database migrations are managed by Flyway",
        "The test suite includes unit integration and e2e tests",
        "Performance monitoring uses Datadog APM traces",
        "The mobile app communicates via REST and gRPC endpoints",
        "Session management uses HttpOnly secure cookies",
        "The CDN caches static assets for 24 hours",
        "Feature flags are managed by LaunchDarkly",
        "The backup system runs daily incremental and weekly full",
    ];

    for (i, text) in topics.iter().enumerate() {
        let mt = [
            MemoryType::Semantic,
            MemoryType::Procedural,
            MemoryType::Episodic,
        ][i % 3];
        let result = engine.store(&mem(text, 5, mt, None)).expect("store");
        assert!(
            matches!(result, StoreResult::Added(_)),
            "memory {i} should be added"
        );
    }

    assert_eq!(engine.count().expect("count"), 25);

    // Verify embedding_status was set (NoopBackend always succeeds)
    let db = engine.database();
    let success_count: i64 = db
        .with_reader(|conn| {
            conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE embedding_status = 'success'",
                [],
                |row| row.get(0),
            )
            .map_err(Into::into)
        })
        .expect("count query");
    assert_eq!(
        success_count, 25,
        "all memories should have embedding_status='success'"
    );

    // Verify vectors were stored
    let vector_count: i64 = db
        .with_reader(|conn| {
            conn.query_row("SELECT COUNT(*) FROM memory_vectors", [], |row| row.get(0))
                .map_err(Into::into)
        })
        .expect("vector count");
    assert_eq!(vector_count, 25, "all memories should have stored vectors");

    // Search with Auto mode (should use hybrid since embedding backend is configured)
    let results = engine
        .search("authentication JWT token")
        .limit(10)
        .execute()
        .expect("search");
    // NoopBackend produces zero vectors (cosine similarity = NaN/0 for all),
    // so hybrid search degrades to FTS5-only ranking. Still should find results.
    assert!(!results.is_empty(), "should find results via FTS5 path");
    assert!(results[0].score > 0.0);

    // Context assembly
    let assembly = engine
        .assemble_context("database query timeout", &ContextBudget::new(4000))
        .expect("assemble");
    assert!(
        !assembly.items.is_empty(),
        "context should contain relevant memories"
    );
    assert!(assembly.total_tokens <= 4000);

    // Search for a topic that requires keyword matching
    let results = engine
        .search("Redis caching TTL")
        .limit(5)
        .execute()
        .expect("search");
    assert!(!results.is_empty(), "should find Redis-related memory");

    // Search with memory type filter
    let results = engine
        .search("authentication")
        .memory_type(MemoryType::Semantic)
        .limit(10)
        .execute()
        .expect("search");
    assert!(
        !results.is_empty(),
        "should find semantic memories about auth"
    );
}

/// Test store_batch with NoopBackend.
#[test]
fn store_batch_with_noop_embedding() {
    use femind::embeddings::NoopBackend;

    let backend = NoopBackend::new(384);
    let engine = MemoryEngine::<Mem>::builder()
        .embedding_backend(backend)
        .build()
        .expect("build");

    let records: Vec<Mem> = (0..30)
        .map(|i| {
            mem(
                &format!("batch memory {i}: topic about engineering and systems"),
                5,
                MemoryType::Semantic,
                None,
            )
        })
        .collect();

    let results = engine.store_batch(&records).expect("batch store");
    assert_eq!(results.len(), 30);
    assert!(results.iter().all(|r| matches!(r, StoreResult::Added(_))));

    // Verify all embeddings stored
    let db = engine.database();
    let vector_count: i64 = db
        .with_reader(|conn| {
            conn.query_row("SELECT COUNT(*) FROM memory_vectors", [], |row| row.get(0))
                .map_err(Into::into)
        })
        .expect("vector count");
    assert_eq!(vector_count, 30);

    // Search works after batch store
    let results = engine
        .search("engineering systems")
        .limit(5)
        .execute()
        .expect("search");
    assert!(!results.is_empty());
}
