//! Full pipeline end-to-end tests.
//!
//! Exercises: store → embed → search hybrid → score → context assembly → graph → activation → prune

use chrono::{DateTime, Duration, Utc};
use mindcore::context::ContextBudget;
use mindcore::engine::MemoryEngine;
use mindcore::memory::activation;
use mindcore::memory::pruning::{self, PruningPolicy};
use mindcore::memory::store::StoreResult;
use mindcore::memory::{GraphMemory, RelationType};
use mindcore::scoring::{CompositeScorer, ImportanceScorer, RecencyScorer};
use mindcore::traits::{MemoryRecord, MemoryType};
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
    fn id(&self) -> Option<i64> { self.id }
    fn searchable_text(&self) -> String { self.text.clone() }
    fn memory_type(&self) -> MemoryType { self.mem_type }
    fn importance(&self) -> u8 { self.importance }
    fn created_at(&self) -> DateTime<Utc> { self.created_at }
    fn category(&self) -> Option<&str> { self.category.as_deref() }
}

fn mem(text: &str, imp: u8, mt: MemoryType, cat: Option<&str>) -> Mem {
    Mem {
        id: None, text: text.into(), importance: imp, mem_type: mt,
        category: cat.map(String::from), created_at: Utc::now(),
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
    let types = [MemoryType::Episodic, MemoryType::Semantic, MemoryType::Procedural];

    for i in 0..100 {
        let cat = categories[i % categories.len()];
        let mt = types[i % types.len()];
        let imp = ((i % 10) + 1) as u8;
        engine.store(&mem(
            &format!("memory {i}: authentication JWT error handling pattern in production system"),
            imp, mt, Some(cat),
        )).expect("store");
    }

    assert_eq!(engine.count().expect("count"), 100);

    // Search with scoring
    let results = engine.search("authentication JWT")
        .limit(10)
        .execute()
        .expect("search");
    assert_eq!(results.len(), 10);

    // Verify results are scored (not just raw FTS5)
    for r in &results {
        assert!(r.score > 0.0, "all results should have positive scores");
    }

    // Context assembly
    let assembly = engine.assemble_context("JWT error", &ContextBudget::new(2000))
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

    let StoreResult::Added(error_id) = engine.store(&mem(
        "error: JWT token expired during auth flow", 7, MemoryType::Procedural, Some("error")
    )).expect("store") else { panic!() };

    let StoreResult::Added(fix_id) = engine.store(&mem(
        "fix: implement token refresh before expiry", 8, MemoryType::Procedural, Some("fix")
    )).expect("store") else { panic!() };

    let StoreResult::Added(cause_id) = engine.store(&mem(
        "root cause: clock drift between auth server and app", 9, MemoryType::Semantic, Some("cause")
    )).expect("store") else { panic!() };

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

    let StoreResult::Added(id) = engine.store(&mem(
        "activation test memory", 5, MemoryType::Semantic, None
    )).expect("store") else { panic!() };

    let db = engine.database();

    // Initial activation (base only)
    let a0 = activation::compute_activation(db, id).expect("activation");

    // Record multiple accesses
    for i in 0..5 {
        activation::record_access(db, id, &format!("query {i}")).expect("access");
    }

    // Activation should increase
    let a5 = activation::compute_activation(db, id).expect("activation");
    assert!(a5 > a0, "activation should increase with accesses: {a0} → {a5}");

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
        id: None, text: "old debug session log".into(), importance: 3,
        mem_type: MemoryType::Episodic, category: Some("log".into()),
        created_at: Utc::now() - Duration::days(60),
    };
    engine.store(&old_ep).expect("store");

    // Old semantic (should NOT be pruned — wrong type)
    let old_sem = Mem {
        id: None, text: "project uses PostgreSQL".into(), importance: 7,
        mem_type: MemoryType::Semantic, category: Some("fact".into()),
        created_at: Utc::now() - Duration::days(90),
    };
    engine.store(&old_sem).expect("store");

    // Recent episodic (should NOT be pruned — too new)
    engine.store(&mem("recent debug session", 3, MemoryType::Episodic, Some("log"))).expect("store");

    // Set low activation on old memories
    db.with_writer(|conn| {
        conn.execute("UPDATE memories SET activation_cache = -3.0 WHERE created_at < datetime('now', '-30 days')", [])?;
        Ok(())
    }).expect("set activation");

    let report = pruning::prune(db, &PruningPolicy::default()).expect("prune");
    assert_eq!(report.pruned, 1, "should prune only the old episodic memory");
    assert_eq!(engine.count().expect("count"), 2, "semantic and recent should survive");
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
    engine.store(&mem("project-specific fact", 5, MemoryType::Semantic, None)).expect("store");

    // Store in global database directly
    let gdb = engine.global_database().expect("global db");
    gdb.with_writer(|conn| {
        conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
             VALUES ('global cross-project learning', 'semantic', 'ghash', '{}')",
            [],
        )?;
        Ok(())
    }).expect("global insert");

    // Both databases accessible
    assert_eq!(engine.count().expect("count"), 1); // project db only
}

/// Test deduplication at scale.
#[test]
fn dedup_at_scale() {
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");

    // Store 50 unique memories
    for i in 0..50 {
        engine.store(&mem(&format!("unique memory {i}"), 5, MemoryType::Semantic, None)).expect("store");
    }

    // Try to store all 50 again — should all be duplicates
    for i in 0..50 {
        let result = engine.store(&mem(&format!("unique memory {i}"), 5, MemoryType::Semantic, None)).expect("store");
        assert!(matches!(result, StoreResult::Duplicate(_)), "memory {i} should be a duplicate");
    }

    assert_eq!(engine.count().expect("count"), 50);
}
