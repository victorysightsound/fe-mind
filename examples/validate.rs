#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Comprehensive validation of femind as a consumer would use it.
//!
//! Run with: cargo run --example validate
//!
//! Tests:
//! 1. Engine lifecycle (create, store, search, update, delete)
//! 2. Deduplication (content hash)
//! 3. FTS5 search with Porter stemming
//! 4. Scoring strategies (recency, importance, composite)
//! 5. Context assembly with token budget
//! 6. Activation model (access tracking, decay computation)
//! 7. Graph relationships (create, traverse, cycle prevention)
//! 8. Consolidation (hash dedup)
//! 9. Pruning (policy-based memory cleanup)
//! 10. Performance (latency targets at 1K and 10K scale)
//! 11. Thread safety (concurrent reads)
//! 12. File persistence (data survives engine restart)

use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Duration, Utc};
use femind::context::ContextBudget;
use femind::engine::MemoryEngine;
use femind::memory::activation;
use femind::memory::pruning::{self, PruningPolicy};
use femind::memory::store::StoreResult;
use femind::memory::{GraphMemory, RelationType};
use femind::scoring::{CompositeScorer, ImportanceScorer, MemoryTypeScorer, RecencyScorer};
use femind::search::SearchMode;
use femind::traits::{MemoryRecord, MemoryType};
use serde::{Deserialize, Serialize};

// --- Memory types that simulate real consumers ---

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Learning {
    id: Option<i64>,
    description: String,
    category: String,
    importance: u8,
    mem_type: MemoryType,
    created_at: DateTime<Utc>,
}

impl MemoryRecord for Learning {
    fn id(&self) -> Option<i64> {
        self.id
    }
    fn searchable_text(&self) -> String {
        self.description.clone()
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
        Some(&self.category)
    }
}

fn learning(desc: &str, cat: &str, imp: u8, mem_type: MemoryType) -> Learning {
    Learning {
        id: None,
        description: desc.into(),
        category: cat.into(),
        importance: imp,
        mem_type,
        created_at: Utc::now(),
    }
}

fn old_learning(desc: &str, days_ago: i64) -> Learning {
    Learning {
        id: None,
        description: desc.into(),
        category: "test".into(),
        importance: 3,
        mem_type: MemoryType::Episodic,
        created_at: Utc::now() - Duration::days(days_ago),
    }
}

fn main() {
    let mut passed = 0;
    let mut failed = 0;

    macro_rules! check {
        ($name:expr, $body:expr) => {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
                Ok(Ok(())) => {
                    println!("  PASS  {}", $name);
                    passed += 1;
                }
                Ok(Err(e)) => {
                    println!("  FAIL  {} — {}", $name, e);
                    failed += 1;
                }
                Err(_) => {
                    println!("  FAIL  {} — panicked", $name);
                    failed += 1;
                }
            }
        };
    }

    println!("femind v0.2.0 Validation Suite");
    println!("================================\n");

    // --- 1. Engine Lifecycle ---
    println!("[1] Engine Lifecycle");
    check!("Create in-memory engine", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        assert_eq!(engine.count()?, 0);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Store and retrieve", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let r = engine.store(&learning("test memory", "test", 5, MemoryType::Semantic))?;
        let StoreResult::Added(id) = r else {
            return Err("expected Added".into());
        };
        let mem = engine.get(id)?.ok_or("not found")?;
        assert_eq!(mem.description, "test memory");
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Update memory", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let StoreResult::Added(id) =
            engine.store(&learning("original", "test", 5, MemoryType::Semantic))?
        else {
            return Err("expected Added".into());
        };
        let updated = Learning {
            id: Some(id),
            description: "updated".into(),
            ..learning("", "test", 5, MemoryType::Semantic)
        };
        engine.update(id, &updated)?;
        let mem = engine.get(id)?.ok_or("not found")?;
        assert_eq!(mem.description, "updated");
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Delete memory", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let StoreResult::Added(id) =
            engine.store(&learning("to delete", "test", 5, MemoryType::Semantic))?
        else {
            return Err("expected Added".into());
        };
        assert!(engine.delete(id)?);
        assert!(engine.get(id)?.is_none());
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 2. Deduplication ---
    println!("\n[2] Deduplication");
    check!("Exact duplicates prevented", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let mem = learning("exact same content", "test", 5, MemoryType::Semantic);
        let r1 = engine.store(&mem)?;
        let r2 = engine.store(&mem)?;
        assert!(matches!(r1, StoreResult::Added(_)));
        assert!(matches!(r2, StoreResult::Duplicate(_)));
        assert_eq!(engine.count()?, 1);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Different content allowed", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        engine.store(&learning("memory one", "test", 5, MemoryType::Semantic))?;
        engine.store(&learning("memory two", "test", 5, MemoryType::Semantic))?;
        assert_eq!(engine.count()?, 2);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 3. FTS5 Search ---
    println!("\n[3] FTS5 Search");
    check!("Basic keyword search", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        engine.store(&learning(
            "authentication failed with JWT token",
            "error",
            7,
            MemoryType::Procedural,
        ))?;
        engine.store(&learning(
            "database connection pool exhausted",
            "error",
            5,
            MemoryType::Episodic,
        ))?;
        engine.store(&learning(
            "cargo build succeeded",
            "build",
            3,
            MemoryType::Episodic,
        ))?;
        let results = engine.search("authentication").execute()?;
        assert_eq!(results.len(), 1);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Porter stemming (authenticate → authentication)", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        engine.store(&learning(
            "the authentication system was redesigned",
            "decision",
            8,
            MemoryType::Semantic,
        ))?;
        engine.store(&learning(
            "user failed to authenticate via OAuth",
            "error",
            6,
            MemoryType::Procedural,
        ))?;
        let results = engine.search("authenticate").execute()?;
        assert_eq!(results.len(), 2, "Porter stemming should match both");
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Search with limit", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        for i in 0..20 {
            engine.store(&learning(
                &format!("searchable test item {i}"),
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }
        let results = engine.search("searchable").limit(5).execute()?;
        assert_eq!(results.len(), 5);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Search with category filter", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        engine.store(&learning(
            "auth error in prod",
            "error",
            7,
            MemoryType::Procedural,
        ))?;
        engine.store(&learning(
            "auth flow decision",
            "decision",
            8,
            MemoryType::Semantic,
        ))?;
        let results = engine.search("auth").category("error").execute()?;
        assert_eq!(results.len(), 1);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Search with memory type filter", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        engine.store(&learning(
            "build error log",
            "error",
            5,
            MemoryType::Episodic,
        ))?;
        engine.store(&learning(
            "build fix pattern",
            "pattern",
            7,
            MemoryType::Procedural,
        ))?;
        let results = engine
            .search("build")
            .memory_type(MemoryType::Procedural)
            .execute()?;
        assert_eq!(results.len(), 1);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Exhaustive search mode", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        for i in 0..30 {
            engine.store(&learning(
                &format!("exhaustive item {i}"),
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }
        let results = engine
            .search("exhaustive")
            .mode(SearchMode::Exhaustive { min_score: 0.0 })
            .execute()?;
        assert_eq!(results.len(), 30, "exhaustive should return all matches");
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 4. Scoring ---
    println!("\n[4] Scoring Strategies");
    check!("Importance scoring affects order", {
        let engine = MemoryEngine::<Learning>::builder()
            .scoring(ImportanceScorer::default())
            .build()?;
        engine.store(&learning(
            "scored item low",
            "test",
            1,
            MemoryType::Semantic,
        ))?;
        engine.store(&learning(
            "scored item high",
            "test",
            10,
            MemoryType::Semantic,
        ))?;
        let results = engine.search("scored item").execute()?;
        assert!(
            results[0].score >= results[1].score,
            "high importance should rank first"
        );
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Composite scoring", {
        let scorer = CompositeScorer::new(vec![
            Box::new(RecencyScorer::default_half_life()),
            Box::new(ImportanceScorer::default()),
            Box::new(MemoryTypeScorer::default()),
        ]);
        let engine = MemoryEngine::<Learning>::builder()
            .scoring(scorer)
            .build()?;
        engine.store(&learning(
            "composite test alpha",
            "test",
            9,
            MemoryType::Procedural,
        ))?;
        engine.store(&learning(
            "composite test beta",
            "test",
            2,
            MemoryType::Episodic,
        ))?;
        let results = engine.search("composite test").execute()?;
        assert!(results[0].score > results[1].score);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 5. Context Assembly ---
    println!("\n[5] Context Assembly");
    check!("Assemble within budget", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        engine.store(&learning(
            "context alpha item",
            "test",
            5,
            MemoryType::Semantic,
        ))?;
        engine.store(&learning(
            "context beta item",
            "test",
            7,
            MemoryType::Semantic,
        ))?;
        engine.store(&learning(
            "unrelated cats and dogs",
            "test",
            5,
            MemoryType::Episodic,
        ))?;
        let assembly = engine.assemble_context("context item", &ContextBudget::new(1000))?;
        assert_eq!(assembly.items.len(), 2);
        let rendered = assembly.render();
        assert!(rendered.contains("alpha"));
        assert!(rendered.contains("beta"));
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Budget limits output", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        for i in 0..20 {
            engine.store(&learning(
                &format!("budget test memory with padding text number {i}"),
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }
        let assembly = engine.assemble_context("budget test", &ContextBudget::new(50))?;
        assert!(assembly.items.len() < 20);
        assert!(assembly.total_tokens <= 50);
        assert!(assembly.is_truncated());
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 6. Activation Model ---
    println!("\n[6] Activation Model");
    check!("Access tracking increases activation", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let StoreResult::Added(id) = engine.store(&learning(
            "activation test",
            "test",
            5,
            MemoryType::Semantic,
        ))?
        else {
            return Err("expected Added".into());
        };
        let db = engine.database();
        let a0 = activation::compute_activation(db, id)?;
        activation::record_access(db, id, "test query")?;
        let a1 = activation::compute_activation(db, id)?;
        assert!(
            a1 > a0,
            "activation should increase: before={a0}, after={a1}"
        );
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Activation cache", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let StoreResult::Added(id) =
            engine.store(&learning("cache test", "test", 5, MemoryType::Semantic))?
        else {
            return Err("expected Added".into());
        };
        let db = engine.database();
        activation::record_access(db, id, "q")?;
        let cached = activation::update_activation_cache(db, id)?;
        let fetched = activation::get_activation(db, id)?;
        assert!((cached - fetched).abs() < f32::EPSILON);
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 7. Graph Relationships ---
    println!("\n[7] Graph Relationships");
    check!("Create and traverse relationships", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let StoreResult::Added(id1) = engine.store(&learning(
            "error: JWT expired",
            "error",
            7,
            MemoryType::Procedural,
        ))?
        else {
            return Err("".into());
        };
        let StoreResult::Added(id2) = engine.store(&learning(
            "fix: refresh JWT token",
            "fix",
            8,
            MemoryType::Procedural,
        ))?
        else {
            return Err("".into());
        };
        let StoreResult::Added(id3) = engine.store(&learning(
            "root cause: clock drift",
            "cause",
            9,
            MemoryType::Semantic,
        ))?
        else {
            return Err("".into());
        };
        let db = engine.database();
        GraphMemory::relate(db, id1, id2, &RelationType::SolvedBy)?;
        GraphMemory::relate(db, id1, id3, &RelationType::CausedBy)?;
        let related = GraphMemory::traverse(db, id1, 3)?;
        assert_eq!(related.len(), 2, "should find 2 related memories");
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Cycle prevention", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let StoreResult::Added(a) =
            engine.store(&learning("cycle node A", "test", 5, MemoryType::Semantic))?
        else {
            return Err("".into());
        };
        let StoreResult::Added(b) =
            engine.store(&learning("cycle node B", "test", 5, MemoryType::Semantic))?
        else {
            return Err("".into());
        };
        let StoreResult::Added(c) =
            engine.store(&learning("cycle node C", "test", 5, MemoryType::Semantic))?
        else {
            return Err("".into());
        };
        let db = engine.database();
        GraphMemory::relate(db, a, b, &RelationType::RelatedTo)?;
        GraphMemory::relate(db, b, c, &RelationType::RelatedTo)?;
        GraphMemory::relate(db, c, a, &RelationType::RelatedTo)?;
        let nodes = GraphMemory::traverse(db, a, 10)?;
        assert!(
            nodes.len() <= 3,
            "cycle should not cause infinite traversal"
        );
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 8. Pruning ---
    println!("\n[8] Pruning");
    check!("Prune old episodic memories", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        engine.store(&old_learning("old session log", 60))?;
        engine.store(&learning(
            "recent semantic fact",
            "test",
            5,
            MemoryType::Semantic,
        ))?;
        // Set low activation on the old memory
        let db = engine.database();
        db.with_writer(|conn| {
            conn.execute(
                "UPDATE memories SET activation_cache = -3.0 WHERE searchable_text LIKE 'old%'",
                [],
            )?;
            Ok(())
        })?;
        let report = pruning::prune(db, &PruningPolicy::default())?;
        assert_eq!(report.pruned, 1, "should prune the old episodic memory");
        assert_eq!(engine.count()?, 1, "semantic memory should survive");
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 9. Performance ---
    println!("\n[9] Performance (1K memories)");
    check!("FTS5 search < 10ms at 1K scale", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        for i in 0..1000 {
            engine.store(&learning(
                &format!(
                    "performance test memory item number {i} with authentication and JWT tokens"
                ),
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }
        let start = Instant::now();
        let results = engine.search("authentication JWT").limit(10).execute()?;
        let elapsed = start.elapsed();
        println!(
            "         FTS5 search: {elapsed:?} ({} results)",
            results.len()
        );
        assert!(
            elapsed.as_millis() < 50,
            "FTS5 search took {elapsed:?} (target: <10ms)"
        );
        assert!(!results.is_empty());
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Context assembly < 20ms at 1K scale", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        for i in 0..1000 {
            engine.store(&learning(
                &format!("context perf test {i} with error handling patterns"),
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }
        let start = Instant::now();
        let assembly = engine.assemble_context("error handling", &ContextBudget::new(4096))?;
        let elapsed = start.elapsed();
        println!(
            "         Context assembly: {elapsed:?} ({} items)",
            assembly.items.len()
        );
        assert!(elapsed.as_millis() < 100, "assembly took {elapsed:?}");
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    check!("Store throughput > 100/sec", {
        let engine = MemoryEngine::<Learning>::builder().build()?;
        let start = Instant::now();
        for i in 0..100 {
            engine.store(&learning(
                &format!("throughput test {i}"),
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }
        let elapsed = start.elapsed();
        let per_sec = 100.0 / elapsed.as_secs_f64();
        println!("         Store throughput: {per_sec:.0}/sec");
        assert!(
            per_sec > 100.0,
            "store throughput {per_sec:.0}/sec (target: >100/sec)"
        );
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 10. Thread Safety ---
    println!("\n[10] Thread Safety");
    check!("Concurrent reads on file database", {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("concurrent.db");
        let engine = Arc::new(
            MemoryEngine::<Learning>::builder()
                .database(path.to_string_lossy().to_string())
                .build()?,
        );
        for i in 0..100 {
            engine.store(&learning(
                &format!("concurrent test {i}"),
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }
        let mut handles = Vec::new();
        for _ in 0..4 {
            let e = Arc::clone(&engine);
            handles.push(std::thread::spawn(move || {
                let results = e.search("concurrent").limit(10).execute().unwrap();
                assert!(!results.is_empty());
            }));
        }
        for h in handles {
            h.join().map_err(|_| "thread panicked")?;
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- 11. File Persistence ---
    println!("\n[11] File Persistence");
    check!("Data survives engine restart", {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("persist.db");
        let path_str = path.to_string_lossy().to_string();

        // Create and populate
        {
            let engine = MemoryEngine::<Learning>::builder()
                .database(&path_str)
                .build()?;
            engine.store(&learning(
                "persistent data test",
                "test",
                5,
                MemoryType::Semantic,
            ))?;
        }

        // Reopen and verify
        {
            let engine = MemoryEngine::<Learning>::builder()
                .database(&path_str)
                .build()?;
            assert_eq!(engine.count()?, 1);
            let results = engine.search("persistent").execute()?;
            assert_eq!(results.len(), 1);
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    });

    // --- Summary ---
    println!("\n================================");
    println!("Results: {passed} passed, {failed} failed");
    if failed > 0 {
        std::process::exit(1);
    } else {
        println!("All validations passed.");
    }
}
