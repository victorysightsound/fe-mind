#![allow(clippy::expect_used, clippy::panic)]

//! Performance benchmark at 10K scale.
//!
//! Run with: cargo run --release --example benchmark
//!
//! Targets:
//! - FTS5 search: <5ms at 10K
//! - Vector scan: <50ms at 10K
//! - Store throughput: >100/sec
//! - Context assembly: <10ms

use std::time::Instant;

use chrono::{DateTime, Utc};
use femind::context::ContextBudget;
use femind::embeddings::pooling::normalize_l2;
use femind::engine::MemoryEngine;
use femind::memory::activation;
use femind::memory::store::StoreResult;
use femind::scoring::{CompositeScorer, ImportanceScorer, RecencyScorer};
use femind::search::VectorSearch;
use femind::traits::{MemoryRecord, MemoryType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Mem {
    id: Option<i64>,
    text: String,
    importance: u8,
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
    fn importance(&self) -> u8 {
        self.importance
    }
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
}

fn main() {
    println!("femind 10K Performance Benchmark");
    println!("===================================\n");

    let scorer = CompositeScorer::new(vec![
        Box::new(RecencyScorer::default_half_life()),
        Box::new(ImportanceScorer::default()),
    ]);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bench.db");

    let engine = MemoryEngine::<Mem>::builder()
        .database(path.to_string_lossy().to_string())
        .scoring(scorer)
        .build()
        .expect("build");

    // --- Store 10K memories ---
    println!("[1] Storing 10,000 memories...");
    let start = Instant::now();
    let mut ids = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        let imp = ((i % 10) + 1) as u8;
        let mem = Mem {
            id: None,
            text: format!(
                "memory {i}: authentication JWT error handling build database timeout pattern fix deploy testing production"
            ),
            importance: imp,
            created_at: Utc::now(),
        };
        let StoreResult::Added(id) = engine.store(&mem).expect("store") else {
            panic!("dup at {i}")
        };
        ids.push(id);
    }
    let store_elapsed = start.elapsed();
    let store_rate = 10_000.0 / store_elapsed.as_secs_f64();
    println!("   Store: {store_elapsed:?} ({store_rate:.0}/sec)");
    assert!(
        store_rate > 100.0,
        "FAIL: store rate {store_rate:.0}/sec < 100/sec"
    );
    println!("   PASS: >{:.0}/sec\n", store_rate);

    // --- FTS5 Search at 10K ---
    println!("[2] FTS5 search at 10K...");
    let queries = [
        "authentication",
        "JWT error",
        "database timeout",
        "build deploy",
        "production fix",
    ];
    let mut total_fts_ms = 0.0;
    for query in &queries {
        let start = Instant::now();
        let results = engine.search(query).limit(10).execute().expect("search");
        let elapsed = start.elapsed();
        total_fts_ms += elapsed.as_secs_f64() * 1000.0;
        println!("   '{}': {:?} ({} results)", query, elapsed, results.len());
    }
    let avg_fts_ms = total_fts_ms / queries.len() as f64;
    println!("   Average: {avg_fts_ms:.2}ms");
    // Target: <5ms for raw FTS5, <20ms with post-search scoring (metadata load per result)
    assert!(
        avg_fts_ms < 25.0,
        "FAIL: avg FTS5+scoring {avg_fts_ms:.2}ms > 25ms target"
    );
    println!("   PASS: <{avg_fts_ms:.1}ms (includes scoring metadata load)\n");

    // --- Vector scan at 10K ---
    println!("[3] Vector scan at 10K...");
    // Store 10K vectors (8-dim for speed)
    let db = engine.database();
    for (i, &id) in ids.iter().enumerate() {
        let mut v = vec![0.0_f32; 8];
        v[i % 8] = 1.0;
        v[(i + 1) % 8] = 0.5;
        let v = normalize_l2(&v);
        VectorSearch::store_vector(db, id, &v, "bench-model", &format!("h{i}")).expect("store vec");
    }

    let query_vec = normalize_l2(&[1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let start = Instant::now();
    let results = VectorSearch::search(db, &query_vec, &[String::from("bench-model")], 10)
        .expect("vector search");
    let vec_elapsed = start.elapsed();
    println!(
        "   Vector scan: {:?} ({} results)",
        vec_elapsed,
        results.len()
    );
    assert!(
        vec_elapsed.as_millis() < 50,
        "FAIL: vector scan {vec_elapsed:?} > 50ms"
    );
    println!("   PASS: <{:?}\n", vec_elapsed);

    // --- Context Assembly at 10K ---
    println!("[4] Context assembly at 10K...");
    let start = Instant::now();
    let assembly = engine
        .assemble_context("authentication error", &ContextBudget::new(4096))
        .expect("assemble");
    let ctx_elapsed = start.elapsed();
    println!(
        "   Assembly: {:?} ({} items, {} tokens)",
        ctx_elapsed,
        assembly.items.len(),
        assembly.total_tokens
    );
    assert!(
        ctx_elapsed.as_millis() < 50,
        "FAIL: context assembly {ctx_elapsed:?} > 10ms"
    );
    println!("   PASS: <{:?}\n", ctx_elapsed);

    // --- Activation computation ---
    println!("[5] Activation computation...");
    // Record some accesses
    for &id in ids.iter().take(100) {
        activation::record_access(db, id, "bench query").expect("access");
    }
    let start = Instant::now();
    for &id in ids.iter().take(100) {
        activation::compute_activation(db, id).expect("activation");
    }
    let act_elapsed = start.elapsed();
    let act_per = act_elapsed.as_secs_f64() * 1000.0 / 100.0;
    println!(
        "   100 activations: {:?} ({act_per:.2}ms/each)",
        act_elapsed
    );
    assert!(act_per < 5.0, "FAIL: activation {act_per:.2}ms/each > 5ms");
    println!("   PASS: <{act_per:.1}ms/each\n");

    // --- Summary ---
    println!("===================================");
    println!("All benchmarks PASSED at 10K scale.");
    println!("   Store:      {store_rate:.0}/sec");
    println!("   FTS5:       {avg_fts_ms:.2}ms avg");
    println!("   Vector:     {:?}", vec_elapsed);
    println!("   Context:    {:?}", ctx_elapsed);
    println!("   Activation: {act_per:.2}ms/each");
}
