#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::useless_format
)]

//! Integration tests for Phase 2: Scoring + Context Assembly

use chrono::{DateTime, Utc};
use femind::context::ContextBudget;
use femind::engine::MemoryEngine;
use femind::scoring::{CompositeScorer, ImportanceScorer, MemoryTypeScorer, RecencyScorer};
use femind::traits::{MemoryRecord, MemoryType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Mem {
    id: Option<i64>,
    text: String,
    importance: u8,
    mem_type: MemoryType,
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
}

fn mem(text: &str, importance: u8, mem_type: MemoryType) -> Mem {
    Mem {
        id: None,
        text: text.into(),
        importance,
        mem_type,
        created_at: Utc::now(),
    }
}

#[test]
fn scoring_affects_result_order() {
    let engine = MemoryEngine::<Mem>::builder()
        .scoring(ImportanceScorer::default())
        .build()
        .expect("build");

    // Store two memories with different importance
    engine
        .store(&mem(
            "search target low importance",
            1,
            MemoryType::Semantic,
        ))
        .expect("store");
    engine
        .store(&mem(
            "search target high importance",
            10,
            MemoryType::Semantic,
        ))
        .expect("store");

    let results = engine
        .search("search target")
        .limit(10)
        .execute()
        .expect("search");
    assert_eq!(results.len(), 2);

    // High importance should rank first after scoring
    let first = &results[0];
    let second = &results[1];
    assert!(
        first.score >= second.score,
        "high importance should score higher: first={}, second={}",
        first.score,
        second.score
    );
}

#[test]
fn composite_scorer_on_engine() {
    let scorer = CompositeScorer::new(vec![
        Box::new(RecencyScorer::default_half_life()),
        Box::new(ImportanceScorer::default()),
        Box::new(MemoryTypeScorer::default()),
    ]);

    let engine = MemoryEngine::<Mem>::builder()
        .scoring(scorer)
        .build()
        .expect("build");

    engine
        .store(&mem(
            "composite scoring test one",
            8,
            MemoryType::Procedural,
        ))
        .expect("store");
    engine
        .store(&mem("composite scoring test two", 3, MemoryType::Episodic))
        .expect("store");

    let results = engine
        .search("composite scoring")
        .limit(10)
        .execute()
        .expect("search");
    assert_eq!(results.len(), 2);

    // Procedural with high importance should beat Episodic with low importance
    // (MemoryTypeScorer: procedural=1.2, episodic=0.8; ImportanceScorer: 8 > 3)
    assert!(results[0].score > results[1].score);
}

#[test]
fn context_assembly_via_engine() {
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");

    engine
        .store(&mem("context assembly test alpha", 5, MemoryType::Semantic))
        .expect("store");
    engine
        .store(&mem("context assembly test beta", 7, MemoryType::Semantic))
        .expect("store");
    engine
        .store(&mem("unrelated memory about cats", 5, MemoryType::Episodic))
        .expect("store");

    let budget = ContextBudget::new(1000);
    let assembly = engine
        .assemble_context("context assembly", &budget)
        .expect("assemble");

    assert_eq!(assembly.items.len(), 2, "should find 2 matching memories");
    assert!(!assembly.is_truncated());

    let rendered = assembly.render();
    assert!(rendered.contains("alpha"));
    assert!(rendered.contains("beta"));
    assert!(!rendered.contains("cats"));
}

#[test]
fn context_assembly_respects_budget() {
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");

    // Store many memories
    for i in 0..20 {
        engine
            .store(&mem(
                &format!(
                    "budget test memory item number {i} with some extra text to take up tokens"
                ),
                5,
                MemoryType::Semantic,
            ))
            .expect("store");
    }

    // Small budget should exclude some
    let budget = ContextBudget::new(50);
    let assembly = engine
        .assemble_context("budget test", &budget)
        .expect("assemble");

    assert!(assembly.items.len() < 20, "budget should limit items");
    assert!(assembly.total_tokens <= 50);
    assert!(assembly.is_truncated());
}

#[test]
fn no_scorer_uses_raw_scores() {
    // Engine without explicit scorer should still work (uses CompositeScorer::empty())
    let engine = MemoryEngine::<Mem>::builder().build().expect("build");

    engine
        .store(&mem("raw score test item", 5, MemoryType::Semantic))
        .expect("store");

    let results = engine.search("raw score").execute().expect("search");
    assert_eq!(results.len(), 1);
    assert!(results[0].score > 0.0);
}
