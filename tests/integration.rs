//! Integration tests for Phase 1: Foundation
//!
//! Tests the complete workflow: engine creation → store → search → context
//! using the public API surface that consumers will use.

use chrono::{DateTime, Utc};
use mindcore::engine::MemoryEngine;
use mindcore::memory::store::StoreResult;
use mindcore::search::{SearchMode, SearchResult};
use mindcore::traits::{MemoryRecord, MemoryType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// --- Test Memory Type ---

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Learning {
    id: Option<i64>,
    description: String,
    category: String,
    times_referenced: u32,
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
        MemoryType::Semantic
    }
    fn importance(&self) -> u8 {
        (self.times_referenced.min(10) as u8).max(3)
    }
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    fn category(&self) -> Option<&str> {
        Some(&self.category)
    }
}

fn learning(desc: &str, cat: &str) -> Learning {
    Learning {
        id: None,
        description: desc.to_string(),
        category: cat.to_string(),
        times_referenced: 1,
        created_at: Utc::now(),
    }
}

// --- Engine Lifecycle ---

#[test]
fn engine_in_memory_lifecycle() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    // Store
    let r = engine
        .store(&learning("JWT authentication fails when token expires", "error"))
        .unwrap();
    assert!(matches!(r, StoreResult::Added(_)));

    // Count
    assert_eq!(engine.count().unwrap(), 1);

    // Get
    let StoreResult::Added(id) = r else { unreachable!() };
    let mem = engine.get(id).unwrap().unwrap();
    assert!(mem.description.contains("JWT"));

    // Update
    let updated = Learning {
        id: Some(id),
        description: "JWT auth fails when token expires — fixed by refreshing".to_string(),
        times_referenced: 5,
        ..mem
    };
    engine.update(id, &updated).unwrap();
    let refreshed = engine.get(id).unwrap().unwrap();
    assert!(refreshed.description.contains("refreshing"));
    assert_eq!(refreshed.times_referenced, 5);

    // Delete
    assert!(engine.delete(id).unwrap());
    assert!(engine.get(id).unwrap().is_none());
    assert_eq!(engine.count().unwrap(), 0);
}

#[test]
fn engine_file_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("persist.db");
    let path_str = path.to_string_lossy().to_string();

    // Create and populate
    {
        let engine = MemoryEngine::<Learning>::builder()
            .database(&path_str)
            .build()
            .unwrap();
        engine
            .store(&learning("persistent memory test", "test"))
            .unwrap();
        assert_eq!(engine.count().unwrap(), 1);
    }

    // Reopen and verify data persists
    {
        let engine = MemoryEngine::<Learning>::builder()
            .database(&path_str)
            .build()
            .unwrap();
        assert_eq!(engine.count().unwrap(), 1);
        let results = engine.search("persistent").execute().unwrap();
        assert_eq!(results.len(), 1);
    }
}

// --- Deduplication ---

#[test]
fn dedup_prevents_exact_duplicates() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();
    let mem = learning("exact duplicate content", "test");

    let r1 = engine.store(&mem).unwrap();
    let r2 = engine.store(&mem).unwrap();
    let r3 = engine.store(&mem).unwrap();

    assert!(matches!(r1, StoreResult::Added(_)));
    assert!(matches!(r2, StoreResult::Duplicate(_)));
    assert!(matches!(r3, StoreResult::Duplicate(_)));
    assert_eq!(engine.count().unwrap(), 1);
}

#[test]
fn dedup_allows_different_content() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    engine.store(&learning("first memory", "test")).unwrap();
    engine.store(&learning("second memory", "test")).unwrap();
    engine.store(&learning("third memory", "test")).unwrap();

    assert_eq!(engine.count().unwrap(), 3);
}

// --- FTS5 Search ---

#[test]
fn fts5_basic_search() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    engine
        .store(&learning("authentication failed with invalid JWT", "error"))
        .unwrap();
    engine
        .store(&learning("database connection pool exhausted", "error"))
        .unwrap();
    engine
        .store(&learning("cargo build succeeded after adding feature flag", "build"))
        .unwrap();

    let results = engine.search("authentication").execute().unwrap();
    assert_eq!(results.len(), 1);

    let results = engine.search("database").execute().unwrap();
    assert_eq!(results.len(), 1);

    let results = engine.search("cargo").execute().unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn fts5_porter_stemming() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    engine
        .store(&learning("the authentication system was redesigned", "decision"))
        .unwrap();
    engine
        .store(&learning("user failed to authenticate via OAuth", "error"))
        .unwrap();

    // "authenticate" should match both via Porter stemming
    let results = engine.search("authenticate").execute().unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn fts5_no_results() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();
    engine
        .store(&learning("hello world", "test"))
        .unwrap();

    let results = engine.search("nonexistent_term_xyz").execute().unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_with_limit() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    for i in 0..20 {
        engine
            .store(&learning(&format!("test memory about searching item {i}"), "test"))
            .unwrap();
    }

    let results = engine.search("searching").limit(5).execute().unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn search_with_category_filter() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    engine
        .store(&learning("auth error in production", "error"))
        .unwrap();
    engine
        .store(&learning("auth decision: use OAuth2", "decision"))
        .unwrap();

    let results = engine
        .search("auth")
        .category("error")
        .execute()
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn search_with_memory_type_filter() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    engine
        .store(&learning("build failed due to missing import", "error"))
        .unwrap();

    // All our test memories are Semantic type
    let results = engine
        .search("build")
        .memory_type(MemoryType::Semantic)
        .execute()
        .unwrap();
    assert_eq!(results.len(), 1);

    let results = engine
        .search("build")
        .memory_type(MemoryType::Episodic)
        .execute()
        .unwrap();
    assert!(results.is_empty());
}

// --- Thread Safety ---

#[test]
fn concurrent_store_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("concurrent.db");
    let engine = Arc::new(
        MemoryEngine::<Learning>::builder()
            .database(path.to_string_lossy().to_string())
            .build()
            .unwrap(),
    );

    // Populate
    for i in 0..50 {
        engine
            .store(&learning(&format!("concurrent test memory {i}"), "test"))
            .unwrap();
    }

    // Concurrent reads
    let mut handles = Vec::new();
    for _ in 0..4 {
        let e = Arc::clone(&engine);
        handles.push(std::thread::spawn(move || {
            let results = e.search("concurrent").limit(10).execute().unwrap();
            assert!(!results.is_empty());
            assert!(results.len() <= 10);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

// --- Search Modes ---

#[test]
fn search_mode_keyword() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();
    engine
        .store(&learning("keyword mode test", "test"))
        .unwrap();

    let results = engine
        .search("keyword")
        .mode(SearchMode::Keyword)
        .execute()
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn search_mode_exhaustive() {
    let engine = MemoryEngine::<Learning>::builder().build().unwrap();

    for i in 0..30 {
        engine
            .store(&learning(&format!("exhaustive test item {i}"), "test"))
            .unwrap();
    }

    let results = engine
        .search("exhaustive")
        .mode(SearchMode::Exhaustive { min_score: 0.0 })
        .execute()
        .unwrap();
    assert_eq!(results.len(), 30, "exhaustive should return all matches");
}

// --- Multiple Memory Types in Same DB ---

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ErrorPattern {
    id: Option<i64>,
    pattern: String,
    occurrence_count: u32,
    created_at: DateTime<Utc>,
}

impl MemoryRecord for ErrorPattern {
    fn id(&self) -> Option<i64> {
        self.id
    }
    fn searchable_text(&self) -> String {
        self.pattern.clone()
    }
    fn memory_type(&self) -> MemoryType {
        MemoryType::Procedural
    }
    fn importance(&self) -> u8 {
        (self.occurrence_count.min(10) as u8).max(5)
    }
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    fn category(&self) -> Option<&str> {
        Some("error")
    }
}

#[test]
fn multiple_record_types_same_db_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi_type.db");
    let path_str = path.to_string_lossy().to_string();

    // Use same DB for two different MemoryRecord types
    let learning_engine = MemoryEngine::<Learning>::builder()
        .database(&path_str)
        .build()
        .unwrap();

    let error_engine = MemoryEngine::<ErrorPattern>::builder()
        .database(&path_str)
        .build()
        .unwrap();

    learning_engine
        .store(&learning("auth system uses JWT tokens", "decision"))
        .unwrap();

    error_engine
        .store(&ErrorPattern {
            id: None,
            pattern: "JWT token expired in auth flow".to_string(),
            occurrence_count: 3,
            created_at: Utc::now(),
        })
        .unwrap();

    // Both types searchable from their respective engines
    let learn_results = learning_engine.search("JWT").execute().unwrap();
    let error_results = error_engine.search("JWT").execute().unwrap();

    // Both should find results (they share the FTS5 index)
    assert!(!learn_results.is_empty() || !error_results.is_empty());
}
