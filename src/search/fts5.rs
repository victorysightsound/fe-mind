use rusqlite::params;

use crate::error::Result;
use crate::storage::Database;

/// Result from an FTS5 keyword search.
#[derive(Debug, Clone)]
pub struct FtsResult {
    /// Memory row ID.
    pub memory_id: i64,
    /// BM25 relevance score (lower = more relevant in SQLite FTS5).
    /// Negated so higher = better (consistent with other scoring).
    pub score: f32,
}

/// FTS5 full-text search with Porter stemming and BM25 ranking.
pub struct FtsSearch;

impl FtsSearch {
    /// Search memories by keyword query.
    ///
    /// Uses FTS5 `MATCH` with Porter stemming (configured at table creation).
    /// Returns results ranked by BM25 score, limited to `limit` results.
    ///
    /// The query is passed directly to FTS5 — consumers can use FTS5 syntax:
    /// - Simple terms: `"authentication error"`
    /// - Phrase: `"\"exact phrase\""`
    /// - Boolean: `"auth AND error"`, `"auth OR login"`
    /// - Prefix: `"auth*"`
    /// - Column filter: `"category:error"`
    pub fn search(
        db: &Database,
        query: &str,
        limit: usize,
        category_filter: Option<&str>,
        memory_type_filter: Option<&str>,
    ) -> Result<Vec<FtsResult>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        db.with_reader(|conn| {
            let mut results = Vec::new();

            // Build query with optional filters
            // BM25 returns negative scores (more negative = more relevant)
            // We negate to make higher = better
            let sql = if category_filter.is_some() || memory_type_filter.is_some() {
                "SELECT m.id, -rank AS score
                 FROM memories_fts fts
                 JOIN memories m ON m.id = fts.rowid
                 WHERE memories_fts MATCH ?1
                   AND (?2 IS NULL OR m.category = ?2)
                   AND (?3 IS NULL OR m.memory_type = ?3)
                 ORDER BY rank
                 LIMIT ?4"
            } else {
                "SELECT m.id, -rank AS score
                 FROM memories_fts fts
                 JOIN memories m ON m.id = fts.rowid
                 WHERE memories_fts MATCH ?1
                   AND (?2 IS NULL OR m.category = ?2)
                   AND (?3 IS NULL OR m.memory_type = ?3)
                 ORDER BY rank
                 LIMIT ?4"
            };

            let mut stmt = conn.prepare(sql)?;
            let rows = stmt.query_map(
                params![query, category_filter, memory_type_filter, limit as i64],
                |row| {
                    Ok(FtsResult {
                        memory_id: row.get(0)?,
                        score: row.get(1)?,
                    })
                },
            )?;

            for row in rows {
                results.push(row?);
            }

            Ok(results)
        })
    }

    /// Search with an over-fetch multiplier (for RRF merge).
    ///
    /// Returns `limit * multiplier` results to give RRF more candidates to work with.
    pub fn search_overfetch(
        db: &Database,
        query: &str,
        limit: usize,
        multiplier: usize,
    ) -> Result<Vec<FtsResult>> {
        Self::search(db, query, limit * multiplier, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStore;
    use crate::storage::migrations;
    use crate::traits::{MemoryRecord, MemoryType};
    use chrono::Utc;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestMem {
        id: Option<i64>,
        text: String,
        category: Option<String>,
        mem_type: String,
        created_at: chrono::DateTime<Utc>,
    }

    impl MemoryRecord for TestMem {
        fn id(&self) -> Option<i64> { self.id }
        fn searchable_text(&self) -> String { self.text.clone() }
        fn memory_type(&self) -> MemoryType {
            MemoryType::from_str(&self.mem_type).unwrap_or(MemoryType::Episodic)
        }
        fn created_at(&self) -> chrono::DateTime<Utc> { self.created_at }
        fn category(&self) -> Option<&str> { self.category.as_deref() }
    }

    fn setup() -> Database {
        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| { migrations::migrate(conn)?; Ok(()) }).expect("migrate failed");
        db
    }

    fn insert(db: &Database, text: &str, category: Option<&str>, mem_type: &str) {
        let store = MemoryStore::<TestMem>::new();
        let record = TestMem {
            id: None,
            text: text.to_string(),
            category: category.map(String::from),
            mem_type: mem_type.to_string(),
            created_at: Utc::now(),
        };
        store.store(db, &record).expect("store failed");
    }

    #[test]
    fn basic_keyword_search() {
        let db = setup();
        insert(&db, "authentication failed with JWT token", None, "procedural");
        insert(&db, "database connection timeout error", None, "episodic");
        insert(&db, "build succeeded after fixing imports", None, "episodic");

        let results = FtsSearch::search(&db, "authentication", 10, None, None)
            .expect("search failed");
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn porter_stemming() {
        let db = setup();
        insert(&db, "authentication failed", None, "semantic");
        insert(&db, "the user authenticated successfully", None, "semantic");

        // "authenticate" should match both via Porter stemming
        let results = FtsSearch::search(&db, "authenticate", 10, None, None)
            .expect("search failed");
        assert_eq!(results.len(), 2, "Porter stemming should match inflections");
    }

    #[test]
    fn empty_query() {
        let db = setup();
        insert(&db, "some memory", None, "semantic");

        let results = FtsSearch::search(&db, "", 10, None, None).expect("search failed");
        assert!(results.is_empty());

        let results = FtsSearch::search(&db, "   ", 10, None, None).expect("search failed");
        assert!(results.is_empty());
    }

    #[test]
    fn no_matches() {
        let db = setup();
        insert(&db, "authentication failed", None, "semantic");

        let results = FtsSearch::search(&db, "xyzzyplugh", 10, None, None)
            .expect("search failed");
        assert!(results.is_empty());
    }

    #[test]
    fn category_filter() {
        let db = setup();
        insert(&db, "auth error in login", Some("error"), "procedural");
        insert(&db, "auth flow redesign decision", Some("decision"), "semantic");

        let results = FtsSearch::search(&db, "auth", 10, Some("error"), None)
            .expect("search failed");
        assert_eq!(results.len(), 1);

        let results = FtsSearch::search(&db, "auth", 10, Some("decision"), None)
            .expect("search failed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn memory_type_filter() {
        let db = setup();
        insert(&db, "build failed with error", None, "episodic");
        insert(&db, "build failures are caused by deps", None, "semantic");

        let results = FtsSearch::search(&db, "build", 10, None, Some("episodic"))
            .expect("search failed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn limit_respected() {
        let db = setup();
        for i in 0..20 {
            insert(&db, &format!("memory about testing item {i}"), None, "semantic");
        }

        let results = FtsSearch::search(&db, "testing", 5, None, None)
            .expect("search failed");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn results_ranked_by_bm25() {
        let db = setup();
        // More relevant: term appears multiple times
        insert(&db, "error error error in authentication", None, "procedural");
        // Less relevant: term appears once
        insert(&db, "minor error in logging", None, "episodic");

        let results = FtsSearch::search(&db, "error", 10, None, None)
            .expect("search failed");
        assert_eq!(results.len(), 2);
        // First result should have higher score (more relevant)
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn overfetch() {
        let db = setup();
        for i in 0..20 {
            insert(&db, &format!("test memory number {i}"), None, "semantic");
        }

        let results = FtsSearch::search_overfetch(&db, "test", 5, 3)
            .expect("search failed");
        assert_eq!(results.len(), 15); // 5 * 3
    }
}
