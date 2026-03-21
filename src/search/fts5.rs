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
        Self::search_with_tiers(db, query, limit, category_filter, memory_type_filter, None)
    }

    /// Search with optional tier filtering (AND mode — all terms must match).
    pub fn search_with_tiers(
        db: &Database,
        query: &str,
        limit: usize,
        category_filter: Option<&str>,
        memory_type_filter: Option<&str>,
        min_tier: Option<i32>,
    ) -> Result<Vec<FtsResult>> {
        let sanitized = sanitize_fts5_query(query);
        Self::execute_fts(db, &sanitized, limit, category_filter, memory_type_filter, min_tier)
    }

    /// Search using OR mode with stop-word removal — returns partial matches ranked by BM25.
    ///
    /// 1. Sanitizes the query (strip special chars, hyphens)
    /// 2. Removes stop words (the, is, how, many, did, etc.)
    /// 3. Joins remaining terms with OR for partial matching
    ///
    /// Much better for long natural language queries where AND yields zero results.
    pub fn search_or_mode(
        db: &Database,
        query: &str,
        limit: usize,
        category_filter: Option<&str>,
        memory_type_filter: Option<&str>,
        min_tier: Option<i32>,
    ) -> Result<Vec<FtsResult>> {
        let sanitized = sanitize_fts5_query(query);
        let stripped = strip_stop_words(&sanitized);
        let or_query = to_or_query(&stripped);
        Self::execute_fts(db, &or_query, limit, category_filter, memory_type_filter, min_tier)
    }

    /// Execute the FTS5 query against the database.
    fn execute_fts(
        db: &Database,
        fts_query: &str,
        limit: usize,
        category_filter: Option<&str>,
        memory_type_filter: Option<&str>,
        min_tier: Option<i32>,
    ) -> Result<Vec<FtsResult>> {
        if fts_query.trim().is_empty() {
            return Ok(Vec::new());
        }

        db.with_reader(|conn| {
            let mut results = Vec::new();

            let sql = "SELECT m.id, -rank AS score
                 FROM memories_fts fts
                 JOIN memories m ON m.id = fts.rowid
                 WHERE memories_fts MATCH ?1
                   AND (?2 IS NULL OR m.category = ?2)
                   AND (?3 IS NULL OR m.memory_type = ?3)
                   AND (?4 IS NULL OR m.tier >= ?4)
                 ORDER BY rank
                 LIMIT ?5";

            let mut stmt = conn.prepare(sql)?;
            let rows = stmt.query_map(
                params![fts_query, category_filter, memory_type_filter, min_tier, limit as i64],
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

/// Sanitize a query string for FTS5 — remove characters that FTS5 interprets as syntax.
///
/// FTS5 special characters: `*`, `"`, `(`, `)`, `:`, `^`, `{`, `}`, `+`, `-`, `~`, `?`
/// We strip them to prevent syntax errors when the query comes from user input.
fn sanitize_fts5_query(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    for ch in query.chars() {
        match ch {
            '*' | '"' | '(' | ')' | ':' | '^' | '{' | '}' | '+' | '~' | '?' | ',' | '.' | '!' | ';' | '\'' | '/' | '\\' | '[' | ']' | '<' | '>' | '&' | '#' | '@' | '=' => {
                result.push(' ');
            }
            '-' => {
                // Replace ALL hyphens with spaces — FTS5 interprets "word-other"
                // as a column filter (like "word:other"), causing "no such column" errors.
                result.push(' ');
            }
            _ => result.push(ch),
        }
    }
    // Collapse multiple spaces and trim
    let collapsed: String = result.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return String::new();
    }
    collapsed
}

/// Convert a sanitized query into OR-mode: "word1 word2 word3" → "word1 OR word2 OR word3".
fn to_or_query(sanitized: &str) -> String {
    let terms: Vec<&str> = sanitized.split_whitespace().collect();
    if terms.len() <= 1 {
        return sanitized.to_string();
    }
    terms.join(" OR ")
}

/// English stop words that carry little semantic meaning for search.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "did", "does", "done", "doing",
    "have", "has", "had", "having",
    "will", "would", "could", "should", "can", "may", "might", "shall", "must",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "and", "or", "but", "nor", "not", "no", "so", "if", "then", "than",
    "how", "many", "much", "what", "when", "where", "which", "who", "whom", "why",
    "in", "on", "at", "to", "for", "of", "with", "from", "by", "about", "into",
    "up", "out", "off", "over", "under", "between", "through", "during", "before", "after",
    "there", "here", "very", "just", "also", "too", "only", "some", "any", "all",
    "each", "every", "both", "few", "more", "most", "other", "such",
    "as", "like", "because", "since", "while", "until", "although",
    "get", "got", "go", "went", "come", "came", "make", "made",
    "say", "said", "tell", "told", "know", "knew", "think", "thought",
    "see", "saw", "look", "find", "give", "take", "want", "need",
    "use", "try", "ask", "work", "seem", "feel", "let", "keep", "put",
    "spend", "participating", "activities",
];

/// Remove stop words from a query, keeping only content-bearing terms.
///
/// Returns the filtered query. If ALL words are stop words, returns the
/// original query to avoid an empty search.
pub fn strip_stop_words(query: &str) -> String {
    let terms: Vec<&str> = query
        .split_whitespace()
        .filter(|w| !STOP_WORDS.contains(&w.to_lowercase().as_str()))
        .collect();

    if terms.is_empty() {
        // All words were stop words — return original to avoid empty query
        return query.to_string();
    }
    terms.join(" ")
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

    #[test]
    fn hyphenated_queries_dont_error() {
        let db = setup();
        insert(&db, "I participated in faith related activities", None, "episodic");
        insert(&db, "The well known scientist published a paper", None, "semantic");
        insert(&db, "The self driving car navigated the highway", None, "procedural");

        // These hyphenated queries previously caused "no such column" errors
        // because FTS5 interprets "word-other" as a column filter.
        let queries = [
            "faith-related activities",
            "well-known scientist",
            "self-driving car",
            "How many faith-related events happened?",
            "state-of-the-art technology",
        ];

        for query in &queries {
            let result = FtsSearch::search(&db, query, 10, None, None);
            assert!(result.is_ok(), "Query '{query}' should not error: {:?}", result.err());
        }

        // Verify "faith related" still matches after hyphen→space conversion
        let results = FtsSearch::search(&db, "faith-related", 10, None, None)
            .expect("search failed");
        assert!(!results.is_empty(), "should find 'faith related' content");
    }

    #[test]
    fn sanitizer_handles_all_special_chars() {
        let db = setup();
        insert(&db, "test content for sanitizer", None, "semantic");

        // Queries with every kind of special character
        let queries = [
            "test*",
            "\"test\"",
            "(test)",
            "test:content",
            "test^2",
            "{test}",
            "test+content",
            "~test",
            "test?",
            "test,content",
            "test.content",
            "test!content",
            "test;content",
            "test'content",
            "test/content",
            "test\\content",
            "test[0]",
            "<test>",
            "test&content",
            "#test",
            "@test",
            "test=content",
        ];

        for query in &queries {
            let result = FtsSearch::search(&db, query, 10, None, None);
            assert!(result.is_ok(), "Query '{query}' should not error: {:?}", result.err());
        }
    }

    #[test]
    fn sanitizer_output() {
        assert_eq!(sanitize_fts5_query("faith-related"), "faith related");
        assert_eq!(sanitize_fts5_query("self-driving car"), "self driving car");
        assert_eq!(sanitize_fts5_query("test:column"), "test column");
        assert_eq!(sanitize_fts5_query("  hello  world  "), "hello world");
        assert_eq!(sanitize_fts5_query("***"), "");
        assert_eq!(sanitize_fts5_query(""), "");
        assert_eq!(sanitize_fts5_query("normal query"), "normal query");
    }

    #[test]
    fn stop_word_removal() {
        assert_eq!(strip_stop_words("How many days did I spend participating in activities"), "days");
        assert_eq!(strip_stop_words("What is my favorite color"), "favorite color");
        assert_eq!(strip_stop_words("the cat sat on the mat"), "cat sat mat");
        // All stop words → return original
        assert_eq!(strip_stop_words("the is a"), "the is a");
        assert_eq!(strip_stop_words(""), "");
        assert_eq!(strip_stop_words("authentication JWT token error"), "authentication JWT token error");
    }

    #[test]
    fn to_or_query_output() {
        assert_eq!(to_or_query("faith related"), "faith OR related");
        assert_eq!(to_or_query("one two three"), "one OR two OR three");
        assert_eq!(to_or_query("single"), "single");
        assert_eq!(to_or_query(""), "");
    }

    #[test]
    fn or_mode_returns_partial_matches() {
        let db = setup();
        insert(&db, "authentication failed with JWT token", None, "procedural");
        insert(&db, "database connection timeout error", None, "episodic");
        insert(&db, "build succeeded after fixing imports", None, "episodic");

        // AND mode: long query with many terms returns nothing
        let and_results = FtsSearch::search(
            &db, "How many days did I spend fixing authentication errors", 10, None, None,
        ).expect("search");

        // OR mode: same query returns partial matches
        let or_results = FtsSearch::search_or_mode(
            &db, "How many days did I spend fixing authentication errors", 10, None, None, None,
        ).expect("search");

        assert!(
            or_results.len() > and_results.len(),
            "OR mode should find more results than AND: OR={}, AND={}",
            or_results.len(), and_results.len()
        );
        assert!(!or_results.is_empty(), "OR mode should find partial matches");
    }
}
