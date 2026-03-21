use std::marker::PhantomData;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::embeddings::EmbeddingBackend;
use crate::error::Result;
use crate::search::fts5::{FtsResult, FtsSearch};
use crate::search::hybrid::rrf_merge;
use crate::search::vector::VectorSearch;
use crate::storage::Database;
use crate::traits::{MemoryMeta, MemoryRecord, MemoryType, ScoringStrategy};

/// Search mode determines which retrieval strategies are used.
#[derive(Debug, Clone)]
pub enum SearchMode {
    /// FTS5 keyword search only (always available).
    Keyword,
    /// Vector similarity search only (requires vector-search feature).
    Vector,
    /// Hybrid: FTS5 + Vector merged via RRF (requires vector-search feature).
    Hybrid,
    /// Auto-detect: Hybrid if vector available, Keyword otherwise.
    Auto,
    /// Return all matches above threshold (for aggregation queries).
    /// Bypasses top-k limits.
    Exhaustive {
        /// Minimum score threshold for inclusion.
        min_score: f32,
    },
}

/// Controls which memory tiers are searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchDepth {
    /// Search summaries and facts only — tiers 1+2 (fastest).
    Standard,
    /// Also search raw episodes if summary results are sparse.
    /// Default until tier-based consolidation is active.
    #[default]
    Deep,
    /// Search all tiers (slowest, most complete, for forensic/audit).
    Forensic,
}

/// A scored search result containing the memory ID and relevance score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Memory row ID.
    pub memory_id: i64,
    /// Combined relevance score (higher = more relevant).
    pub score: f32,
}

/// Fluent builder for constructing and executing memory searches.
///
/// # Example
///
/// ```rust,ignore
/// let results = engine.search("authentication error")
///     .mode(SearchMode::Auto)
///     .limit(10)
///     .category("error")
///     .execute()?;
/// ```
pub struct SearchBuilder<'a, T: MemoryRecord> {
    db: &'a Database,
    query: String,
    mode: SearchMode,
    depth: SearchDepth,
    limit: usize,
    category: Option<String>,
    memory_type: Option<MemoryType>,
    tier: Option<u8>,
    min_score: Option<f32>,
    valid_at: Option<DateTime<Utc>>,
    scoring: Option<Arc<dyn ScoringStrategy>>,
    embedding: Option<Arc<dyn EmbeddingBackend>>,
    _phantom: PhantomData<T>,
}

impl<'a, T: MemoryRecord> SearchBuilder<'a, T> {
    /// Create a new search builder.
    pub fn new(db: &'a Database, query: impl Into<String>) -> Self {
        Self {
            db,
            query: query.into(),
            mode: SearchMode::Auto,
            depth: SearchDepth::default(),
            limit: 10,
            category: None,
            memory_type: None,
            tier: None,
            min_score: None,
            valid_at: None,
            scoring: None,
            embedding: None,
            _phantom: PhantomData,
        }
    }

    /// Attach a scoring strategy (called by MemoryEngine).
    pub fn with_scoring(mut self, scoring: Arc<dyn ScoringStrategy>) -> Self {
        self.scoring = Some(scoring);
        self
    }

    /// Attach an embedding backend for vector search (called by MemoryEngine).
    pub fn with_embedding(mut self, embedding: Arc<dyn EmbeddingBackend>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the search mode.
    pub fn mode(mut self, mode: SearchMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the search depth (which tiers to search).
    pub fn depth(mut self, depth: SearchDepth) -> Self {
        self.depth = depth;
        self
    }

    /// Set the maximum number of results to return.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// Filter by category.
    pub fn category(mut self, cat: impl Into<String>) -> Self {
        self.category = Some(cat.into());
        self
    }

    /// Filter by memory type.
    pub fn memory_type(mut self, t: MemoryType) -> Self {
        self.memory_type = Some(t);
        self
    }

    /// Filter by tier (0=episode, 1=summary, 2=fact).
    pub fn tier(mut self, tier: u8) -> Self {
        self.tier = Some(tier);
        self
    }

    /// Set minimum score threshold.
    pub fn min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Filter to memories valid at the specified time.
    ///
    /// Only returns memories where:
    /// - `valid_from` is NULL or <= the specified time, AND
    /// - `valid_until` is NULL or > the specified time
    pub fn valid_at(mut self, time: DateTime<Utc>) -> Self {
        self.valid_at = Some(time);
        self
    }

    /// Execute the search and return scored results.
    ///
    /// Synchronous — uses pre-computed embeddings from the background indexer
    /// for vector search, not inline inference.
    pub fn execute(self) -> Result<Vec<SearchResult>> {
        match &self.mode {
            SearchMode::Keyword => self.execute_keyword(),
            SearchMode::Vector => self.execute_vector(),
            SearchMode::Hybrid => self.execute_hybrid(),
            SearchMode::Auto => {
                if self.embedding.is_some() {
                    self.execute_hybrid()
                } else {
                    self.execute_keyword()
                }
            }
            SearchMode::Exhaustive { min_score } => {
                let threshold = *min_score;
                self.execute_exhaustive(threshold)
            }
        }
    }

    /// Execute keyword-only search via FTS5.
    fn execute_keyword(&self) -> Result<Vec<SearchResult>> {
        let category_filter = self.category.as_deref();
        let type_filter = self.memory_type.map(|t| t.as_str());
        let min_tier = self.depth_to_min_tier();

        let fts_results = FtsSearch::search_with_tiers(
            self.db,
            &self.query,
            self.limit,
            category_filter,
            type_filter,
            min_tier,
        )?;

        let mut results = self.apply_filters(fts_results);

        // Apply min_score filter
        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }

        results.truncate(self.limit);
        Ok(results)
    }

    /// Execute exhaustive search — return all matches above threshold.
    fn execute_exhaustive(&self, min_score: f32) -> Result<Vec<SearchResult>> {
        let category_filter = self.category.as_deref();
        let type_filter = self.memory_type.map(|t| t.as_str());
        let min_tier = self.depth_to_min_tier();

        let fts_results = FtsSearch::search_with_tiers(
            self.db,
            &self.query,
            10_000,
            category_filter,
            type_filter,
            min_tier,
        )?;

        let mut results = self.apply_filters(fts_results);
        results.retain(|r| r.score >= min_score);
        Ok(results)
    }

    /// Execute vector-only search.
    fn execute_vector(&self) -> Result<Vec<SearchResult>> {
        let Some(ref embedding) = self.embedding else {
            // No embedding backend — fall back to keyword
            return self.execute_keyword();
        };

        if !embedding.is_available() {
            return self.execute_keyword();
        }

        let query_vec = embedding.embed(&self.query)?;
        let model = embedding.model_name();

        let vector_results = VectorSearch::search(
            self.db,
            &query_vec,
            model,
            self.limit * 3,
        )?;

        let mut results = self.apply_filters(vector_results);
        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }
        results.truncate(self.limit);
        Ok(results)
    }

    /// Execute hybrid search: FTS5 + vector merged via RRF.
    fn execute_hybrid(&self) -> Result<Vec<SearchResult>> {
        let Some(ref embedding) = self.embedding else {
            return self.execute_keyword();
        };

        if !embedding.is_available() {
            return self.execute_keyword();
        }

        let category_filter = self.category.as_deref();
        let type_filter = self.memory_type.map(|t| t.as_str());
        let min_tier = self.depth_to_min_tier();

        // FTS5 keyword search (OR mode + stop-word removal for better recall, over-fetch 3x for RRF)
        let fts_results = FtsSearch::search_or_mode(
            self.db,
            &self.query,
            self.limit * 3,
            category_filter,
            type_filter,
            min_tier,
        )?;

        // Vector similarity search
        let query_vec = embedding.embed(&self.query)?;
        let model = embedding.model_name();
        let vector_results = VectorSearch::search(
            self.db,
            &query_vec,
            model,
            self.limit * 3,
        )?;

        // Merge via RRF
        let merged = rrf_merge(&fts_results, &vector_results, &self.query, self.limit * 2);

        let mut results = self.apply_filters(merged);

        // Lightweight reranking: boost results with more query content words, penalize short memories
        rerank_results(self.db, &mut results, &self.query);

        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }
        results.truncate(self.limit);
        Ok(results)
    }

    /// Convert search depth to minimum tier filter.
    fn depth_to_min_tier(&self) -> Option<i32> {
        match self.depth {
            SearchDepth::Standard => Some(1), // Tiers 1+2 (summaries and facts)
            SearchDepth::Deep => Some(0),     // All tiers including raw episodes
            SearchDepth::Forensic => None,    // No filter (same as Deep, but conceptually includes archived)
        }
    }

    /// Apply scoring and filters to FTS results.
    fn apply_filters(&self, fts_results: Vec<FtsResult>) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = fts_results
            .into_iter()
            .map(|r| SearchResult {
                memory_id: r.memory_id,
                score: r.score,
            })
            .collect();

        // Apply post-search scoring if a strategy is configured
        if let Some(ref scoring) = self.scoring {
            self.apply_scoring(&mut results, scoring);
        }

        // Re-sort by final score (descending)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Apply scoring strategy to results by loading memory metadata.
    fn apply_scoring(&self, results: &mut [SearchResult], scoring: &Arc<dyn ScoringStrategy>) {
        for result in results.iter_mut() {
            // Load metadata for scoring
            let meta = self.db.with_reader(|conn| {
                let row = conn.query_row(
                    "SELECT searchable_text, memory_type, importance, category, created_at
                     FROM memories WHERE id = ?1",
                    [result.memory_id],
                    |row| {
                        Ok(MemoryMeta {
                            id: Some(result.memory_id),
                            searchable_text: row.get(0)?,
                            memory_type: crate::traits::MemoryType::from_str(
                                &row.get::<_, String>(1)?
                            ).unwrap_or(crate::traits::MemoryType::Episodic),
                            importance: row.get::<_, i32>(2)? as u8,
                            category: row.get(3)?,
                            created_at: chrono::DateTime::parse_from_rfc3339(
                                &row.get::<_, String>(4)?
                            )
                            .map(|dt| dt.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                            metadata: std::collections::HashMap::new(),
                        })
                    },
                );
                match row {
                    Ok(meta) => Ok(Some(meta)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(e.into()),
                }
            });

            if let Ok(Some(meta)) = meta {
                let multiplier = scoring.score_multiplier(&meta, &self.query, result.score);
                result.score *= multiplier;
            }
        }
    }
}

/// Lightweight reranking pass after RRF merge.
///
/// Adjusts scores based on:
/// 1. Query content word overlap — boost memories containing more query terms
/// 2. Length penalty — penalize very short memories (<20 chars) that are likely noise
fn rerank_results(db: &Database, results: &mut [SearchResult], query: &str) {
    use crate::search::fts5::strip_stop_words;

    let query_words: Vec<String> = strip_stop_words(query)
        .to_lowercase()
        .split_whitespace()
        .map(String::from)
        .collect();

    if query_words.is_empty() {
        return;
    }

    for result in results.iter_mut() {
        // Load memory text
        let text = db.with_reader(|conn| {
            conn.query_row(
                "SELECT searchable_text FROM memories WHERE id = ?1",
                [result.memory_id],
                |row| row.get::<_, String>(0),
            ).map_err(|e| crate::error::MindCoreError::Database(e))
        });

        let Ok(text) = text else { continue };
        let text_lower = text.to_lowercase();

        // Boost by query word overlap ratio (0.0 to 1.0)
        let matches = query_words.iter()
            .filter(|w| text_lower.contains(w.as_str()))
            .count();
        let overlap = matches as f32 / query_words.len() as f32;
        result.score *= 1.0 + overlap * 0.5; // up to 50% boost

        // Penalize very short memories
        if text.len() < 20 {
            result.score *= 0.3;
        } else if text.len() < 50 {
            result.score *= 0.7;
        }
    }

    // Re-sort after reranking
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryStore;
    use crate::storage::migrations;
    use chrono::Utc;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestMem {
        id: Option<i64>,
        text: String,
        category: Option<String>,
        created_at: chrono::DateTime<Utc>,
    }

    impl MemoryRecord for TestMem {
        fn id(&self) -> Option<i64> { self.id }
        fn searchable_text(&self) -> String { self.text.clone() }
        fn memory_type(&self) -> MemoryType { MemoryType::Semantic }
        fn created_at(&self) -> chrono::DateTime<Utc> { self.created_at }
        fn category(&self) -> Option<&str> { self.category.as_deref() }
    }

    fn setup() -> Database {
        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| { migrations::migrate(conn)?; Ok(()) }).expect("migrate");
        let store = MemoryStore::<TestMem>::new();
        for text in [
            "authentication failed with JWT token",
            "database connection timeout",
            "build succeeded after fixing imports",
            "authentication flow redesigned",
        ] {
            store.store(&db, &TestMem {
                id: None,
                text: text.to_string(),
                category: None,
                created_at: Utc::now(),
            }).expect("store");
        }
        db
    }

    #[test]
    fn builder_basic_search() {
        let db = setup();
        let results = SearchBuilder::<TestMem>::new(&db, "authentication")
            .execute()
            .expect("search failed");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn builder_with_limit() {
        let db = setup();
        let results = SearchBuilder::<TestMem>::new(&db, "authentication")
            .limit(1)
            .execute()
            .expect("search failed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn builder_keyword_mode() {
        let db = setup();
        let results = SearchBuilder::<TestMem>::new(&db, "database")
            .mode(SearchMode::Keyword)
            .execute()
            .expect("search failed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn builder_empty_query() {
        let db = setup();
        let results = SearchBuilder::<TestMem>::new(&db, "")
            .execute()
            .expect("search failed");
        assert!(results.is_empty());
    }

    #[test]
    fn builder_no_matches() {
        let db = setup();
        let results = SearchBuilder::<TestMem>::new(&db, "xyznonexistent")
            .execute()
            .expect("search failed");
        assert!(results.is_empty());
    }

    #[test]
    fn builder_min_score() {
        let db = setup();
        let results = SearchBuilder::<TestMem>::new(&db, "authentication")
            .min_score(999.0)
            .execute()
            .expect("search failed");
        assert!(results.is_empty(), "no results should pass a very high min_score");
    }

    #[test]
    fn builder_exhaustive_mode() {
        let db = setup();
        let results = SearchBuilder::<TestMem>::new(&db, "authentication")
            .mode(SearchMode::Exhaustive { min_score: 0.0 })
            .execute()
            .expect("search failed");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn builder_chaining() {
        let db = setup();
        // Test that all builder methods chain properly
        let results = SearchBuilder::<TestMem>::new(&db, "build")
            .mode(SearchMode::Keyword)
            .depth(SearchDepth::Forensic)
            .limit(5)
            .min_score(0.0)
            .execute()
            .expect("search failed");
        assert!(!results.is_empty());
    }
}
