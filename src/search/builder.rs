use std::marker::PhantomData;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::embeddings::EmbeddingBackend;
use crate::engine::VectorSearchMode;
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
    vector_search_mode: VectorSearchMode,
    #[cfg(feature = "ann")]
    ann_index: Option<Arc<crate::search::AnnIndex>>,
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
            vector_search_mode: VectorSearchMode::default(),
            #[cfg(feature = "ann")]
            ann_index: None,
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

    /// Attach the engine's vector search mode.
    pub fn with_vector_search_mode(mut self, mode: VectorSearchMode) -> Self {
        self.vector_search_mode = mode;
        self
    }

    /// Attach the engine's shared ANN index.
    #[cfg(feature = "ann")]
    pub fn with_ann_index(mut self, ann_index: Arc<crate::search::AnnIndex>) -> Self {
        self.ann_index = Some(ann_index);
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
            SearchMode::Auto => self.execute_auto(),
            SearchMode::Exhaustive { min_score } => {
                let threshold = *min_score;
                self.execute_exhaustive(threshold)
            }
        }
    }

    fn execute_auto(&self) -> Result<Vec<SearchResult>> {
        if self.vector_search_mode == VectorSearchMode::Off || self.embedding.is_none() {
            self.execute_keyword()
        } else {
            self.execute_hybrid()
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
        if self.vector_search_mode == VectorSearchMode::Off {
            return self.execute_keyword();
        }

        let Some(ref embedding) = self.embedding else {
            // No embedding backend — fall back to keyword
            return self.execute_keyword();
        };

        if !embedding.is_available() {
            return self.execute_keyword();
        }

        let query_vec = embedding.embed_query(&self.query)?;
        let model = embedding.model_name();
        let vector_results = self.vector_results(&query_vec, model, self.limit * 3)?;

        let mut results = self.apply_filters(vector_results);
        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }
        results.truncate(self.limit);
        Ok(results)
    }

    /// Execute hybrid search: FTS5 + vector merged via RRF.
    fn execute_hybrid(&self) -> Result<Vec<SearchResult>> {
        if self.vector_search_mode == VectorSearchMode::Off {
            return self.execute_keyword();
        }

        let Some(ref embedding) = self.embedding else {
            return self.execute_keyword();
        };

        if !embedding.is_available() {
            return self.execute_keyword();
        }

        let category_filter = self.category.as_deref();
        let type_filter = self.memory_type.map(|t| t.as_str());
        let min_tier = self.depth_to_min_tier();

        // FTS5 keyword search (OR mode + stop-word removal, over-fetch 3x for RRF)
        let fts_results = FtsSearch::search_or_mode(
            self.db,
            &self.query,
            self.limit * 3,
            category_filter,
            type_filter,
            min_tier,
        )?;

        // Vector similarity search (over-fetch 3x)
        let query_vec = embedding.embed_query(&self.query)?;
        let model = embedding.model_name();
        let vector_results = self.vector_results(&query_vec, model, self.limit * 3)?;

        // Merge via RRF
        let merged = rrf_merge(&fts_results, &vector_results, &self.query, self.limit * 2);

        let mut results = self.apply_filters(merged);

        // Reranking disabled — RRF + vector weighting provides better ranking
        // rerank_results(self.db, &mut results, &self.query);

        // Near-duplicate filtering: remove results >0.95 similar to higher-ranked ones
        deduplicate_by_vector_similarity(self.db, &mut results, model, 0.95);

        apply_strict_detail_query_filter(self.db, &self.query, &mut results);

        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }
        results.truncate(self.limit);
        Ok(results)
    }

    fn vector_results(
        &self,
        query_vec: &[f32],
        model: &str,
        limit: usize,
    ) -> Result<Vec<FtsResult>> {
        match self.vector_search_mode {
            VectorSearchMode::Off => Ok(Vec::new()),
            VectorSearchMode::Exact => VectorSearch::search(self.db, query_vec, model, limit),
            VectorSearchMode::Ann => self.execute_ann_vector_search(query_vec, model, limit),
        }
    }

    #[cfg(feature = "ann")]
    fn execute_ann_vector_search(
        &self,
        query_vec: &[f32],
        model: &str,
        limit: usize,
    ) -> Result<Vec<FtsResult>> {
        let Some(ref ann_index) = self.ann_index else {
            return VectorSearch::search(self.db, query_vec, model, limit);
        };

        let expected_count = VectorSearch::count_vectors(self.db, model)?;
        if expected_count == 0 {
            return Ok(Vec::new());
        }

        let needs_rebuild = !ann_index.is_built()
            || ann_index.len() != expected_count
            || ann_index.model_name().as_deref() != Some(model);

        if needs_rebuild {
            ann_index.build(self.db, model)?;
        }

        let results = ann_index.search(query_vec, limit)?;
        if results.is_empty() {
            VectorSearch::search(self.db, query_vec, model, limit)
        } else {
            Ok(results)
        }
    }

    #[cfg(not(feature = "ann"))]
    fn execute_ann_vector_search(
        &self,
        query_vec: &[f32],
        model: &str,
        limit: usize,
    ) -> Result<Vec<FtsResult>> {
        VectorSearch::search(self.db, query_vec, model, limit)
    }

    /// Convert search depth to minimum tier filter.
    fn depth_to_min_tier(&self) -> Option<i32> {
        match self.depth {
            SearchDepth::Standard => Some(1), // Tiers 1+2 (summaries and facts)
            SearchDepth::Deep => Some(0),     // All tiers including raw episodes
            SearchDepth::Forensic => None, // No filter (same as Deep, but conceptually includes archived)
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
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
                                &row.get::<_, String>(1)?,
                            )
                            .unwrap_or(crate::traits::MemoryType::Episodic),
                            importance: row.get::<_, i32>(2)? as u8,
                            category: row.get(3)?,
                            created_at: chrono::DateTime::parse_from_rfc3339(
                                &row.get::<_, String>(4)?,
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

/// Reranking pass after RRF merge with bigram overlap and length penalty.
///
/// 1. Unigram + bigram overlap scoring — captures phrase-level matches
/// 2. Length penalty — penalizes very short memories as noise
#[allow(dead_code)]
fn rerank_results(db: &Database, results: &mut [SearchResult], query: &str) {
    use crate::search::fts5::strip_stop_words;

    let stripped = strip_stop_words(query).to_lowercase();
    let query_words: Vec<&str> = stripped.split_whitespace().collect();

    if query_words.is_empty() {
        return;
    }

    // Extract query bigrams for phrase-level matching
    let query_bigrams: Vec<String> = query_words
        .windows(2)
        .map(|w| format!("{} {}", w[0], w[1]))
        .collect();

    for result in results.iter_mut() {
        let text = db.with_reader(|conn| {
            conn.query_row(
                "SELECT searchable_text FROM memories WHERE id = ?1",
                [result.memory_id],
                |row| row.get::<_, String>(0),
            )
            .map_err(crate::error::FemindError::Database)
        });

        let Ok(text) = text else { continue };
        let text_lower = text.to_lowercase();

        // Unigram overlap (0.0 to 1.0)
        let unigram_matches = query_words
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count();
        let unigram_ratio = unigram_matches as f32 / query_words.len() as f32;

        // Bigram overlap (0.0 to 1.0) — captures phrase matching
        let bigram_ratio = if !query_bigrams.is_empty() {
            let bigram_matches = query_bigrams
                .iter()
                .filter(|bg| text_lower.contains(bg.as_str()))
                .count();
            bigram_matches as f32 / query_bigrams.len() as f32
        } else {
            0.0
        };

        // Combined boost: unigrams (30%) + bigrams (20%)
        result.score *= 1.0 + unigram_ratio * 0.3 + bigram_ratio * 0.2;

        // Penalize very short memories (noise)
        if text.len() < 20 {
            result.score *= 0.3;
        } else if text.len() < 50 {
            result.score *= 0.7;
        }
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Filter near-duplicate results based on stored vector cosine similarity.
///
/// Removes results with >0.95 similarity to any higher-ranked result.
/// Maximizes information density in the context budget.
fn deduplicate_by_vector_similarity(
    db: &Database,
    results: &mut Vec<SearchResult>,
    model_name: &str,
    threshold: f32,
) {
    use crate::embeddings::pooling::{bytes_to_vec, cosine_similarity};

    if results.len() < 2 {
        return;
    }

    // Load vectors for all results
    let vectors: Vec<Option<Vec<f32>>> = results
        .iter()
        .map(|r| {
            db.with_reader(|conn| {
                let blob = conn.query_row(
                    "SELECT embedding FROM memory_vectors WHERE memory_id = ?1 AND model_name = ?2",
                    rusqlite::params![r.memory_id, model_name],
                    |row| row.get::<_, Vec<u8>>(0),
                );
                match blob {
                    Ok(b) => Ok(Some(bytes_to_vec(&b))),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(crate::error::FemindError::Database(e)),
                }
            })
            .ok()
            .flatten()
        })
        .collect();

    // Greedy deduplication: keep item if not too similar to any already-kept item
    let mut keep = vec![true; results.len()];
    for i in 1..results.len() {
        let Some(ref vi) = vectors[i] else { continue };
        for j in 0..i {
            if !keep[j] {
                continue;
            }
            let Some(ref vj) = vectors[j] else { continue };
            if cosine_similarity(vi, vj) > threshold {
                keep[i] = false;
                break;
            }
        }
    }

    let mut idx = 0;
    results.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

fn apply_strict_detail_query_filter(
    db: &Database,
    query: &str,
    results: &mut Vec<SearchResult>,
) {
    if !query_requires_strict_grounding(query) || results.is_empty() {
        return;
    }

    results.retain(|result| {
        let text = db
            .with_reader(|conn| {
                conn.query_row(
                    "SELECT searchable_text FROM memories WHERE id = ?1",
                    [result.memory_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(crate::error::FemindError::Database)
            })
            .ok();

        let Some(text) = text else { return false };
        lexical_grounding_ok(query, &text)
    });
}

fn query_requires_strict_grounding(query: &str) -> bool {
    let normalized = normalize_text(query);
    let tokens: Vec<_> = normalized.split_whitespace().collect();

    if tokens.is_empty() {
        return false;
    }

    let has_exact_signal = tokens.iter().any(|token| {
        matches!(
            *token,
            "exact" | "precise" | "specific" | "total" | "cost" | "token" | "tokens"
                | "price" | "version" | "number" | "id"
        )
    });
    let asks_how_many = tokens.windows(2).any(|pair| pair == ["how", "many"]);

    has_exact_signal || asks_how_many || query_asks_for_combined_capability(query)
}

fn lexical_grounding_ok(query: &str, text: &str) -> bool {
    if query_asks_for_combined_capability(query) && text_implies_exclusion(text) {
        return false;
    }

    let query_tokens = meaning_tokens(query);
    let text_tokens = meaning_tokens(text);

    if query_tokens.is_empty() || text_tokens.is_empty() {
        return false;
    }

    let overlap = query_tokens
        .iter()
        .filter(|token| text_tokens.contains(*token))
        .count();

    let detail_overlap = detail_tokens(query)
        .into_iter()
        .filter(|token| text_tokens.contains(token))
        .count();
    let recall = overlap as f32 / query_tokens.len() as f32;

    if query_requires_strict_grounding(query) && !detail_tokens(query).is_empty() && detail_overlap == 0 {
        return false;
    }

    overlap >= 2 || recall >= 0.5
}

fn meaning_tokens(value: &str) -> Vec<String> {
    normalize_text(value)
        .split_whitespace()
        .filter_map(canonical_token)
        .collect()
}

fn canonical_token(token: &str) -> Option<String> {
    let token = match token {
        "the" | "a" | "an" | "is" | "are" | "was" | "were" | "be" | "been" | "being"
        | "to" | "for" | "of" | "in" | "on" | "at" | "by" | "and" | "or" | "that"
        | "this" | "it" | "its" | "still" | "then" | "than" | "because" | "what"
        | "which" | "who" | "should" | "not" | "do" | "does" | "did" | "yet"
        | "after" | "before" | "over" | "under" | "with" | "without" | "from"
        | "into" | "about" | "no" | "current" | "earlier" => return None,
        "keep" | "used" | "use" => "prefer",
        "tried" | "try" => "first",
        "improved" | "good" | "looked" => "better",
        "happen" | "performed" => "run",
        "built" | "build" => "build",
        "preferred" | "prefer" => "prefer",
        "superseded" => "superseded",
        other => other,
    };

    let stemmed = token
        .trim_end_matches("ing")
        .trim_end_matches("ed")
        .trim_end_matches('s');
    if stemmed.is_empty() {
        None
    } else {
        Some(stemmed.to_string())
    }
}

fn detail_tokens(value: &str) -> Vec<String> {
    meaning_tokens(value)
        .into_iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "exact"
                    | "precise"
                    | "specific"
                    | "total"
                    | "cost"
                    | "token"
                    | "price"
                    | "version"
                    | "number"
                    | "id"
                    | "publish"
                    | "published"
                    | "release"
            )
        })
        .collect()
}

fn query_asks_for_combined_capability(query: &str) -> bool {
    let normalized = normalize_text(query);
    let tokens: Vec<_> = normalized.split_whitespace().collect();
    tokens.contains(&"together") || tokens.contains(&"both")
}

fn text_implies_exclusion(text: &str) -> bool {
    let normalized = normalize_text(text);
    normalized.contains("except")
        || normalized.contains("without")
        || normalized.contains("excluding")
        || normalized.contains("not with")
}

fn normalize_text(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
        fn id(&self) -> Option<i64> {
            self.id
        }
        fn searchable_text(&self) -> String {
            self.text.clone()
        }
        fn memory_type(&self) -> MemoryType {
            MemoryType::Semantic
        }
        fn created_at(&self) -> chrono::DateTime<Utc> {
            self.created_at
        }
        fn category(&self) -> Option<&str> {
            self.category.as_deref()
        }
    }

    fn setup() -> Database {
        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migrate");
        let store = MemoryStore::<TestMem>::new();
        for text in [
            "authentication failed with JWT token",
            "database connection timeout",
            "build succeeded after fixing imports",
            "authentication flow redesigned",
        ] {
            store
                .store(
                    &db,
                    &TestMem {
                        id: None,
                        text: text.to_string(),
                        category: None,
                        created_at: Utc::now(),
                    },
                )
                .expect("store");
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
        assert!(
            results.is_empty(),
            "no results should pass a very high min_score"
        );
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

    #[test]
    fn strict_grounding_detects_exact_detail_queries() {
        assert!(query_requires_strict_grounding(
            "What was the exact total token cost of the last Nemotron run?"
        ));
        assert!(!query_requires_strict_grounding(
            "Is desktop-first still the active plan?"
        ));
    }

    #[test]
    fn lexical_grounding_rejects_single_token_semantic_neighbor() {
        let query = "What was the exact total token cost of the last Nemotron run?";
        let weak_hit = "Need to compare extraction quality between gpt-oss-120b and Nemotron after the smoke test.";
        let grounded_hit = "The last Nemotron run cost 1832 input tokens and 411 output tokens total.";

        assert!(!lexical_grounding_ok(query, weak_hit));
        assert!(lexical_grounding_ok(query, grounded_hit));
    }

    #[test]
    fn lexical_grounding_rejects_exact_version_without_version_detail() {
        let query = "What exact crates.io version has already been published for femind?";
        let weak_hit = "The local crate and repo are now femind / fe-mind. The package rename is complete locally, and publication work is the remaining external packaging step.";

        assert!(!lexical_grounding_ok(query, weak_hit));
    }

    #[test]
    fn lexical_grounding_rejects_combination_query_when_text_says_except() {
        let query = "Which feature flag enables SQLCipher encryption and MCP server together?";
        let weak_hit = "Compile-time feature flags include api-embeddings for the DeepInfra embedding API, api-llm for the chat completions API, ann for HNSW approximate nearest neighbor, and full for everything except encryption and mcp-server.";

        assert!(!lexical_grounding_ok(query, weak_hit));
    }
}
