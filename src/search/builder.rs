use std::marker::PhantomData;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::embeddings::EmbeddingBackend;
use crate::engine::VectorSearchMode;
use crate::error::Result;
use crate::memory::GraphMemory;
use crate::search::fts5::{FtsResult, FtsSearch};
use crate::search::hybrid::rrf_merge;
use crate::search::vector::VectorSearch;
use crate::storage::Database;
use crate::traits::{
    MemoryMeta, MemoryRecord, MemoryType, RerankCandidate, RerankerBackend, ScoringStrategy,
};

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

/// High-level query classes used to route retrieval behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QueryIntent {
    General,
    ExactDetail,
    CurrentState,
    HistoricalState,
    Aggregation,
    AbstentionRisk,
}

impl QueryIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::ExactDetail => "exact-detail",
            Self::CurrentState => "current-state",
            Self::HistoricalState => "historical-state",
            Self::Aggregation => "aggregation",
            Self::AbstentionRisk => "abstention-risk",
        }
    }
}

/// Route-level temporal preference applied after first-stage retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TemporalPolicy {
    Neutral,
    PreferNewer,
    PreferOlder,
}

impl TemporalPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::PreferNewer => "prefer-newer",
            Self::PreferOlder => "prefer-older",
        }
    }
}

impl std::fmt::Display for TemporalPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Route-level state/conflict preference applied after first-stage retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StateConflictPolicy {
    Neutral,
    PreferCurrent,
    PreferHistorical,
}

impl StateConflictPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::PreferCurrent => "prefer-current",
            Self::PreferHistorical => "prefer-historical",
        }
    }
}

impl std::fmt::Display for StateConflictPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Display for QueryIntent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Effective routed search plan for a query.
#[derive(Debug, Clone)]
pub struct QueryRoute {
    pub intent: QueryIntent,
    pub mode: SearchMode,
    pub depth: SearchDepth,
    pub temporal_policy: TemporalPolicy,
    pub state_conflict_policy: StateConflictPolicy,
    pub strict_grounding: bool,
    pub query_alignment: bool,
    pub rerank_limit: usize,
    pub note: &'static str,
}

impl QueryRoute {
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            SearchMode::Keyword => "keyword",
            SearchMode::Vector => "vector",
            SearchMode::Hybrid => "hybrid",
            SearchMode::Auto => "auto",
            SearchMode::Exhaustive { .. } => "exhaustive",
        }
    }

    pub fn depth_name(&self) -> &'static str {
        match self.depth {
            SearchDepth::Standard => "standard",
            SearchDepth::Deep => "deep",
            SearchDepth::Forensic => "forensic",
        }
    }

    pub fn temporal_policy_name(&self) -> &'static str {
        self.temporal_policy.as_str()
    }

    pub fn state_conflict_policy_name(&self) -> &'static str {
        self.state_conflict_policy.as_str()
    }
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
    reranker: Option<Arc<dyn RerankerBackend>>,
    rerank_limit: usize,
    vector_search_mode: VectorSearchMode,
    strict_grounding_enabled: bool,
    query_alignment_enabled: bool,
    routing_enabled: bool,
    query_intent: Option<QueryIntent>,
    mode_overridden: bool,
    depth_overridden: bool,
    strict_grounding_overridden: bool,
    query_alignment_overridden: bool,
    rerank_limit_overridden: bool,
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
            reranker: None,
            rerank_limit: 20,
            vector_search_mode: VectorSearchMode::default(),
            strict_grounding_enabled: true,
            query_alignment_enabled: true,
            routing_enabled: true,
            query_intent: None,
            mode_overridden: false,
            depth_overridden: false,
            strict_grounding_overridden: false,
            query_alignment_overridden: false,
            rerank_limit_overridden: false,
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

    /// Attach a reranker backend for second-stage candidate refinement.
    pub fn with_reranker(mut self, reranker: Arc<dyn RerankerBackend>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Set the maximum number of candidates to pass through the reranker.
    pub fn with_rerank_limit(mut self, limit: usize) -> Self {
        self.rerank_limit = limit;
        self.rerank_limit_overridden = true;
        self
    }

    /// Attach the engine's vector search mode.
    pub fn with_vector_search_mode(mut self, mode: VectorSearchMode) -> Self {
        self.vector_search_mode = mode;
        self
    }

    /// Enable or disable strict post-search grounding filters.
    pub fn with_strict_grounding(mut self, enabled: bool) -> Self {
        self.strict_grounding_enabled = enabled;
        self.strict_grounding_overridden = true;
        self
    }

    /// Enable or disable query-shape-aware reranking heuristics.
    pub fn with_query_alignment(mut self, enabled: bool) -> Self {
        self.query_alignment_enabled = enabled;
        self.query_alignment_overridden = true;
        self
    }

    /// Attach default grounding behavior without marking it as a caller override.
    pub(crate) fn with_default_strict_grounding(mut self, enabled: bool) -> Self {
        self.strict_grounding_enabled = enabled;
        self
    }

    /// Attach default query-alignment behavior without marking it as a caller override.
    pub(crate) fn with_default_query_alignment(mut self, enabled: bool) -> Self {
        self.query_alignment_enabled = enabled;
        self
    }

    /// Attach the engine default rerank limit without marking it as a caller override.
    pub(crate) fn with_default_rerank_limit(mut self, limit: usize) -> Self {
        self.rerank_limit = limit;
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
        self.mode_overridden = true;
        self
    }

    /// Set the search depth (which tiers to search).
    pub fn depth(mut self, depth: SearchDepth) -> Self {
        self.depth = depth;
        self.depth_overridden = true;
        self
    }

    /// Enable or disable query-intent routing.
    pub fn with_routing(mut self, enabled: bool) -> Self {
        self.routing_enabled = enabled;
        self
    }

    /// Override the inferred query intent.
    pub fn with_query_intent(mut self, intent: QueryIntent) -> Self {
        self.query_intent = Some(intent);
        self
    }

    /// Return the effective routed plan for this query.
    pub fn query_route(&self) -> QueryRoute {
        let intent = self
            .query_intent
            .unwrap_or_else(|| infer_query_intent(&self.query));
        if !self.routing_enabled {
            return QueryRoute {
                intent,
                mode: self.mode.clone(),
                depth: self.depth,
                temporal_policy: TemporalPolicy::Neutral,
                state_conflict_policy: StateConflictPolicy::Neutral,
                strict_grounding: self.strict_grounding_enabled,
                query_alignment: self.query_alignment_enabled,
                rerank_limit: self.rerank_limit,
                note: "routing disabled; using configured search settings",
            };
        }

        let mode = if self.mode_overridden {
            self.mode.clone()
        } else {
            match intent {
                QueryIntent::AbstentionRisk => SearchMode::Keyword,
                _ => self.mode.clone(),
            }
        };

        let depth = if self.depth_overridden {
            self.depth
        } else {
            match intent {
                QueryIntent::HistoricalState | QueryIntent::Aggregation => SearchDepth::Deep,
                _ => self.depth,
            }
        };

        let strict_grounding = if self.strict_grounding_overridden {
            self.strict_grounding_enabled
        } else {
            matches!(
                intent,
                QueryIntent::ExactDetail | QueryIntent::AbstentionRisk
            )
        };

        let temporal_policy = match intent {
            QueryIntent::CurrentState => TemporalPolicy::PreferNewer,
            QueryIntent::HistoricalState => TemporalPolicy::PreferOlder,
            QueryIntent::General
            | QueryIntent::ExactDetail
            | QueryIntent::Aggregation
            | QueryIntent::AbstentionRisk => TemporalPolicy::Neutral,
        };

        let state_conflict_policy = match intent {
            QueryIntent::CurrentState => StateConflictPolicy::PreferCurrent,
            QueryIntent::HistoricalState => StateConflictPolicy::PreferHistorical,
            QueryIntent::General
            | QueryIntent::ExactDetail
            | QueryIntent::Aggregation
            | QueryIntent::AbstentionRisk => StateConflictPolicy::Neutral,
        };

        let query_alignment = if self.query_alignment_overridden {
            self.query_alignment_enabled
        } else {
            !matches!(
                intent,
                QueryIntent::Aggregation | QueryIntent::AbstentionRisk
            )
        };

        let rerank_limit = if self.rerank_limit_overridden {
            self.rerank_limit
        } else {
            match intent {
                QueryIntent::ExactDetail => self.rerank_limit.clamp(8, 16),
                QueryIntent::CurrentState | QueryIntent::HistoricalState => {
                    self.rerank_limit.max(30)
                }
                QueryIntent::Aggregation | QueryIntent::AbstentionRisk => 0,
                QueryIntent::General => self.rerank_limit,
            }
        };

        let note = match intent {
            QueryIntent::General => "default hybrid-style retrieval",
            QueryIntent::ExactDetail => "exact-detail query; keep grounding and compact reranking",
            QueryIntent::CurrentState => {
                "current-state query; widen reranking and favor current-state evidence"
            }
            QueryIntent::HistoricalState => {
                "historical-state query; widen reranking and favor earlier-state evidence"
            }
            QueryIntent::Aggregation => {
                "aggregation query; preserve broad coverage and bypass reranking"
            }
            QueryIntent::AbstentionRisk => {
                "abstention-risk query; use keyword-first strict grounding"
            }
        };

        QueryRoute {
            intent,
            mode,
            depth,
            temporal_policy,
            state_conflict_policy,
            strict_grounding,
            query_alignment,
            rerank_limit,
            note,
        }
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
        let route = self.query_route();
        match &route.mode {
            SearchMode::Keyword => self.execute_keyword(&route),
            SearchMode::Vector => self.execute_vector(&route),
            SearchMode::Hybrid => self.execute_hybrid(&route),
            SearchMode::Auto => self.execute_auto(&route),
            SearchMode::Exhaustive { min_score } => {
                let threshold = *min_score;
                self.execute_exhaustive(&route, threshold)
            }
        }
    }

    fn execute_auto(&self, route: &QueryRoute) -> Result<Vec<SearchResult>> {
        if self.vector_search_mode == VectorSearchMode::Off || self.embedding.is_none() {
            self.execute_keyword(route)
        } else {
            self.execute_hybrid(route)
        }
    }

    /// Execute keyword-only search via FTS5.
    fn execute_keyword(&self, route: &QueryRoute) -> Result<Vec<SearchResult>> {
        let category_filter = self.category.as_deref();
        let type_filter = self.memory_type.map(|t| t.as_str());
        let min_tier = self.depth_to_min_tier(route.depth);

        let fts_results = FtsSearch::search_with_tiers(
            self.db,
            &self.query,
            self.limit,
            category_filter,
            type_filter,
            min_tier,
        )?;
        let fts_results = self.maybe_rerank_results(fts_results, route.rerank_limit)?;

        let mut results = self.apply_filters(fts_results);
        apply_temporal_route_bias(self.db, &mut results, route.temporal_policy);
        apply_state_conflict_route_bias(self.db, &mut results, route.state_conflict_policy);

        // Apply min_score filter
        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }

        results.truncate(self.limit);
        Ok(results)
    }

    /// Execute exhaustive search — return all matches above threshold.
    fn execute_exhaustive(&self, route: &QueryRoute, min_score: f32) -> Result<Vec<SearchResult>> {
        let category_filter = self.category.as_deref();
        let type_filter = self.memory_type.map(|t| t.as_str());
        let min_tier = self.depth_to_min_tier(route.depth);

        let fts_results = FtsSearch::search_with_tiers(
            self.db,
            &self.query,
            10_000,
            category_filter,
            type_filter,
            min_tier,
        )?;
        let fts_results = self.maybe_rerank_results(fts_results, route.rerank_limit)?;

        let mut results = self.apply_filters(fts_results);
        apply_temporal_route_bias(self.db, &mut results, route.temporal_policy);
        apply_state_conflict_route_bias(self.db, &mut results, route.state_conflict_policy);
        results.retain(|r| r.score >= min_score);
        Ok(results)
    }

    /// Execute vector-only search.
    fn execute_vector(&self, route: &QueryRoute) -> Result<Vec<SearchResult>> {
        if self.vector_search_mode == VectorSearchMode::Off {
            return self.execute_keyword(route);
        }

        let Some(ref embedding) = self.embedding else {
            // No embedding backend — fall back to keyword
            return self.execute_keyword(route);
        };

        if !embedding.is_available() {
            return self.execute_keyword(route);
        }

        let query_vec = embedding.embed_query(&self.query)?;
        let model_names = embedding.compatibility_model_names();
        let vector_results = self.vector_results(&query_vec, &model_names, self.limit * 3)?;
        let vector_results = self.maybe_rerank_results(vector_results, route.rerank_limit)?;

        let mut results = self.apply_filters(vector_results);
        apply_temporal_route_bias(self.db, &mut results, route.temporal_policy);
        apply_state_conflict_route_bias(self.db, &mut results, route.state_conflict_policy);
        if route.strict_grounding {
            apply_strict_detail_query_filter(self.db, &self.query, &mut results);
        }
        if route.query_alignment {
            rerank_for_query_alignment(self.db, &self.query, &mut results);
        }
        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }
        results.truncate(self.limit);
        Ok(results)
    }

    /// Execute hybrid search: FTS5 + vector merged via RRF.
    fn execute_hybrid(&self, route: &QueryRoute) -> Result<Vec<SearchResult>> {
        if self.vector_search_mode == VectorSearchMode::Off {
            return self.execute_keyword(route);
        }

        let Some(ref embedding) = self.embedding else {
            return self.execute_keyword(route);
        };

        if !embedding.is_available() {
            return self.execute_keyword(route);
        }

        let category_filter = self.category.as_deref();
        let type_filter = self.memory_type.map(|t| t.as_str());
        let min_tier = self.depth_to_min_tier(route.depth);

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
        let model_names = embedding.compatibility_model_names();
        let vector_results = self.vector_results(&query_vec, &model_names, self.limit * 3)?;

        // Merge via RRF
        let merged = rrf_merge(&fts_results, &vector_results, &self.query, self.limit * 2);
        let merged = self.maybe_rerank_results(merged, route.rerank_limit)?;

        let mut results = self.apply_filters(merged);
        apply_temporal_route_bias(self.db, &mut results, route.temporal_policy);
        apply_state_conflict_route_bias(self.db, &mut results, route.state_conflict_policy);

        // Reranking disabled — RRF + vector weighting provides better ranking
        // rerank_results(self.db, &mut results, &self.query);

        // Near-duplicate filtering: remove results >0.95 similar to higher-ranked ones
        deduplicate_by_vector_similarity(self.db, &mut results, &model_names, 0.95);

        if route.strict_grounding {
            apply_strict_detail_query_filter(self.db, &self.query, &mut results);
        }
        if route.query_alignment {
            rerank_for_query_alignment(self.db, &self.query, &mut results);
        }

        if let Some(threshold) = self.min_score {
            results.retain(|r| r.score >= threshold);
        }
        results.truncate(self.limit);
        Ok(results)
    }

    fn vector_results(
        &self,
        query_vec: &[f32],
        model_names: &[String],
        limit: usize,
    ) -> Result<Vec<FtsResult>> {
        match self.vector_search_mode {
            VectorSearchMode::Off => Ok(Vec::new()),
            VectorSearchMode::Exact => VectorSearch::search(self.db, query_vec, model_names, limit),
            VectorSearchMode::Ann => self.execute_ann_vector_search(query_vec, model_names, limit),
        }
    }

    #[cfg(feature = "ann")]
    fn execute_ann_vector_search(
        &self,
        query_vec: &[f32],
        model_names: &[String],
        limit: usize,
    ) -> Result<Vec<FtsResult>> {
        let Some(ref ann_index) = self.ann_index else {
            return VectorSearch::search(self.db, query_vec, model_names, limit);
        };

        let canonical_model = model_names.first().map_or("", String::as_str);

        let expected_count = VectorSearch::count_vectors_for_models(self.db, model_names)?;
        if expected_count == 0 {
            return Ok(Vec::new());
        }

        let needs_rebuild = !ann_index.is_built()
            || ann_index.len() != expected_count
            || ann_index.model_name().as_deref() != Some(canonical_model);

        if needs_rebuild {
            ann_index.build(self.db, canonical_model, model_names)?;
        }

        let results = ann_index.search(query_vec, limit)?;
        if results.is_empty() {
            VectorSearch::search(self.db, query_vec, model_names, limit)
        } else {
            Ok(results)
        }
    }

    #[cfg(not(feature = "ann"))]
    fn execute_ann_vector_search(
        &self,
        query_vec: &[f32],
        model_names: &[String],
        limit: usize,
    ) -> Result<Vec<FtsResult>> {
        VectorSearch::search(self.db, query_vec, model_names, limit)
    }

    /// Convert search depth to minimum tier filter.
    fn depth_to_min_tier(&self, depth: SearchDepth) -> Option<i32> {
        match depth {
            SearchDepth::Standard => Some(1), // Tiers 1+2 (summaries and facts)
            SearchDepth::Deep => Some(0),     // All tiers including raw episodes
            SearchDepth::Forensic => None, // No filter (same as Deep, but conceptually includes archived)
        }
    }

    fn maybe_rerank_results(
        &self,
        results: Vec<FtsResult>,
        rerank_limit: usize,
    ) -> Result<Vec<FtsResult>> {
        let Some(reranker) = self.reranker.as_ref() else {
            return Ok(results);
        };
        if rerank_limit == 0 || results.len() < 2 {
            return Ok(results);
        }

        let rerank_count = rerank_limit.min(results.len());
        let mut candidates = Vec::with_capacity(rerank_count);
        for result in results.iter().take(rerank_count) {
            let text = self.db.with_reader(|conn| {
                conn.query_row(
                    "SELECT searchable_text FROM memories WHERE id = ?1",
                    [result.memory_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(crate::error::FemindError::Database)
            })?;
            candidates.push(RerankCandidate {
                memory_id: result.memory_id,
                text,
                score: result.score,
                raw_score: result.score,
                score_multiplier: 1.0,
            });
        }

        let reranked = reranker.rerank(&self.query, candidates)?;
        let mut reranked_results = reranked
            .into_iter()
            .map(|result| FtsResult {
                memory_id: result.memory_id,
                score: result.score,
            })
            .collect::<Vec<_>>();
        reranked_results.extend(results.into_iter().skip(rerank_count));
        Ok(reranked_results)
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

        if let Some(valid_at) = self.valid_at {
            results.retain(|result| {
                GraphMemory::state_conflict_snapshot(self.db, result.memory_id)
                    .ok()
                    .flatten()
                    .is_none_or(|snapshot| snapshot.is_valid_at(valid_at))
            });
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
    model_names: &[String],
    threshold: f32,
) {
    use crate::embeddings::pooling::{bytes_to_vec, cosine_similarity};
    use rusqlite::params_from_iter;

    if results.len() < 2 {
        return;
    }

    if model_names.is_empty() {
        return;
    }

    let placeholders = std::iter::repeat_n("?", model_names.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT embedding FROM memory_vectors WHERE memory_id = ? AND model_name IN ({placeholders})"
    );

    // Load vectors for all results
    let vectors: Vec<Option<Vec<f32>>> = results
        .iter()
        .map(|r| {
            db.with_reader(|conn| {
                let blob = conn.query_row(
                    &sql,
                    params_from_iter(
                        std::iter::once(r.memory_id.to_string()).chain(model_names.iter().cloned()),
                    ),
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

fn apply_temporal_route_bias(
    db: &Database,
    results: &mut [SearchResult],
    temporal_policy: TemporalPolicy,
) {
    if results.len() < 2 || matches!(temporal_policy, TemporalPolicy::Neutral) {
        return;
    }

    let timestamps: Vec<Option<i64>> = results
        .iter()
        .map(|result| {
            db.with_reader(|conn| {
                conn.query_row(
                    "SELECT created_at FROM memories WHERE id = ?1",
                    [result.memory_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(crate::error::FemindError::Database)
            })
            .ok()
            .and_then(|value| {
                chrono::DateTime::parse_from_rfc3339(&value)
                    .ok()
                    .map(|dt| dt.timestamp())
            })
        })
        .collect();

    let Some(min_ts) = timestamps.iter().flatten().copied().min() else {
        return;
    };
    let Some(max_ts) = timestamps.iter().flatten().copied().max() else {
        return;
    };
    if min_ts == max_ts {
        return;
    }

    let spread = (max_ts - min_ts) as f32;
    for (result, maybe_ts) in results.iter_mut().zip(timestamps.iter()) {
        let Some(ts) = *maybe_ts else { continue };
        let normalized_recent = (ts - min_ts) as f32 / spread;
        let multiplier = match temporal_policy {
            TemporalPolicy::Neutral => 1.0,
            TemporalPolicy::PreferNewer => 0.9 + 0.2 * normalized_recent,
            TemporalPolicy::PreferOlder => 0.9 + 0.2 * (1.0 - normalized_recent),
        };
        result.score *= multiplier;
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn apply_state_conflict_route_bias(
    db: &Database,
    results: &mut [SearchResult],
    policy: StateConflictPolicy,
) {
    if results.len() < 2 || matches!(policy, StateConflictPolicy::Neutral) {
        return;
    }

    let now = Utc::now();
    for result in results.iter_mut() {
        let Some(snapshot) = GraphMemory::state_conflict_snapshot(db, result.memory_id)
            .ok()
            .flatten()
        else {
            continue;
        };

        let mut multiplier = match policy {
            StateConflictPolicy::Neutral => 1.0,
            StateConflictPolicy::PreferCurrent => {
                let mut value = 1.0;
                if snapshot.is_superseded {
                    value *= 0.42;
                }
                if snapshot.supersedes_other {
                    value *= 1.18;
                }
                if snapshot
                    .valid_until
                    .is_some_and(|valid_until| valid_until <= now)
                {
                    value *= 0.6;
                }
                if snapshot
                    .valid_from
                    .is_some_and(|valid_from| valid_from > now)
                {
                    value *= 0.75;
                }
                value
            }
            StateConflictPolicy::PreferHistorical => {
                let mut value = 1.0;
                if snapshot.is_superseded {
                    value *= 1.3;
                }
                if snapshot.supersedes_other {
                    value *= 0.82;
                }
                if snapshot
                    .valid_until
                    .is_some_and(|valid_until| valid_until <= now)
                {
                    value *= 1.08;
                }
                if snapshot
                    .valid_from
                    .is_some_and(|valid_from| valid_from > now)
                {
                    value *= 0.7;
                }
                value
            }
        };

        if snapshot.has_conflict {
            multiplier *= 0.97;
        }

        result.score *= multiplier;
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(crate) fn apply_strict_detail_query_filter(
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

pub(crate) fn rerank_for_query_alignment(db: &Database, query: &str, results: &mut [SearchResult]) {
    if results.is_empty() {
        return;
    }

    for result in results.iter_mut() {
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

        let Some(text) = text else { continue };
        result.score *= query_alignment_multiplier(query, &text);
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub(crate) fn query_requires_strict_grounding(query: &str) -> bool {
    let normalized = normalize_text(query);
    let tokens: Vec<_> = normalized.split_whitespace().collect();

    if tokens.is_empty() {
        return false;
    }

    let has_exact_signal = tokens.iter().any(|token| {
        matches!(
            *token,
            "exact"
                | "precise"
                | "specific"
                | "total"
                | "cost"
                | "token"
                | "tokens"
                | "price"
                | "version"
                | "number"
                | "id"
                | "reserved"
                | "removed"
                | "remove"
                | "failed"
                | "fail"
                | "hour"
                | "minute"
                | "day"
                | "date"
                | "month"
                | "year"
                | "dollar"
                | "duration"
                | "filename"
                | "label"
                | "header"
                | "deployment"
                | "parameter"
                | "value"
                | "hnsw"
                | "subtype"
                | "plist"
                | "hash"
                | "threshold"
        )
    });
    let asks_how_many = tokens.windows(2).any(|pair| pair == ["how", "many"]);
    let has_ordinal_signal = ordinal_detail_signal(&tokens)
        || (contains_ordinal_token(&tokens)
            && tokens.iter().any(|token| {
                matches!(
                    *token,
                    "type" | "subtype" | "value" | "model" | "version" | "candidate" | "memory"
                )
            }));

    has_exact_signal
        || asks_how_many
        || has_ordinal_signal
        || query_asks_for_combined_capability(query)
        || query_mentions_artifact_detail(query)
}

pub fn infer_query_intent(query: &str) -> QueryIntent {
    if query_requests_aggregation(query) {
        QueryIntent::Aggregation
    } else if query_requests_abstention_risk(query) {
        QueryIntent::AbstentionRisk
    } else if query_requires_strict_grounding(query) {
        QueryIntent::ExactDetail
    } else if query_requests_historical_state(query) {
        QueryIntent::HistoricalState
    } else if query_requests_current_state(query) {
        QueryIntent::CurrentState
    } else {
        QueryIntent::General
    }
}

pub(crate) fn lexical_grounding_ok(query: &str, text: &str) -> bool {
    if query_asks_for_combined_capability(query) && text_implies_exclusion(text) {
        return query_is_yes_no(query);
    }

    let normalized_query = normalize_text(query);

    if query_requires_strict_grounding(query)
        && query_mentions_artifact_detail(query)
        && !normalized_query.contains("hnsw")
        && !normalized_query.contains("subtype")
        && text_contains_artifact_detail(text)
    {
        let normalized_text = normalize_text(text);
        let asks_source_of_truth = normalized_query.contains("source of truth");
        let asks_summary_file =
            normalized_query.contains("summary file") || normalized_query.contains("artifact file");
        let asks_script = normalized_query.contains("script")
            || normalized_query.contains("entry point")
            || normalized_query.contains("runner");

        if asks_source_of_truth && normalized_text.contains("source of truth") {
            return true;
        }
        if asks_summary_file && normalized_text.contains("summary") {
            return true;
        }
        if asks_script
            && (text.to_lowercase().contains("scripts/") || normalized_text.contains("script"))
        {
            return true;
        }
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

    let detail = detail_tokens(query);
    let specific_detail = specific_detail_tokens(query);
    let detail_overlap = detail
        .into_iter()
        .filter(|token| text_tokens.contains(token))
        .count();
    let specific_detail_overlap = specific_detail
        .iter()
        .filter(|token| text_tokens.contains(*token))
        .count();
    let recall = overlap as f32 / query_tokens.len() as f32;

    let query_numeric = numeric_tokens(query);
    let text_numeric = numeric_tokens(text);
    let query_units = unit_tokens(query);

    if query_requires_strict_grounding(query)
        && !specific_detail.is_empty()
        && specific_detail_overlap < specific_detail.len()
    {
        return false;
    }

    if query_requires_strict_grounding(query)
        && specific_detail.is_empty()
        && !detail_tokens(query).is_empty()
        && detail_overlap == 0
    {
        return false;
    }

    if query_requires_strict_grounding(query)
        && !query_units.is_empty()
        && !query_units.iter().all(|token| text_tokens.contains(token))
    {
        return false;
    }

    if query_requires_strict_grounding(query)
        && !query_numeric.is_empty()
        && !query_numeric
            .iter()
            .all(|token| text_numeric.contains(token))
    {
        return false;
    }

    overlap >= 2 || recall >= 0.5
}

fn query_alignment_multiplier(query: &str, text: &str) -> f32 {
    let normalized_query = normalize_text(query);
    let normalized_text = normalize_text(text);
    let lowered_text = text.to_lowercase();
    let intent = infer_query_intent(query);

    let mut multiplier = 1.0_f32;

    if query_mentions_artifact_detail(query) {
        multiplier *= if text_contains_artifact_detail(text) {
            1.45
        } else {
            0.82
        };

        if normalized_query.contains("source of truth") {
            multiplier *= if normalized_text.contains("source of truth") {
                1.35
            } else {
                0.82
            };
        }
    }

    if query_requests_support_state(query) {
        multiplier *= if text_indicates_support_state(&normalized_text, &lowered_text) {
            1.45
        } else {
            0.82
        };
    }

    if query_requests_next_step(query) {
        multiplier *= if text_indicates_next_step(&normalized_text) {
            1.5
        } else {
            0.8
        };
    }

    if query_requests_prerequisite(query) {
        multiplier *= if text_indicates_prerequisite(&normalized_text) {
            1.45
        } else {
            0.84
        };
    }

    if query_requests_negative_limit(query) {
        multiplier *= if text_indicates_negative_limit(&normalized_text) {
            2.4
        } else {
            0.45
        };
    }

    if query_requests_preference(query) {
        multiplier *= if text_indicates_preference(&normalized_text) {
            1.65
        } else {
            0.78
        };
    }

    match intent {
        QueryIntent::CurrentState => {
            let current = text_indicates_current_state(&normalized_text);
            let historical = text_indicates_historical_state(&normalized_text);
            if current {
                multiplier *= 1.55;
            } else if historical {
                multiplier *= 0.8;
            }
        }
        QueryIntent::HistoricalState => {
            let historical = text_indicates_historical_state(&normalized_text);
            let current = text_indicates_current_state(&normalized_text);
            if historical {
                multiplier *= 1.5;
            } else if current {
                multiplier *= 0.82;
            }
        }
        QueryIntent::AbstentionRisk => {
            multiplier *= if text_implies_exclusion(&normalized_text)
                || text_indicates_negative_limit(&normalized_text)
            {
                1.1
            } else {
                0.95
            };
        }
        QueryIntent::Aggregation | QueryIntent::ExactDetail | QueryIntent::General => {}
    }

    multiplier
}

fn meaning_tokens(value: &str) -> Vec<String> {
    normalize_text(value)
        .split_whitespace()
        .filter_map(canonical_token)
        .collect()
}

fn canonical_token(token: &str) -> Option<String> {
    let token = match token {
        "the" | "a" | "an" | "is" | "are" | "was" | "were" | "be" | "been" | "being" | "to"
        | "for" | "of" | "in" | "on" | "at" | "by" | "and" | "or" | "that" | "this" | "it"
        | "its" | "still" | "then" | "than" | "because" | "what" | "which" | "who" | "should"
        | "not" | "do" | "does" | "did" | "yet" | "after" | "before" | "over" | "under"
        | "with" | "without" | "from" | "into" | "about" | "no" | "current" | "earlier" => {
            return None;
        }
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
                    | "first"
                    | "second"
                    | "third"
                    | "fourth"
                    | "fifth"
                    | "reserved"
                    | "remove"
                    | "removed"
                    | "fail"
                    | "failed"
                    | "hour"
                    | "minute"
                    | "day"
                    | "date"
                    | "month"
                    | "year"
                    | "dollar"
                    | "duration"
                    | "filename"
                    | "file"
                    | "path"
                    | "script"
                    | "runner"
                    | "label"
                    | "header"
                    | "deployment"
                    | "parameter"
                    | "value"
                    | "hnsw"
                    | "subtype"
                    | "plist"
                    | "hash"
                    | "threshold"
                    | "emotion"
                    | "publish"
                    | "published"
                    | "release"
            )
        })
        .collect()
}

fn specific_detail_tokens(value: &str) -> Vec<String> {
    detail_tokens(value)
        .into_iter()
        .filter(|token| {
            !matches!(
                token.as_str(),
                "exact" | "precise" | "specific" | "total" | "publish" | "published" | "release"
            )
        })
        .collect()
}

fn numeric_tokens(value: &str) -> std::collections::BTreeSet<String> {
    normalize_text(value)
        .split_whitespace()
        .filter(|token| token.chars().any(|c| c.is_ascii_digit()))
        .map(|token| token.to_string())
        .collect()
}

fn unit_tokens(value: &str) -> std::collections::BTreeSet<String> {
    meaning_tokens(value)
        .into_iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "hour" | "minute" | "day" | "date" | "month" | "year" | "dollar"
            )
        })
        .collect()
}

fn query_asks_for_combined_capability(query: &str) -> bool {
    let normalized = normalize_text(query);
    let tokens: Vec<_> = normalized.split_whitespace().collect();
    tokens.contains(&"together") || tokens.contains(&"both")
}

fn query_is_yes_no(query: &str) -> bool {
    let normalized = normalize_text(query);
    matches!(
        normalized.split_whitespace().next(),
        Some(
            "is" | "are"
                | "was"
                | "were"
                | "does"
                | "do"
                | "did"
                | "can"
                | "could"
                | "should"
                | "would"
        )
    )
}

fn ordinal_detail_signal(tokens: &[&str]) -> bool {
    const ORDINALS: &[&str] = &["first", "second", "third", "fourth", "fifth"];

    tokens.windows(2).any(|pair| {
        matches!(pair[0], "what" | "which" | "who" | "when" | "where")
            && ORDINALS.contains(&pair[1])
    })
}

fn contains_ordinal_token(tokens: &[&str]) -> bool {
    const ORDINALS: &[&str] = &["first", "second", "third", "fourth", "fifth"];
    tokens.iter().any(|token| ORDINALS.contains(token))
}

fn text_implies_exclusion(text: &str) -> bool {
    let normalized = normalize_text(text);
    normalized.contains("except")
        || normalized.contains("without")
        || normalized.contains("excluding")
        || normalized.contains("not with")
}

fn query_mentions_artifact_detail(query: &str) -> bool {
    let normalized = normalize_text(query);
    let tokens: Vec<_> = normalized.split_whitespace().collect();

    normalized.contains("source of truth")
        || normalized.contains("summary file")
        || normalized.contains("artifact file")
        || normalized.contains("entry point")
        || tokens.iter().any(|token| {
            matches!(
                *token,
                "script" | "runner" | "file" | "filename" | "artifact"
            )
        })
        || query_requests_literal_path(query, &tokens)
}

fn text_contains_artifact_detail(text: &str) -> bool {
    let lowered = text.to_lowercase();
    lowered.contains("scripts/")
        || lowered.contains("target/")
        || lowered.contains(".db")
        || lowered.contains(".json")
        || lowered.contains(".sh")
        || lowered.contains(".sql")
        || lowered.contains(".md")
        || lowered.contains(".toml")
}

fn query_requests_support_state(query: &str) -> bool {
    let normalized = normalize_text(query);
    normalized.contains("support")
        || normalized.contains("enabled")
        || normalized.contains("stateful")
}

fn query_requests_aggregation(query: &str) -> bool {
    let normalized = normalize_text(query);
    let tokens: Vec<_> = normalized.split_whitespace().collect();
    let rollup_nouns = |token: &str| {
        matches!(
            token,
            "sessions"
                | "runs"
                | "times"
                | "entries"
                | "items"
                | "questions"
                | "scenarios"
                | "checks"
                | "memories"
                | "results"
        )
    };
    let totalization = tokens.contains(&"sum")
        || (tokens.contains(&"count") && tokens.iter().any(|token| rollup_nouns(token)))
        || (tokens.contains(&"total")
            && tokens
                .iter()
                .any(|token| matches!(*token, "number" | "count") || rollup_nouns(token)));

    tokens.windows(2).any(|pair| pair == ["how", "many"])
        || tokens.windows(2).any(|pair| pair == ["list", "all"])
        || tokens.windows(2).any(|pair| pair == ["which", "ones"])
        || totalization
        || (tokens.contains(&"all") && tokens.contains(&"sessions"))
}

fn query_requests_abstention_risk(query: &str) -> bool {
    let normalized = normalize_text(query);
    normalized.contains("ever mention")
        || normalized.contains("ever say")
        || normalized.contains("did i ever")
        || normalized.contains("did we ever")
        || normalized.contains("if any")
        || normalized.contains("at all")
        || normalized.contains("anything about")
}

fn query_requests_current_state(query: &str) -> bool {
    let normalized = normalize_text(query);
    normalized.contains("current")
        || normalized.contains("currently")
        || normalized.contains("right now")
        || normalized.contains("latest")
        || normalized.contains("today")
        || normalized.contains("active")
        || normalized.contains("going with")
        || normalized.contains("using now")
        || normalized.contains("preferred now")
        || normalized.contains("final")
}

fn query_requests_historical_state(query: &str) -> bool {
    let normalized = normalize_text(query);
    normalized.contains("earlier")
        || normalized.contains("previous")
        || normalized.contains("previously")
        || normalized.contains("before")
        || normalized.contains("prior")
        || normalized.contains("used to")
        || normalized.contains("originally")
        || normalized.contains("initial")
        || normalized.contains("formerly")
        || normalized.contains("at first")
}

fn text_indicates_support_state(normalized_text: &str, lowered_text: &str) -> bool {
    normalized_text.contains("enabled")
        || normalized_text.contains("stateful")
        || normalized_text.contains("support")
        || lowered_text.contains("codex-cli")
}

fn text_indicates_current_state(normalized_text: &str) -> bool {
    normalized_text.contains("current")
        || normalized_text.contains("currently")
        || normalized_text.contains("right now")
        || normalized_text.contains("latest")
        || normalized_text.contains("active")
        || normalized_text.contains("preferred")
        || normalized_text.contains("prefer")
        || normalized_text.contains("going with")
        || normalized_text.contains("using now")
        || normalized_text.contains("final")
}

fn text_indicates_historical_state(normalized_text: &str) -> bool {
    normalized_text.contains("earlier")
        || normalized_text.contains("previous")
        || normalized_text.contains("previously")
        || normalized_text.contains("before")
        || normalized_text.contains("prior")
        || normalized_text.contains("used to")
        || normalized_text.contains("originally")
        || normalized_text.contains("initial")
        || normalized_text.contains("formerly")
        || normalized_text.contains("at first")
}

fn query_requests_next_step(query: &str) -> bool {
    normalize_text(query)
        .split_whitespace()
        .any(|token| token == "next")
}

fn text_indicates_next_step(normalized_text: &str) -> bool {
    normalized_text.contains("next validation step")
        || normalized_text.contains("come next")
        || normalized_text.contains("should come next")
        || normalized_text.contains("rather than continuing")
}

fn query_requests_prerequisite(query: &str) -> bool {
    let normalized = normalize_text(query);
    normalized.contains("before") || normalized.contains("had to be completed")
}

fn text_indicates_prerequisite(normalized_text: &str) -> bool {
    normalized_text.contains("before")
        || normalized_text.contains("had to be completed")
        || normalized_text.contains("complete ann")
        || normalized_text.contains("pending explicit user approval")
}

fn query_requests_negative_limit(query: &str) -> bool {
    let normalized = normalize_text(query);
    normalized.contains("did not")
        || normalized.contains("not test")
        || normalized.contains("not prove")
        || normalized.contains("alone prove")
}

fn text_indicates_negative_limit(normalized_text: &str) -> bool {
    normalized_text.contains("did not")
        || normalized_text.contains("not test")
        || normalized_text.contains("not prove")
        || normalized_text.contains("cannot")
        || normalized_text.contains("can not")
}

fn query_requests_preference(query: &str) -> bool {
    let normalized = normalize_text(query);
    normalized.contains("matters more than")
        || normalized.contains("rather than")
        || normalized.contains("prioritize")
        || normalized
            .split_whitespace()
            .any(|token| matches!(token, "prefer" | "preferred" | "preference"))
}

fn text_indicates_preference(normalized_text: &str) -> bool {
    normalized_text.contains("matters more")
        || normalized_text.contains("matter more")
        || normalized_text.contains("rather than")
        || normalized_text.contains("prioritize")
        || normalized_text
            .split_whitespace()
            .any(|token| matches!(token, "prefer" | "preferred" | "preference"))
}

fn query_requests_literal_path(query: &str, tokens: &[&str]) -> bool {
    if !tokens.contains(&"path") {
        return false;
    }

    tokens.iter().any(|token| {
        matches!(
            *token,
            "file"
                | "filename"
                | "script"
                | "runner"
                | "artifact"
                | "summary"
                | "db"
                | "database"
                | "config"
                | "plist"
                | "entry"
        )
    }) || query.contains('/')
        || query.contains(".db")
        || query.contains(".json")
        || query.contains(".sh")
        || query.contains(".sql")
        || query.contains(".toml")
        || query.contains(".md")
}

fn normalize_text(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() {
                c
            } else {
                ' '
            }
        })
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
    use rusqlite::params;

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
        assert!(!query_requires_strict_grounding(
            "What embedding path is preferred for evaluation workflows?"
        ));
    }

    #[test]
    fn lexical_grounding_rejects_single_token_semantic_neighbor() {
        let query = "What was the exact total token cost of the last Nemotron run?";
        let weak_hit = "Need to compare extraction quality between gpt-oss-120b and Nemotron after the smoke test.";
        let grounded_hit =
            "The last Nemotron run cost 1832 input tokens and 411 output tokens total.";

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

    #[test]
    fn lexical_grounding_accepts_yes_no_combination_query_with_exclusion_evidence() {
        let query = "Does the full feature set include encryption and mcp-server together?";
        let supporting_hit = "Compile-time feature flags include api-embeddings for the DeepInfra embedding API, api-llm for the chat completions API, ann for HNSW approximate nearest neighbor, and full for everything except encryption and mcp-server.";

        assert!(lexical_grounding_ok(query, supporting_hit));
    }

    #[test]
    fn lexical_grounding_rejects_unsupported_ordinal_detail_query() {
        let query = "What fourth cognitive memory type is reserved for emotions?";
        let weak_hit = "The optimized memory model uses three cognitive memory types: episodic, semantic, and procedural.";

        assert!(query_requires_strict_grounding(query));
        assert!(!lexical_grounding_ok(query, weak_hit));
    }

    #[test]
    fn lexical_grounding_rejects_removed_candidate_query_without_removal_evidence() {
        let query = "Did Nemotron fail the broader live-usage sample and get removed from the candidate set?";
        let weak_hit = "The broader live-usage sample passes 11/11 across four tested extraction models: openai/gpt-oss-120b, zai-org/GLM-4.7-Flash, gpt-5.4-mini, and gpt-5.1-codex-mini.";

        assert!(query_requires_strict_grounding(query));
        assert!(!lexical_grounding_ok(query, weak_hit));
    }

    #[test]
    fn lexical_grounding_rejects_exact_hour_without_hour_detail() {
        let query = "What exact hour was version 0.2.0 published?";
        let weak_hit = "Version 0.2.0 was released on 2026-03-26.";

        assert!(query_requires_strict_grounding(query));
        assert!(!lexical_grounding_ok(query, weak_hit));
    }

    #[test]
    fn lexical_grounding_rejects_mismatched_numeric_date_detail() {
        let query = "What version was released on 2026-03-25?";
        let weak_hit = "Version 0.2.0 was released on 2026-03-26.";

        assert!(query_requires_strict_grounding(query));
        assert!(!lexical_grounding_ok(query, weak_hit));
    }

    #[test]
    fn lexical_grounding_rejects_filename_without_filename_detail() {
        let query = "What exact artifact filename stored the earlier summary?";
        let weak_hit = "The earlier run scored 36 out of 44 on the larger corpus.";

        assert!(query_requires_strict_grounding(query));
        assert!(!lexical_grounding_ok(query, weak_hit));
    }

    #[test]
    fn lexical_grounding_rejects_deployment_without_deployment_detail() {
        let query = "Which Azure deployment name is the default CLI extraction backend?";
        let weak_hit = "The default CLI extraction model is gpt-5.4-mini via codex-cli.";

        assert!(query_requires_strict_grounding(query));
        assert!(!lexical_grounding_ok(query, weak_hit));
    }

    #[test]
    fn lexical_grounding_accepts_source_of_truth_db_path() {
        let query = "Is the primary local source of truth file femind_architecture.sql?";
        let supporting_hit = "The primary local source of truth is .docs/femind_spec.db. The markdown docs in the repo should stay aligned with that database.";

        assert!(query_requires_strict_grounding(query));
        assert!(lexical_grounding_ok(query, supporting_hit));
    }

    #[test]
    fn query_alignment_boosts_negative_limit_text() {
        let negative_query = "What did benchmark work not prove reliably?";
        let negative_hit = "Benchmarks did not test LLM fact extraction quality, graph-based retrieval, or real conversation memory.";
        let positive_hit = "Benchmarks validated the core search pipeline.";

        assert!(
            query_alignment_multiplier(negative_query, negative_hit)
                > query_alignment_multiplier(negative_query, positive_hit)
        );
    }

    #[test]
    fn query_alignment_treats_prefer_as_preference_signal() {
        let query = "What embedding path is preferred for evaluation workflows?";
        let preference_hit = "Update: for benchmark and evaluation workflows, prefer DeepInfra MiniLM embeddings because they are much faster and cheap enough.";
        let non_preference_hit =
            "Benchmark and evaluation workflows were discussed during the last planning review.";

        assert!(
            query_alignment_multiplier(query, preference_hit)
                > query_alignment_multiplier(query, non_preference_hit)
        );
    }

    #[test]
    fn lexical_grounding_allows_generic_embedding_path_preference_text() {
        let query = "What embedding path is preferred for evaluation workflows?";
        let supporting_hit = "Update: for benchmark and evaluation workflows, prefer DeepInfra MiniLM embeddings because they are much faster and cheap enough.";

        assert!(lexical_grounding_ok(query, supporting_hit));
    }

    #[test]
    fn infer_query_intent_covers_core_routes() {
        assert_eq!(
            infer_query_intent("How many practical scenarios are there?"),
            QueryIntent::Aggregation
        );
        assert_eq!(
            infer_query_intent("What was the exact total token cost of the last Nemotron run?"),
            QueryIntent::ExactDetail
        );
        assert_eq!(
            infer_query_intent(
                "What exact token count threshold triggers the lexical-grounding filter?"
            ),
            QueryIntent::ExactDetail
        );
        assert_eq!(
            infer_query_intent("Did we ever mention Redis at all?"),
            QueryIntent::AbstentionRisk
        );
        assert_eq!(
            infer_query_intent("What is the current preferred embedding path?"),
            QueryIntent::CurrentState
        );
        assert_eq!(
            infer_query_intent("What did we use before the rename?"),
            QueryIntent::HistoricalState
        );
        assert_eq!(
            infer_query_intent("What exact plist filename was used?"),
            QueryIntent::ExactDetail
        );
        assert_eq!(
            infer_query_intent(
                "What exact dataset hash value is stored in cache_meta for the current benchmark corpus?"
            ),
            QueryIntent::ExactDetail
        );
        assert_eq!(
            infer_query_intent(
                "What exact request header value was sent in the failing initialize call before the transport fix?"
            ),
            QueryIntent::ExactDetail
        );
    }

    #[test]
    fn query_route_adjusts_plan_for_abstention_queries() {
        let db = setup();
        let route =
            SearchBuilder::<TestMem>::new(&db, "Did we ever mention Redis at all?").query_route();
        assert_eq!(route.intent, QueryIntent::AbstentionRisk);
        assert!(matches!(route.mode, SearchMode::Keyword));
        assert!(route.strict_grounding);
        assert!(!route.query_alignment);
        assert_eq!(route.rerank_limit, 0);
    }

    #[test]
    fn query_route_widens_reranking_for_state_queries() {
        let db = setup();
        let current =
            SearchBuilder::<TestMem>::new(&db, "What is the current preferred path?").query_route();
        let historical =
            SearchBuilder::<TestMem>::new(&db, "What was the preferred path before?").query_route();
        assert_eq!(current.intent, QueryIntent::CurrentState);
        assert_eq!(historical.intent, QueryIntent::HistoricalState);
        assert_eq!(current.temporal_policy, TemporalPolicy::PreferNewer);
        assert_eq!(historical.temporal_policy, TemporalPolicy::PreferOlder);
        assert_eq!(
            current.state_conflict_policy,
            StateConflictPolicy::PreferCurrent
        );
        assert_eq!(
            historical.state_conflict_policy,
            StateConflictPolicy::PreferHistorical
        );
        assert!(current.query_alignment);
        assert!(historical.query_alignment);
        assert_eq!(current.rerank_limit, 30);
        assert_eq!(historical.rerank_limit, 30);
    }

    #[test]
    fn query_alignment_boosts_current_state_text() {
        let query = "What is the current preferred embedding path?";
        let current_hit = "The current preferred path is remote MiniLM with local fallback.";
        let historical_hit = "Previously we used the earlier local-only path before the switch.";
        assert!(
            query_alignment_multiplier(query, current_hit)
                > query_alignment_multiplier(query, historical_hit)
        );
    }

    #[test]
    fn query_alignment_boosts_historical_state_text() {
        let query = "What did we use before the rename?";
        let historical_hit =
            "Before the rename, the initial plan used mindcore and local-only retrieval.";
        let current_hit = "The current path uses femind with remote fallback.";
        assert!(
            query_alignment_multiplier(query, historical_hit)
                > query_alignment_multiplier(query, current_hit)
        );
    }

    #[test]
    fn temporal_bias_prefers_newer_for_current_state_queries() {
        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migrate");
        let store = MemoryStore::<TestMem>::new();

        let older = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "preferred path was local cpu".to_string(),
                    category: None,
                    created_at: Utc::now() - chrono::Duration::days(10),
                },
            )
            .expect("store old");
        let newer = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "current preferred path is remote fallback".to_string(),
                    category: None,
                    created_at: Utc::now(),
                },
            )
            .expect("store new");

        let older_id = match older {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };
        let newer_id = match newer {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };

        let mut results = vec![
            SearchResult {
                memory_id: older_id,
                score: 1.0,
            },
            SearchResult {
                memory_id: newer_id,
                score: 1.0,
            },
        ];
        apply_temporal_route_bias(&db, &mut results, TemporalPolicy::PreferNewer);
        assert_eq!(results[0].memory_id, newer_id);
    }

    #[test]
    fn temporal_bias_prefers_older_for_historical_queries() {
        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migrate");
        let store = MemoryStore::<TestMem>::new();

        let older = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "before the rename we used mindcore".to_string(),
                    category: None,
                    created_at: Utc::now() - chrono::Duration::days(30),
                },
            )
            .expect("store old");
        let newer = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "current repo name is fe mind".to_string(),
                    category: None,
                    created_at: Utc::now(),
                },
            )
            .expect("store new");

        let older_id = match older {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };
        let newer_id = match newer {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };

        let mut results = vec![
            SearchResult {
                memory_id: newer_id,
                score: 1.0,
            },
            SearchResult {
                memory_id: older_id,
                score: 1.0,
            },
        ];
        apply_temporal_route_bias(&db, &mut results, TemporalPolicy::PreferOlder);
        assert_eq!(results[0].memory_id, older_id);
    }

    #[test]
    fn state_conflict_bias_prefers_current_for_current_queries() {
        use crate::memory::{GraphMemory, RelationType};

        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migrate");
        let store = MemoryStore::<TestMem>::new();

        let older = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "desktop-first was the earlier plan".to_string(),
                    category: None,
                    created_at: Utc::now() - chrono::Duration::days(7),
                },
            )
            .expect("store old");
        let newer = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "current plan starts with femind".to_string(),
                    category: None,
                    created_at: Utc::now(),
                },
            )
            .expect("store new");

        let older_id = match older {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };
        let newer_id = match newer {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };

        GraphMemory::relate(&db, older_id, newer_id, &RelationType::SupersededBy).expect("relate");

        let mut results = vec![
            SearchResult {
                memory_id: older_id,
                score: 1.0,
            },
            SearchResult {
                memory_id: newer_id,
                score: 1.0,
            },
        ];
        apply_state_conflict_route_bias(&db, &mut results, StateConflictPolicy::PreferCurrent);
        assert_eq!(results[0].memory_id, newer_id);
    }

    #[test]
    fn state_conflict_bias_prefers_prior_state_for_historical_queries() {
        use crate::memory::{GraphMemory, RelationType};

        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migrate");
        let store = MemoryStore::<TestMem>::new();

        let older = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "before the rename the repo was mindcore".to_string(),
                    category: None,
                    created_at: Utc::now() - chrono::Duration::days(30),
                },
            )
            .expect("store old");
        let newer = store
            .store(
                &db,
                &TestMem {
                    id: None,
                    text: "the repo is now fe-mind".to_string(),
                    category: None,
                    created_at: Utc::now(),
                },
            )
            .expect("store new");

        let older_id = match older {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };
        let newer_id = match newer {
            crate::memory::store::StoreResult::Added(id) => id,
            other => panic!("unexpected store result: {other:?}"),
        };

        GraphMemory::relate(&db, older_id, newer_id, &RelationType::SupersededBy).expect("relate");

        let mut results = vec![
            SearchResult {
                memory_id: newer_id,
                score: 1.0,
            },
            SearchResult {
                memory_id: older_id,
                score: 1.0,
            },
        ];
        apply_state_conflict_route_bias(&db, &mut results, StateConflictPolicy::PreferHistorical);
        assert_eq!(results[0].memory_id, older_id);
    }

    #[test]
    fn valid_at_filters_out_memories_outside_validity_window() {
        let db = setup();
        let historical = "2026-03-01T00:00:00Z";
        let current = "2026-03-25T00:00:00Z";
        db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO memories (
                    searchable_text, memory_type, content_hash, created_at, valid_from, valid_until, record_json
                 ) VALUES (?1, 'semantic', ?2, ?3, ?4, ?5, '{}')",
                params![
                    "repo was mindcore before the rename",
                    "valid_old",
                    historical,
                    historical,
                    current,
                ],
            )?;
            conn.execute(
                "INSERT INTO memories (
                    searchable_text, memory_type, content_hash, created_at, valid_from, record_json
                 ) VALUES (?1, 'semantic', ?2, ?3, ?4, '{}')",
                params![
                    "repo is now fe-mind",
                    "valid_new",
                    current,
                    current,
                ],
            )?;
            Ok::<(), crate::error::FemindError>(())
        })
        .expect("insert");

        let results = SearchBuilder::<TestMem>::new(&db, "repo")
            .valid_at(
                chrono::DateTime::parse_from_rfc3339("2026-03-20T00:00:00Z")
                    .expect("parse")
                    .with_timezone(&Utc),
            )
            .limit(10)
            .execute()
            .expect("search");

        assert_eq!(results.len(), 1);
        let text = db
            .with_reader(|conn| {
                conn.query_row(
                    "SELECT searchable_text FROM memories WHERE id = ?1",
                    [results[0].memory_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(crate::error::FemindError::Database)
            })
            .expect("load");
        assert!(text.contains("mindcore"));
    }
}
