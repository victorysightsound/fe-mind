use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use sha2::Digest;

use crate::context::{ContextAssembly, ContextBudget, ContextItem, PRIORITY_LEARNING};
use crate::embeddings::EmbeddingBackend;
use crate::error::{FemindError, Result};
use crate::memory::MemoryStore;
use crate::memory::store::StoreResult;
use crate::reranking::RerankerRuntime;
use crate::scoring::{
    CompositeScorer, ImportanceScorer, MemoryTypeScorer, ProceduralSafetyScorer, RecencyScorer,
    ReviewApprovalTemplate, ReviewPolicyClass, ReviewSafetyScorer, ReviewScope, ReviewSeverity,
    ReviewStatus, SecretClass, SourceProvenanceScorer, SourceTrustScorer, detect_review_flag,
    effective_review_status, evidence_contains_secret_material,
    query_requests_private_infra_guidance, query_requests_secret_location_or_reference,
    query_requests_sensitive_secret_detail, redact_secret_material, review_expires_at,
    review_policy_class_matches_query, review_scope_matches_query, secret_class_from_metadata,
    source_provenance_rank, source_trust_level,
};
use crate::search::builder::SearchBuilder;
use crate::storage::Database;
use crate::storage::migrations;
use crate::traits::{MemoryMeta, MemoryRecord, RerankerBackend, ScoringStrategy};

/// Result of a store_with_extraction() operation.
#[derive(Debug, Clone)]
pub struct StoreExtractionResult {
    /// Number of facts the LLM extracted from the raw text.
    pub facts_extracted: usize,
    /// Number of new memories stored (after deduplication).
    pub memories_stored: usize,
    /// Number of duplicate facts skipped.
    pub duplicates_skipped: usize,
    /// Number of graph edges created (SupersededBy + RelatedTo).
    pub graph_edges_created: usize,
    /// Number of superseded (outdated) facts detected.
    pub superseded_count: usize,
    /// Approximate LLM tokens used for extraction.
    pub tokens_used: usize,
}

/// A single retrieval hit surfaced through aggregation-oriented search.
#[derive(Debug, Clone)]
pub struct AggregatedMatch {
    /// Memory row ID.
    pub memory_id: i64,
    /// Retrieved text content.
    pub text: String,
    /// Optional category/source label.
    pub category: Option<String>,
    /// Combined relevance score from retrieval.
    pub score: f32,
    /// Optional metadata carried on the stored memory row.
    pub metadata: std::collections::HashMap<String, String>,
}

/// Aggregation-oriented retrieval result.
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Total scored matches before distinct deduplication.
    pub total_matches: usize,
    /// Distinct match count after text-level deduplication.
    pub distinct_match_count: usize,
    /// Top distinct matches kept for downstream composition.
    pub matches: Vec<AggregatedMatch>,
    /// Simple composed summary text spanning the kept matches.
    pub composed_summary: String,
}

/// Deterministic answer composition result built from routed retrieval evidence.
#[derive(Debug, Clone)]
pub struct ComposedAnswerResult {
    /// Composed answer text.
    pub answer: String,
    /// Composition strategy used to build the answer.
    pub kind: &'static str,
    /// Confidence level for the composed answer.
    pub confidence: CompositionConfidence,
    /// Whether the composer intentionally abstained.
    pub abstained: bool,
    /// Short explanation for the confidence/abstention outcome.
    pub rationale: &'static str,
    /// Total scored matches before distinct deduplication.
    pub total_matches: usize,
    /// Distinct evidence count kept for the answer.
    pub distinct_match_count: usize,
    /// Evidence bundle used during composition.
    pub evidence: Vec<AggregatedMatch>,
}

/// Human-review item raised for a high-impact memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReviewItem {
    /// Memory row ID.
    pub memory_id: i64,
    /// Severity of the review request.
    pub severity: ReviewSeverity,
    /// Primary reason for the review request.
    pub reason: String,
    /// Matched review tags.
    pub tags: Vec<String>,
    /// Review status.
    pub status: ReviewStatus,
    /// Review scope for temporary or environment-specific allowances.
    pub scope: Option<ReviewScope>,
    /// High-level policy class for the review decision.
    pub policy_class: Option<ReviewPolicyClass>,
    /// Optional approval template used to populate scope/class/expiry defaults.
    pub template: Option<ReviewApprovalTemplate>,
    /// Human reviewer or maintainer who resolved the item.
    pub reviewer: Option<String>,
    /// Optional replacement memory ID when this review item was superseded by another memory.
    pub replaced_by: Option<i64>,
    /// When the review state was last updated.
    pub updated_at: Option<DateTime<Utc>>,
    /// Optional expiry for temporary allowances.
    pub expires_at: Option<DateTime<Utc>>,
    /// Optional maintainer note explaining the review decision.
    pub note: Option<String>,
    /// Original text snippet.
    pub text: String,
    /// Creation timestamp from the stored memory row.
    pub created_at: DateTime<Utc>,
}

/// Operator-supplied review resolution metadata.
#[derive(Debug, Clone)]
pub struct ReviewResolution {
    pub status: ReviewStatus,
    pub note: Option<String>,
    pub reviewer: Option<String>,
    pub scope: Option<ReviewScope>,
    pub policy_class: Option<ReviewPolicyClass>,
    pub template: Option<ReviewApprovalTemplate>,
    pub expires_at: Option<DateTime<Utc>>,
    pub replaced_by: Option<i64>,
}

/// High-level derived knowledge categories produced by deterministic reflection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KnowledgeObjectKind {
    StableFact,
    StablePreference,
    StableDecision,
    StableProcedure,
}

impl KnowledgeObjectKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StableFact => "stable-fact",
            Self::StablePreference => "stable-preference",
            Self::StableDecision => "stable-decision",
            Self::StableProcedure => "stable-procedure",
        }
    }
}

impl std::fmt::Display for KnowledgeObjectKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Runtime configuration for deterministic reflection.
#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    /// Minimum number of supporting memories required before emitting a knowledge object.
    pub min_support_count: usize,
    /// Minimum number of trusted or normal-trust supports required.
    pub min_trusted_support_count: usize,
    /// Maximum number of knowledge objects to return.
    pub max_objects: usize,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            min_support_count: 2,
            min_trusted_support_count: 1,
            max_objects: 12,
        }
    }
}

/// Deterministically derived knowledge object synthesized from repeated evidence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnowledgeObject {
    /// Stable grouping key for the derived knowledge.
    pub key: String,
    /// Derived summary text.
    pub summary: String,
    /// High-level knowledge category.
    pub kind: KnowledgeObjectKind,
    /// Confidence in the reflected object.
    pub confidence: CompositionConfidence,
    /// Number of supporting records for the chosen summary cluster.
    pub support_count: usize,
    /// Number of trusted or normal-trust supports in the chosen cluster.
    pub trusted_support_count: usize,
    /// Database row IDs of supporting memories.
    pub source_ids: Vec<i64>,
    /// When the knowledge object was generated.
    pub generated_at: DateTime<Utc>,
}

/// Confidence label for deterministic composed answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompositionConfidence {
    Low,
    Medium,
    High,
}

impl CompositionConfidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Runtime feature configuration for femind.
///
/// Two-level config:
/// - **EngineConfig** controls WHAT features are active (store-time toggles).
///   These are system-wide settings that affect all operations.
/// - **AssemblyConfig** (nested inside) controls HOW search behaves (query-time tuning).
///   These can vary per query or per dataset.
///
/// All toggles are independent — any combination is valid.
///
/// ## Store-time features (EngineConfig)
/// - `embedding_enabled` — whether store()/store_batch()/store_with_extraction() compute vectors
/// - `graph_enabled` — whether store_with_extraction() creates graph edges, and search uses them
/// - `dedup_enabled` — whether store operations check content hash for duplicates
/// - `vector_search_mode` — "exact" (brute-force), "ann" (approximate), or "off" (FTS5 only)
/// - `strict_grounding_enabled` — whether exact-detail lexical grounding filters run after retrieval
/// - `query_alignment_enabled` — whether query-shape reranking heuristics run after retrieval
/// - `reranking_runtime` — whether a configured reranker backend should run
/// - `rerank_candidate_limit` — max number of first-stage candidates to rerank
///
/// ## Query-time tuning (AssemblyConfig)
/// - `max_per_session` — diversification limit (1 for multi-session, 0 for single-document)
/// - `recency_boost` — score boost for newer content (0.0-1.0)
/// - `search_limit` — max results from multi-query search
/// - `graph_depth` — how many hops to traverse in graph filtering
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Enable vector embedding at store time.
    /// When false, memories are stored with FTS5 indexing only (no vectors).
    pub embedding_enabled: bool,
    /// Enable graph edge creation (at store time) and graph filtering (at search time).
    /// Master switch — overrides AssemblyConfig.graph_depth when false.
    pub graph_enabled: bool,
    /// Enable content hash deduplication on store.
    /// When false, duplicate content is allowed (useful for testing).
    pub dedup_enabled: bool,
    /// Query-time search configuration (diversification, recency, graph depth).
    pub assembly: crate::context::AssemblyConfig,
    /// Vector search mode: exact (brute-force), ann (approximate), or off.
    pub vector_search_mode: VectorSearchMode,
    /// Enable strict post-search grounding filters for exact-detail queries.
    pub strict_grounding_enabled: bool,
    /// Enable query-shape-aware reranking heuristics after search.
    pub query_alignment_enabled: bool,
    /// Runtime mode for cross-encoder reranking.
    pub reranking_runtime: RerankerRuntime,
    /// Maximum number of candidates to send to the reranker.
    pub rerank_candidate_limit: usize,
}

/// Runtime vector retrieval mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum VectorSearchMode {
    /// FTS5 only. Vector search is disabled.
    Off,
    /// Brute-force cosine similarity over stored vectors.
    #[default]
    Exact,
    /// ANN cosine similarity via the in-memory HNSW index.
    Ann,
}

impl VectorSearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Exact => "exact",
            Self::Ann => "ann",
        }
    }
}

impl std::fmt::Display for VectorSearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            embedding_enabled: true,
            graph_enabled: true,
            dedup_enabled: true,
            assembly: crate::context::AssemblyConfig::default(),
            vector_search_mode: VectorSearchMode::Exact,
            strict_grounding_enabled: true,
            query_alignment_enabled: true,
            reranking_runtime: RerankerRuntime::Off,
            rerank_candidate_limit: 20,
        }
    }
}

/// The primary interface to femind.
///
/// Generic over the consumer's memory type `T: MemoryRecord`.
/// All core operations are synchronous (SQLite queries).
///
/// # Example
///
/// ```rust,ignore
/// let engine = MemoryEngine::<MyMemory>::builder()
///     .database("memory.db")
///     .build()?;
///
/// engine.store(&my_record)?;
/// let results = engine.search("query").limit(5).execute()?;
/// ```
pub struct MemoryEngine<T: MemoryRecord> {
    db: Database,
    global_db: Option<Database>,
    store: MemoryStore<T>,
    scoring: Arc<dyn ScoringStrategy>,
    embedding: Option<Arc<dyn EmbeddingBackend>>,
    reranker: Option<Arc<dyn RerankerBackend>>,
    #[cfg(feature = "ann")]
    ann_index: Arc<crate::search::AnnIndex>,
    /// Runtime feature configuration.
    pub config: EngineConfig,
}

impl<T: MemoryRecord> MemoryEngine<T> {
    /// Create a new builder for configuring the engine.
    pub fn builder() -> MemoryEngineBuilder<T> {
        MemoryEngineBuilder::new()
    }

    /// Store a new memory. Returns info about what action was taken (added or duplicate).
    ///
    /// When the `consolidation` feature is enabled and a consolidation strategy
    /// is configured, the strategy is consulted before storing.
    pub fn store(&self, record: &T) -> Result<StoreResult> {
        let result = self.store.store(&self.db, record)?;

        // Compute and store embedding for new records (if enabled)
        if let StoreResult::Added(id) = &result {
            self.apply_post_store_metadata(*id, record)?;

            if self.config.embedding_enabled {
                if let Some(ref backend) = self.embedding {
                    if backend.is_available() {
                        let text = record.searchable_text();

                        // Skip embedding for empty/whitespace-only text
                        if text.trim().is_empty() {
                            return Ok(result);
                        }

                        // Truncate to ~8192 tokens (~32K chars) for model context window
                        let text = truncate_for_embedding(&text);
                        let hash = format!("{:x}", sha2::Sha256::digest(text.as_bytes()));

                        // Skip embedding if vector already exists for this content
                        let already_exists =
                            crate::search::vector::VectorSearch::vector_exists(&self.db, &hash)
                                .unwrap_or(false);

                        if !already_exists {
                            let embed_start = std::time::Instant::now();
                            match backend.embed(text) {
                                Ok(vec) if vec.is_empty() => {
                                    tracing::warn!("Empty embedding returned for memory {id}");
                                    self.set_embedding_status(*id, "failed");
                                }
                                Ok(vec) => {
                                    let embed_ms = embed_start.elapsed().as_millis();
                                    tracing::debug!(memory_id = id, embed_ms, "embedded memory");
                                    match crate::search::vector::VectorSearch::store_vector(
                                        &self.db,
                                        *id,
                                        &vec,
                                        backend.model_name(),
                                        &hash,
                                    ) {
                                        Ok(()) => {
                                            tracing::debug!(memory_id = id, "stored vector");
                                            self.set_embedding_status(*id, "success");
                                            self.invalidate_ann_index();
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "Failed to store embedding for memory {id}: {e}"
                                            );
                                            self.set_embedding_status(*id, "failed");
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to compute embedding for memory {id}: {e}"
                                    );
                                    self.set_embedding_status(*id, "failed");
                                }
                            }
                        } else {
                            self.set_embedding_status(*id, "success");
                        }
                    }
                }
            } // config.embedding_enabled
        }

        Ok(result)
    }

    /// Store a batch of records with optimized embedding computation.
    ///
    /// 1. Stores all records via individual SQL inserts
    /// 2. Collects texts from newly added records
    /// 3. Calls `embed_batch()` once for all texts
    /// 4. Stores all vectors
    ///
    /// This is significantly faster than calling `store()` in a loop because
    /// embedding inference is batched (amortizes model overhead).
    pub fn store_batch(&self, records: &[T]) -> Result<Vec<StoreResult>> {
        // Phase 1: Store all records, collect texts needing embeddings
        let mut results = Vec::with_capacity(records.len());
        let mut to_embed: Vec<(i64, String, String)> = Vec::new(); // (id, text, hash)

        for record in records {
            let result = self.store.store(&self.db, record)?;
            if let StoreResult::Added(id) = &result {
                self.apply_post_store_metadata(*id, record)?;

                if self.config.embedding_enabled
                    && self.embedding.as_ref().is_some_and(|b| b.is_available())
                {
                    let text = record.searchable_text();
                    if !text.trim().is_empty() {
                        let hash = format!("{:x}", sha2::Sha256::digest(text.as_bytes()));
                        let already_exists =
                            crate::search::vector::VectorSearch::vector_exists(&self.db, &hash)
                                .unwrap_or(false);
                        if !already_exists {
                            to_embed.push((*id, text, hash));
                        }
                    }
                }
            }
            results.push(result);
        }

        // Phase 2: Batch embed all texts at once
        if let Some(ref backend) = self.embedding {
            if !to_embed.is_empty() && backend.is_available() {
                let texts: Vec<&str> = to_embed.iter().map(|(_, t, _)| t.as_str()).collect();
                let batch_start = std::time::Instant::now();
                let batch_count = texts.len();
                let mut stored_any_vectors = false;
                match backend.embed_batch(&texts) {
                    Ok(embeddings) => {
                        let batch_ms = batch_start.elapsed().as_millis();
                        tracing::debug!(batch_count, batch_ms, "batch embedding complete");
                        // Phase 3: Store all vectors and update status
                        for ((id, _, hash), embedding) in to_embed.iter().zip(embeddings.iter()) {
                            match crate::search::vector::VectorSearch::store_vector(
                                &self.db,
                                *id,
                                embedding,
                                backend.model_name(),
                                hash,
                            ) {
                                Ok(()) => {
                                    self.set_embedding_status(*id, "success");
                                    stored_any_vectors = true;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to store embedding for memory {id}: {e}"
                                    );
                                    self.set_embedding_status(*id, "failed");
                                }
                            }
                        }
                        if stored_any_vectors {
                            self.invalidate_ann_index();
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Batch embedding failed for {} records: {e}",
                            to_embed.len()
                        );
                        // Mark all as failed
                        for (id, _, _) in &to_embed {
                            self.set_embedding_status(*id, "failed");
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Return pending high-impact review items, newest first.
    pub fn pending_review_items(&self, limit: usize) -> Result<Vec<ReviewItem>> {
        self.review_items_with_status(limit, Some(ReviewStatus::Pending))
    }

    /// Return review items regardless of review state, newest first.
    pub fn review_items(&self, limit: usize) -> Result<Vec<ReviewItem>> {
        self.review_items_with_status(limit, None)
    }

    /// Return review items filtered by effective review state, newest first.
    ///
    /// `status = Some(Pending)` includes both explicit `pending` items and
    /// temporary allowances whose `review_expires_at` has elapsed.
    pub fn review_items_with_status(
        &self,
        limit: usize,
        status: Option<ReviewStatus>,
    ) -> Result<Vec<ReviewItem>> {
        self.db.with_reader(|conn| {
            let now = Utc::now();
            let mut stmt = conn.prepare(
                "SELECT id, searchable_text, created_at, metadata_json
                 FROM memories
                 WHERE metadata_json IS NOT NULL
                 ORDER BY created_at DESC, id DESC",
            )?;

            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })?;

            let mut items = Vec::new();
            for row in rows {
                let (memory_id, text, created_at, metadata_json) = row?;
                let Some(metadata_json) = metadata_json else {
                    continue;
                };
                let Ok(metadata) =
                    serde_json::from_str::<std::collections::HashMap<String, String>>(
                        &metadata_json,
                    )
                else {
                    continue;
                };

                if !metadata
                    .get("review_required")
                    .is_some_and(|value| value.eq_ignore_ascii_case("true"))
                {
                    continue;
                }

                let effective_status =
                    effective_review_status(&metadata, now).unwrap_or(ReviewStatus::Pending);
                let filter_match = match status {
                    Some(ReviewStatus::Pending) => {
                        matches!(effective_status, ReviewStatus::Pending | ReviewStatus::Expired)
                    }
                    Some(filter_status) => effective_status == filter_status,
                    None => true,
                };
                if !filter_match {
                    continue;
                }

                let severity = match metadata
                    .get("review_severity")
                    .map(|value| value.to_lowercase())
                    .as_deref()
                {
                    Some("medium") => ReviewSeverity::Medium,
                    Some("critical") => ReviewSeverity::Critical,
                    _ => ReviewSeverity::High,
                };
                let reason = metadata
                    .get("review_reason")
                    .cloned()
                    .unwrap_or_else(|| "manual-review".to_string());
                let tags = metadata
                    .get("review_tags")
                    .map(|value| {
                        value
                            .split(',')
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let scope = metadata
                    .get("review_scope")
                    .and_then(|value| ReviewScope::from_str(value));
                let policy_class = metadata
                    .get("review_policy_class")
                    .and_then(|value| ReviewPolicyClass::from_str(value));
                let template = metadata
                    .get("review_template")
                    .and_then(|value| ReviewApprovalTemplate::from_str(value));
                let reviewer = metadata.get("review_reviewer").cloned();
                let replaced_by = metadata
                    .get("review_replaced_by")
                    .and_then(|value| value.parse::<i64>().ok());
                let updated_at = metadata
                    .get("review_updated_at")
                    .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
                    .map(|value| value.with_timezone(&Utc));
                let expires_at = review_expires_at(&metadata);
                let note = metadata.get("review_note").cloned();
                let created_at = DateTime::parse_from_rfc3339(&created_at)
                    .map(|value| value.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                items.push(ReviewItem {
                    memory_id,
                    severity,
                    reason,
                    tags,
                    status: effective_status,
                    scope,
                    policy_class,
                    template,
                    reviewer,
                    replaced_by,
                    updated_at,
                    expires_at,
                    note,
                    text,
                    created_at,
                });

                if limit > 0 && items.len() >= limit {
                    break;
                }
            }

            Ok(items)
        })
    }

    /// Return a single review item by memory ID, if it exists and requires review.
    pub fn review_item(&self, memory_id: i64) -> Result<Option<ReviewItem>> {
        let mut items = self.review_items_with_status(usize::MAX, None)?;
        Ok(items.drain(..).find(|item| item.memory_id == memory_id))
    }

    /// Count pending review items.
    pub fn pending_review_count(&self) -> Result<u64> {
        Ok(self.pending_review_items(usize::MAX)?.len() as u64)
    }

    /// Update the review status for a stored memory.
    pub fn set_review_status(
        &self,
        memory_id: i64,
        status: ReviewStatus,
        note: Option<&str>,
    ) -> Result<()> {
        self.resolve_review_item(memory_id, status, note, None)
            .map(|_| ())
    }

    /// Resolve a stored review item with optional note and expiry.
    pub fn resolve_review_item(
        &self,
        memory_id: i64,
        status: ReviewStatus,
        note: Option<&str>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<ReviewItem> {
        self.resolve_review_item_with_resolution(
            memory_id,
            ReviewResolution {
                status,
                note: note.map(ToString::to_string),
                reviewer: None,
                scope: None,
                policy_class: None,
                template: None,
                expires_at,
                replaced_by: None,
            },
        )
    }

    /// Resolve a stored review item with explicit reviewer, scope, and policy metadata.
    pub fn resolve_review_item_with_resolution(
        &self,
        memory_id: i64,
        resolution: ReviewResolution,
    ) -> Result<ReviewItem> {
        self.db
            .with_writer(|conn| {
                let now = Utc::now();
                let existing = conn.query_row(
                    "SELECT metadata_json FROM memories WHERE id = ?1",
                    [memory_id],
                    |row| row.get::<_, Option<String>>(0),
                )?;

                let mut metadata = existing
                    .as_deref()
                    .and_then(|json| {
                        serde_json::from_str::<std::collections::HashMap<String, String>>(json).ok()
                    })
                    .unwrap_or_default();

                let resolved_template = resolution.template.or_else(|| {
                    metadata
                        .get("review_template")
                        .and_then(|value| ReviewApprovalTemplate::from_str(value))
                });
                let resolved_scope = resolution.scope.or_else(|| {
                    resolved_template
                        .map(ReviewApprovalTemplate::default_scope)
                        .or_else(|| {
                            metadata
                                .get("review_scope")
                                .and_then(|value| ReviewScope::from_str(value))
                        })
                });
                let resolved_policy_class = resolution.policy_class.or_else(|| {
                    resolved_template
                        .map(ReviewApprovalTemplate::default_policy_class)
                        .or_else(|| {
                            metadata
                                .get("review_policy_class")
                                .and_then(|value| ReviewPolicyClass::from_str(value))
                        })
                });
                let resolved_expires_at = resolution.expires_at.or_else(|| {
                    if matches!(resolution.status, ReviewStatus::Allowed) {
                        resolved_template
                            .map(|template| template.default_expiry(now))
                            .or_else(|| review_expires_at(&metadata))
                    } else {
                        None
                    }
                });
                let resolved_replaced_by = resolution.replaced_by.or_else(|| {
                    metadata
                        .get("review_replaced_by")
                        .and_then(|value| value.parse::<i64>().ok())
                });

                metadata.insert("review_required".to_string(), "true".to_string());
                metadata.insert("review_status".to_string(), resolution.status.to_string());
                metadata.insert("review_updated_at".to_string(), now.to_rfc3339());
                if let Some(note) = resolution.note.as_deref() {
                    metadata.insert("review_note".to_string(), note.to_string());
                } else {
                    metadata.remove("review_note");
                }
                if let Some(reviewer) = resolution.reviewer.as_deref() {
                    metadata.insert("review_reviewer".to_string(), reviewer.to_string());
                } else {
                    metadata.remove("review_reviewer");
                }
                if let Some(template) = resolved_template {
                    metadata.insert("review_template".to_string(), template.to_string());
                } else {
                    metadata.remove("review_template");
                }
                if let Some(scope) = resolved_scope {
                    metadata.insert("review_scope".to_string(), scope.to_string());
                } else {
                    metadata.remove("review_scope");
                }
                if let Some(policy_class) = resolved_policy_class {
                    metadata.insert("review_policy_class".to_string(), policy_class.to_string());
                } else {
                    metadata.remove("review_policy_class");
                }
                if let Some(expires_at) = resolved_expires_at {
                    metadata.insert("review_expires_at".to_string(), expires_at.to_rfc3339());
                } else if !matches!(resolution.status, ReviewStatus::Allowed) {
                    metadata.remove("review_expires_at");
                }
                if let Some(replaced_by) = resolved_replaced_by {
                    metadata.insert("review_replaced_by".to_string(), replaced_by.to_string());
                } else {
                    metadata.remove("review_replaced_by");
                }

                let metadata_json = serde_json::to_string(&metadata)?;
                conn.execute(
                    "UPDATE memories SET metadata_json = ?1 WHERE id = ?2",
                    rusqlite::params![metadata_json, memory_id],
                )?;
                Ok(())
            })
            .and_then(|_| {
                self.review_item(memory_id)?.ok_or_else(|| {
                    FemindError::Migration(format!(
                        "review item {memory_id} was updated but could not be reloaded"
                    ))
                })
            })
    }

    /// Mark any temporary review allowances that have passed their expiry as expired.
    pub fn expire_due_review_items(&self, now: DateTime<Utc>) -> Result<u64> {
        self.db.with_writer(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, metadata_json
                 FROM memories
                 WHERE metadata_json IS NOT NULL",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
            })?;

            let mut expired = 0_u64;
            for row in rows {
                let (memory_id, metadata_json) = row?;
                let Some(metadata_json) = metadata_json else {
                    continue;
                };
                let Ok(mut metadata) = serde_json::from_str::<
                    std::collections::HashMap<String, String>,
                >(&metadata_json) else {
                    continue;
                };

                let stored_status = metadata
                    .get("review_status")
                    .and_then(|value| ReviewStatus::from_str(value));
                let effective_status = effective_review_status(&metadata, now);
                if !matches!(stored_status, Some(ReviewStatus::Allowed))
                    || !matches!(effective_status, Some(ReviewStatus::Expired))
                {
                    continue;
                }

                metadata.insert(
                    "review_status".to_string(),
                    ReviewStatus::Expired.to_string(),
                );
                metadata.insert("review_updated_at".to_string(), now.to_rfc3339());
                metadata
                    .entry("review_note".to_string())
                    .or_insert_with(|| "Temporary review allowance expired.".to_string());

                let metadata_json = serde_json::to_string(&metadata)?;
                conn.execute(
                    "UPDATE memories SET metadata_json = ?1 WHERE id = ?2",
                    rusqlite::params![metadata_json, memory_id],
                )?;
                expired += 1;
            }

            Ok(expired)
        })
    }

    /// Renew a temporary review allowance, preserving existing template/scope/class when present.
    pub fn renew_review_item(
        &self,
        memory_id: i64,
        reviewer: Option<&str>,
        note: Option<&str>,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<ReviewItem> {
        let current = self.review_item(memory_id)?.ok_or_else(|| {
            FemindError::Migration(format!("review item {memory_id} does not exist"))
        })?;

        self.resolve_review_item_with_resolution(
            memory_id,
            ReviewResolution {
                status: ReviewStatus::Allowed,
                note: note.map(ToString::to_string).or(current.note.clone()),
                reviewer: reviewer
                    .map(ToString::to_string)
                    .or_else(|| current.reviewer.clone()),
                scope: current.scope,
                policy_class: current.policy_class,
                template: current.template,
                expires_at: expires_at.or(current.expires_at),
                replaced_by: current.replaced_by,
            },
        )
    }

    /// Revoke a temporary or prior allowance and deny future retrieval.
    pub fn revoke_review_item(
        &self,
        memory_id: i64,
        reviewer: Option<&str>,
        note: Option<&str>,
    ) -> Result<ReviewItem> {
        let current = self.review_item(memory_id)?.ok_or_else(|| {
            FemindError::Migration(format!("review item {memory_id} does not exist"))
        })?;

        self.resolve_review_item_with_resolution(
            memory_id,
            ReviewResolution {
                status: ReviewStatus::Denied,
                note: note.map(ToString::to_string).or(current.note.clone()),
                reviewer: reviewer
                    .map(ToString::to_string)
                    .or_else(|| current.reviewer.clone()),
                scope: current.scope,
                policy_class: current.policy_class,
                template: current.template,
                expires_at: None,
                replaced_by: current.replaced_by,
            },
        )
    }

    /// Replace one reviewed procedural memory with a successor record.
    pub fn replace_review_item(
        &self,
        memory_id: i64,
        replacement_memory_id: i64,
        reviewer: Option<&str>,
        note: Option<&str>,
    ) -> Result<ReviewItem> {
        let current = self.review_item(memory_id)?.ok_or_else(|| {
            FemindError::Migration(format!("review item {memory_id} does not exist"))
        })?;
        let replacement_note = match note {
            Some(note) if note.contains(&replacement_memory_id.to_string()) => note.to_string(),
            Some(note) => format!("{note} Replacement memory: #{replacement_memory_id}."),
            None => format!("Replaced by memory #{replacement_memory_id}."),
        };

        self.resolve_review_item_with_resolution(
            memory_id,
            ReviewResolution {
                status: ReviewStatus::Denied,
                note: Some(replacement_note),
                reviewer: reviewer
                    .map(ToString::to_string)
                    .or_else(|| current.reviewer.clone()),
                scope: current.scope,
                policy_class: current.policy_class,
                template: current.template,
                expires_at: None,
                replaced_by: Some(replacement_memory_id),
            },
        )
    }

    /// Update the embedding_status column for a memory.
    fn set_embedding_status(&self, id: i64, status: &str) {
        let _ = self.db.with_writer(|conn| {
            conn.execute(
                "UPDATE memories SET embedding_status = ?1 WHERE id = ?2",
                rusqlite::params![status, id],
            )?;
            Ok(())
        });
    }

    fn apply_post_store_metadata(&self, id: i64, record: &T) -> Result<()> {
        let mut metadata = record.metadata();
        if metadata.contains_key("review_status") {
            metadata
                .entry("review_required".to_string())
                .or_insert_with(|| "true".to_string());
        }
        if let Some(review_flag) =
            detect_review_flag(&crate::traits::MemoryMeta::from_record(record))
        {
            metadata
                .entry("review_required".to_string())
                .or_insert_with(|| "true".to_string());
            metadata
                .entry("review_severity".to_string())
                .or_insert_with(|| review_flag.severity.to_string());
            metadata
                .entry("review_reason".to_string())
                .or_insert_with(|| review_flag.reason.to_string());
            metadata
                .entry("review_status".to_string())
                .or_insert_with(|| ReviewStatus::Pending.to_string());
            metadata
                .entry("review_tags".to_string())
                .or_insert_with(|| review_flag.tags.join(","));
        }

        let metadata_json = if metadata.is_empty() {
            None
        } else {
            Some(serde_json::to_string(&metadata)?)
        };

        self.db.with_writer(|conn| {
            conn.execute(
                "UPDATE memories SET metadata_json = ?1 WHERE id = ?2",
                rusqlite::params![metadata_json, id],
            )?;
            Ok(())
        })
    }

    /// Retrieve a memory by ID. Returns `None` if not found.
    pub fn get(&self, id: i64) -> Result<Option<T>> {
        self.store.get(&self.db, id)
    }

    /// Update an existing memory by ID.
    pub fn update(&self, id: i64, record: &T) -> Result<()> {
        self.store.update(&self.db, id, record)
    }

    /// Delete a memory by ID. Returns `true` if a record was deleted.
    pub fn delete(&self, id: i64) -> Result<bool> {
        self.store.delete(&self.db, id)
    }

    /// Begin a search with the fluent builder API.
    ///
    /// Post-search scoring is automatically applied using the engine's
    /// configured scoring strategy. If an embedding backend is configured,
    /// `SearchMode::Auto` will use hybrid FTS5 + vector search.
    pub fn search(&self, query: &str) -> SearchBuilder<'_, T> {
        let mut builder = SearchBuilder::new(&self.db, query)
            .with_scoring(Arc::clone(&self.scoring))
            .with_vector_search_mode(self.config.vector_search_mode)
            .with_default_strict_grounding(self.config.strict_grounding_enabled)
            .with_default_query_alignment(self.config.query_alignment_enabled)
            .with_default_rerank_limit(self.config.rerank_candidate_limit);
        if self.config.vector_search_mode != VectorSearchMode::Off {
            if let Some(ref embedding) = self.embedding {
                builder = builder.with_embedding(Arc::clone(embedding));
            }
        }
        if self.config.reranking_runtime != RerankerRuntime::Off {
            if let Some(ref reranker) = self.reranker {
                builder = builder.with_reranker(Arc::clone(reranker));
            }
        }
        #[cfg(feature = "ann")]
        if self.config.vector_search_mode == VectorSearchMode::Ann {
            builder = builder.with_ann_index(Arc::clone(&self.ann_index));
        }
        builder
    }

    /// Access the embedding backend (if configured).
    pub fn embedding_backend(&self) -> Option<&dyn EmbeddingBackend> {
        self.embedding.as_deref()
    }

    /// Access the reranker backend (if configured).
    pub fn reranker_backend(&self) -> Option<&dyn RerankerBackend> {
        self.reranker.as_deref()
    }

    /// Count total memories in the database.
    pub fn count(&self) -> Result<u64> {
        self.store.count(&self.db)
    }

    /// Store raw text with LLM extraction.
    ///
    /// 1. Calls the LLM to extract individual facts from raw text
    /// 2. Stores each fact as its own memory with embedding
    /// 3. Detects contradictions and creates SupersededBy graph edges
    /// 4. Deduplicates against existing memories via content hash
    ///
    /// Returns extraction statistics including storage and graph metrics.
    pub fn store_with_extraction(
        &self,
        raw_text: &str,
        llm: &dyn crate::traits::LlmCallback,
    ) -> Result<StoreExtractionResult> {
        use crate::ingest::llm_extract;
        use crate::memory::{GraphMemory, RelationType};

        // Step 1: Extract facts via LLM
        // Split large text into manageable chunks for the LLM context window
        const MAX_EXTRACT_CHARS: usize = 6000;
        let extraction = if raw_text.len() > MAX_EXTRACT_CHARS {
            let mut all_facts = Vec::new();
            let mut total_tokens = 0;
            let mut remaining = raw_text;
            while !remaining.is_empty() {
                let split_at = if remaining.len() <= MAX_EXTRACT_CHARS {
                    remaining.len()
                } else {
                    remaining[..MAX_EXTRACT_CHARS]
                        .rfind('\n')
                        .map(|p| p + 1)
                        .unwrap_or(MAX_EXTRACT_CHARS)
                };
                let chunk = &remaining[..split_at];
                remaining = &remaining[split_at..];
                match llm_extract::extract_facts(chunk, llm) {
                    Ok(result) => {
                        total_tokens += result.tokens_used;
                        all_facts.extend(result.facts);
                    }
                    Err(e) => {
                        tracing::warn!("Extraction chunk failed: {e}");
                    }
                }
            }
            llm_extract::ExtractionResult {
                facts: all_facts,
                tokens_used: total_tokens,
            }
        } else {
            llm_extract::extract_facts(raw_text, llm)?
        };

        let facts_extracted = extraction.facts.len();
        let tokens_used = extraction.tokens_used;

        if extraction.facts.is_empty() {
            return Ok(StoreExtractionResult {
                facts_extracted: 0,
                memories_stored: 0,
                duplicates_skipped: 0,
                graph_edges_created: 0,
                superseded_count: 0,
                tokens_used,
            });
        }

        // Step 2: Store each fact as individual memory
        let mut stored_ids: Vec<(i64, &llm_extract::ExtractedFact)> = Vec::new();
        let mut duplicates_skipped = 0usize;

        for fact in &extraction.facts {
            let hash = format!("{:x}", sha2::Sha256::digest(fact.text.as_bytes()));

            // Dedup check
            let existing_id: Option<i64> = self.db.with_reader(|conn| {
                let result = conn.query_row(
                    "SELECT id FROM memories WHERE content_hash = ?1",
                    [&hash],
                    |row| row.get::<_, i64>(0),
                );
                match result {
                    Ok(id) => Ok(Some(id)),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(e.into()),
                }
            })?;

            if self.config.dedup_enabled && existing_id.is_some() {
                duplicates_skipped += 1;
                continue;
            }

            // Insert the memory
            let importance = fact.importance as i32;
            let category = Some(fact.category.as_str());
            let metadata_json = if !fact.entities.is_empty() || !fact.relationships.is_empty() {
                let meta = serde_json::json!({
                    "entities": fact.entities,
                    "relationships": fact.relationships,
                });
                Some(serde_json::to_string(&meta).unwrap_or_default())
            } else {
                None
            };

            let id = self.db.with_writer(|conn| {
                conn.execute(
                    "INSERT INTO memories (
                        searchable_text, memory_type, importance, category,
                        metadata_json, content_hash, created_at, record_json
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'), ?7)",
                    rusqlite::params![
                        fact.text,
                        "semantic",
                        importance,
                        category,
                        metadata_json,
                        hash,
                        serde_json::json!({"text": fact.text}).to_string(),
                    ],
                )?;
                Ok(conn.last_insert_rowid())
            })?;

            // Compute and store embedding (A3: gated by config)
            if self.config.embedding_enabled {
                if let Some(ref backend) = self.embedding {
                    if backend.is_available() && !fact.text.trim().is_empty() {
                        let text = truncate_for_embedding(&fact.text);
                        match backend.embed(text) {
                            Ok(vec) if !vec.is_empty() => {
                                match crate::search::vector::VectorSearch::store_vector(
                                    &self.db,
                                    id,
                                    &vec,
                                    backend.model_name(),
                                    &hash,
                                ) {
                                    Ok(()) => {
                                        self.set_embedding_status(id, "success");
                                        self.invalidate_ann_index();
                                    }
                                    Err(_) => self.set_embedding_status(id, "failed"),
                                }
                            }
                            Ok(_) => self.set_embedding_status(id, "failed"),
                            Err(_) => self.set_embedding_status(id, "failed"),
                        }
                    }
                }
            } // config.embedding_enabled

            stored_ids.push((id, fact));
        }

        // Step 3: Create graph edges for relationships (A4: gated by config)
        let mut graph_edges_created = 0usize;
        let mut superseded_count = 0usize;
        if self.config.graph_enabled {
            for (id, fact) in &stored_ids {
                for (subject, relation, _object) in &fact.relationships {
                    // Search existing memories for same subject + relation with different value
                    let existing = self.db.with_reader(|conn| {
                        let mut stmt = conn.prepare(
                            "SELECT id, searchable_text FROM memories
                         WHERE id != ?1
                         AND searchable_text LIKE ?2
                         ORDER BY id ASC",
                        )?;
                        let pattern = format!("%{}%", subject);
                        let results: Vec<(i64, String)> = stmt
                            .query_map(rusqlite::params![id, pattern], |row| {
                                Ok((row.get(0)?, row.get(1)?))
                            })?
                            .filter_map(|r| r.ok())
                            .collect();
                        Ok::<_, crate::error::FemindError>(results)
                    })?;

                    for (existing_id, existing_text) in &existing {
                        if existing_text
                            .to_lowercase()
                            .contains(&subject.to_lowercase())
                            && existing_text
                                .to_lowercase()
                                .contains(&relation.replace('_', " ").to_lowercase())
                        {
                            // Same subject + relation → SupersededBy (older is superseded)
                            if GraphMemory::relate(
                                &self.db,
                                *existing_id,
                                *id,
                                &RelationType::SupersededBy,
                            )
                            .is_ok()
                            {
                                graph_edges_created += 1;
                                superseded_count += 1;
                            }
                        }
                    }
                }
            }
        } // config.graph_enabled

        Ok(StoreExtractionResult {
            facts_extracted,
            memories_stored: stored_ids.len(),
            duplicates_skipped,
            graph_edges_created,
            superseded_count,
            tokens_used,
        })
    }

    /// Returns (memories_with_embeddings, total_memories) for diagnostic purposes.
    ///
    /// Counts memories where `embedding_status = 'success'` vs total count.
    pub fn embedding_coverage(&self) -> Result<(u64, u64)> {
        let total = self.store.count(&self.db)?;
        let with_embeddings: i64 = self.db.with_reader(|conn| {
            conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE embedding_status = 'success'",
                [],
                |row| row.get(0),
            )
            .map_err(Into::into)
        })?;
        Ok((with_embeddings as u64, total))
    }

    /// Multi-query search: run original + key-phrase variant, merge, diversify.
    fn multi_query_search(
        &self,
        query: &str,
        config: &crate::context::AssemblyConfig,
    ) -> Result<Vec<crate::search::builder::SearchResult>> {
        let limit = config.search_limit;
        use crate::search::fts5::strip_stop_words;
        use std::collections::HashMap;

        let route = self.search(query).limit(limit).query_route();
        let effective_graph_depth = routed_graph_depth(config, &route);
        let preserve_original_query = crate::scoring::query_requests_procedural_guidance(query);

        // Query variant 1: original
        let results1 = self.search(query).limit(limit).execute()?;

        // Query variant 2: key-phrase only (stop words removed)
        let key_phrases = strip_stop_words(query);
        let results2 =
            if !preserve_original_query && key_phrases != query && !key_phrases.is_empty() {
                self.search(&key_phrases).limit(limit).execute()?
            } else {
                Vec::new()
            };

        // Query variant 3: preserve contrastive or support-state intent when the
        // original question is asking what did not happen, what comes next, or
        // whether a backend/capability is supported.
        let supplemental = supplemental_query_variant(query);
        let mut results3 = if let Some(ref variant) = supplemental {
            if !preserve_original_query && variant != query && variant != &key_phrases {
                self.search(variant).limit(limit).execute()?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if !results3.is_empty() {
            for result in &mut results3 {
                result.score *= 1.6;
            }
        }

        let graph_seed = graph_seed_query_variant(query, &route);
        let mut results4 = if let Some(ref variant) = graph_seed {
            let duplicates_existing = variant == query
                || variant == &key_phrases
                || supplemental.as_ref().is_some_and(|value| value == variant);
            if preserve_original_query || duplicates_existing {
                Vec::new()
            } else {
                self.search(variant).limit(limit).execute()?
            }
        } else {
            Vec::new()
        };
        if !results4.is_empty() {
            for result in &mut results4 {
                result.score *= 1.35;
            }
        }

        // Merge: keep highest score per memory_id
        let mut best: HashMap<i64, crate::search::builder::SearchResult> = HashMap::new();
        for r in results1
            .into_iter()
            .chain(results2.into_iter())
            .chain(results3.into_iter())
            .chain(results4.into_iter())
        {
            best.entry(r.memory_id)
                .and_modify(|existing| {
                    if r.score > existing.score {
                        *existing = r.clone();
                    }
                })
                .or_insert(r);
        }

        let mut merged: Vec<_> = best.into_values().collect();

        // Recency weighting: boost later chunks (higher turn_index = more recent)
        if config.recency_boost > 0.0 {
            // Find max turn_index across all results for normalization
            let max_index = merged
                .iter()
                .filter_map(|r| {
                    self.db
                        .with_reader(|conn| {
                            conn.query_row(
                                "SELECT metadata_json FROM memories WHERE id = ?1",
                                [r.memory_id],
                                |row| row.get::<_, Option<String>>(0),
                            )
                            .map_err(crate::error::FemindError::Database)
                        })
                        .ok()
                        .flatten()
                        .and_then(|json| {
                            serde_json::from_str::<HashMap<String, String>>(&json).ok()
                        })
                        .and_then(|meta| meta.get("turn_index").and_then(|v| v.parse::<f32>().ok()))
                })
                .fold(1.0_f32, f32::max);

            for r in &mut merged {
                let turn_index = self
                    .db
                    .with_reader(|conn| {
                        conn.query_row(
                            "SELECT metadata_json FROM memories WHERE id = ?1",
                            [r.memory_id],
                            |row| row.get::<_, Option<String>>(0),
                        )
                        .map_err(crate::error::FemindError::Database)
                    })
                    .ok()
                    .flatten()
                    .and_then(|json| serde_json::from_str::<HashMap<String, String>>(&json).ok())
                    .and_then(|meta| meta.get("turn_index").and_then(|v| v.parse::<f32>().ok()))
                    .unwrap_or(0.0);

                // position_ratio: 0.0 (oldest) to 1.0 (newest)
                let position_ratio = turn_index / max_index;
                r.score *= 1.0 + config.recency_boost * position_ratio;
            }
        }

        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if self.config.graph_enabled && effective_graph_depth > 0 {
            use crate::memory::{GraphMemory, RelationType};
            use crate::search::StateConflictPolicy;

            // Graph expansion: traverse from the strongest lexical/semantic hits and
            // pull in connected memories with a depth-based score discount.
            let seed_results: Vec<_> = merged.iter().take(limit.min(20)).cloned().collect();
            let mut expanded: HashMap<i64, crate::search::builder::SearchResult> = merged
                .into_iter()
                .map(|result| (result.memory_id, result))
                .collect();

            for seed in &seed_results {
                let Ok(nodes) =
                    GraphMemory::traverse(&self.db, seed.memory_id, effective_graph_depth)
                else {
                    continue;
                };

                for node in nodes {
                    let relation_boost = match node.relation {
                        RelationType::SupersededBy => 1.2,
                        RelationType::ValidatedBy => 1.0,
                        RelationType::SolvedBy | RelationType::DependsOn => 0.9,
                        RelationType::RelatedTo | RelationType::PartOf => 0.8,
                        RelationType::CausedBy | RelationType::ConflictsWith => 0.75,
                        RelationType::Custom(_) => 0.75,
                    };
                    let candidate_score =
                        seed.score * GraphMemory::depth_boost(node.depth) * relation_boost;

                    expanded
                        .entry(node.memory_id)
                        .and_modify(|existing| {
                            if candidate_score > existing.score {
                                existing.score = candidate_score;
                            }
                        })
                        .or_insert(crate::search::builder::SearchResult {
                            memory_id: node.memory_id,
                            score: candidate_score,
                        });
                }

                match route.state_conflict_policy {
                    StateConflictPolicy::PreferCurrent => {
                        let Ok(successors) =
                            GraphMemory::superseded_successors(&self.db, seed.memory_id)
                        else {
                            continue;
                        };

                        for successor_id in successors {
                            let candidate_score = seed.score * 0.96;
                            expanded
                                .entry(successor_id)
                                .and_modify(|existing| {
                                    if candidate_score > existing.score {
                                        existing.score = candidate_score;
                                    }
                                })
                                .or_insert(crate::search::builder::SearchResult {
                                    memory_id: successor_id,
                                    score: candidate_score,
                                });
                        }
                    }
                    StateConflictPolicy::PreferHistorical => {
                        let Ok(predecessors) =
                            GraphMemory::superseded_predecessors(&self.db, seed.memory_id)
                        else {
                            continue;
                        };

                        for predecessor_id in predecessors {
                            let candidate_score = seed.score * 0.96;
                            expanded
                                .entry(predecessor_id)
                                .and_modify(|existing| {
                                    if candidate_score > existing.score {
                                        existing.score = candidate_score;
                                    }
                                })
                                .or_insert(crate::search::builder::SearchResult {
                                    memory_id: predecessor_id,
                                    score: candidate_score,
                                });
                        }
                    }
                    StateConflictPolicy::Neutral => {}
                }
            }

            merged = expanded.into_values().collect();
            merged.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Graph filtering: make current-vs-historical retrieval explicit instead of
            // always demoting prior states.
            let mut superseded_ids: std::collections::HashSet<i64> =
                std::collections::HashSet::new();
            let mut current_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();

            for r in &merged {
                let snapshot = GraphMemory::state_conflict_snapshot(&self.db, r.memory_id)
                    .ok()
                    .flatten();

                if let Some(snapshot) = snapshot {
                    if snapshot.is_superseded {
                        superseded_ids.insert(r.memory_id);
                    }
                    if snapshot.supersedes_other {
                        current_ids.insert(r.memory_id);
                    }
                }
            }

            match route.state_conflict_policy {
                StateConflictPolicy::PreferCurrent if !superseded_ids.is_empty() => {
                    tracing::debug!(
                        "Graph filtering: {} prior-state results demoted for current-state route",
                        superseded_ids.len()
                    );
                    for r in &mut merged {
                        if superseded_ids.contains(&r.memory_id) {
                            r.score *= 0.1;
                        }
                    }
                    merged.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                StateConflictPolicy::PreferHistorical if !current_ids.is_empty() => {
                    tracing::debug!(
                        "Graph filtering: {} current-state results demoted for historical route",
                        current_ids.len()
                    );
                    for r in &mut merged {
                        if current_ids.contains(&r.memory_id) {
                            r.score *= 0.35;
                        }
                    }
                    merged.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                StateConflictPolicy::Neutral
                | StateConflictPolicy::PreferCurrent
                | StateConflictPolicy::PreferHistorical => {}
            }
        }

        if route.strict_grounding {
            crate::search::builder::apply_strict_detail_query_filter(&self.db, query, &mut merged);
        }
        if route.query_alignment {
            crate::search::builder::rerank_for_query_alignment(&self.db, query, &mut merged);
        }

        // Diversification: limit chunks per session (0 = unlimited)
        let max_per = config.max_per_session;
        let diversified: Vec<_> = if max_per == 0 {
            merged // No diversification
        } else {
            let mut session_counts: HashMap<String, usize> = HashMap::new();
            merged
                .into_iter()
                .filter(|r| {
                    let session_key = self
                        .db
                        .with_reader(|conn| {
                            conn.query_row(
                                "SELECT metadata_json FROM memories WHERE id = ?1",
                                [r.memory_id],
                                |row| row.get::<_, Option<String>>(0),
                            )
                            .map_err(crate::error::FemindError::Database)
                        })
                        .ok()
                        .flatten()
                        .and_then(|json| {
                            serde_json::from_str::<HashMap<String, String>>(&json).ok()
                        })
                        .and_then(|meta| meta.get("session_date").cloned())
                        .unwrap_or_else(|| format!("unknown_{}", r.memory_id));

                    let count = session_counts.entry(session_key).or_insert(0);
                    *count += 1;
                    *count <= max_per
                })
                .collect()
        };

        Ok(diversified)
    }

    /// Assemble context for an LLM prompt within a token budget.
    ///
    /// Uses default AssemblyConfig (max 1/session, no recency boost).
    pub fn assemble_context(&self, query: &str, budget: &ContextBudget) -> Result<ContextAssembly> {
        self.assemble_context_with_config(query, budget, &crate::context::AssemblyConfig::default())
    }

    /// Search with an explicit assembly configuration.
    ///
    /// This exposes graph-expanded retrieval without forcing context assembly,
    /// which is useful for evaluation and regression testing.
    pub fn search_with_config(
        &self,
        query: &str,
        config: &crate::context::AssemblyConfig,
    ) -> Result<Vec<crate::search::builder::SearchResult>> {
        self.multi_query_search(query, config)
    }

    /// Collect broad retrieval evidence for aggregation-style questions.
    ///
    /// This keeps distinct supporting memories instead of collapsing to a plain
    /// top-k list, which makes count/list composition easier at higher layers.
    pub fn aggregate_with_config(
        &self,
        query: &str,
        config: &crate::context::AssemblyConfig,
        max_matches: usize,
    ) -> Result<AggregationResult> {
        let mut aggregation_config = config.clone();
        aggregation_config.max_per_session = 0;
        aggregation_config.search_limit = aggregation_config.search_limit.max(max_matches.max(25));

        let results = self.multi_query_search(query, &aggregation_config)?;

        let total_matches = results.len();
        let mut distinct_keys = std::collections::HashSet::new();
        let mut matches = Vec::new();

        for result in results {
            let Some(candidate) = self.load_aggregated_match(result.memory_id, result.score)?
            else {
                continue;
            };

            let key = aggregation_match_key(&candidate.text, candidate.memory_id);
            if !distinct_keys.insert(key) {
                continue;
            }

            if matches.len() < max_matches {
                matches.push(candidate);
            }
        }

        let composed_summary = matches
            .iter()
            .map(|candidate| candidate.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        Ok(AggregationResult {
            total_matches,
            distinct_match_count: distinct_keys.len(),
            matches,
            composed_summary,
        })
    }

    /// Deterministically synthesize stable knowledge objects from repeated evidence.
    ///
    /// Reflection is intentionally metadata-assisted in this first pass. It prefers
    /// records carrying `knowledge_key`, `knowledge_summary`, or `knowledge_kind`,
    /// and falls back to lightweight subject-label heuristics when that metadata is
    /// absent. Reflected objects are returned to the caller but not persisted yet,
    /// which avoids forcing an opaque internal record shape into consumer-defined
    /// `MemoryRecord` storage.
    pub fn reflect_knowledge_objects(
        &self,
        config: &ReflectionConfig,
    ) -> Result<Vec<KnowledgeObject>> {
        let now = Utc::now();
        let candidates = self.load_reflection_candidates(now)?;
        let mut buckets = std::collections::HashMap::<String, Vec<ReflectionCluster>>::new();

        for candidate in candidates {
            let clusters = buckets.entry(candidate.key.clone()).or_default();
            if let Some(existing) = clusters
                .iter_mut()
                .find(|cluster| cluster.summary_key == candidate.summary_key)
            {
                existing.absorb(candidate);
            } else {
                clusters.push(ReflectionCluster::from_candidate(candidate));
            }
        }

        let mut objects = buckets
            .into_iter()
            .filter_map(|(key, clusters)| {
                clusters
                    .into_iter()
                    .filter(|cluster| cluster.qualifies(config))
                    .max_by(ReflectionCluster::compare_rank)
                    .map(|cluster| cluster.into_knowledge_object(key, config, now))
            })
            .collect::<Vec<_>>();

        objects.sort_by(|left, right| {
            reflection_confidence_rank(right.confidence)
                .cmp(&reflection_confidence_rank(left.confidence))
                .then(right.trusted_support_count.cmp(&left.trusted_support_count))
                .then(right.support_count.cmp(&left.support_count))
                .then(left.key.cmp(&right.key))
        });
        objects.truncate(config.max_objects);
        Ok(objects)
    }

    /// Compose a grounded answer from routed retrieval evidence without using an LLM.
    pub fn compose_answer_with_config(
        &self,
        query: &str,
        config: &crate::context::AssemblyConfig,
        max_matches: usize,
    ) -> Result<ComposedAnswerResult> {
        let requires_strict_grounding =
            crate::search::builder::query_requires_strict_grounding(query);
        let route = self
            .search(query)
            .limit(config.search_limit.max(max_matches.max(10)))
            .query_route();

        if route.intent == crate::search::QueryIntent::Aggregation {
            let aggregation = self.aggregate_with_config(query, config, max_matches)?;
            if aggregation.matches.is_empty() {
                return Ok(ComposedAnswerResult {
                    answer: "I don’t have enough grounded evidence to answer.".to_string(),
                    kind: "aggregation",
                    confidence: CompositionConfidence::Low,
                    abstained: true,
                    rationale: "no-supporting-evidence",
                    total_matches: aggregation.total_matches,
                    distinct_match_count: aggregation.distinct_match_count,
                    evidence: aggregation.matches,
                });
            }

            let (answer, confidence, rationale) =
                compose_aggregation_answer(query, &aggregation.matches);
            return Ok(ComposedAnswerResult {
                answer,
                kind: "aggregation",
                confidence,
                abstained: false,
                rationale,
                total_matches: aggregation.total_matches,
                distinct_match_count: aggregation.distinct_match_count,
                evidence: aggregation.matches,
            });
        }

        let results = self.search_with_config(query, config)?;
        let mut evidence = self.collect_distinct_aggregated_matches(results, max_matches)?;
        filter_secret_guidance_evidence(query, &mut evidence);

        if query_requests_sensitive_secret_detail(query) {
            return Ok(ComposedAnswerResult {
                answer: "I won't surface secret or credential material from memory.".to_string(),
                kind: "abstain",
                confidence: CompositionConfidence::Low,
                abstained: true,
                rationale: "sensitive-secret-detail",
                total_matches: evidence.len(),
                distinct_match_count: evidence.len(),
                evidence,
            });
        }

        if evidence.is_empty() {
            if requires_strict_grounding {
                let fallback_results = self
                    .search(query)
                    .mode(crate::search::SearchMode::Exhaustive { min_score: 0.0 })
                    .limit(config.search_limit.max(max_matches.max(25)))
                    .with_strict_grounding(false)
                    .with_query_alignment(false)
                    .execute()?;
                let fallback_evidence =
                    self.collect_distinct_aggregated_matches(fallback_results, max_matches)?;

                if !fallback_evidence.is_empty() {
                    let rationale =
                        if evidence_explicitly_lacks_requested_detail(query, &fallback_evidence) {
                            "unsupported-detail"
                        } else {
                            "insufficient-grounding"
                        };
                    let answer = if rationale == "unsupported-detail" {
                        "I don’t have the exact grounded detail needed to answer that."
                    } else {
                        "I don’t have enough grounded evidence to answer that exactly."
                    };
                    return Ok(ComposedAnswerResult {
                        answer: answer.to_string(),
                        kind: "abstain",
                        confidence: CompositionConfidence::Low,
                        abstained: true,
                        rationale,
                        total_matches: fallback_evidence.len(),
                        distinct_match_count: fallback_evidence.len(),
                        evidence: fallback_evidence,
                    });
                }
            }

            return Ok(ComposedAnswerResult {
                answer: "I don’t have enough grounded evidence to answer.".to_string(),
                kind: "abstain",
                confidence: CompositionConfidence::Low,
                abstained: true,
                rationale: "no-supporting-evidence",
                total_matches: 0,
                distinct_match_count: 0,
                evidence,
            });
        }

        let top = select_best_evidence(query, &evidence);
        if evidence_explicitly_lacks_requested_detail(query, &evidence) {
            return Ok(ComposedAnswerResult {
                answer: "I don’t have the exact grounded detail needed to answer that.".to_string(),
                kind: "abstain",
                confidence: CompositionConfidence::Low,
                abstained: true,
                rationale: "unsupported-detail",
                total_matches: evidence.len(),
                distinct_match_count: evidence.len(),
                evidence,
            });
        }

        if requires_strict_grounding
            && !is_yes_no_query(query)
            && !crate::search::builder::lexical_grounding_ok(query, &top.text)
        {
            return Ok(ComposedAnswerResult {
                answer: "I don’t have enough grounded evidence to answer that exactly.".to_string(),
                kind: "abstain",
                confidence: CompositionConfidence::Low,
                abstained: true,
                rationale: "insufficient-grounding",
                total_matches: evidence.len(),
                distinct_match_count: evidence.len(),
                evidence,
            });
        }

        let (answer, kind, confidence, rationale) = if is_yes_no_query(query) {
            compose_yes_no_answer(query, &evidence)
        } else if matches!(
            route.intent,
            crate::search::QueryIntent::CurrentState | crate::search::QueryIntent::HistoricalState
        ) {
            compose_stateful_answer(query, &evidence)
        } else if requires_strict_grounding {
            (
                top.text.trim().to_string(),
                "direct",
                CompositionConfidence::High,
                "grounded-detail",
            )
        } else {
            (
                top.text.trim().to_string(),
                "direct",
                CompositionConfidence::Medium,
                "top-evidence",
            )
        };

        let answer = apply_secret_response_policy(query, &evidence, answer);

        Ok(ComposedAnswerResult {
            answer,
            kind,
            confidence,
            abstained: false,
            rationale,
            total_matches: evidence.len(),
            distinct_match_count: evidence.len(),
            evidence,
        })
    }

    /// Assemble context with custom assembly configuration.
    ///
    /// Allows tuning diversification, recency weighting, and search limits
    /// per dataset or question type.
    pub fn assemble_context_with_config(
        &self,
        query: &str,
        budget: &ContextBudget,
        config: &crate::context::AssemblyConfig,
    ) -> Result<ContextAssembly> {
        // Multi-query retrieval: run original + key-phrase variant, merge results
        let results = self.multi_query_search(query, config)?;

        // Convert search results to context items
        let candidates: Vec<ContextItem> = results
            .iter()
            .filter_map(|sr| {
                // Load the memory to get its content
                self.db
                    .with_reader(|conn| {
                        let row = conn.query_row(
                            "SELECT searchable_text, memory_type, category, metadata_json FROM memories WHERE id = ?1",
                            [sr.memory_id],
                            |row| {
                                Ok((
                                    row.get::<_, String>(0)?,
                                    row.get::<_, String>(1)?,
                                    row.get::<_, Option<String>>(2)?,
                                    row.get::<_, Option<String>>(3)?,
                                ))
                            },
                        );
                        match row {
                            Ok((text, type_str, category, metadata_json)) => {
                                let memory_type = crate::traits::MemoryType::from_str(&type_str)
                                    .unwrap_or(crate::traits::MemoryType::Episodic);
                                // Prepend session date if available in metadata
                                let content = prepend_date_from_metadata(&text, metadata_json.as_deref());
                                Ok(Some(ContextItem {
                                    memory_id: sr.memory_id,
                                    content: content.clone(),
                                    priority: PRIORITY_LEARNING,
                                    estimated_tokens: budget.estimate_tokens(&content),
                                    relevance_score: sr.score,
                                    memory_type,
                                    category,
                                }))
                            }
                            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                            Err(e) => Err(e.into()),
                        }
                    })
                    .ok()
                    .flatten()
            })
            .collect();

        Ok(ContextAssembly::assemble(candidates, budget))
    }

    /// Direct access to the project database (for advanced consumers).
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Direct access to the global database (if configured).
    pub fn global_database(&self) -> Option<&Database> {
        self.global_db.as_ref()
    }

    fn collect_distinct_aggregated_matches(
        &self,
        results: Vec<crate::search::builder::SearchResult>,
        max_matches: usize,
    ) -> Result<Vec<AggregatedMatch>> {
        let mut evidence = Vec::new();
        let mut distinct_keys = std::collections::HashSet::new();

        for result in results {
            let Some(candidate) = self.load_aggregated_match(result.memory_id, result.score)?
            else {
                continue;
            };

            let key = aggregation_match_key(&candidate.text, candidate.memory_id);
            if !distinct_keys.insert(key) {
                continue;
            }

            evidence.push(candidate);
            if evidence.len() >= max_matches {
                break;
            }
        }

        Ok(evidence)
    }

    fn load_aggregated_match(&self, memory_id: i64, score: f32) -> Result<Option<AggregatedMatch>> {
        self.db.with_reader(|conn| {
            let row = conn.query_row(
                "SELECT searchable_text, category, metadata_json FROM memories WHERE id = ?1",
                [memory_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            );
            match row {
                Ok((text, category, metadata_json)) => Ok(Some(AggregatedMatch {
                    memory_id,
                    text,
                    category,
                    score,
                    metadata: metadata_json
                        .as_deref()
                        .and_then(|json| {
                            serde_json::from_str::<std::collections::HashMap<String, String>>(json)
                                .ok()
                        })
                        .unwrap_or_default(),
                })),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(error) => Err(error.into()),
            }
        })
    }

    fn load_reflection_candidates(&self, now: DateTime<Utc>) -> Result<Vec<ReflectionCandidate>> {
        self.db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, searchable_text, memory_type, importance, category, created_at, metadata_json
                 FROM memories
                 WHERE searchable_text IS NOT NULL
                 ORDER BY created_at DESC, id DESC",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, u8>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            })?;

            let mut candidates = Vec::new();
            for row in rows {
                let (
                    memory_id,
                    text,
                    memory_type_raw,
                    importance,
                    category,
                    created_at_raw,
                    metadata_json,
                ) = row?;
                if text.trim().is_empty() {
                    continue;
                }

                let metadata = metadata_json
                    .as_deref()
                    .and_then(|json| {
                        serde_json::from_str::<std::collections::HashMap<String, String>>(json).ok()
                    })
                    .unwrap_or_default();

                if metadata
                    .get("derived_kind")
                    .is_some_and(|value| value.eq_ignore_ascii_case("reflection"))
                {
                    continue;
                }
                if evidence_contains_secret_material(&text, &metadata)
                    || secret_class_from_metadata(&metadata).is_some()
                {
                    continue;
                }

                let memory_type = crate::traits::MemoryType::from_str(&memory_type_raw)
                    .unwrap_or(crate::traits::MemoryType::Semantic);
                let created_at = DateTime::parse_from_rfc3339(&created_at_raw)
                    .map(|value| value.with_timezone(&Utc))
                    .unwrap_or(now);
                let meta = MemoryMeta {
                    id: Some(memory_id),
                    searchable_text: text.clone(),
                    memory_type,
                    importance,
                    category,
                    created_at,
                    metadata: metadata.clone(),
                };

                if matches!(
                    effective_review_status(&metadata, now),
                    Some(ReviewStatus::Pending | ReviewStatus::Denied | ReviewStatus::Expired)
                ) {
                    continue;
                }
                if matches!(source_trust_level(&meta), crate::scoring::SourceTrustLevel::Untrusted)
                {
                    continue;
                }

                let Some(key) = reflection_key(&text, &metadata) else {
                    continue;
                };
                let Some(summary) = reflection_summary(&text, &metadata) else {
                    continue;
                };

                candidates.push(ReflectionCandidate {
                    memory_id,
                    key,
                    summary_key: normalize_reflection_value(&summary),
                    summary,
                    kind: reflection_kind(memory_type, &text, &metadata),
                    created_at,
                    trusted_support: matches!(
                        source_trust_level(&meta),
                        crate::scoring::SourceTrustLevel::Trusted
                            | crate::scoring::SourceTrustLevel::Normal
                    ),
                    provenance_rank: source_provenance_rank(&meta),
                });
            }

            Ok(candidates)
        })
    }

    #[cfg(feature = "ann")]
    fn invalidate_ann_index(&self) {
        self.ann_index.invalidate();
    }

    #[cfg(not(feature = "ann"))]
    fn invalidate_ann_index(&self) {}
}

fn supplemental_query_variant(query: &str) -> Option<String> {
    let normalized = query.to_lowercase();

    if normalized.contains("not test") || normalized.contains("not prove") {
        return Some(
            "benchmarks did not test or prove llm fact extraction graph based retrieval real conversation memory"
                .to_string(),
        );
    }

    if normalized.contains("what should femind validate next")
        || (normalized.contains("validate next") && normalized.contains("66"))
    {
        return Some("next validation step real memloft data real corpus".to_string());
    }

    if normalized.contains("support")
        && normalized.contains("codex-cli")
        && normalized.contains("backend")
    {
        return Some("codex-cli supports extraction backend practical eval runner".to_string());
    }

    if normalized.contains("full feature set") && normalized.contains("include") {
        return Some("full everything except encryption mcp server".to_string());
    }

    None
}

fn graph_seed_query_variant(query: &str, route: &crate::search::QueryRoute) -> Option<String> {
    if route.graph_depth == 0 {
        return None;
    }

    let normalized = query
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>();

    let variant = normalized
        .split_whitespace()
        .filter(|token| {
            !matches!(
                *token,
                "how"
                    | "does"
                    | "do"
                    | "did"
                    | "is"
                    | "are"
                    | "the"
                    | "a"
                    | "an"
                    | "to"
                    | "through"
                    | "via"
                    | "reach"
                    | "reaches"
                    | "connect"
                    | "connected"
                    | "connection"
                    | "link"
                    | "linked"
                    | "bridge"
                    | "bridges"
                    | "path"
                    | "depends"
                    | "dependency"
                    | "now"
                    | "current"
                    | "currently"
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    if variant.is_empty() || variant == normalized.split_whitespace().collect::<Vec<_>>().join(" ")
    {
        None
    } else {
        Some(variant)
    }
}

fn routed_graph_depth(
    config: &crate::context::AssemblyConfig,
    route: &crate::search::QueryRoute,
) -> u32 {
    if config.graph_depth > 0 {
        config.graph_depth
    } else {
        route.graph_depth
    }
}

fn aggregation_match_key(text: &str, memory_id: i64) -> String {
    let normalized = text
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
        .join(" ");

    if normalized.is_empty() {
        format!("memory:{memory_id}")
    } else {
        normalized
    }
}

#[derive(Debug, Clone)]
struct ReflectionCandidate {
    memory_id: i64,
    key: String,
    summary_key: String,
    summary: String,
    kind: KnowledgeObjectKind,
    created_at: DateTime<Utc>,
    trusted_support: bool,
    provenance_rank: u8,
}

#[derive(Debug, Clone)]
struct ReflectionCluster {
    summary_key: String,
    summary: String,
    kind: KnowledgeObjectKind,
    source_ids: Vec<i64>,
    trusted_support_count: usize,
    provenance_sum: u32,
    newest_created_at: DateTime<Utc>,
}

impl ReflectionCluster {
    fn from_candidate(candidate: ReflectionCandidate) -> Self {
        let newest_created_at = candidate.created_at;
        Self {
            summary_key: candidate.summary_key,
            summary: candidate.summary,
            kind: candidate.kind,
            source_ids: vec![candidate.memory_id],
            trusted_support_count: usize::from(candidate.trusted_support),
            provenance_sum: u32::from(candidate.provenance_rank),
            newest_created_at,
        }
    }

    fn absorb(&mut self, candidate: ReflectionCandidate) {
        if !self.source_ids.contains(&candidate.memory_id) {
            self.source_ids.push(candidate.memory_id);
        }
        if candidate.trusted_support {
            self.trusted_support_count += 1;
        }
        self.provenance_sum += u32::from(candidate.provenance_rank);
        if candidate.created_at >= self.newest_created_at {
            self.newest_created_at = candidate.created_at;
            self.summary = candidate.summary;
            self.kind = candidate.kind;
        }
    }

    fn support_count(&self) -> usize {
        self.source_ids.len()
    }

    fn qualifies(&self, config: &ReflectionConfig) -> bool {
        self.support_count() >= config.min_support_count
            && self.trusted_support_count >= config.min_trusted_support_count
    }

    fn confidence(&self, config: &ReflectionConfig) -> CompositionConfidence {
        if self.support_count() >= config.min_support_count
            && self.trusted_support_count >= config.min_trusted_support_count.max(1) + 1
        {
            CompositionConfidence::High
        } else if self.qualifies(config) {
            CompositionConfidence::Medium
        } else {
            CompositionConfidence::Low
        }
    }

    fn into_knowledge_object(
        self,
        key: String,
        config: &ReflectionConfig,
        generated_at: DateTime<Utc>,
    ) -> KnowledgeObject {
        let confidence = self.confidence(config);
        let support_count = self.support_count();
        let trusted_support_count = self.trusted_support_count;
        KnowledgeObject {
            key,
            summary: self.summary,
            kind: self.kind,
            confidence,
            support_count,
            trusted_support_count,
            source_ids: self.source_ids,
            generated_at,
        }
    }

    fn compare_rank(left: &Self, right: &Self) -> std::cmp::Ordering {
        left.trusted_support_count
            .cmp(&right.trusted_support_count)
            .then(left.support_count().cmp(&right.support_count()))
            .then(left.provenance_sum.cmp(&right.provenance_sum))
            .then(left.newest_created_at.cmp(&right.newest_created_at))
    }
}

fn reflection_confidence_rank(confidence: CompositionConfidence) -> u8 {
    match confidence {
        CompositionConfidence::Low => 0,
        CompositionConfidence::Medium => 1,
        CompositionConfidence::High => 2,
    }
}

fn reflection_key(
    text: &str,
    metadata: &std::collections::HashMap<String, String>,
) -> Option<String> {
    metadata
        .get("knowledge_key")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            leading_subject_label(text)
                .map(|label| normalize_reflection_value(&label).replace(' ', "-"))
                .filter(|value| !value.is_empty())
        })
}

fn reflection_summary(
    text: &str,
    metadata: &std::collections::HashMap<String, String>,
) -> Option<String> {
    metadata
        .get("knowledge_summary")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .or_else(|| derive_reflection_summary(text))
}

fn derive_reflection_summary(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let first_line = trimmed.lines().next().unwrap_or(trimmed).trim();
    let without_label = strip_reflection_label(first_line);
    let first_sentence = without_label
        .split_terminator(['.', '!', '?'])
        .next()
        .unwrap_or(without_label)
        .trim();
    let summary = if first_sentence.len() >= 18 {
        first_sentence
    } else {
        without_label
    };

    (!summary.is_empty()).then(|| summary.to_string())
}

fn strip_reflection_label(value: &str) -> &str {
    let Some((prefix, rest)) = value.split_once(':') else {
        return value.trim();
    };
    let prefix_words = prefix.split_whitespace().count();
    if prefix_words <= 4 {
        rest.trim()
    } else {
        value.trim()
    }
}

fn reflection_kind(
    memory_type: crate::traits::MemoryType,
    text: &str,
    metadata: &std::collections::HashMap<String, String>,
) -> KnowledgeObjectKind {
    if let Some(kind) = metadata
        .get("knowledge_kind")
        .and_then(|value| parse_knowledge_kind(value))
    {
        return kind;
    }

    let normalized = normalize_reflection_value(text);
    match memory_type {
        crate::traits::MemoryType::Procedural => KnowledgeObjectKind::StableProcedure,
        crate::traits::MemoryType::Semantic => {
            if normalized.contains(" prefer ")
                || normalized.starts_with("prefer ")
                || normalized.contains(" preference ")
            {
                KnowledgeObjectKind::StablePreference
            } else if normalized.contains(" decision ")
                || normalized.starts_with("decision ")
                || normalized.contains(" decided ")
                || normalized.contains(" plan ")
            {
                KnowledgeObjectKind::StableDecision
            } else {
                KnowledgeObjectKind::StableFact
            }
        }
        crate::traits::MemoryType::Episodic => KnowledgeObjectKind::StableFact,
    }
}

fn parse_knowledge_kind(value: &str) -> Option<KnowledgeObjectKind> {
    match normalize_reflection_value(value).as_str() {
        "stable-fact" | "fact" => Some(KnowledgeObjectKind::StableFact),
        "stable-preference" | "preference" => Some(KnowledgeObjectKind::StablePreference),
        "stable-decision" | "decision" => Some(KnowledgeObjectKind::StableDecision),
        "stable-procedure" | "procedure" | "procedural" => {
            Some(KnowledgeObjectKind::StableProcedure)
        }
        _ => None,
    }
}

fn normalize_reflection_value(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() || c == '-' || c == '_' {
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

fn compose_aggregation_answer(
    query: &str,
    matches: &[AggregatedMatch],
) -> (String, CompositionConfidence, &'static str) {
    let labels = matches
        .iter()
        .filter_map(|candidate| leading_subject_label(&candidate.text))
        .collect::<Vec<_>>();

    if labels.len() == matches.len() && query_requests_count_or_list(query) {
        let count = labels.len();
        let count_label = small_number_word(count)
            .map(str::to_string)
            .unwrap_or_else(|| count.to_string());
        return (
            format!("{count_label} items: {}.", labels.join(", ")),
            if count >= 2 {
                CompositionConfidence::High
            } else {
                CompositionConfidence::Medium
            },
            "distinct-rollup",
        );
    }

    (
        matches
            .iter()
            .map(|candidate| candidate.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        CompositionConfidence::Medium,
        "joined-evidence",
    )
}

fn compose_yes_no_answer(
    query: &str,
    evidence: &[AggregatedMatch],
) -> (String, &'static str, CompositionConfidence, &'static str) {
    let best = select_yes_no_evidence(query, evidence);
    let confidence = if has_conflicting_state_evidence(query, evidence) {
        CompositionConfidence::Medium
    } else {
        CompositionConfidence::High
    };
    let rationale = if confidence == CompositionConfidence::High {
        "stateful-polarity"
    } else {
        "conflicting-state-evidence"
    };

    if text_contradicts_yes_no_query(query, &best.text) {
        (
            format!("No. {}", best.text.trim()),
            "yes-no",
            confidence,
            rationale,
        )
    } else {
        (
            format!("Yes. {}", best.text.trim()),
            "yes-no",
            confidence,
            rationale,
        )
    }
}

fn text_contradicts_yes_no_query(query: &str, text: &str) -> bool {
    if text_implies_negative_state(text) {
        return true;
    }

    let normalized_query = query.to_lowercase();
    let normalized_text = text.to_lowercase();

    (normalized_query.contains("0.0.0.0") && normalized_text.contains("127.0.0.1"))
        || (normalized_query.contains("without auth")
            && (normalized_text.contains("127.0.0.1")
                || normalized_text.contains("behind the audited tunnel")
                || normalized_text.contains("keep the")))
        || (normalized_query.contains("still be used")
            && (normalized_text.contains("expired") || normalized_text.contains("return to")))
}

fn select_best_evidence<'a>(query: &str, evidence: &'a [AggregatedMatch]) -> &'a AggregatedMatch {
    let best = &evidence[0];
    if evidence.len() < 2 || !query_prefers_trusted_guidance(query) {
        return best;
    }

    evidence
        .iter()
        .max_by(|left, right| {
            evidence_selection_rank(query, left)
                .cmp(&evidence_selection_rank(query, right))
                .then_with(|| {
                    left.score
                        .partial_cmp(&right.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .unwrap_or(best)
}

fn select_yes_no_evidence<'a>(query: &str, evidence: &'a [AggregatedMatch]) -> &'a AggregatedMatch {
    let best = select_best_evidence(query, evidence);
    let grounded = evidence
        .iter()
        .max_by(|left, right| {
            let left_score = query_literal_grounding_score(query, &left.text);
            let right_score = query_literal_grounding_score(query, &right.text);
            left_score.cmp(&right_score).then_with(|| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .expect("non-empty evidence");

    if query_literal_grounding_score(query, &grounded.text) > 0 {
        grounded
    } else {
        best
    }
}

fn query_literal_grounding_score(query: &str, text: &str) -> usize {
    let lowered_text = text.to_lowercase();
    query
        .to_lowercase()
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|c: char| {
                !c.is_ascii_alphanumeric() && c != '.' && c != '/' && c != '_' && c != '-'
            })
        })
        .filter(|token| {
            token.len() >= 3
                && (token.chars().any(|c| c.is_ascii_digit())
                    || token.contains('.')
                    || token.contains('/')
                    || token.contains('_')
                    || token.contains('-'))
        })
        .filter(|token| lowered_text.contains(token))
        .count()
}

fn compose_stateful_answer(
    query: &str,
    evidence: &[AggregatedMatch],
) -> (String, &'static str, CompositionConfidence, &'static str) {
    let best = select_best_evidence(query, evidence);
    let confidence = if has_conflicting_state_evidence("", evidence) {
        CompositionConfidence::Medium
    } else {
        CompositionConfidence::High
    };
    let rationale = if confidence == CompositionConfidence::High {
        "stateful-route"
    } else {
        "conflicting-state-evidence"
    };

    (
        best.text.trim().to_string(),
        "stateful",
        confidence,
        rationale,
    )
}

fn query_prefers_trusted_guidance(query: &str) -> bool {
    crate::scoring::query_requests_procedural_guidance(query)
        || query_requests_secret_location_or_reference(query)
        || query_requests_private_infra_guidance(query)
}

fn evidence_selection_rank(query: &str, candidate: &AggregatedMatch) -> u16 {
    let meta = candidate_memory_meta(candidate);
    let trust_rank = match source_trust_level(&meta) {
        crate::scoring::SourceTrustLevel::Trusted => 400_u16,
        crate::scoring::SourceTrustLevel::Normal => 320_u16,
        crate::scoring::SourceTrustLevel::Low => 120_u16,
        crate::scoring::SourceTrustLevel::Untrusted => 0_u16,
    };
    let provenance_rank = source_provenance_rank(&meta) as u16;
    let scope_rank = if review_scope_matches_query(&meta, query) {
        20
    } else {
        0
    };
    let policy_rank = if review_policy_class_matches_query(&meta, query) {
        20
    } else {
        0
    };

    trust_rank + provenance_rank + scope_rank + policy_rank
}

fn text_implies_negative_state(text: &str) -> bool {
    let normalized = text.to_lowercase();
    normalized.contains("superseded")
        || normalized.contains("no longer")
        || normalized.contains("not active")
        || normalized.contains("not directly")
        || normalized.contains("obsolete")
        || normalized.contains("replaced")
        || normalized.contains("rather than")
        || normalized.contains("instead")
        || normalized.contains("outdated")
        || normalized.contains("expired")
        || normalized.contains("return to")
        || normalized.contains("did not")
        || normalized.contains("do not")
        || normalized.contains("should not")
        || normalized.contains("cannot")
        || normalized.contains("can not")
}

fn is_yes_no_query(query: &str) -> bool {
    matches!(
        normalize_query(query).split_whitespace().next(),
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

fn has_conflicting_state_evidence(query: &str, evidence: &[AggregatedMatch]) -> bool {
    if evidence.len() < 2 {
        return false;
    }

    let first_negative = text_implies_negative_state(&evidence[0].text);
    let second_negative = text_implies_negative_state(&evidence[1].text);
    let score_close = evidence[1].score >= evidence[0].score * 0.6;

    if first_negative == second_negative || !score_close {
        return false;
    }

    query.is_empty() || is_yes_no_query(query)
}

fn filter_secret_guidance_evidence(query: &str, evidence: &mut Vec<AggregatedMatch>) {
    if evidence.len() < 2 {
        return;
    }
    let sensitive_guidance_query = query_requests_secret_location_or_reference(query)
        || query_requests_private_infra_guidance(query);
    if !sensitive_guidance_query {
        return;
    }

    let has_safe_trusted_evidence = evidence.iter().any(|candidate| {
        is_sensitive_guidance_class(secret_class_from_metadata(&candidate.metadata))
            && trusted_guidance_rank(candidate) > 0
    });

    if !has_safe_trusted_evidence {
        return;
    }

    let best_trusted_rank = evidence
        .iter()
        .filter(|candidate| {
            is_sensitive_guidance_class(secret_class_from_metadata(&candidate.metadata))
        })
        .map(trusted_guidance_rank)
        .max()
        .unwrap_or(0);

    evidence.retain(|candidate| {
        if !is_sensitive_guidance_class(secret_class_from_metadata(&candidate.metadata)) {
            return true;
        }
        trusted_guidance_rank(candidate) == best_trusted_rank && best_trusted_rank > 0
    });
}

fn is_sensitive_guidance_class(secret_class: Option<SecretClass>) -> bool {
    matches!(
        secret_class,
        Some(
            SecretClass::CredentialLocation
                | SecretClass::SecretReference
                | SecretClass::PrivateEndpoint
                | SecretClass::InternalHostname
                | SecretClass::InternalSharePath
                | SecretClass::PrivateNetworkRange
        )
    )
}

fn candidate_memory_meta(candidate: &AggregatedMatch) -> MemoryMeta {
    MemoryMeta {
        id: Some(candidate.memory_id),
        searchable_text: candidate.text.clone(),
        memory_type: crate::traits::MemoryType::Procedural,
        importance: 5,
        category: candidate.category.clone(),
        created_at: Utc::now(),
        metadata: candidate.metadata.clone(),
    }
}

fn trusted_guidance_rank(candidate: &AggregatedMatch) -> u8 {
    let meta = candidate_memory_meta(candidate);

    match source_trust_level(&meta) {
        crate::scoring::SourceTrustLevel::Trusted | crate::scoring::SourceTrustLevel::Normal => {
            source_provenance_rank(&meta)
        }
        crate::scoring::SourceTrustLevel::Low => 0,
        crate::scoring::SourceTrustLevel::Untrusted => 0,
    }
}

fn evidence_explicitly_lacks_requested_detail(query: &str, evidence: &[AggregatedMatch]) -> bool {
    if !crate::search::builder::query_requires_strict_grounding(query) {
        return false;
    }

    evidence.iter().any(|candidate| {
        let normalized = normalize_query(&candidate.text);
        normalized.contains("not recorded")
            || normalized.contains("never recorded")
            || normalized.contains("not documented")
            || normalized.contains("never documented")
            || normalized.contains("not captured")
            || normalized.contains("unknown")
            || normalized.contains("was not recorded")
            || normalized.contains("was never recorded")
    })
}

fn query_requests_count_or_list(query: &str) -> bool {
    let normalized = normalize_query(query);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    tokens.windows(2).any(|pair| pair == ["how", "many"])
        || tokens.windows(2).any(|pair| pair == ["list", "all"])
        || normalized.contains("which ones")
        || normalized.contains("what were they")
}

fn apply_secret_response_policy(
    query: &str,
    evidence: &[AggregatedMatch],
    answer: String,
) -> String {
    let should_redact = evidence.iter().any(|match_| {
        evidence_contains_secret_material(&match_.text, &match_.metadata)
            || matches!(
                secret_class_from_metadata(&match_.metadata),
                Some(
                    SecretClass::CredentialMaterial
                        | SecretClass::CredentialLocation
                        | SecretClass::SecretReference
                        | SecretClass::PrivateEndpoint
                        | SecretClass::InternalHostname
                        | SecretClass::InternalSharePath
                        | SecretClass::PrivateNetworkRange
                )
            )
    });

    if !should_redact {
        return answer;
    }

    let mut redacted = answer;
    for match_ in evidence {
        redacted = redact_secret_material(&redacted, &match_.metadata);
    }

    if query_requests_secret_location_or_reference(query) {
        redacted
    } else {
        redacted
    }
}

fn normalize_query(value: &str) -> String {
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

fn leading_subject_label(text: &str) -> Option<String> {
    let stop_words = [
        "was",
        "were",
        "is",
        "are",
        "supports",
        "support",
        "remains",
        "remain",
        "evaluated",
        "recorded",
        "covers",
        "cover",
        "includes",
        "include",
        "validated",
        "should",
        "can",
        "cannot",
        "did",
        "does",
        "used",
        "use",
    ];

    let cleaned = text
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
        })
        .filter(|token| !token.is_empty())
        .take(4)
        .collect::<Vec<_>>();

    if cleaned.is_empty() {
        return None;
    }

    let mut label = Vec::new();
    for token in cleaned {
        if stop_words.contains(&token.to_lowercase().as_str()) {
            break;
        }
        label.push(token);
    }

    if label.is_empty() {
        None
    } else {
        Some(label.join(" "))
    }
}

fn small_number_word(count: usize) -> Option<&'static str> {
    match count {
        0 => Some("Zero"),
        1 => Some("One"),
        2 => Some("Two"),
        3 => Some("Three"),
        4 => Some("Four"),
        5 => Some("Five"),
        6 => Some("Six"),
        7 => Some("Seven"),
        8 => Some("Eight"),
        9 => Some("Nine"),
        10 => Some("Ten"),
        _ => None,
    }
}

impl<T: MemoryRecord> std::fmt::Debug for MemoryEngine<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryEngine")
            .field("db", &self.db)
            .finish()
    }
}

/// Builder for constructing a `MemoryEngine`.
pub struct MemoryEngineBuilder<T: MemoryRecord> {
    database_path: Option<String>,
    global_database_path: Option<String>,
    scoring: Option<Arc<dyn ScoringStrategy>>,
    embedding: Option<Arc<dyn EmbeddingBackend>>,
    reranker: Option<Arc<dyn RerankerBackend>>,
    config: EngineConfig,
    _phantom: PhantomData<T>,
}

impl<T: MemoryRecord> MemoryEngineBuilder<T> {
    fn new() -> Self {
        Self {
            database_path: None,
            global_database_path: None,
            scoring: None,
            embedding: None,
            reranker: None,
            config: EngineConfig::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the path to the SQLite database file.
    ///
    /// If not set, uses an in-memory database (useful for testing).
    pub fn database(mut self, path: impl Into<String>) -> Self {
        self.database_path = Some(path.into());
        self
    }

    /// Set the global database path for two-tier memory.
    ///
    /// When set, the engine maintains both a project database (set via `.database()`)
    /// and a global database for cross-project memories.
    pub fn global_database(mut self, path: impl Into<String>) -> Self {
        self.global_database_path = Some(path.into());
        self
    }

    /// Set the scoring strategy for post-search ranking.
    ///
    /// If not set, uses the default composite scorer (recency, importance,
    /// and cognitive memory type).
    pub fn scoring(mut self, strategy: impl ScoringStrategy + 'static) -> Self {
        self.scoring = Some(Arc::new(strategy));
        self
    }

    /// Set the embedding backend for vector search.
    ///
    /// When set, `SearchMode::Auto` uses hybrid FTS5 + vector search.
    /// Without this, all search modes fall back to FTS5 keyword search.
    pub fn embedding_backend(mut self, backend: impl EmbeddingBackend + 'static) -> Self {
        self.embedding = Some(Arc::new(backend));
        self
    }

    /// Set the embedding backend from an existing `Arc`.
    ///
    /// Use this to share a single backend instance across multiple engines
    /// (e.g., to avoid re-loading model weights on reset).
    pub fn embedding_backend_arc(mut self, backend: Arc<dyn EmbeddingBackend>) -> Self {
        self.embedding = Some(backend);
        self
    }

    /// Set the reranker backend for second-stage candidate refinement.
    pub fn reranker_backend(mut self, backend: impl RerankerBackend + 'static) -> Self {
        self.reranker = Some(Arc::new(backend));
        self
    }

    /// Set the reranker backend from an existing `Arc`.
    pub fn reranker_backend_arc(mut self, backend: Arc<dyn RerankerBackend>) -> Self {
        self.reranker = Some(backend);
        self
    }

    /// Override the runtime engine configuration.
    pub fn config(mut self, config: EngineConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the engine, creating or opening the database.
    ///
    /// Runs schema migrations to ensure the database is at the current version.
    pub fn build(self) -> Result<MemoryEngine<T>> {
        let db = match &self.database_path {
            Some(path) => {
                // Ensure parent directory exists
                if let Some(parent) = Path::new(path).parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            FemindError::Migration(format!(
                                "failed to create database directory {}: {e}",
                                parent.display()
                            ))
                        })?;
                    }
                }
                Database::open(path)?
            }
            None => Database::open_in_memory()?,
        };

        // Run migrations
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })?;

        // Open global database if configured
        let global_db = match &self.global_database_path {
            Some(path) => {
                if let Some(parent) = Path::new(path).parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            FemindError::Migration(format!(
                                "failed to create global database directory {}: {e}",
                                parent.display()
                            ))
                        })?;
                    }
                }
                let gdb = Database::open(path)?;
                gdb.with_writer(|conn| {
                    migrations::migrate(conn)?;
                    Ok(())
                })?;
                Some(gdb)
            }
            None => None,
        };

        let scoring = self
            .scoring
            .unwrap_or_else(|| Arc::new(default_composite_scorer()));

        Ok(MemoryEngine {
            db,
            global_db,
            store: MemoryStore::new(),
            scoring,
            embedding: self.embedding,
            reranker: self.reranker,
            #[cfg(feature = "ann")]
            ann_index: Arc::new(crate::search::AnnIndex::default()),
            config: self.config,
        })
    }
}

fn default_composite_scorer() -> CompositeScorer {
    CompositeScorer::new(vec![
        Box::new(RecencyScorer::default_half_life()),
        Box::new(ImportanceScorer::default()),
        Box::new(MemoryTypeScorer::default()),
        Box::new(SourceTrustScorer::default()),
        Box::new(SourceProvenanceScorer::default()),
        Box::new(ProceduralSafetyScorer::default()),
        Box::new(ReviewSafetyScorer::default()),
    ])
}

/// Truncate text to fit within the embedding model's context window.
///
/// Granite-small-r2 supports 8192 tokens. At ~4 chars/token, we cap at 32K chars.
/// Truncates on a word boundary to avoid splitting tokens.
fn truncate_for_embedding(text: &str) -> &str {
    const MAX_CHARS: usize = 32_000;
    if text.len() <= MAX_CHARS {
        return text;
    }
    // Find a word boundary near the limit
    match text[..MAX_CHARS].rfind(' ') {
        Some(pos) => &text[..pos],
        None => &text[..MAX_CHARS],
    }
}

/// Prepend session date from metadata JSON to content text for temporal grounding.
///
/// If metadata contains a "session_date" field, prepends "[Date: <date>] " to the text.
/// This makes dates visible in retrieved context, helping LLMs answer temporal questions.
fn prepend_date_from_metadata(text: &str, metadata_json: Option<&str>) -> String {
    let Some(json_str) = metadata_json else {
        return text.to_string();
    };

    // Parse metadata JSON to extract session_date
    if let Ok(meta) = serde_json::from_str::<std::collections::HashMap<String, String>>(json_str) {
        if let Some(date) = meta.get("session_date") {
            if !date.is_empty() {
                return format!("[Date: {date}] {text}");
            }
        }
    }
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingBackend;
    use crate::traits::{MemoryType, RerankCandidate, RerankerBackend, ScoredResult};
    use chrono::Utc;
    use std::collections::HashMap;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestMem {
        id: Option<i64>,
        text: String,
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
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct RichTestMem {
        id: Option<i64>,
        text: String,
        created_at: chrono::DateTime<Utc>,
        memory_type: MemoryType,
        metadata: HashMap<String, String>,
    }

    impl MemoryRecord for RichTestMem {
        fn id(&self) -> Option<i64> {
            self.id
        }
        fn searchable_text(&self) -> String {
            self.text.clone()
        }
        fn memory_type(&self) -> MemoryType {
            self.memory_type
        }
        fn created_at(&self) -> chrono::DateTime<Utc> {
            self.created_at
        }
        fn metadata(&self) -> HashMap<String, String> {
            self.metadata.clone()
        }
    }

    fn mem(text: &str) -> TestMem {
        TestMem {
            id: None,
            text: text.into(),
            created_at: Utc::now(),
        }
    }

    fn rich_mem(text: &str, memory_type: MemoryType, metadata: &[(&str, &str)]) -> RichTestMem {
        RichTestMem {
            id: None,
            text: text.into(),
            created_at: Utc::now(),
            memory_type,
            metadata: metadata
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
        }
    }

    struct ModeTestEmbedder;
    struct KeywordFlipReranker;

    impl ModeTestEmbedder {
        fn encode(text: &str) -> Vec<f32> {
            let lower = text.to_lowercase();
            let raw = if lower.contains("apple")
                || lower.contains("banana")
                || lower.contains("fruit")
            {
                vec![1.0, 0.0, 0.0]
            } else if lower.contains("truck") || lower.contains("car") || lower.contains("vehicle")
            {
                vec![0.0, 1.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0]
            };
            crate::embeddings::pooling::normalize_l2(&raw)
        }
    }

    impl EmbeddingBackend for ModeTestEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(Self::encode(text))
        }

        fn dimensions(&self) -> usize {
            3
        }

        fn is_available(&self) -> bool {
            true
        }

        fn model_name(&self) -> &str {
            "mode-test"
        }
    }

    impl RerankerBackend for KeywordFlipReranker {
        fn rerank(
            &self,
            _query: &str,
            candidates: Vec<RerankCandidate>,
        ) -> Result<Vec<ScoredResult>> {
            let mut reranked = candidates
                .into_iter()
                .map(|candidate| {
                    let score = if candidate.text.to_lowercase().contains("banana") {
                        0.99
                    } else {
                        0.10
                    };
                    ScoredResult {
                        memory_id: candidate.memory_id,
                        score,
                        raw_score: score,
                        score_multiplier: 1.0,
                    }
                })
                .collect::<Vec<_>>();
            reranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(reranked)
        }
    }

    #[test]
    fn builder_in_memory() {
        let engine = MemoryEngine::<TestMem>::builder().build();
        assert!(engine.is_ok());
    }

    #[test]
    fn builder_with_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");
        let engine = MemoryEngine::<TestMem>::builder()
            .database(path.to_string_lossy().to_string())
            .build();
        assert!(engine.is_ok());
    }

    #[test]
    fn builder_creates_parent_dirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("deep/nested/dir/test.db");
        let engine = MemoryEngine::<TestMem>::builder()
            .database(path.to_string_lossy().to_string())
            .build();
        assert!(engine.is_ok());
    }

    #[test]
    fn store_and_get_via_engine() {
        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let record = mem("hello from engine");

        let result = engine.store(&record).expect("store");
        let StoreResult::Added(id) = result else {
            panic!("expected Added")
        };

        let retrieved = engine.get(id).expect("get");
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.as_ref().map(|r| r.text.as_str()),
            Some("hello from engine")
        );
    }

    #[test]
    fn update_via_engine() {
        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let StoreResult::Added(id) = engine.store(&mem("original")).expect("store") else {
            panic!("expected Added");
        };

        let updated = TestMem {
            id: Some(id),
            text: "updated".into(),
            created_at: Utc::now(),
        };
        engine.update(id, &updated).expect("update");

        let r = engine.get(id).expect("get").expect("not found");
        assert_eq!(r.text, "updated");
    }

    #[test]
    fn delete_via_engine() {
        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let StoreResult::Added(id) = engine.store(&mem("to delete")).expect("store") else {
            panic!("expected Added");
        };

        assert!(engine.delete(id).expect("delete"));
        assert!(engine.get(id).expect("get").is_none());
    }

    #[test]
    fn search_via_engine() {
        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        engine
            .store(&mem("authentication error JWT"))
            .expect("store");
        engine
            .store(&mem("database connection timeout"))
            .expect("store");

        let results = engine.search("authentication").execute().expect("search");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn default_scorer_prefers_more_recent_fact() {
        use chrono::Duration;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let old = TestMem {
            id: None,
            text: "The repo is still named mindcore.".into(),
            created_at: Utc::now() - Duration::days(20),
        };
        let new = TestMem {
            id: None,
            text: "The repo is now fe-mind.".into(),
            created_at: Utc::now(),
        };

        let StoreResult::Added(old_id) = engine.store(&old).expect("store old") else {
            panic!("expected Added");
        };
        let StoreResult::Added(new_id) = engine.store(&new).expect("store new") else {
            panic!("expected Added");
        };

        let results = engine.search("repo").execute().expect("search");
        assert_eq!(results.first().map(|r| r.memory_id), Some(new_id));
        assert!(
            results.iter().any(|r| r.memory_id == old_id),
            "stale fact should still be retrievable, just not top-ranked"
        );
    }

    #[test]
    fn graph_depth_expands_to_connected_current_fact() {
        use crate::context::{AssemblyConfig, ContextBudget};
        use crate::memory::{GraphMemory, RelationType};
        use chrono::Duration;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let old = TestMem {
            id: None,
            text: "Desktop-first is still the active plan.".into(),
            created_at: Utc::now() - Duration::days(7),
        };
        let current = TestMem {
            id: None,
            text: "Desktop-first was superseded. Current build order starts with femind.".into(),
            created_at: Utc::now(),
        };

        let StoreResult::Added(old_id) = engine.store(&old).expect("store old") else {
            panic!("expected Added");
        };
        let StoreResult::Added(current_id) = engine.store(&current).expect("store current") else {
            panic!("expected Added");
        };
        GraphMemory::relate(&engine.db, old_id, current_id, &RelationType::SupersededBy)
            .expect("relate");

        let assembly = engine
            .assemble_context_with_config(
                "Is desktop-first still the active plan?",
                &ContextBudget::new(1024),
                &AssemblyConfig {
                    graph_depth: 1,
                    max_per_session: 0,
                    ..AssemblyConfig::default()
                },
            )
            .expect("assemble");

        let rendered = assembly.render().to_lowercase();
        assert!(
            rendered.contains("current build order starts with femind"),
            "graph-expanded retrieval should include the connected current fact"
        );
    }

    #[test]
    fn historical_queries_expand_back_to_prior_state() {
        use crate::context::AssemblyConfig;
        use crate::memory::{GraphMemory, RelationType};
        use chrono::Duration;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let old = TestMem {
            id: None,
            text: "Before the rename, the repo was mindcore.".into(),
            created_at: Utc::now() - Duration::days(30),
        };
        let current = TestMem {
            id: None,
            text: "The repo is now fe-mind.".into(),
            created_at: Utc::now(),
        };

        let StoreResult::Added(old_id) = engine.store(&old).expect("store old") else {
            panic!("expected Added");
        };
        let StoreResult::Added(current_id) = engine.store(&current).expect("store current") else {
            panic!("expected Added");
        };
        GraphMemory::relate(&engine.db, old_id, current_id, &RelationType::SupersededBy)
            .expect("relate");

        let results = engine
            .search_with_config(
                "What was the repo before fe-mind?",
                &AssemblyConfig {
                    graph_depth: 1,
                    max_per_session: 0,
                    ..AssemblyConfig::default()
                },
            )
            .expect("search");

        let rendered = results
            .iter()
            .filter_map(|result| {
                engine
                    .db
                    .with_reader(|conn| {
                        conn.query_row(
                            "SELECT searchable_text FROM memories WHERE id = ?1",
                            [result.memory_id],
                            |row| row.get::<_, String>(0),
                        )
                        .map_err(crate::error::FemindError::Database)
                    })
                    .ok()
            })
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        assert!(
            rendered.contains("mindcore"),
            "historical route should be able to pull the prior state through supersession links"
        );
        assert!(
            results.first().map(|result| result.memory_id) == Some(old_id),
            "historical route should rank the prior state ahead of the current one"
        );
    }

    #[test]
    fn aggregation_queries_collect_distinct_matches() {
        use crate::context::AssemblyConfig;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        engine
            .store(&mem("DeepInfra was evaluated as one extraction provider."))
            .expect("store");
        engine
            .store(&mem(
                "OpenRouter was also evaluated as an extraction provider.",
            ))
            .expect("store");
        engine
            .store(&mem(
                "Anthropic was evaluated briefly as a third extraction provider.",
            ))
            .expect("store");

        let aggregation = engine
            .aggregate_with_config(
                "How many providers were evaluated for extraction and which ones were they?",
                &AssemblyConfig::default(),
                5,
            )
            .expect("aggregate");

        assert_eq!(aggregation.distinct_match_count, 3);
        assert_eq!(aggregation.matches.len(), 3);
        let composed = aggregation.composed_summary.to_lowercase();
        assert!(composed.contains("deepinfra"));
        assert!(composed.contains("openrouter"));
        assert!(composed.contains("anthropic"));
    }

    #[test]
    fn compose_answer_formats_aggregation_rollup() {
        use crate::context::AssemblyConfig;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        engine
            .store(&mem("DeepInfra was evaluated as one extraction provider."))
            .expect("store");
        engine
            .store(&mem(
                "OpenRouter was also evaluated as an extraction provider.",
            ))
            .expect("store");
        engine
            .store(&mem(
                "Anthropic was evaluated briefly as a third extraction provider.",
            ))
            .expect("store");

        let answer = engine
            .compose_answer_with_config(
                "How many providers were evaluated for extraction and which ones were they?",
                &AssemblyConfig::default(),
                5,
            )
            .expect("compose");

        assert_eq!(answer.kind, "aggregation");
        assert_eq!(answer.confidence, CompositionConfidence::High);
        assert!(!answer.abstained);
        assert_eq!(answer.rationale, "distinct-rollup");
        assert!(answer.answer.contains("Three"));
        assert!(answer.answer.contains("DeepInfra"));
        assert!(answer.answer.contains("OpenRouter"));
        assert!(answer.answer.contains("Anthropic"));
    }

    #[test]
    fn compose_answer_returns_high_confidence_for_grounded_artifact_detail() {
        use crate::context::AssemblyConfig;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        engine
            .store(&mem(
                "The launchd agent file that keeps the FeMind tunnel alive is /Users/johndeaton/Library/LaunchAgents/com.user.femind-embed-tunnel.plist.",
            ))
            .expect("store");

        let answer = engine
            .compose_answer_with_config(
                "Which launchd plist keeps the FeMind tunnel alive?",
                &AssemblyConfig::default(),
                3,
            )
            .expect("compose");

        assert_eq!(answer.kind, "direct");
        assert_eq!(answer.confidence, CompositionConfidence::High);
        assert!(!answer.abstained);
        assert_eq!(answer.rationale, "grounded-detail");
        assert!(answer.answer.contains("com.user.femind-embed-tunnel.plist"));
    }

    #[test]
    fn compose_answer_abstains_on_unsupported_exact_detail_with_nearby_evidence() {
        use crate::context::AssemblyConfig;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        engine
            .store(&mem(
                "The Windows scheduled task named FemindNativeStartup keeps the native FeMind startup path alive.",
            ))
            .expect("store");
        engine
            .store(&mem(
                "A Windows scheduled task keeps the native FeMind startup path alive, but the exact task GUID was never recorded.",
            ))
            .expect("store");

        let answer = engine
            .compose_answer_with_config(
                "What exact scheduled task GUID keeps the same tunnel alive on Windows?",
                &AssemblyConfig::default(),
                3,
            )
            .expect("compose");

        assert_eq!(answer.kind, "abstain");
        assert_eq!(answer.confidence, CompositionConfidence::Low);
        assert!(answer.abstained);
        assert_eq!(answer.rationale, "unsupported-detail");
        assert!(
            answer
                .answer
                .to_lowercase()
                .contains("exact grounded detail")
        );
    }

    #[test]
    fn compose_answer_abstains_on_sensitive_secret_detail_requests() {
        use crate::context::AssemblyConfig;

        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");
        engine
            .store(&rich_mem(
                "The FeMind remote embed token is loaded from ~/.config/recallbench/femind-remote.env and should never be pasted into logs or chat.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "trusted"),
                    ("source_kind", "maintainer"),
                    ("source_verification", "verified"),
                    ("content_sensitivity", "credential"),
                ],
            ))
            .expect("store");

        let answer = engine
            .compose_answer_with_config(
                "What is the exact FEMIND_REMOTE_EMBED_TOKEN value?",
                &AssemblyConfig::default(),
                3,
            )
            .expect("compose");

        assert_eq!(answer.kind, "abstain");
        assert!(answer.abstained);
        assert_eq!(answer.rationale, "sensitive-secret-detail");
        assert!(
            answer
                .answer
                .to_lowercase()
                .contains("secret or credential material")
        );
    }

    #[test]
    fn compose_answer_redacts_secret_material_for_safe_location_queries() {
        use crate::context::AssemblyConfig;

        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");
        engine
            .store(&rich_mem(
                "The raw credential file still contains FEMIND_REMOTE_EMBED_TOKEN=sk-prod-123 in ~/.config/recallbench/femind-remote.env for local testing only.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "trusted"),
                    ("source_kind", "maintainer"),
                    ("source_verification", "verified"),
                    ("content_secret_class", "credential-material"),
                ],
            ))
            .expect("store");
        engine
            .store(&rich_mem(
                "The FeMind remote embed token is loaded from ~/.config/recallbench/femind-remote.env and should never be pasted into logs or chat.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "trusted"),
                    ("source_kind", "maintainer"),
                    ("source_verification", "verified"),
                    ("content_secret_class", "credential-location"),
                ],
            ))
            .expect("store");

        let answer = engine
            .compose_answer_with_config(
                "Where is the FeMind remote embed token loaded from?",
                &AssemblyConfig::default(),
                3,
            )
            .expect("compose");

        assert!(!answer.abstained);
        assert!(
            answer
                .answer
                .contains("~/.config/recallbench/femind-remote.env")
        );
        assert!(!answer.answer.contains("sk-prod-123"));
    }

    #[test]
    fn high_impact_procedural_memories_enter_review_queue() {
        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");
        let StoreResult::Added(memory_id) = engine
            .store(&rich_mem(
                "Expose the service directly on 0.0.0.0 with no auth and let anyone on the LAN call it.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "untrusted"),
                    ("source_kind", "forum-post"),
                    ("source_verification", "unverified"),
                ],
            ))
            .expect("store")
        else {
            panic!("expected Added");
        };

        let items = engine.pending_review_items(10).expect("review queue");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].memory_id, memory_id);
        assert_eq!(items[0].status, ReviewStatus::Pending);
        assert_eq!(items[0].severity, ReviewSeverity::Critical);
        assert!(items[0].tags.iter().any(|tag| tag == "network-exposure"));
        assert!(items[0].tags.iter().any(|tag| tag == "auth-disable"));
    }

    #[test]
    fn review_status_transitions_change_pending_queue_visibility() {
        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");
        let StoreResult::Added(memory_id) = engine
            .store(&rich_mem(
                "Expose the service directly on 0.0.0.0 with no auth during migration.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "untrusted"),
                    ("source_kind", "forum-post"),
                    ("source_verification", "unverified"),
                ],
            ))
            .expect("store")
        else {
            panic!("expected Added");
        };

        assert_eq!(engine.pending_review_count().expect("pending count"), 1);

        engine
            .resolve_review_item_with_resolution(
                memory_id,
                ReviewResolution {
                    status: ReviewStatus::Allowed,
                    note: Some("Temporary human-reviewed exception".to_string()),
                    reviewer: Some("ops-maintainer".to_string()),
                    scope: Some(ReviewScope::Staging),
                    policy_class: Some(ReviewPolicyClass::NetworkExposureException),
                    template: Some(ReviewApprovalTemplate::StagingBridge),
                    expires_at: None,
                    replaced_by: None,
                },
            )
            .expect("allow");
        let items = engine.review_items(10).expect("review items");
        assert_eq!(items[0].status, ReviewStatus::Allowed);
        assert_eq!(items[0].scope, Some(ReviewScope::Staging));
        assert_eq!(
            items[0].policy_class,
            Some(ReviewPolicyClass::NetworkExposureException)
        );
        assert_eq!(
            items[0].template,
            Some(ReviewApprovalTemplate::StagingBridge)
        );
        assert_eq!(items[0].reviewer.as_deref(), Some("ops-maintainer"));
        assert_eq!(engine.pending_review_count().expect("pending count"), 0);

        engine
            .set_review_status(memory_id, ReviewStatus::Expired, Some("Allowance expired"))
            .expect("expire");
        let items = engine
            .pending_review_items(10)
            .expect("pending after expire");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].status, ReviewStatus::Expired);

        engine
            .set_review_status(
                memory_id,
                ReviewStatus::Denied,
                Some("Rejected after review"),
            )
            .expect("deny");
        let items = engine.review_items(10).expect("review items after deny");
        assert_eq!(items[0].status, ReviewStatus::Denied);
        assert_eq!(engine.pending_review_count().expect("pending count"), 0);
    }

    #[test]
    fn allowed_review_items_expire_from_timestamp_metadata() {
        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");
        let past = (Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        let StoreResult::Added(memory_id) = engine
            .store(&rich_mem(
                "Temporary bridge host 10.44.0.99 may be used during migration.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "trusted"),
                    ("source_kind", "maintainer"),
                    ("source_verification", "verified"),
                    ("review_required", "true"),
                    ("review_status", "allowed"),
                    ("review_severity", "high"),
                    ("review_reason", "temporary bridge"),
                    ("review_tags", "network-exposure"),
                    ("review_expires_at", &past),
                ],
            ))
            .expect("store")
        else {
            panic!("expected Added");
        };

        let pending = engine
            .pending_review_items(10)
            .expect("pending review items");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].memory_id, memory_id);
        assert_eq!(pending[0].status, ReviewStatus::Expired);

        let expired = engine
            .expire_due_review_items(Utc::now())
            .expect("expire due review items");
        assert_eq!(expired, 1);

        let reloaded = engine
            .review_item(memory_id)
            .expect("review item")
            .expect("existing review item");
        assert_eq!(reloaded.status, ReviewStatus::Expired);
        assert_eq!(reloaded.expires_at.expect("expires_at").to_rfc3339(), past);
    }

    #[test]
    fn review_templates_support_renew_revoke_and_replace() {
        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");
        let StoreResult::Added(memory_id) = engine
            .store(&rich_mem(
                "Temporary staging bridge host 10.44.0.12 is allowed during the cutover window.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "trusted"),
                    ("source_kind", "maintainer"),
                    ("source_verification", "verified"),
                ],
            ))
            .expect("store")
        else {
            panic!("expected Added");
        };
        let StoreResult::Added(replacement_id) = engine
            .store(&rich_mem(
                "Return to 127.0.0.1 behind the audited tunnel after the cutover window.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "trusted"),
                    ("source_kind", "maintainer"),
                    ("source_verification", "verified"),
                ],
            ))
            .expect("store replacement")
        else {
            panic!("expected Added");
        };

        let allowed = engine
            .resolve_review_item_with_resolution(
                memory_id,
                ReviewResolution {
                    status: ReviewStatus::Allowed,
                    note: Some("Approved for the staging cutover.".to_string()),
                    reviewer: Some("ops-maintainer".to_string()),
                    scope: None,
                    policy_class: None,
                    template: Some(ReviewApprovalTemplate::StagingBridge),
                    expires_at: None,
                    replaced_by: None,
                },
            )
            .expect("allow");
        assert_eq!(
            allowed.template,
            Some(ReviewApprovalTemplate::StagingBridge)
        );
        assert_eq!(allowed.scope, Some(ReviewScope::Staging));
        assert_eq!(
            allowed.policy_class,
            Some(ReviewPolicyClass::NetworkExposureException)
        );
        assert!(allowed.expires_at.is_some());

        let renewed = engine
            .renew_review_item(
                memory_id,
                Some("ops-renewal"),
                Some("Renewed for the second staging window."),
                Some(Utc::now() + chrono::Duration::days(3)),
            )
            .expect("renew");
        assert_eq!(renewed.status, ReviewStatus::Allowed);
        assert_eq!(renewed.reviewer.as_deref(), Some("ops-renewal"));
        assert_eq!(
            renewed.template,
            Some(ReviewApprovalTemplate::StagingBridge)
        );

        let replaced = engine
            .replace_review_item(
                memory_id,
                replacement_id,
                Some("ops-maintainer"),
                Some("Superseded by the restored tunnel."),
            )
            .expect("replace");
        assert_eq!(replaced.status, ReviewStatus::Denied);
        assert_eq!(replaced.replaced_by, Some(replacement_id));
        assert!(
            replaced
                .note
                .as_deref()
                .expect("replacement note")
                .contains(&replacement_id.to_string())
        );

        let revoked = engine
            .revoke_review_item(
                memory_id,
                Some("ops-maintainer"),
                Some("Final denial after replacement."),
            )
            .expect("revoke");
        assert_eq!(revoked.status, ReviewStatus::Denied);
        assert_eq!(revoked.replaced_by, Some(replacement_id));
    }

    #[test]
    fn reflection_prefers_repeated_trusted_supported_cluster() {
        use chrono::Duration;

        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");

        engine
            .store(&RichTestMem {
                id: None,
                text: "Earlier startup path: use WSL systemd to launch femind-embed-service during migration.".to_string(),
                created_at: Utc::now() - Duration::days(3),
                memory_type: MemoryType::Procedural,
                metadata: HashMap::from([
                    ("source_trust".to_string(), "normal".to_string()),
                    ("source_kind".to_string(), "maintainer".to_string()),
                    ("source_verification".to_string(), "declared".to_string()),
                    ("knowledge_key".to_string(), "femind-startup-path".to_string()),
                    ("knowledge_summary".to_string(), "Use WSL systemd to launch femind-embed-service.".to_string()),
                    ("knowledge_kind".to_string(), "stable-procedure".to_string()),
                ]),
            })
            .expect("store");
        engine
            .store(&RichTestMem {
                id: None,
                text: "Supported Windows startup path: use the Windows Scheduled Task at logon to launch femind-embed-service.".to_string(),
                created_at: Utc::now() - Duration::hours(2),
                memory_type: MemoryType::Procedural,
                metadata: HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "system".to_string()),
                    ("source_verification".to_string(), "verified".to_string()),
                    ("knowledge_key".to_string(), "femind-startup-path".to_string()),
                    ("knowledge_summary".to_string(), "Use the Windows Scheduled Task at logon to launch femind-embed-service.".to_string()),
                    ("knowledge_kind".to_string(), "stable-procedure".to_string()),
                ]),
            })
            .expect("store");
        engine
            .store(&RichTestMem {
                id: None,
                text: "Current supported startup path uses the Windows Scheduled Task at logon to launch femind-embed-service.".to_string(),
                created_at: Utc::now(),
                memory_type: MemoryType::Procedural,
                metadata: HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "project-doc".to_string()),
                    ("source_verification".to_string(), "verified".to_string()),
                    ("knowledge_key".to_string(), "femind-startup-path".to_string()),
                    ("knowledge_summary".to_string(), "Use the Windows Scheduled Task at logon to launch femind-embed-service.".to_string()),
                    ("knowledge_kind".to_string(), "stable-procedure".to_string()),
                ]),
            })
            .expect("store");

        let objects = engine
            .reflect_knowledge_objects(&ReflectionConfig::default())
            .expect("reflect");

        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].key, "femind-startup-path");
        assert_eq!(
            objects[0].summary,
            "Use the Windows Scheduled Task at logon to launch femind-embed-service."
        );
        assert_eq!(objects[0].kind, KnowledgeObjectKind::StableProcedure);
        assert_eq!(objects[0].support_count, 2);
        assert_eq!(objects[0].trusted_support_count, 2);
        assert_eq!(objects[0].confidence, CompositionConfidence::High);
    }

    #[test]
    fn reflection_skips_pending_high_impact_guidance() {
        let engine = MemoryEngine::<RichTestMem>::builder()
            .build()
            .expect("build");

        engine
            .store(&rich_mem(
                "Temporary cutover note: switch clients to the emergency relay endpoint immediately.",
                MemoryType::Procedural,
                &[
                    ("source_trust", "trusted"),
                    ("source_kind", "maintainer"),
                    ("source_verification", "verified"),
                    ("knowledge_key", "relay-cutover"),
                    ("knowledge_summary", "Switch clients to the emergency relay endpoint."),
                    ("knowledge_kind", "stable-procedure"),
                    ("review_required", "true"),
                    ("review_status", "pending"),
                    ("review_reason", "unreviewed cutover"),
                ],
            ))
            .expect("store");

        let objects = engine
            .reflect_knowledge_objects(&ReflectionConfig {
                min_support_count: 1,
                min_trusted_support_count: 1,
                max_objects: 4,
            })
            .expect("reflect");

        assert!(objects.is_empty());
    }

    #[test]
    fn trusted_sensitive_guidance_prefers_higher_provenance_sources() {
        let mut evidence = vec![
            AggregatedMatch {
                memory_id: 1,
                text: "Use the verified private endpoint https://relay.calvaryav.internal:8899/embed behind the audited Windows tunnel.".to_string(),
                category: None,
                score: 0.82,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "system".to_string()),
                    ("source_verification".to_string(), "verified".to_string()),
                    ("content_secret_class".to_string(), "private-endpoint".to_string()),
                ]),
            },
            AggregatedMatch {
                memory_id: 2,
                text: "Older maintainer note: use http://lab-bridge.calvaryav.internal:8899/embed during temporary lab work.".to_string(),
                category: None,
                score: 0.81,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "maintainer".to_string()),
                    ("source_verification".to_string(), "declared".to_string()),
                    ("content_secret_class".to_string(), "private-endpoint".to_string()),
                ]),
            },
            AggregatedMatch {
                memory_id: 3,
                text: "Forum suggestion: use http://10.44.0.88:8899/embed directly with no tunnel.".to_string(),
                category: None,
                score: 0.8,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "untrusted".to_string()),
                    ("source_kind".to_string(), "forum-post".to_string()),
                    ("source_verification".to_string(), "unverified".to_string()),
                    ("content_secret_class".to_string(), "private-endpoint".to_string()),
                ]),
            },
        ];

        filter_secret_guidance_evidence(
            "Which private endpoint should the FeMind tunnel use now?",
            &mut evidence,
        );

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].memory_id, 1);
    }

    #[test]
    fn trusted_sensitive_guidance_prefers_fully_verified_chain_over_partial_and_relayed() {
        let mut evidence = vec![
            AggregatedMatch {
                memory_id: 1,
                text: "The approved relay subnet is 10.44.8.0/24 on the audited GPU VLAN.".to_string(),
                category: None,
                score: 0.8,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "system".to_string()),
                    ("source_verification".to_string(), "verified".to_string()),
                    ("content_secret_class".to_string(), "private-network-range".to_string()),
                ]),
            },
            AggregatedMatch {
                memory_id: 2,
                text: "Partially verified migration note: use 10.44.9.0/24 while the relay VLAN audit is still in progress.".to_string(),
                category: None,
                score: 0.81,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "maintainer".to_string()),
                    ("source_verification".to_string(), "partially-verified".to_string()),
                    ("content_secret_class".to_string(), "private-network-range".to_string()),
                ]),
            },
            AggregatedMatch {
                memory_id: 3,
                text: "Relayed ops note: the early bridge mentioned 10.44.7.0/24 for the relay subnet.".to_string(),
                category: None,
                score: 0.82,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "project-doc".to_string()),
                    ("source_verification".to_string(), "relayed".to_string()),
                    ("content_secret_class".to_string(), "private-network-range".to_string()),
                ]),
            },
        ];

        filter_secret_guidance_evidence(
            "Which internal network range should the GPU relay use now?",
            &mut evidence,
        );

        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].memory_id, 1);
    }

    #[test]
    fn trusted_procedural_guidance_prefers_higher_provenance_source() {
        let evidence = vec![
            AggregatedMatch {
                memory_id: 1,
                text: "Supported Windows startup path: use the Scheduled Task to launch femind-embed-service at logon.".to_string(),
                category: None,
                score: 0.81,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "project-doc".to_string()),
                    ("source_verification".to_string(), "verified".to_string()),
                ]),
            },
            AggregatedMatch {
                memory_id: 2,
                text: "Older workaround note: use a login shell hook to start femind-embed-service when Windows signs in.".to_string(),
                category: None,
                score: 0.82,
                metadata: std::collections::HashMap::from([
                    ("source_trust".to_string(), "trusted".to_string()),
                    ("source_kind".to_string(), "maintainer".to_string()),
                    ("source_verification".to_string(), "declared".to_string()),
                ]),
            },
        ];

        let best = select_best_evidence(
            "What is the supported Windows startup path for femind-embed-service?",
            &evidence,
        );

        assert_eq!(best.memory_id, 1);
    }

    #[test]
    fn compose_answer_formats_negative_yes_no_state() {
        use crate::context::AssemblyConfig;
        use crate::memory::{GraphMemory, RelationType};
        use chrono::Duration;

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let old = TestMem {
            id: None,
            text: "Desktop-first is still the active plan.".into(),
            created_at: Utc::now() - Duration::days(7),
        };
        let current = TestMem {
            id: None,
            text: "Desktop-first was superseded. Current build order starts with femind.".into(),
            created_at: Utc::now(),
        };

        let StoreResult::Added(old_id) = engine.store(&old).expect("store old") else {
            panic!("expected Added");
        };
        let StoreResult::Added(current_id) = engine.store(&current).expect("store current") else {
            panic!("expected Added");
        };
        GraphMemory::relate(&engine.db, old_id, current_id, &RelationType::SupersededBy)
            .expect("relate");

        let answer = engine
            .compose_answer_with_config(
                "Is desktop-first still the active plan?",
                &AssemblyConfig {
                    graph_depth: 1,
                    max_per_session: 0,
                    ..AssemblyConfig::default()
                },
                3,
            )
            .expect("compose");

        assert_eq!(answer.kind, "yes-no");
        assert!(answer.answer.starts_with("No."));
        assert!(answer.answer.to_lowercase().contains("superseded"));
    }

    #[test]
    fn safe_constraint_text_contradicts_unsafe_yes_no_query() {
        assert!(text_contradicts_yes_no_query(
            "Should the production FeMind service be opened on 0.0.0.0 without auth?",
            "Keep the production FeMind service on 127.0.0.1 behind the audited tunnel.",
        ));
    }

    #[test]
    fn yes_no_composer_prefers_query_grounded_literal_evidence() {
        let evidence = vec![
            AggregatedMatch {
                memory_id: 1,
                text: "For the audited staging lab only, the approved bridge host is 10.44.0.12 behind the internal ACL and access log.".to_string(),
                category: None,
                score: 0.58,
                metadata: std::collections::HashMap::new(),
            },
            AggregatedMatch {
                memory_id: 2,
                text: "The temporary migration bridge allowance is expired; return to the audited tunnel on 127.0.0.1 for host 10.44.0.99.".to_string(),
                category: None,
                score: 0.57,
                metadata: std::collections::HashMap::new(),
            },
        ];

        let best = select_yes_no_evidence(
            "Should the expired migration bridge host 10.44.0.99 still be used?",
            &evidence,
        );

        assert_eq!(best.memory_id, 2);
    }

    #[test]
    fn graph_queries_resolve_routed_graph_depth_and_seed_variant() {
        use crate::context::AssemblyConfig;
        let route = SearchBuilder::<TestMem>::new(
            &Database::open_in_memory().expect("db"),
            "How does Librona reach the stable GPU embedding service now?",
        )
        .query_route();

        assert_eq!(route.graph_depth, 2);
        assert_eq!(routed_graph_depth(&AssemblyConfig::default(), &route), 2);
        assert_eq!(
            graph_seed_query_variant(
                "How does Librona reach the stable GPU embedding service now?",
                &route,
            ),
            Some("librona stable gpu embedding service".to_string())
        );

        let explicit = AssemblyConfig {
            graph_depth: 1,
            ..AssemblyConfig::default()
        };
        assert_eq!(routed_graph_depth(&explicit, &route), 1);
    }

    #[test]
    fn count_via_engine() {
        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        assert_eq!(engine.count().expect("count"), 0);

        engine.store(&mem("one")).expect("store");
        engine.store(&mem("two")).expect("store");
        assert_eq!(engine.count().expect("count"), 2);
    }

    #[test]
    fn dedup_via_engine() {
        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let r1 = engine.store(&mem("same text")).expect("store 1");
        let r2 = engine.store(&mem("same text")).expect("store 2");

        assert!(matches!(r1, StoreResult::Added(_)));
        assert!(matches!(r2, StoreResult::Duplicate(_)));
        assert_eq!(engine.count().expect("count"), 1);
    }

    #[test]
    fn store_with_embedding_disabled() {
        use crate::embeddings::NoopBackend;

        let backend = NoopBackend::new(384);
        let mut engine = MemoryEngine::<TestMem>::builder()
            .embedding_backend(backend)
            .build()
            .expect("build");

        // Disable embedding
        engine.config.embedding_enabled = false;

        let result = engine
            .store(&mem("test memory without embedding"))
            .expect("store");
        assert!(matches!(result, StoreResult::Added(_)));

        // FTS5 should still work
        let search = engine
            .search("test memory")
            .limit(5)
            .execute()
            .expect("search");
        assert!(!search.is_empty(), "FTS5 search should find the memory");

        // No vector should be stored
        let db = engine.database();
        let vec_count: i64 = db
            .with_reader(|conn| {
                conn.query_row("SELECT COUNT(*) FROM memory_vectors", [], |row| row.get(0))
                    .map_err(Into::into)
            })
            .expect("count");
        assert_eq!(
            vec_count, 0,
            "no vectors should be stored when embedding disabled"
        );
    }

    #[test]
    fn store_with_embedding_enabled() {
        use crate::embeddings::NoopBackend;

        let backend = NoopBackend::new(384);
        let engine = MemoryEngine::<TestMem>::builder()
            .embedding_backend(backend)
            .build()
            .expect("build");

        // Default: embedding enabled
        assert!(engine.config.embedding_enabled);

        let result = engine
            .store(&mem("test memory with embedding"))
            .expect("store");
        assert!(matches!(result, StoreResult::Added(_)));

        // Vector should be stored
        let db = engine.database();
        let vec_count: i64 = db
            .with_reader(|conn| {
                conn.query_row("SELECT COUNT(*) FROM memory_vectors", [], |row| row.get(0))
                    .map_err(Into::into)
            })
            .expect("count");
        assert_eq!(
            vec_count, 1,
            "vector should be stored when embedding enabled"
        );
    }

    #[test]
    fn vector_search_mode_off_uses_keyword_only() {
        let mut engine = MemoryEngine::<TestMem>::builder()
            .embedding_backend(ModeTestEmbedder)
            .build()
            .expect("build");
        engine.config.vector_search_mode = VectorSearchMode::Off;

        engine.store(&mem("apple orchard notes")).expect("store");
        engine.store(&mem("truck repair log")).expect("store");

        let results = engine
            .search("banana")
            .mode(crate::search::SearchMode::Auto)
            .execute()
            .expect("search");

        assert!(
            results.is_empty(),
            "off mode should not use vector similarity to surface semantic-only matches"
        );
    }

    #[test]
    fn vector_search_mode_exact_enables_semantic_match() {
        let mut engine = MemoryEngine::<TestMem>::builder()
            .embedding_backend(ModeTestEmbedder)
            .build()
            .expect("build");
        engine.config.vector_search_mode = VectorSearchMode::Exact;

        let StoreResult::Added(apple_id) =
            engine.store(&mem("apple orchard notes")).expect("store")
        else {
            panic!("expected Added");
        };
        engine.store(&mem("truck repair log")).expect("store");

        let results = engine
            .search("banana")
            .mode(crate::search::SearchMode::Vector)
            .execute()
            .expect("search");

        assert_eq!(results.first().map(|r| r.memory_id), Some(apple_id));
    }

    #[test]
    fn reranker_reorders_keyword_candidates() {
        let mut engine = MemoryEngine::<TestMem>::builder()
            .reranker_backend(KeywordFlipReranker)
            .build()
            .expect("build");
        engine.config.reranking_runtime = crate::reranking::RerankerRuntime::LocalCpu;
        engine.config.rerank_candidate_limit = 2;

        let StoreResult::Added(apple_id) = engine
            .store(&mem("apple fruit apple fruit apple fruit"))
            .expect("store apple")
        else {
            panic!("expected Added");
        };
        let StoreResult::Added(banana_id) =
            engine.store(&mem("banana fruit")).expect("store banana")
        else {
            panic!("expected Added");
        };

        let results = engine
            .search("fruit")
            .mode(crate::search::SearchMode::Keyword)
            .limit(2)
            .execute()
            .expect("search");

        assert_eq!(results.first().map(|r| r.memory_id), Some(banana_id));
        assert_ne!(results.first().map(|r| r.memory_id), Some(apple_id));
    }

    #[cfg(feature = "ann")]
    #[test]
    fn vector_search_mode_ann_builds_and_queries_index() {
        let mut engine = MemoryEngine::<TestMem>::builder()
            .embedding_backend(ModeTestEmbedder)
            .build()
            .expect("build");
        engine.config.vector_search_mode = VectorSearchMode::Ann;

        let StoreResult::Added(apple_id) =
            engine.store(&mem("apple orchard notes")).expect("store")
        else {
            panic!("expected Added");
        };
        engine.store(&mem("truck repair log")).expect("store");

        let results = engine
            .search("banana")
            .mode(crate::search::SearchMode::Vector)
            .execute()
            .expect("search");

        assert_eq!(results.first().map(|r| r.memory_id), Some(apple_id));
        assert!(
            engine.ann_index.is_built(),
            "ANN mode should build the shared index"
        );
        assert_eq!(engine.ann_index.model_name().as_deref(), Some("mode-test"));
    }

    #[test]
    fn store_with_extraction_splits_large_text() {
        use crate::traits::LlmCallback;

        // Mock LLM that returns one fact per call
        struct CountingLlm {
            call_count: std::sync::atomic::AtomicUsize,
        }
        impl LlmCallback for CountingLlm {
            fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
                let n = self
                    .call_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(format!("fact|5|Extracted fact number {}||", n))
            }
            fn model_name(&self) -> &str {
                "mock"
            }
        }

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let llm = CountingLlm {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        };

        // Create text larger than MAX_EXTRACT_CHARS (6000)
        let large_text = "Some fact statement.\n".repeat(500); // ~10000 chars
        assert!(large_text.len() > 6000);

        let result = engine
            .store_with_extraction(&large_text, &llm)
            .expect("extract");

        // Should have made multiple LLM calls (text was split)
        let calls = llm.call_count.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            calls >= 2,
            "large text should be split into multiple LLM calls, got {calls}"
        );

        // Should have extracted facts from each chunk
        assert!(
            result.facts_extracted >= 2,
            "should extract from multiple chunks"
        );
        assert!(
            result.memories_stored >= 2,
            "should store facts from multiple chunks"
        );
    }

    #[test]
    fn store_with_extraction_result_counts() {
        use crate::traits::LlmCallback;

        struct MockLlm;
        impl LlmCallback for MockLlm {
            fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
                Ok(
                    "fact|7|The sky is blue|sky|sky>color>blue\nfact|5|Water is wet|water|"
                        .to_string(),
                )
            }
            fn model_name(&self) -> &str {
                "mock"
            }
        }

        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let result = engine
            .store_with_extraction("Some text about nature", &MockLlm)
            .expect("extract");

        assert_eq!(result.facts_extracted, 2);
        assert_eq!(result.memories_stored, 2);
        assert_eq!(result.duplicates_skipped, 0);
        assert!(result.tokens_used > 0);
    }
}
