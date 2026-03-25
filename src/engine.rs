use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use sha2::Digest;

use crate::context::{ContextAssembly, ContextBudget, ContextItem, PRIORITY_LEARNING};
use crate::embeddings::EmbeddingBackend;
use crate::error::{MindCoreError, Result};
use crate::memory::MemoryStore;
use crate::memory::store::StoreResult;
use crate::scoring::CompositeScorer;
use crate::search::builder::SearchBuilder;
use crate::storage::Database;
use crate::storage::migrations;
use crate::traits::{MemoryRecord, ScoringStrategy};

/// Runtime feature configuration for MindCore.
///
/// Controls which features are active. All toggles are independent —
/// any combination is valid. Defaults to all features enabled.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Enable vector embedding at store time.
    pub embedding_enabled: bool,
    /// Enable graph edge creation and traversal.
    pub graph_enabled: bool,
    /// Enable content hash deduplication on store.
    pub dedup_enabled: bool,
    /// Assembly configuration (diversification, recency, graph depth).
    pub assembly: crate::context::AssemblyConfig,
    /// Vector search mode: "ann" (default), "exact" (brute-force), "off".
    pub vector_search_mode: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            embedding_enabled: true,
            graph_enabled: true,
            dedup_enabled: true,
            assembly: crate::context::AssemblyConfig::default(),
            vector_search_mode: "exact".to_string(), // brute-force until ANN is implemented
        }
    }
}

/// The primary interface to MindCore.
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

        // Compute and store embedding for new records
        if let StoreResult::Added(id) = &result {
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
                    let already_exists = crate::search::vector::VectorSearch::vector_exists(
                        &self.db, &hash,
                    ).unwrap_or(false);

                    if !already_exists {
                        let embed_start = std::time::Instant::now();
                        match backend.embed(&text) {
                            Ok(vec) if vec.is_empty() => {
                                tracing::warn!("Empty embedding returned for memory {id}");
                                self.set_embedding_status(*id, "failed");
                            }
                            Ok(vec) => {
                                let embed_ms = embed_start.elapsed().as_millis();
                                tracing::debug!(memory_id = id, embed_ms, "embedded memory");
                                match crate::search::vector::VectorSearch::store_vector(
                                    &self.db, *id, &vec, backend.model_name(), &hash,
                                ) {
                                    Ok(()) => {
                                        tracing::debug!(memory_id = id, "stored vector");
                                        self.set_embedding_status(*id, "success");
                                    }
                                    Err(e) => {
                                        tracing::warn!("Failed to store embedding for memory {id}: {e}");
                                        self.set_embedding_status(*id, "failed");
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to compute embedding for memory {id}: {e}");
                                self.set_embedding_status(*id, "failed");
                            }
                        }
                    } else {
                        self.set_embedding_status(*id, "success");
                    }
                }
            }
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
                if self.embedding.as_ref().is_some_and(|b| b.is_available()) {
                    let text = record.searchable_text();
                    if !text.trim().is_empty() {
                        let hash = format!("{:x}", sha2::Sha256::digest(text.as_bytes()));
                        let already_exists = crate::search::vector::VectorSearch::vector_exists(
                            &self.db, &hash,
                        ).unwrap_or(false);
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
                match backend.embed_batch(&texts) {
                    Ok(embeddings) => {
                        let batch_ms = batch_start.elapsed().as_millis();
                        tracing::debug!(batch_count, batch_ms, "batch embedding complete");
                        // Phase 3: Store all vectors and update status
                        for ((id, _, hash), embedding) in to_embed.iter().zip(embeddings.iter()) {
                            match crate::search::vector::VectorSearch::store_vector(
                                &self.db, *id, embedding, backend.model_name(), hash,
                            ) {
                                Ok(()) => self.set_embedding_status(*id, "success"),
                                Err(e) => {
                                    tracing::warn!("Failed to store embedding for memory {id}: {e}");
                                    self.set_embedding_status(*id, "failed");
                                }
                            }
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
            .with_scoring(Arc::clone(&self.scoring));
        if let Some(ref embedding) = self.embedding {
            builder = builder.with_embedding(Arc::clone(embedding));
        }
        builder
    }

    /// Access the embedding backend (if configured).
    pub fn embedding_backend(&self) -> Option<&dyn EmbeddingBackend> {
        self.embedding.as_deref()
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
    /// Returns extraction statistics.
    pub fn store_with_extraction(
        &self,
        raw_text: &str,
        llm: &dyn crate::traits::LlmCallback,
    ) -> Result<crate::ingest::llm_extract::ExtractionResult> {
        use crate::ingest::llm_extract;
        use crate::memory::{GraphMemory, RelationType};

        // Step 1: Extract facts via LLM
        let extraction = llm_extract::extract_facts(raw_text, llm)?;

        if extraction.facts.is_empty() {
            return Ok(extraction);
        }

        // Step 2: Store each fact as individual memory
        let mut stored_ids: Vec<(i64, &llm_extract::ExtractedFact)> = Vec::new();

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

            if existing_id.is_some() {
                continue; // Skip duplicate
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

            // Compute and store embedding
            if let Some(ref backend) = self.embedding {
                if backend.is_available() && !fact.text.trim().is_empty() {
                    let text = truncate_for_embedding(&fact.text);
                    match backend.embed(text) {
                        Ok(vec) if !vec.is_empty() => {
                            let _ = crate::search::vector::VectorSearch::store_vector(
                                &self.db, id, &vec, backend.model_name(), &hash,
                            );
                            self.set_embedding_status(id, "success");
                        }
                        Ok(_) => self.set_embedding_status(id, "failed"),
                        Err(_) => self.set_embedding_status(id, "failed"),
                    }
                }
            }

            stored_ids.push((id, fact));
        }

        // Step 3: Create graph edges for relationships
        for (id, fact) in &stored_ids {
            for (subject, relation, _object) in &fact.relationships {
                // Search existing memories for same subject + relation with different value
                let search_text = format!("{} {}", subject, relation);
                let existing = self.db.with_reader(|conn| {
                    let mut stmt = conn.prepare(
                        "SELECT id, searchable_text FROM memories
                         WHERE id != ?1
                         AND searchable_text LIKE ?2
                         ORDER BY id ASC",
                    )?;
                    let pattern = format!("%{}%", subject);
                    let results: Vec<(i64, String)> = stmt.query_map(
                        rusqlite::params![id, pattern],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )?.filter_map(|r| r.ok()).collect();
                    Ok::<_, crate::error::MindCoreError>(results)
                })?;

                for (existing_id, existing_text) in &existing {
                    if existing_text.to_lowercase().contains(&subject.to_lowercase())
                       && existing_text.to_lowercase().contains(&relation.replace('_', " ").to_lowercase()) {
                        // Same subject + relation → SupersededBy (older is superseded)
                        let _ = GraphMemory::relate(
                            &self.db, *existing_id, *id, &RelationType::SupersededBy,
                        );
                    }
                }
            }
        }

        Ok(extraction)
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
            ).map_err(Into::into)
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
        use std::collections::HashMap;
        use crate::search::fts5::strip_stop_words;

        // Query variant 1: original
        let results1 = self.search(query).limit(limit).execute()?;

        // Query variant 2: key-phrase only (stop words removed)
        let key_phrases = strip_stop_words(query);
        let results2 = if key_phrases != query && !key_phrases.is_empty() {
            self.search(&key_phrases).limit(limit).execute()?
        } else {
            Vec::new()
        };

        // Merge: keep highest score per memory_id
        let mut best: HashMap<i64, crate::search::builder::SearchResult> = HashMap::new();
        for r in results1.into_iter().chain(results2.into_iter()) {
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
            let max_index = merged.iter().filter_map(|r| {
                self.db.with_reader(|conn| {
                    conn.query_row(
                        "SELECT metadata_json FROM memories WHERE id = ?1",
                        [r.memory_id],
                        |row| row.get::<_, Option<String>>(0),
                    ).map_err(|e| crate::error::MindCoreError::Database(e))
                }).ok().flatten().and_then(|json| {
                    serde_json::from_str::<HashMap<String, String>>(&json).ok()
                }).and_then(|meta| meta.get("turn_index").and_then(|v| v.parse::<f32>().ok()))
            }).fold(1.0_f32, f32::max);

            for r in &mut merged {
                let turn_index = self.db.with_reader(|conn| {
                    conn.query_row(
                        "SELECT metadata_json FROM memories WHERE id = ?1",
                        [r.memory_id],
                        |row| row.get::<_, Option<String>>(0),
                    ).map_err(|e| crate::error::MindCoreError::Database(e))
                }).ok().flatten().and_then(|json| {
                    serde_json::from_str::<HashMap<String, String>>(&json).ok()
                }).and_then(|meta| meta.get("turn_index").and_then(|v| v.parse::<f32>().ok()))
                .unwrap_or(0.0);

                // position_ratio: 0.0 (oldest) to 1.0 (newest)
                let position_ratio = turn_index / max_index;
                r.score *= 1.0 + config.recency_boost * position_ratio;
            }
        }

        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Graph filtering: demote results that have been superseded by newer facts.
        // If memory A has a SupersededBy edge to memory B, A is outdated — reduce its score.
        if config.graph_depth > 0 {
            use crate::memory::{GraphMemory, RelationType};

            let mut superseded_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();

            // Check each result: does anything supersede it?
            for r in &merged {
                // Look for incoming SupersededBy edges (this memory IS the old one)
                let is_superseded = self.db.with_reader(|conn| {
                    let count: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM memory_relations
                         WHERE source_id = ?1 AND relation = 'superseded_by'",
                        [r.memory_id],
                        |row| row.get(0),
                    ).unwrap_or(0);
                    Ok::<bool, crate::error::MindCoreError>(count > 0)
                }).unwrap_or(false);

                if is_superseded {
                    superseded_ids.insert(r.memory_id);
                }
            }

            if !superseded_ids.is_empty() {
                tracing::debug!("Graph filtering: {} results demoted as superseded", superseded_ids.len());
                for r in &mut merged {
                    if superseded_ids.contains(&r.memory_id) {
                        r.score *= 0.1; // Heavily demote outdated facts
                    }
                }
                merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        // Diversification: limit chunks per session (0 = unlimited)
        let max_per = config.max_per_session;
        let diversified: Vec<_> = if max_per == 0 {
            merged // No diversification
        } else {
            let mut session_counts: HashMap<String, usize> = HashMap::new();
            merged.into_iter().filter(|r| {
                let session_key = self.db.with_reader(|conn| {
                    conn.query_row(
                        "SELECT metadata_json FROM memories WHERE id = ?1",
                        [r.memory_id],
                        |row| row.get::<_, Option<String>>(0),
                    ).map_err(|e| crate::error::MindCoreError::Database(e))
                }).ok().flatten().and_then(|json| {
                    serde_json::from_str::<HashMap<String, String>>(&json).ok()
                }).and_then(|meta| meta.get("session_date").cloned())
                .unwrap_or_else(|| format!("unknown_{}", r.memory_id));

                let count = session_counts.entry(session_key).or_insert(0);
                *count += 1;
                *count <= max_per
            }).collect()
        };

        Ok(diversified)
    }

    /// Assemble context for an LLM prompt within a token budget.
    ///
    /// Uses default AssemblyConfig (max 1/session, no recency boost).
    pub fn assemble_context(
        &self,
        query: &str,
        budget: &ContextBudget,
    ) -> Result<ContextAssembly> {
        self.assemble_context_with_config(query, budget, &crate::context::AssemblyConfig::default())
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
    _phantom: PhantomData<T>,
}

impl<T: MemoryRecord> MemoryEngineBuilder<T> {
    fn new() -> Self {
        Self {
            database_path: None,
            global_database_path: None,
            scoring: None,
            embedding: None,
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
    /// If not set, uses a no-op scorer (raw retrieval scores only).
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
                            MindCoreError::Migration(format!(
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
                            MindCoreError::Migration(format!(
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
            .unwrap_or_else(|| Arc::new(CompositeScorer::empty()));

        Ok(MemoryEngine {
            db,
            global_db,
            store: MemoryStore::new(),
            scoring,
            embedding: self.embedding,
            config: EngineConfig::default(),
        })
    }
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
    let Some(json_str) = metadata_json else { return text.to_string() };

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
    use crate::traits::MemoryType;
    use chrono::Utc;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestMem {
        id: Option<i64>,
        text: String,
        created_at: chrono::DateTime<Utc>,
    }

    impl MemoryRecord for TestMem {
        fn id(&self) -> Option<i64> { self.id }
        fn searchable_text(&self) -> String { self.text.clone() }
        fn memory_type(&self) -> MemoryType { MemoryType::Semantic }
        fn created_at(&self) -> chrono::DateTime<Utc> { self.created_at }
    }

    fn mem(text: &str) -> TestMem {
        TestMem { id: None, text: text.into(), created_at: Utc::now() }
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
        let StoreResult::Added(id) = result else { panic!("expected Added") };

        let retrieved = engine.get(id).expect("get");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.as_ref().map(|r| r.text.as_str()), Some("hello from engine"));
    }

    #[test]
    fn update_via_engine() {
        let engine = MemoryEngine::<TestMem>::builder().build().expect("build");
        let StoreResult::Added(id) = engine.store(&mem("original")).expect("store") else {
            panic!("expected Added");
        };

        let updated = TestMem { id: Some(id), text: "updated".into(), created_at: Utc::now() };
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
        engine.store(&mem("authentication error JWT")).expect("store");
        engine.store(&mem("database connection timeout")).expect("store");

        let results = engine.search("authentication").execute().expect("search");
        assert_eq!(results.len(), 1);
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
}
