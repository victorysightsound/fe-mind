use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use crate::context::{ContextAssembly, ContextBudget, ContextItem, PRIORITY_LEARNING};
use crate::error::{MindCoreError, Result};
use crate::memory::MemoryStore;
use crate::memory::store::StoreResult;
use crate::scoring::CompositeScorer;
use crate::search::builder::SearchBuilder;
use crate::storage::Database;
use crate::storage::migrations;
use crate::traits::{MemoryRecord, ScoringStrategy};

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
    store: MemoryStore<T>,
    scoring: Arc<dyn ScoringStrategy>,
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
        self.store.store(&self.db, record)
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
    /// configured scoring strategy.
    pub fn search(&self, query: &str) -> SearchBuilder<'_, T> {
        SearchBuilder::new(&self.db, query).with_scoring(Arc::clone(&self.scoring))
    }

    /// Count total memories in the database.
    pub fn count(&self) -> Result<u64> {
        self.store.count(&self.db)
    }

    /// Assemble context for an LLM prompt within a token budget.
    ///
    /// Searches for relevant memories, converts them to context items,
    /// and assembles within the budget using priority-ranked selection.
    pub fn assemble_context(
        &self,
        query: &str,
        budget: &ContextBudget,
    ) -> Result<ContextAssembly> {
        // Search for relevant memories (3x budget to have plenty of candidates)
        let results = self.search(query).limit(50).execute()?;

        // Convert search results to context items
        let candidates: Vec<ContextItem> = results
            .iter()
            .filter_map(|sr| {
                // Load the memory to get its content
                self.db
                    .with_reader(|conn| {
                        let row = conn.query_row(
                            "SELECT searchable_text, memory_type, category FROM memories WHERE id = ?1",
                            [sr.memory_id],
                            |row| {
                                Ok((
                                    row.get::<_, String>(0)?,
                                    row.get::<_, String>(1)?,
                                    row.get::<_, Option<String>>(2)?,
                                ))
                            },
                        );
                        match row {
                            Ok((text, type_str, category)) => {
                                let memory_type = crate::traits::MemoryType::from_str(&type_str)
                                    .unwrap_or(crate::traits::MemoryType::Episodic);
                                Ok(Some(ContextItem {
                                    memory_id: sr.memory_id,
                                    content: text.clone(),
                                    priority: PRIORITY_LEARNING,
                                    estimated_tokens: budget.estimate_tokens(&text),
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

    /// Direct access to the database (for advanced consumers).
    pub fn database(&self) -> &Database {
        &self.db
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
    scoring: Option<Arc<dyn ScoringStrategy>>,
    _phantom: PhantomData<T>,
}

impl<T: MemoryRecord> MemoryEngineBuilder<T> {
    fn new() -> Self {
        Self {
            database_path: None,
            scoring: None,
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

    /// Set the scoring strategy for post-search ranking.
    ///
    /// If not set, uses a no-op scorer (raw retrieval scores only).
    pub fn scoring(mut self, strategy: impl ScoringStrategy + 'static) -> Self {
        self.scoring = Some(Arc::new(strategy));
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

        let scoring = self
            .scoring
            .unwrap_or_else(|| Arc::new(CompositeScorer::empty()));

        Ok(MemoryEngine {
            db,
            store: MemoryStore::new(),
            scoring,
        })
    }
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
