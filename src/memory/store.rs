use rusqlite::params;
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

use crate::error::{MindCoreError, Result};
use crate::storage::Database;
use crate::traits::{MemoryRecord, MemoryType};

/// Result of a store operation, indicating what action was taken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreResult {
    /// New memory was inserted. Contains the assigned row ID.
    Added(i64),
    /// Exact duplicate detected (same content hash). Contains existing ID.
    Duplicate(i64),
}

/// CRUD operations for memories.
///
/// Generic over the consumer's `MemoryRecord` implementation.
/// All operations are synchronous (SQLite queries).
pub struct MemoryStore<T: MemoryRecord> {
    _phantom: PhantomData<T>,
}

impl<T: MemoryRecord> Default for MemoryStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: MemoryRecord> MemoryStore<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Store a new memory. Returns `StoreResult::Duplicate` if content hash exists.
    pub fn store(&self, db: &Database, record: &T) -> Result<StoreResult> {
        let searchable_text = record.searchable_text();
        let content_hash = Self::compute_hash(&searchable_text);

        // Check for exact duplicate via content hash
        let existing_id = db.with_reader(|conn| {
            let result = conn.query_row(
                "SELECT id FROM memories WHERE content_hash = ?1",
                [&content_hash],
                |row| row.get::<_, i64>(0),
            );
            match result {
                Ok(id) => Ok(Some(id)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })?;

        if let Some(id) = existing_id {
            return Ok(StoreResult::Duplicate(id));
        }

        let record_json = serde_json::to_string(record)?;
        let memory_type = record.memory_type().as_str().to_string();
        let importance = record.importance() as i32;
        let category = record.category().map(String::from);
        let metadata_json = if record.metadata().is_empty() {
            None
        } else {
            Some(serde_json::to_string(&record.metadata())?)
        };
        let created_at = record.created_at().to_rfc3339();

        db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO memories (
                    searchable_text, memory_type, importance, category,
                    metadata_json, content_hash, created_at, record_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    searchable_text,
                    memory_type,
                    importance,
                    category,
                    metadata_json,
                    content_hash,
                    created_at,
                    record_json,
                ],
            )?;
            let id = conn.last_insert_rowid();
            Ok(StoreResult::Added(id))
        })
    }

    /// Retrieve a memory by ID.
    pub fn get(&self, db: &Database, id: i64) -> Result<Option<T>> {
        db.with_reader(|conn| {
            let result = conn.query_row(
                "SELECT record_json FROM memories WHERE id = ?1",
                [id],
                |row| row.get::<_, String>(0),
            );
            match result {
                Ok(json) => {
                    let record: T = serde_json::from_str(&json)?;
                    Ok(Some(record))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })
    }

    /// Update an existing memory by ID.
    ///
    /// Replaces the searchable text, metadata, and record JSON.
    /// Updates the content hash and FTS5 index (via trigger).
    pub fn update(&self, db: &Database, id: i64, record: &T) -> Result<()> {
        let searchable_text = record.searchable_text();
        let content_hash = Self::compute_hash(&searchable_text);
        let record_json = serde_json::to_string(record)?;
        let memory_type = record.memory_type().as_str().to_string();
        let importance = record.importance() as i32;
        let category = record.category().map(String::from);
        let metadata_json = if record.metadata().is_empty() {
            None
        } else {
            Some(serde_json::to_string(&record.metadata())?)
        };

        db.with_writer(|conn| {
            let rows = conn.execute(
                "UPDATE memories SET
                    searchable_text = ?1,
                    memory_type = ?2,
                    importance = ?3,
                    category = ?4,
                    metadata_json = ?5,
                    content_hash = ?6,
                    embedding_status = 'pending',
                    updated_at = datetime('now'),
                    record_json = ?7
                WHERE id = ?8",
                params![
                    searchable_text,
                    memory_type,
                    importance,
                    category,
                    metadata_json,
                    content_hash,
                    record_json,
                    id,
                ],
            )?;

            if rows == 0 {
                return Err(MindCoreError::Database(rusqlite::Error::QueryReturnedNoRows));
            }
            Ok(())
        })
    }

    /// Delete a memory by ID.
    ///
    /// Also removes associated FTS5 entries (via trigger), vectors, and access logs
    /// (via ON DELETE CASCADE).
    pub fn delete(&self, db: &Database, id: i64) -> Result<bool> {
        db.with_writer(|conn| {
            let rows = conn.execute("DELETE FROM memories WHERE id = ?1", [id])?;
            Ok(rows > 0)
        })
    }

    /// Count total memories in the database.
    pub fn count(&self, db: &Database) -> Result<u64> {
        db.with_reader(|conn| {
            let count: i64 =
                conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
            Ok(count as u64)
        })
    }

    /// Count memories by type.
    pub fn count_by_type(&self, db: &Database, memory_type: MemoryType) -> Result<u64> {
        db.with_reader(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE memory_type = ?1",
                [memory_type.as_str()],
                |row| row.get(0),
            )?;
            Ok(count as u64)
        })
    }

    /// Compute SHA-256 hash of content for deduplication.
    fn compute_hash(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::migrations;
    use chrono::Utc;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct TestMemory {
        id: Option<i64>,
        text: String,
        category: Option<String>,
        created_at: chrono::DateTime<Utc>,
    }

    impl MemoryRecord for TestMemory {
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

    fn setup() -> (Database, MemoryStore<TestMemory>) {
        let db = Database::open_in_memory().expect("open failed");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migration failed");
        (db, MemoryStore::new())
    }

    fn test_record(text: &str) -> TestMemory {
        TestMemory {
            id: None,
            text: text.to_string(),
            category: None,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn store_and_get() {
        let (db, store) = setup();
        let record = test_record("hello world");

        let result = store.store(&db, &record).expect("store failed");
        let StoreResult::Added(id) = result else {
            panic!("expected Added, got {result:?}");
        };

        let retrieved = store.get(&db, id).expect("get failed");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.as_ref().map(|r| r.text.as_str()), Some("hello world"));
    }

    #[test]
    fn get_nonexistent() {
        let (db, store) = setup();
        let result = store.get(&db, 999).expect("get failed");
        assert!(result.is_none());
    }

    #[test]
    fn store_dedup() {
        let (db, store) = setup();
        let record = test_record("duplicate text");

        let r1 = store.store(&db, &record).expect("first store failed");
        let r2 = store.store(&db, &record).expect("second store failed");

        assert!(matches!(r1, StoreResult::Added(_)));
        assert!(matches!(r2, StoreResult::Duplicate(_)));

        assert_eq!(store.count(&db).expect("count failed"), 1);
    }

    #[test]
    fn update_record() {
        let (db, store) = setup();
        let record = test_record("original text");
        let StoreResult::Added(id) = store.store(&db, &record).expect("store failed") else {
            panic!("expected Added");
        };

        let updated = TestMemory {
            id: Some(id),
            text: "updated text".to_string(),
            category: Some("decision".to_string()),
            created_at: record.created_at,
        };
        store.update(&db, id, &updated).expect("update failed");

        let retrieved = store.get(&db, id).expect("get failed").expect("not found");
        assert_eq!(retrieved.text, "updated text");
        assert_eq!(retrieved.category.as_deref(), Some("decision"));
    }

    #[test]
    fn update_nonexistent() {
        let (db, store) = setup();
        let record = test_record("ghost");
        let result = store.update(&db, 999, &record);
        assert!(result.is_err());
    }

    #[test]
    fn delete_record() {
        let (db, store) = setup();
        let record = test_record("to be deleted");
        let StoreResult::Added(id) = store.store(&db, &record).expect("store failed") else {
            panic!("expected Added");
        };

        let deleted = store.delete(&db, id).expect("delete failed");
        assert!(deleted);

        let retrieved = store.get(&db, id).expect("get failed");
        assert!(retrieved.is_none());
    }

    #[test]
    fn delete_nonexistent() {
        let (db, store) = setup();
        let deleted = store.delete(&db, 999).expect("delete failed");
        assert!(!deleted);
    }

    #[test]
    fn count_operations() {
        let (db, store) = setup();
        assert_eq!(store.count(&db).expect("count failed"), 0);

        store.store(&db, &test_record("one")).expect("store 1");
        store.store(&db, &test_record("two")).expect("store 2");
        store.store(&db, &test_record("three")).expect("store 3");

        assert_eq!(store.count(&db).expect("count failed"), 3);
        assert_eq!(
            store
                .count_by_type(&db, MemoryType::Semantic)
                .expect("count_by_type failed"),
            3
        );
        assert_eq!(
            store
                .count_by_type(&db, MemoryType::Episodic)
                .expect("count_by_type failed"),
            0
        );
    }

    #[test]
    fn hash_deterministic() {
        let h1 = MemoryStore::<TestMemory>::compute_hash("same content");
        let h2 = MemoryStore::<TestMemory>::compute_hash("same content");
        let h3 = MemoryStore::<TestMemory>::compute_hash("different content");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
