use rusqlite::Connection;

use crate::error::Result;

/// Initial schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// Create all core tables, indexes, and triggers.
///
/// Idempotent — safe to call on an existing database (uses IF NOT EXISTS).
pub fn create_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(CORE_SCHEMA)?;
    Ok(())
}

const CORE_SCHEMA: &str = "
-- Main memory table
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    searchable_text TEXT NOT NULL,
    memory_type     TEXT NOT NULL CHECK(memory_type IN ('episodic','semantic','procedural')),
    importance      INTEGER NOT NULL DEFAULT 5 CHECK(importance BETWEEN 1 AND 10),
    category        TEXT,
    metadata_json   TEXT,
    content_hash    TEXT NOT NULL,
    embedding_status TEXT DEFAULT 'pending' CHECK(embedding_status IN ('pending','success','failed')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    tier            INTEGER NOT NULL DEFAULT 0 CHECK(tier BETWEEN 0 AND 2),
    source_ids      TEXT,
    valid_from      TEXT,
    valid_until     TEXT,
    activation_cache REAL,
    activation_updated TEXT,
    record_json     TEXT NOT NULL
);

-- FTS5 full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    searchable_text,
    category,
    tokenize='porter'
);

-- FTS5 sync triggers
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, searchable_text, category)
    VALUES (new.id, new.searchable_text, new.category);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
    UPDATE memories_fts SET searchable_text = new.searchable_text,
                            category = new.category
    WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE rowid = old.id;
END;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);

-- Vector storage (feature: vector-search)
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id   INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    model_name  TEXT NOT NULL,
    dimensions  INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Graph relationships (feature: graph-memory)
CREATE TABLE IF NOT EXISTS memory_relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    valid_from  TEXT,
    valid_until TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON memory_relations(relation);

-- Access log for activation model (feature: activation-model)
CREATE TABLE IF NOT EXISTS memory_access_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
    query_text  TEXT
);

CREATE INDEX IF NOT EXISTS idx_access_log_memory ON memory_access_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_access_log_time ON memory_access_log(accessed_at);
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_schema_succeeds() {
        let conn = Connection::open_in_memory().expect("open failed");
        let result = create_schema(&conn);
        assert!(result.is_ok(), "schema creation failed: {result:?}");
    }

    #[test]
    fn schema_is_idempotent() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("first create failed");
        create_schema(&conn).expect("second create should succeed (idempotent)");
    }

    #[test]
    fn memories_table_exists() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("schema failed");

        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(count, 1);
    }

    #[test]
    fn fts5_table_exists() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("schema failed");

        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memories_fts'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(count, 1);
    }

    #[test]
    fn fts5_triggers_sync_insert() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("schema failed");

        conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
             VALUES ('test memory text', 'semantic', 'abc123', '{}')",
            [],
        )
        .expect("insert failed");

        let fts_count: i32 = conn
            .query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))
            .expect("fts query failed");
        assert_eq!(fts_count, 1, "FTS5 trigger should have inserted a row");
    }

    #[test]
    fn fts5_triggers_sync_delete() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("schema failed");

        conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
             VALUES ('delete me', 'episodic', 'def456', '{}')",
            [],
        )
        .expect("insert failed");

        conn.execute("DELETE FROM memories WHERE id = 1", [])
            .expect("delete failed");

        let fts_count: i32 = conn
            .query_row("SELECT COUNT(*) FROM memories_fts", [], |row| row.get(0))
            .expect("fts query failed");
        assert_eq!(fts_count, 0, "FTS5 trigger should have deleted the row");
    }

    #[test]
    fn fts5_search_works() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("schema failed");

        conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
             VALUES ('authentication failed with JWT token', 'procedural', 'hash1', '{}')",
            [],
        )
        .expect("insert failed");

        conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
             VALUES ('database connection timeout', 'episodic', 'hash2', '{}')",
            [],
        )
        .expect("insert failed");

        // FTS5 search with Porter stemming
        let results: Vec<String> = conn
            .prepare("SELECT searchable_text FROM memories_fts WHERE memories_fts MATCH 'authenticate'")
            .expect("prepare failed")
            .query_map([], |row| row.get(0))
            .expect("query failed")
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(results.len(), 1);
        assert!(results[0].contains("authentication"));
    }

    #[test]
    fn memory_type_constraint() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("schema failed");

        let result = conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
             VALUES ('test', 'invalid_type', 'hash', '{}')",
            [],
        );
        assert!(result.is_err(), "invalid memory_type should be rejected");
    }

    #[test]
    fn importance_constraint() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_schema(&conn).expect("schema failed");

        let result = conn.execute(
            "INSERT INTO memories (searchable_text, memory_type, importance, content_hash, record_json)
             VALUES ('test', 'semantic', 11, 'hash', '{}')",
            [],
        );
        assert!(result.is_err(), "importance > 10 should be rejected");
    }
}
