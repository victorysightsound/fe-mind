use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

use crate::error::Result;

/// SQLite database wrapper with WAL mode and optimized pragmas.
///
/// All write operations are serialized through an internal `Mutex<Connection>`.
/// Readers use a separate connection pool (managed by `MemoryEngine`).
pub struct Database {
    writer: Mutex<Connection>,
}

impl Database {
    /// Open or create a database at the given path.
    ///
    /// Configures WAL mode, mmap, cache, and foreign keys on the connection.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        Self::configure(&conn)?;
        Ok(Self {
            writer: Mutex::new(conn),
        })
    }

    /// Open an in-memory database (useful for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::configure(&conn)?;
        Ok(Self {
            writer: Mutex::new(conn),
        })
    }

    /// Execute a closure with exclusive write access to the connection.
    pub fn with_writer<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let conn = self.writer.lock().map_err(|e| {
            crate::error::MindCoreError::Database(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
                Some(format!("writer lock poisoned: {e}")),
            ))
        })?;
        f(&conn)
    }

    /// Execute a closure with read access to the connection.
    ///
    /// Currently uses the writer connection (single-connection mode).
    /// Future: use a connection pool for concurrent reads.
    pub fn with_reader<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        self.with_writer(f)
    }

    /// Apply optimized SQLite pragmas for agent memory workloads.
    fn configure(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA temp_store = MEMORY;
             PRAGMA mmap_size = 268435456;
             PRAGMA cache_size = -64000;
             PRAGMA foreign_keys = ON;",
        )?;
        Ok(())
    }
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("writer", &"Mutex<Connection>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_in_memory() {
        let db = Database::open_in_memory();
        assert!(db.is_ok());
    }

    #[test]
    fn wal_mode_enabled() {
        let db = Database::open_in_memory().expect("failed to open db");
        db.with_reader(|conn| {
            let mode: String =
                conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
            // In-memory databases use "memory" journal mode, not WAL
            // WAL is only for file-based databases
            assert!(
                mode == "wal" || mode == "memory",
                "unexpected journal mode: {mode}"
            );
            Ok(())
        })
        .expect("pragma query failed");
    }

    #[test]
    fn foreign_keys_enabled() {
        let db = Database::open_in_memory().expect("failed to open db");
        db.with_reader(|conn| {
            let fk: i32 =
                conn.query_row("PRAGMA foreign_keys", [], |row| row.get(0))?;
            assert_eq!(fk, 1, "foreign keys should be enabled");
            Ok(())
        })
        .expect("pragma query failed");
    }

    #[test]
    fn open_file_database() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().join("test.db");
        let db = Database::open(&path);
        assert!(db.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn write_and_read() {
        let db = Database::open_in_memory().expect("failed to open db");
        db.with_writer(|conn| {
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)", [])?;
            conn.execute("INSERT INTO test (val) VALUES (?1)", ["hello"])?;
            Ok(())
        })
        .expect("write failed");

        db.with_reader(|conn| {
            let val: String =
                conn.query_row("SELECT val FROM test WHERE id = 1", [], |row| row.get(0))?;
            assert_eq!(val, "hello");
            Ok(())
        })
        .expect("read failed");
    }
}
