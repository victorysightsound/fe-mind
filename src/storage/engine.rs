use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::Result;

/// SQLite database wrapper with WAL mode and optimized pragmas.
///
/// Thread-safe via:
/// - **Writer:** A single `Mutex<Connection>` serializes all writes.
/// - **Readers:** For file-based databases, a pool of `Mutex<Connection>`
///   provides concurrent read access. In-memory databases use the writer.
///
/// `Database` is `Send + Sync` because all `Connection` access is behind `Mutex`.
/// (`Mutex<T>` is `Sync` when `T: Send`, and `rusqlite::Connection: Send`.)
pub struct Database {
    writer: Mutex<Connection>,
    /// Pool of read-only connections (file-based DBs only).
    /// Each wrapped in Mutex so the Vec itself is Sync.
    reader_pool: Mutex<Vec<Connection>>,
    /// `None` for in-memory databases.
    path: Option<PathBuf>,
}

impl Database {
    /// Open or create a database at the given path.
    ///
    /// Configures WAL mode, mmap, cache, and foreign keys.
    /// Creates a pool of read-only connections sized to available parallelism.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)?;
        Self::configure(&conn)?;

        let pool_size = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2)
            .max(2);

        let mut readers = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let reader = Connection::open_with_flags(
                &path,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?;
            Self::configure(&reader)?;
            readers.push(reader);
        }

        Ok(Self {
            writer: Mutex::new(conn),
            reader_pool: Mutex::new(readers),
            path: Some(path),
        })
    }

    /// Open an in-memory database (useful for testing).
    ///
    /// In-memory databases don't support a reader pool — reads use the writer.
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::configure(&conn)?;
        Ok(Self {
            writer: Mutex::new(conn),
            reader_pool: Mutex::new(Vec::new()),
            path: None,
        })
    }

    /// Execute a closure with exclusive write access to the connection.
    pub fn with_writer<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let conn = self.lock_writer()?;
        f(&conn)
    }

    /// Execute a closure with read access.
    ///
    /// For file-based databases, borrows a connection from the reader pool.
    /// For in-memory databases, uses the writer connection.
    pub fn with_reader<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        if self.path.is_some() {
            // File-based: try to get a reader from the pool
            let reader = self.borrow_reader();
            match reader {
                Some(conn) => {
                    let result = f(&conn);
                    self.return_reader(conn);
                    result
                }
                None => {
                    // Pool exhausted, fall back to writer
                    let conn = self.lock_writer()?;
                    f(&conn)
                }
            }
        } else {
            // In-memory: use the writer
            let conn = self.lock_writer()?;
            f(&conn)
        }
    }

    /// Lock the writer connection.
    fn lock_writer(&self) -> Result<std::sync::MutexGuard<'_, Connection>> {
        self.writer.lock().map_err(|e| {
            crate::error::MindCoreError::Database(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
                Some(format!("writer lock poisoned: {e}")),
            ))
        })
    }

    /// Borrow a reader connection from the pool.
    fn borrow_reader(&self) -> Option<Connection> {
        let mut pool = self.reader_pool.lock().ok()?;
        pool.pop()
    }

    /// Return a reader connection to the pool.
    fn return_reader(&self, conn: Connection) {
        if let Ok(mut pool) = self.reader_pool.lock() {
            pool.push(conn);
        }
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

    /// Whether this is a file-based database (vs in-memory).
    pub fn is_file_based(&self) -> bool {
        self.path.is_some()
    }
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("path", &self.path)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn open_in_memory() {
        let db = Database::open_in_memory();
        assert!(db.is_ok());
        assert!(!db.as_ref().map(|d| d.is_file_based()).unwrap_or(true));
    }

    #[test]
    fn wal_mode_enabled() {
        let db = Database::open_in_memory().expect("open failed");
        db.with_reader(|conn| {
            let mode: String =
                conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
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
        let db = Database::open_in_memory().expect("open failed");
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
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");
        let db = Database::open(&path);
        assert!(db.is_ok());
        assert!(path.exists());
        assert!(db.as_ref().map(|d| d.is_file_based()).unwrap_or(false));
    }

    #[test]
    fn write_and_read() {
        let db = Database::open_in_memory().expect("open failed");
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

    #[test]
    fn file_db_reader_pool() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pool_test.db");
        let db = Database::open(&path).expect("open failed");

        db.with_writer(|conn| {
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)", [])?;
            conn.execute("INSERT INTO test (val) VALUES (?1)", ["pool_data"])?;
            Ok(())
        })
        .expect("write failed");

        db.with_reader(|conn| {
            let val: String =
                conn.query_row("SELECT val FROM test WHERE id = 1", [], |row| row.get(0))?;
            assert_eq!(val, "pool_data");
            Ok(())
        })
        .expect("pool read failed");
    }

    #[test]
    fn concurrent_reads_file_db() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("concurrent.db");
        let db = Arc::new(Database::open(&path).expect("open failed"));

        db.with_writer(|conn| {
            conn.execute("CREATE TABLE nums (id INTEGER PRIMARY KEY, val INTEGER)", [])?;
            for i in 0..100 {
                conn.execute("INSERT INTO nums (val) VALUES (?1)", [i])?;
            }
            Ok(())
        })
        .expect("write failed");

        let mut handles = Vec::new();
        for _ in 0..4 {
            let db = Arc::clone(&db);
            handles.push(std::thread::spawn(move || {
                db.with_reader(|conn| {
                    let count: i64 =
                        conn.query_row("SELECT COUNT(*) FROM nums", [], |row| row.get(0))?;
                    assert_eq!(count, 100);
                    Ok(())
                })
                .expect("concurrent read failed");
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked");
        }
    }

    #[test]
    fn database_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Database>();
    }
}
