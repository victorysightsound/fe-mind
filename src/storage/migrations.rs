use rusqlite::Connection;

use crate::error::{MindCoreError, Result};
use crate::storage::schema;

/// Current schema version. Incremented with each migration.
pub const CURRENT_VERSION: u32 = schema::SCHEMA_VERSION;

/// Run all pending migrations to bring the database up to `CURRENT_VERSION`.
///
/// On a fresh database, creates the schema from scratch. On an existing database,
/// checks the stored version and runs any pending migrations sequentially.
///
/// Migrations run inside a transaction — if any step fails, the database
/// rolls back to its previous state.
pub fn migrate(conn: &Connection) -> Result<()> {
    create_meta_table(conn)?;
    let version = get_version(conn)?;

    if version == 0 {
        // Fresh database — create schema from scratch
        schema::create_schema(conn)?;
        set_version(conn, CURRENT_VERSION)?;
        return Ok(());
    }

    if version > CURRENT_VERSION {
        return Err(MindCoreError::Migration(format!(
            "database schema version ({version}) is newer than this build supports ({CURRENT_VERSION}). \
             Upgrade mindcore to open this database."
        )));
    }

    if version < CURRENT_VERSION {
        run_migrations(conn, version)?;
    }

    Ok(())
}

/// Create the metadata table if it doesn't exist.
fn create_meta_table(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS mindcore_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )?;
    Ok(())
}

/// Get the current schema version (0 if no version set = fresh database).
fn get_version(conn: &Connection) -> Result<u32> {
    let result = conn.query_row(
        "SELECT value FROM mindcore_meta WHERE key = 'schema_version'",
        [],
        |row| row.get::<_, String>(0),
    );

    match result {
        Ok(v) => v
            .parse::<u32>()
            .map_err(|e| MindCoreError::Migration(format!("invalid schema version '{v}': {e}"))),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(0),
        Err(e) => Err(e.into()),
    }
}

/// Set the schema version.
fn set_version(conn: &Connection, version: u32) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO mindcore_meta (key, value) VALUES ('schema_version', ?1)",
        [version.to_string()],
    )?;
    Ok(())
}

/// Type for individual migration functions.
type Migration = fn(&Connection) -> Result<()>;

/// Ordered list of migrations. Index 0 = migration from v1→v2, etc.
/// Each migration upgrades the schema by one version.
const MIGRATIONS: &[Migration] = &[
    // Future migrations will be added here:
    // v1 → v2: |conn| { conn.execute_batch("ALTER TABLE ..."); Ok(()) },
];

/// Run all migrations from `from_version` to `CURRENT_VERSION`.
fn run_migrations(conn: &Connection, from_version: u32) -> Result<()> {
    for (i, migration) in MIGRATIONS.iter().enumerate() {
        let migration_version = (i as u32) + 1;
        if migration_version >= from_version && migration_version < CURRENT_VERSION {
            tracing::info!(
                from = migration_version,
                to = migration_version + 1,
                "running schema migration"
            );

            // Run each migration in a transaction
            let tx = conn.unchecked_transaction().map_err(|e| {
                MindCoreError::Migration(format!("failed to start migration transaction: {e}"))
            })?;

            migration(&tx)?;
            set_version(&tx, migration_version + 1)?;

            tx.commit().map_err(|e| {
                MindCoreError::Migration(format!("migration commit failed: {e}"))
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_fresh_database() {
        let conn = Connection::open_in_memory().expect("open failed");
        let result = migrate(&conn);
        assert!(result.is_ok(), "migration failed: {result:?}");

        let version = get_version(&conn).expect("get_version failed");
        assert_eq!(version, CURRENT_VERSION);
    }

    #[test]
    fn migrate_is_idempotent() {
        let conn = Connection::open_in_memory().expect("open failed");
        migrate(&conn).expect("first migration failed");
        migrate(&conn).expect("second migration should succeed");

        let version = get_version(&conn).expect("get_version failed");
        assert_eq!(version, CURRENT_VERSION);
    }

    #[test]
    fn migrate_creates_meta_table() {
        let conn = Connection::open_in_memory().expect("open failed");
        migrate(&conn).expect("migration failed");

        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='mindcore_meta'",
                [],
                |row| row.get(0),
            )
            .expect("query failed");
        assert_eq!(count, 1);
    }

    #[test]
    fn migrate_creates_memories_table() {
        let conn = Connection::open_in_memory().expect("open failed");
        migrate(&conn).expect("migration failed");

        // Verify memories table was created by the schema
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
    fn rejects_newer_version() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_meta_table(&conn).expect("meta table failed");
        set_version(&conn, CURRENT_VERSION + 1).expect("set version failed");

        let result = migrate(&conn);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("newer than this build"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn version_roundtrip() {
        let conn = Connection::open_in_memory().expect("open failed");
        create_meta_table(&conn).expect("meta table failed");

        set_version(&conn, 42).expect("set failed");
        let v = get_version(&conn).expect("get failed");
        assert_eq!(v, 42);
    }
}
