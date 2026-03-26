use std::path::{Path, PathBuf};

use crate::error::{FemindError, Result};
use crate::storage::migrations;
use crate::storage::Database;

/// Two-tier memory database manager.
///
/// Manages a global database (`~/.femind/global.db`) for cross-project memories
/// and a project database (`./.femind/memory.db`) for project-specific memories.
///
/// Both databases share the same schema. Queries can target either or both,
/// with project memories receiving a scoring boost.
pub struct TwoTierManager {
    global_db: Database,
    project_db: Database,
    /// Scoring boost for project-specific memories (default: 1.5x).
    pub project_boost: f32,
}

impl TwoTierManager {
    /// Open or create both databases.
    ///
    /// - `global_path`: path to the global database (typically `~/.femind/global.db`)
    /// - `project_path`: path to the project database (typically `./.femind/memory.db`)
    pub fn open(global_path: impl AsRef<Path>, project_path: impl AsRef<Path>) -> Result<Self> {
        let global_path = global_path.as_ref();
        let project_path = project_path.as_ref();

        // Ensure parent directories exist
        for path in [global_path, project_path] {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        FemindError::Migration(format!(
                            "failed to create directory {}: {e}",
                            parent.display()
                        ))
                    })?;
                }
            }
        }

        let global_db = Database::open(global_path)?;
        global_db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })?;

        let project_db = Database::open(project_path)?;
        project_db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })?;

        Ok(Self {
            global_db,
            project_db,
            project_boost: 1.5,
        })
    }

    /// Get the global database.
    pub fn global(&self) -> &Database {
        &self.global_db
    }

    /// Get the project database.
    pub fn project(&self) -> &Database {
        &self.project_db
    }

    /// Default global database path.
    pub fn default_global_path() -> PathBuf {
        let femind = dirs_home().join(".femind").join("global.db");
        let legacy = dirs_home().join(".mindcore").join("global.db");
        if legacy.exists() && !femind.exists() {
            legacy
        } else {
            femind
        }
    }

    /// Default project database path (relative to current directory).
    pub fn default_project_path() -> PathBuf {
        let femind = PathBuf::from(".femind").join("memory.db");
        let legacy = PathBuf::from(".mindcore").join("memory.db");
        if legacy.exists() && !femind.exists() {
            legacy
        } else {
            femind
        }
    }
}

/// Get user home directory.
fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_two_tier() {
        let dir = tempfile::tempdir().expect("tempdir");
        let global = dir.path().join("global.db");
        let project = dir.path().join("project.db");

        let manager = TwoTierManager::open(&global, &project).expect("open");
        assert!(global.exists());
        assert!(project.exists());

        // Both databases should be usable
        manager
            .global()
            .with_writer(|conn| {
                conn.execute(
                    "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
                 VALUES ('global memory', 'semantic', 'gh', '{}')",
                    [],
                )?;
                Ok(())
            })
            .expect("global insert");

        manager
            .project()
            .with_writer(|conn| {
                conn.execute(
                    "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json)
                 VALUES ('project memory', 'semantic', 'ph', '{}')",
                    [],
                )?;
                Ok(())
            })
            .expect("project insert");

        // Verify isolation
        let global_count: i64 = manager
            .global()
            .with_reader(|conn| {
                Ok(conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?)
            })
            .expect("count");
        let project_count: i64 = manager
            .project()
            .with_reader(|conn| {
                Ok(conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?)
            })
            .expect("count");

        assert_eq!(global_count, 1);
        assert_eq!(project_count, 1);
    }

    #[test]
    fn default_paths() {
        let global = TwoTierManager::default_global_path();
        let project = TwoTierManager::default_project_path();

        assert!(global.to_string_lossy().contains(".femind"));
        assert!(project.to_string_lossy().contains(".femind"));
    }
}
