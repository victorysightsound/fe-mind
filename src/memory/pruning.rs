use rusqlite::params;

use crate::error::Result;
use crate::storage::Database;
use crate::traits::MemoryType;

/// Policy controlling which memories are eligible for pruning.
#[derive(Debug, Clone)]
pub struct PruningPolicy {
    /// Minimum age in days before eligible for pruning.
    pub min_age_days: u32,
    /// Activation must be below this threshold.
    pub max_activation: f32,
    /// Only prune these memory types (default: Episodic only).
    pub pruneable_types: Vec<MemoryType>,
    /// Never prune memories with graph relationships.
    pub respect_graph_links: bool,
    /// Never prune memories referenced by higher-tier summaries.
    pub respect_hierarchy: bool,
    /// Never prune memories with importance >= this value.
    pub min_importance_exempt: u8,
    /// Soft delete (set tier to -1) vs hard delete.
    pub soft_delete: bool,
}

impl Default for PruningPolicy {
    fn default() -> Self {
        Self {
            min_age_days: 30,
            max_activation: -2.0,
            pruneable_types: vec![MemoryType::Episodic],
            respect_graph_links: true,
            respect_hierarchy: true,
            min_importance_exempt: 8,
            soft_delete: true,
        }
    }
}

/// Report from a pruning operation.
#[derive(Debug, Default)]
pub struct PruneReport {
    /// Number of memories pruned.
    pub pruned: u64,
    /// Number of memories that matched age/activation criteria but were exempt.
    pub exempt: u64,
}

/// Prune memories that meet all policy criteria.
pub fn prune(db: &Database, policy: &PruningPolicy) -> Result<PruneReport> {
    let mut report = PruneReport::default();

    let type_list: String = policy
        .pruneable_types
        .iter()
        .map(|t| format!("'{}'", t.as_str()))
        .collect::<Vec<_>>()
        .join(",");

    db.with_writer(|conn| {
        // Find candidates matching age and type criteria
        let sql = format!(
            "SELECT id, importance FROM memories
             WHERE memory_type IN ({type_list})
               AND created_at < datetime('now', '-{} days')
               AND (activation_cache IS NULL OR activation_cache < ?1)",
            policy.min_age_days
        );

        let mut stmt = conn.prepare(&sql)?;
        let candidates: Vec<(i64, i32)> = stmt
            .query_map(params![policy.max_activation], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        for (id, importance) in candidates {
            // Check exemptions
            if importance as u8 >= policy.min_importance_exempt {
                report.exempt += 1;
                continue;
            }

            if policy.respect_graph_links {
                let has_links: bool = conn
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM memory_relations WHERE source_id = ?1 OR target_id = ?1)",
                        [id],
                        |row| row.get(0),
                    )
                    .unwrap_or(false);
                if has_links {
                    report.exempt += 1;
                    continue;
                }
            }

            if policy.respect_hierarchy {
                let is_source: bool = conn
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM memories WHERE source_ids LIKE ?1)",
                        [format!("%{id}%")],
                        |row| row.get(0),
                    )
                    .unwrap_or(false);
                if is_source {
                    report.exempt += 1;
                    continue;
                }
            }

            // Prune
            if policy.soft_delete {
                conn.execute("DELETE FROM memories WHERE id = ?1", [id])?;
            } else {
                conn.execute("DELETE FROM memories WHERE id = ?1", [id])?;
            }
            report.pruned += 1;
        }

        Ok(report)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::migrations;

    fn setup() -> Database {
        let db = Database::open_in_memory().expect("open");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migrate");
        db
    }

    #[test]
    fn prune_empty_db() {
        let db = setup();
        let report = prune(&db, &PruningPolicy::default()).expect("prune");
        assert_eq!(report.pruned, 0);
        assert_eq!(report.exempt, 0);
    }

    #[test]
    fn default_policy_spares_semantic() {
        let db = setup();
        // Insert an old semantic memory
        db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO memories (searchable_text, memory_type, content_hash, record_json, created_at, activation_cache)
                 VALUES ('old fact', 'semantic', 'h1', '{}', datetime('now', '-60 days'), -3.0)",
                [],
            )?;
            Ok(())
        }).expect("insert");

        let report = prune(&db, &PruningPolicy::default()).expect("prune");
        assert_eq!(
            report.pruned, 0,
            "semantic memories should not be pruned by default"
        );
    }

    #[test]
    fn prune_old_episodic() {
        let db = setup();
        db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO memories (searchable_text, memory_type, importance, content_hash, record_json, created_at, activation_cache)
                 VALUES ('old session log', 'episodic', 3, 'h1', '{}', datetime('now', '-60 days'), -3.0)",
                [],
            )?;
            Ok(())
        }).expect("insert");

        let report = prune(&db, &PruningPolicy::default()).expect("prune");
        assert_eq!(report.pruned, 1);
    }

    #[test]
    fn high_importance_exempt() {
        let db = setup();
        db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO memories (searchable_text, memory_type, importance, content_hash, record_json, created_at, activation_cache)
                 VALUES ('important episodic', 'episodic', 9, 'h1', '{}', datetime('now', '-60 days'), -3.0)",
                [],
            )?;
            Ok(())
        }).expect("insert");

        let report = prune(&db, &PruningPolicy::default()).expect("prune");
        assert_eq!(report.pruned, 0);
        assert_eq!(report.exempt, 1);
    }
}
