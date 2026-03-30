use chrono::{DateTime, NaiveDateTime, Utc};
use rusqlite::params;

use crate::error::Result;
use crate::storage::Database;

/// Standard relationship types between memories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationType {
    /// "X caused Y" (error → root cause)
    CausedBy,
    /// "X was solved by Y" (error → fix)
    SolvedBy,
    /// "X depends on Y" (task → prerequisite)
    DependsOn,
    /// "X was replaced by Y" (old → new approach)
    SupersededBy,
    /// Generic association
    RelatedTo,
    /// "X is part of Y" (subtask → parent)
    PartOf,
    /// "X contradicts Y" (opposing learnings)
    ConflictsWith,
    /// "X was confirmed by Y" (learning → evidence)
    ValidatedBy,
    /// User-defined relationship type
    Custom(String),
}

impl RelationType {
    /// Convert to string for storage.
    pub fn as_str(&self) -> &str {
        match self {
            Self::CausedBy => "caused_by",
            Self::SolvedBy => "solved_by",
            Self::DependsOn => "depends_on",
            Self::SupersededBy => "superseded_by",
            Self::RelatedTo => "related_to",
            Self::PartOf => "part_of",
            Self::ConflictsWith => "conflicts_with",
            Self::ValidatedBy => "validated_by",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Parse from stored string.
    pub fn from_str(s: &str) -> Self {
        match s {
            "caused_by" => Self::CausedBy,
            "solved_by" => Self::SolvedBy,
            "depends_on" => Self::DependsOn,
            "superseded_by" => Self::SupersededBy,
            "related_to" => Self::RelatedTo,
            "part_of" => Self::PartOf,
            "conflicts_with" => Self::ConflictsWith,
            "validated_by" => Self::ValidatedBy,
            other => Self::Custom(other.to_string()),
        }
    }
}

/// Graph relationship operations on memories.
pub struct GraphMemory;

/// Snapshot of a memory's current state/conflict markers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StateConflictSnapshot {
    /// This memory has been superseded by a newer memory.
    pub is_superseded: bool,
    /// This memory supersedes at least one older memory.
    pub supersedes_other: bool,
    /// This memory participates in at least one explicit conflict edge.
    pub has_conflict: bool,
    /// Optional temporal validity start.
    pub valid_from: Option<DateTime<Utc>>,
    /// Optional temporal validity end.
    pub valid_until: Option<DateTime<Utc>>,
}

impl StateConflictSnapshot {
    /// Returns true when the memory is valid at the specified time.
    pub fn is_valid_at(&self, at: DateTime<Utc>) -> bool {
        let starts_ok = self.valid_from.is_none_or(|value| value <= at);
        let ends_ok = self.valid_until.is_none_or(|value| value > at);
        starts_ok && ends_ok
    }
}

impl GraphMemory {
    /// Create a relationship between two memories.
    pub fn relate(
        db: &Database,
        source_id: i64,
        target_id: i64,
        relation: &RelationType,
    ) -> Result<()> {
        db.with_writer(|conn| {
            conn.execute(
                "INSERT OR IGNORE INTO memory_relations (source_id, target_id, relation)
                 VALUES (?1, ?2, ?3)",
                params![source_id, target_id, relation.as_str()],
            )?;
            Ok(())
        })
    }

    /// Remove a relationship.
    pub fn unrelate(
        db: &Database,
        source_id: i64,
        target_id: i64,
        relation: &RelationType,
    ) -> Result<bool> {
        db.with_writer(|conn| {
            let rows = conn.execute(
                "DELETE FROM memory_relations WHERE source_id = ?1 AND target_id = ?2 AND relation = ?3",
                params![source_id, target_id, relation.as_str()],
            )?;
            Ok(rows > 0)
        })
    }

    /// Find memories related to a given memory via recursive CTE traversal.
    ///
    /// Returns memory IDs with their relationship type and hop distance.
    /// Includes cycle prevention and depth limits.
    pub fn traverse(db: &Database, start_id: i64, max_depth: u32) -> Result<Vec<GraphNode>> {
        db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "WITH RECURSIVE chain(id, relation, depth, path) AS (
                    SELECT target_id, relation, 1, source_id || '→' || target_id
                    FROM memory_relations
                    WHERE source_id = ?1
                      AND (valid_until IS NULL OR valid_until > datetime('now'))

                    UNION ALL

                    SELECT r.target_id, r.relation, c.depth + 1,
                           c.path || '→' || r.target_id
                    FROM memory_relations r
                    JOIN chain c ON r.source_id = c.id
                    WHERE c.depth < ?2
                      AND c.path NOT LIKE '%' || r.target_id || '%'
                      AND (r.valid_until IS NULL OR r.valid_until > datetime('now'))
                )
                SELECT DISTINCT id, relation, depth
                FROM chain
                ORDER BY depth ASC",
            )?;

            let nodes: Vec<GraphNode> = stmt
                .query_map(params![start_id, max_depth], |row| {
                    Ok(GraphNode {
                        memory_id: row.get(0)?,
                        relation: RelationType::from_str(&row.get::<_, String>(1)?),
                        depth: row.get(2)?,
                    })
                })?
                .filter_map(|r| r.ok())
                .collect();

            Ok(nodes)
        })
    }

    /// Get direct relationships from a memory.
    pub fn direct_relations(db: &Database, memory_id: i64) -> Result<Vec<GraphNode>> {
        db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT target_id, relation FROM memory_relations
                 WHERE source_id = ?1
                   AND (valid_until IS NULL OR valid_until > datetime('now'))",
            )?;

            let nodes: Vec<GraphNode> = stmt
                .query_map([memory_id], |row| {
                    Ok(GraphNode {
                        memory_id: row.get(0)?,
                        relation: RelationType::from_str(&row.get::<_, String>(1)?),
                        depth: 1,
                    })
                })?
                .filter_map(|r| r.ok())
                .collect();

            Ok(nodes)
        })
    }

    /// Get direct supersession successors (`old -> new`) for a memory.
    pub fn superseded_successors(db: &Database, memory_id: i64) -> Result<Vec<i64>> {
        db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT target_id FROM memory_relations
                 WHERE source_id = ?1
                   AND relation = 'superseded_by'
                   AND (valid_until IS NULL OR valid_until > datetime('now'))",
            )?;

            let ids: Vec<i64> = stmt
                .query_map([memory_id], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();

            Ok(ids)
        })
    }

    /// Get direct supersession predecessors (`old -> new`) for a memory.
    pub fn superseded_predecessors(db: &Database, memory_id: i64) -> Result<Vec<i64>> {
        db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT source_id FROM memory_relations
                 WHERE target_id = ?1
                   AND relation = 'superseded_by'
                   AND (valid_until IS NULL OR valid_until > datetime('now'))",
            )?;

            let ids: Vec<i64> = stmt
                .query_map([memory_id], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();

            Ok(ids)
        })
    }

    /// Get directly conflicting neighbors for a memory.
    pub fn conflict_neighbors(db: &Database, memory_id: i64) -> Result<Vec<i64>> {
        db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT CASE
                        WHEN source_id = ?1 THEN target_id
                        ELSE source_id
                    END AS neighbor_id
                 FROM memory_relations
                 WHERE (source_id = ?1 OR target_id = ?1)
                   AND relation = 'conflicts_with'
                   AND (valid_until IS NULL OR valid_until > datetime('now'))",
            )?;

            let ids: Vec<i64> = stmt
                .query_map([memory_id], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();

            Ok(ids)
        })
    }

    /// Load the current state/conflict markers for a memory.
    pub fn state_conflict_snapshot(
        db: &Database,
        memory_id: i64,
    ) -> Result<Option<StateConflictSnapshot>> {
        db.with_reader(|conn| {
            let result = conn.query_row(
                "SELECT
                    EXISTS(
                        SELECT 1 FROM memory_relations r
                        WHERE r.source_id = m.id
                          AND r.relation = 'superseded_by'
                          AND (r.valid_until IS NULL OR r.valid_until > datetime('now'))
                    ) AS is_superseded,
                    EXISTS(
                        SELECT 1 FROM memory_relations r
                        WHERE r.target_id = m.id
                          AND r.relation = 'superseded_by'
                          AND (r.valid_until IS NULL OR r.valid_until > datetime('now'))
                    ) AS supersedes_other,
                    EXISTS(
                        SELECT 1 FROM memory_relations r
                        WHERE (r.source_id = m.id OR r.target_id = m.id)
                          AND r.relation = 'conflicts_with'
                          AND (r.valid_until IS NULL OR r.valid_until > datetime('now'))
                    ) AS has_conflict,
                    m.valid_from,
                    m.valid_until
                 FROM memories m
                 WHERE m.id = ?1",
                [memory_id],
                |row| {
                    Ok(StateConflictSnapshot {
                        is_superseded: row.get::<_, bool>(0)?,
                        supersedes_other: row.get::<_, bool>(1)?,
                        has_conflict: row.get::<_, bool>(2)?,
                        valid_from: parse_optional_temporal(row.get(3)?),
                        valid_until: parse_optional_temporal(row.get(4)?),
                    })
                },
            );

            match result {
                Ok(snapshot) => Ok(Some(snapshot)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })
    }

    /// Compute a scoring boost for connected memories.
    ///
    /// Returns `1.0 / (depth + 1)`:
    /// - Direct connection (depth 1): 0.5x boost
    /// - Two hops (depth 2): 0.33x boost
    pub fn depth_boost(depth: u32) -> f32 {
        1.0 / (depth as f32 + 1.0)
    }
}

fn parse_optional_temporal(value: Option<String>) -> Option<DateTime<Utc>> {
    let value = value?;
    if let Ok(parsed) = DateTime::parse_from_rfc3339(&value) {
        return Some(parsed.with_timezone(&Utc));
    }
    if let Ok(parsed) = NaiveDateTime::parse_from_str(&value, "%Y-%m-%d %H:%M:%S") {
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(parsed, Utc));
    }
    None
}

/// A node in the memory graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Memory ID of the connected memory.
    pub memory_id: i64,
    /// Relationship type from the source.
    pub relation: RelationType,
    /// Number of hops from the starting memory.
    pub depth: u32,
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

        // Insert test memories
        for i in 1..=5 {
            db.with_writer(|conn| {
                conn.execute(
                    "INSERT INTO memories (id, searchable_text, memory_type, content_hash, record_json)
                     VALUES (?1, ?2, 'semantic', ?3, '{}')",
                    params![i, format!("memory {i}"), format!("h{i}")],
                )?;
                Ok(())
            }).expect("insert");
        }
        db
    }

    #[test]
    fn create_relationship() {
        let db = setup();
        GraphMemory::relate(&db, 1, 2, &RelationType::CausedBy).expect("relate");

        let rels = GraphMemory::direct_relations(&db, 1).expect("direct");
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].memory_id, 2);
        assert_eq!(rels[0].relation, RelationType::CausedBy);
    }

    #[test]
    fn duplicate_relation_ignored() {
        let db = setup();
        GraphMemory::relate(&db, 1, 2, &RelationType::SolvedBy).expect("first");
        GraphMemory::relate(&db, 1, 2, &RelationType::SolvedBy)
            .expect("duplicate should be ignored");

        let rels = GraphMemory::direct_relations(&db, 1).expect("direct");
        assert_eq!(rels.len(), 1);
    }

    #[test]
    fn remove_relationship() {
        let db = setup();
        GraphMemory::relate(&db, 1, 2, &RelationType::RelatedTo).expect("relate");
        let removed = GraphMemory::unrelate(&db, 1, 2, &RelationType::RelatedTo).expect("unrelate");
        assert!(removed);

        let rels = GraphMemory::direct_relations(&db, 1).expect("direct");
        assert!(rels.is_empty());
    }

    #[test]
    fn traverse_chain() {
        let db = setup();
        // 1 → 2 → 3
        GraphMemory::relate(&db, 1, 2, &RelationType::CausedBy).expect("1→2");
        GraphMemory::relate(&db, 2, 3, &RelationType::SolvedBy).expect("2→3");

        let nodes = GraphMemory::traverse(&db, 1, 3).expect("traverse");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].memory_id, 2);
        assert_eq!(nodes[0].depth, 1);
        assert_eq!(nodes[1].memory_id, 3);
        assert_eq!(nodes[1].depth, 2);
    }

    #[test]
    fn traverse_depth_limit() {
        let db = setup();
        // 1 → 2 → 3 → 4
        GraphMemory::relate(&db, 1, 2, &RelationType::RelatedTo).expect("1→2");
        GraphMemory::relate(&db, 2, 3, &RelationType::RelatedTo).expect("2→3");
        GraphMemory::relate(&db, 3, 4, &RelationType::RelatedTo).expect("3→4");

        let nodes = GraphMemory::traverse(&db, 1, 2).expect("traverse");
        assert_eq!(nodes.len(), 2); // Only 2 and 3, not 4
    }

    #[test]
    fn traverse_cycle_prevention() {
        let db = setup();
        // 1 → 2 → 3 → 1 (cycle)
        GraphMemory::relate(&db, 1, 2, &RelationType::RelatedTo).expect("1→2");
        GraphMemory::relate(&db, 2, 3, &RelationType::RelatedTo).expect("2→3");
        GraphMemory::relate(&db, 3, 1, &RelationType::RelatedTo).expect("3→1 cycle");

        let nodes = GraphMemory::traverse(&db, 1, 10).expect("traverse");
        // Should not loop infinitely
        assert!(nodes.len() <= 3);
    }

    #[test]
    fn depth_boost_calculation() {
        assert!((GraphMemory::depth_boost(1) - 0.5).abs() < 0.01);
        assert!((GraphMemory::depth_boost(2) - 0.333).abs() < 0.01);
        assert!((GraphMemory::depth_boost(0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn supersession_helpers_expose_predecessors_and_successors() {
        let db = setup();
        GraphMemory::relate(&db, 1, 2, &RelationType::SupersededBy).expect("1→2");

        let successors = GraphMemory::superseded_successors(&db, 1).expect("successors");
        let predecessors = GraphMemory::superseded_predecessors(&db, 2).expect("predecessors");

        assert_eq!(successors, vec![2]);
        assert_eq!(predecessors, vec![1]);
    }

    #[test]
    fn state_snapshot_reports_supersession_flags() {
        let db = setup();
        GraphMemory::relate(&db, 1, 2, &RelationType::SupersededBy).expect("1→2");

        let old = GraphMemory::state_conflict_snapshot(&db, 1)
            .expect("snapshot")
            .expect("present");
        let current = GraphMemory::state_conflict_snapshot(&db, 2)
            .expect("snapshot")
            .expect("present");

        assert!(old.is_superseded);
        assert!(!old.supersedes_other);
        assert!(!current.is_superseded);
        assert!(current.supersedes_other);
    }

    #[test]
    fn conflict_neighbors_include_both_directions() {
        let db = setup();
        GraphMemory::relate(&db, 2, 4, &RelationType::ConflictsWith).expect("2↔4");

        let from_source = GraphMemory::conflict_neighbors(&db, 2).expect("neighbors");
        let from_target = GraphMemory::conflict_neighbors(&db, 4).expect("neighbors");

        assert_eq!(from_source, vec![4]);
        assert_eq!(from_target, vec![2]);
    }

    #[test]
    fn relation_type_roundtrip() {
        let types = vec![
            RelationType::CausedBy,
            RelationType::SolvedBy,
            RelationType::DependsOn,
            RelationType::SupersededBy,
            RelationType::RelatedTo,
            RelationType::PartOf,
            RelationType::ConflictsWith,
            RelationType::ValidatedBy,
            RelationType::Custom("my_custom".to_string()),
        ];
        for rt in &types {
            let s = rt.as_str();
            let parsed = RelationType::from_str(s);
            assert_eq!(&parsed, rt, "roundtrip failed for {s}");
        }
    }
}
