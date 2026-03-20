use rusqlite::params;

use crate::error::{MindCoreError, Result};
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
    pub fn traverse(
        db: &Database,
        start_id: i64,
        max_depth: u32,
    ) -> Result<Vec<GraphNode>> {
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

    /// Compute a scoring boost for connected memories.
    ///
    /// Returns `1.0 / (depth + 1)`:
    /// - Direct connection (depth 1): 0.5x boost
    /// - Two hops (depth 2): 0.33x boost
    pub fn depth_boost(depth: u32) -> f32 {
        1.0 / (depth as f32 + 1.0)
    }
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
        db.with_writer(|conn| { migrations::migrate(conn)?; Ok(()) }).expect("migrate");

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
        GraphMemory::relate(&db, 1, 2, &RelationType::SolvedBy).expect("duplicate should be ignored");

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
    fn relation_type_roundtrip() {
        let types = vec![
            RelationType::CausedBy, RelationType::SolvedBy, RelationType::DependsOn,
            RelationType::SupersededBy, RelationType::RelatedTo, RelationType::PartOf,
            RelationType::ConflictsWith, RelationType::ValidatedBy,
            RelationType::Custom("my_custom".to_string()),
        ];
        for rt in &types {
            let s = rt.as_str();
            let parsed = RelationType::from_str(s);
            assert_eq!(&parsed, rt, "roundtrip failed for {s}");
        }
    }
}
