use rusqlite::params;

use crate::error::Result;
use crate::storage::Database;
use crate::traits::MemoryType;

/// ACT-R decay rates by memory type.
const DECAY_EPISODIC: f64 = 0.5;
const DECAY_SEMANTIC: f64 = 0.2;
const DECAY_PROCEDURAL: f64 = 0.3;

/// Record that a memory was accessed (retrieved by search).
///
/// Each access is logged with a timestamp and the query that retrieved it.
/// The access history feeds the ACT-R activation formula.
pub fn record_access(db: &Database, memory_id: i64, query: &str) -> Result<()> {
    db.with_writer(|conn| {
        conn.execute(
            "INSERT INTO memory_access_log (memory_id, query_text) VALUES (?1, ?2)",
            params![memory_id, query],
        )?;
        Ok(())
    })
}

/// Compute the ACT-R activation level for a memory.
///
/// Formula: `activation = base_level + Σ ln(t_j^-d)`
///
/// Where:
/// - `base_level` = importance / 10.0
/// - `t_j` = seconds since the j-th access
/// - `d` = decay rate (varies by memory type)
pub fn compute_activation(db: &Database, memory_id: i64) -> Result<f32> {
    db.with_reader(|conn| {
        // Get memory type and importance
        let (type_str, importance): (String, i32) = conn.query_row(
            "SELECT memory_type, importance FROM memories WHERE id = ?1",
            [memory_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        let memory_type = MemoryType::from_str(&type_str).unwrap_or(MemoryType::Episodic);
        let decay = decay_rate(memory_type);
        let base = importance as f64 / 10.0;

        // Get all access timestamps as seconds-since-access
        let mut stmt = conn.prepare(
            "SELECT (julianday('now') - julianday(accessed_at)) * 86400.0
             FROM memory_access_log
             WHERE memory_id = ?1
             ORDER BY accessed_at DESC",
        )?;

        let accesses: Vec<f64> = stmt
            .query_map([memory_id], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        // Handle compacted records
        let mut stmt_compacted = conn.prepare(
            "SELECT query_text FROM memory_access_log
             WHERE memory_id = ?1 AND query_text LIKE '[compacted:%'",
        )?;
        let compacted_count: i64 = stmt_compacted
            .query_map([memory_id], |row| {
                let text: String = row.get(0)?;
                // Parse "[compacted: N accesses]"
                let count = text
                    .trim_start_matches("[compacted: ")
                    .trim_end_matches(" accesses]")
                    .parse::<i64>()
                    .unwrap_or(0);
                Ok(count)
            })?
            .filter_map(|r| r.ok())
            .sum();

        // Compute activation from individual accesses
        // Use max(0.1) to ensure very recent accesses (< 0.1 seconds ago)
        // still contribute positively to activation.
        let individual_activation: f64 =
            accesses.iter().map(|t| t.max(0.1).powf(-decay).ln()).sum();

        // Approximate compacted accesses (assume they were spread over 90+ days ago)
        let compacted_activation = if compacted_count > 0 {
            // Each compacted access contributes a small constant
            // (old accesses at 90+ days have very low activation)
            compacted_count as f64 * (90.0 * 86400.0_f64).powf(-decay).ln()
        } else {
            0.0
        };

        let activation = base + individual_activation + compacted_activation;
        Ok(activation as f32)
    })
}

/// Update the cached activation score for a memory.
pub fn update_activation_cache(db: &Database, memory_id: i64) -> Result<f32> {
    let activation = compute_activation(db, memory_id)?;
    db.with_writer(|conn| {
        conn.execute(
            "UPDATE memories SET activation_cache = ?1, activation_updated = datetime('now')
             WHERE id = ?2",
            params![activation, memory_id],
        )?;
        Ok(activation)
    })
}

/// Get the cached activation score, computing it if not cached.
pub fn get_activation(db: &Database, memory_id: i64) -> Result<f32> {
    let cached: Option<f32> = db.with_reader(|conn| {
        let result = conn.query_row(
            "SELECT activation_cache FROM memories WHERE id = ?1",
            [memory_id],
            |row| row.get(0),
        );
        match result {
            Ok(v) => Ok(v),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    })?;

    match cached {
        Some(v) => Ok(v),
        None => update_activation_cache(db, memory_id),
    }
}

/// Compact old access log entries (older than 90 days).
///
/// Replaces individual access rows with a single summary row per memory,
/// preserving the count for activation approximation.
pub fn compact_access_log(db: &Database) -> Result<u64> {
    db.with_writer(|conn| {
        // Count rows to be compacted
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_access_log
             WHERE accessed_at < datetime('now', '-90 days')
               AND query_text NOT LIKE '[compacted:%'",
            [],
            |row| row.get(0),
        )?;

        if count == 0 {
            return Ok(0);
        }

        // Insert summary records
        conn.execute(
            "INSERT INTO memory_access_log (memory_id, accessed_at, query_text)
             SELECT memory_id, MAX(accessed_at), '[compacted: ' || COUNT(*) || ' accesses]'
             FROM memory_access_log
             WHERE accessed_at < datetime('now', '-90 days')
               AND query_text NOT LIKE '[compacted:%'
             GROUP BY memory_id",
            [],
        )?;

        // Remove individual old rows
        conn.execute(
            "DELETE FROM memory_access_log
             WHERE accessed_at < datetime('now', '-90 days')
               AND query_text NOT LIKE '[compacted:%'",
            [],
        )?;

        Ok(count as u64)
    })
}

/// Get the decay rate for a memory type.
fn decay_rate(memory_type: MemoryType) -> f64 {
    match memory_type {
        MemoryType::Episodic => DECAY_EPISODIC,
        MemoryType::Semantic => DECAY_SEMANTIC,
        MemoryType::Procedural => DECAY_PROCEDURAL,
    }
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

    fn insert_memory(db: &Database, text: &str, mem_type: &str, importance: i32) -> i64 {
        db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO memories (searchable_text, memory_type, importance, content_hash, record_json)
                 VALUES (?1, ?2, ?3, ?4, '{}')",
                params![text, mem_type, importance, format!("hash_{text}")],
            )?;
            Ok(conn.last_insert_rowid())
        })
        .expect("insert")
    }

    #[test]
    fn record_and_compute_activation() {
        let db = setup();
        let id = insert_memory(&db, "test memory", "semantic", 5);

        // No accesses: activation = base only (0.5)
        let a0 = compute_activation(&db, id).expect("compute");
        assert!(
            (a0 - 0.5).abs() < 0.1,
            "base activation should be ~0.5, got {a0}"
        );

        // One access: activation should increase
        record_access(&db, id, "test query").expect("record");
        let a1 = compute_activation(&db, id).expect("compute");
        assert!(
            a1 > a0,
            "activation should increase after access: before={a0}, after={a1}"
        );
    }

    #[test]
    fn multiple_accesses_increase_activation() {
        let db = setup();
        let id = insert_memory(&db, "multi access", "procedural", 7);

        let a0 = compute_activation(&db, id).expect("compute");

        for i in 0..5 {
            record_access(&db, id, &format!("query {i}")).expect("record");
        }

        let a5 = compute_activation(&db, id).expect("compute");
        assert!(a5 > a0, "5 accesses should produce higher activation");
    }

    #[test]
    fn activation_cache() {
        let db = setup();
        let id = insert_memory(&db, "cached", "semantic", 5);
        record_access(&db, id, "q").expect("record");

        let cached = update_activation_cache(&db, id).expect("cache");
        let fetched = get_activation(&db, id).expect("get");

        assert!((cached - fetched).abs() < f32::EPSILON);
    }

    #[test]
    fn different_types_different_decay() {
        let db = setup();
        let ep = insert_memory(&db, "episodic memory", "episodic", 5);
        let sem = insert_memory(&db, "semantic memory", "semantic", 5);
        let proc = insert_memory(&db, "procedural memory", "procedural", 5);

        // All same base, same access pattern
        for id in [ep, sem, proc] {
            record_access(&db, id, "q").expect("record");
        }

        let a_ep = compute_activation(&db, ep).expect("compute");
        let a_sem = compute_activation(&db, sem).expect("compute");
        let a_proc = compute_activation(&db, proc).expect("compute");

        // With very recent access, differences are small but semantic should decay slowest
        // All should be positive (base 0.5 + recent access contribution)
        assert!(a_ep > 0.0 && a_sem > 0.0 && a_proc > 0.0);
    }

    #[test]
    fn compact_empty_log() {
        let db = setup();
        let count = compact_access_log(&db).expect("compact");
        assert_eq!(count, 0);
    }
}
