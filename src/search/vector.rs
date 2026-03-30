use rusqlite::params;
use rusqlite::params_from_iter;

use crate::embeddings::pooling::{bytes_to_vec, cosine_similarity};
use crate::error::Result;
use crate::search::fts5::FtsResult;
use crate::storage::Database;

/// Brute-force vector similarity search.
///
/// Loads all vectors matching the current model, computes cosine similarity
/// with the query vector, and returns the top-k results.
pub struct VectorSearch;

impl VectorSearch {
    /// Search for memories similar to the query vector.
    ///
    /// Only considers vectors produced by the specified model (Decision 020).
    pub fn search(
        db: &Database,
        query_vector: &[f32],
        model_names: &[String],
        limit: usize,
    ) -> Result<Vec<FtsResult>> {
        if model_names.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = std::iter::repeat_n("?", model_names.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT memory_id, embedding FROM memory_vectors WHERE model_name IN ({placeholders})"
        );

        db.with_reader(|conn| {
            let mut stmt = conn.prepare(&sql)?;

            let mut scored: Vec<(i64, f32)> = stmt
                .query_map(params_from_iter(model_names.iter()), |row| {
                    let id: i64 = row.get(0)?;
                    let blob: Vec<u8> = row.get(1)?;
                    Ok((id, blob))
                })?
                .filter_map(|r| r.ok())
                .map(|(id, blob)| {
                    let stored = bytes_to_vec(&blob);
                    let sim = cosine_similarity(query_vector, &stored);
                    (id, sim)
                })
                .collect();

            // Sort by similarity descending
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(limit);

            Ok(scored
                .into_iter()
                .map(|(id, score)| FtsResult {
                    memory_id: id,
                    score,
                })
                .collect())
        })
    }

    /// Store a vector for a memory.
    pub fn store_vector(
        db: &Database,
        memory_id: i64,
        embedding: &[f32],
        model_name: &str,
        content_hash: &str,
    ) -> Result<()> {
        let blob = crate::embeddings::pooling::vec_to_bytes(embedding);
        let dims = embedding.len() as i32;

        db.with_writer(|conn| {
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors (memory_id, embedding, model_name, dimensions, content_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![memory_id, blob, model_name, dims, content_hash],
            )?;
            Ok(())
        })
    }

    /// Check if a vector exists for a content hash (skip re-embedding).
    pub fn vector_exists(db: &Database, content_hash: &str) -> Result<bool> {
        db.with_reader(|conn| {
            let count: i32 = conn.query_row(
                "SELECT COUNT(*) FROM memory_vectors WHERE content_hash = ?1",
                [content_hash],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })
    }

    /// Count vectors stored for the given model.
    pub fn count_vectors(db: &Database, model_name: &str) -> Result<usize> {
        Self::count_vectors_for_models(db, &[model_name.to_string()])
    }

    /// Count vectors stored for any of the compatible model labels.
    pub fn count_vectors_for_models(db: &Database, model_names: &[String]) -> Result<usize> {
        if model_names.is_empty() {
            return Ok(0);
        }

        let placeholders = std::iter::repeat_n("?", model_names.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql =
            format!("SELECT COUNT(*) FROM memory_vectors WHERE model_name IN ({placeholders})");

        db.with_reader(|conn| {
            let count: i64 =
                conn.query_row(&sql, params_from_iter(model_names.iter()), |row| row.get(0))?;
            Ok(count as usize)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::pooling::normalize_l2;
    use crate::storage::migrations;

    fn setup() -> Database {
        let db = Database::open_in_memory().expect("open");
        db.with_writer(|conn| {
            migrations::migrate(conn)?;
            Ok(())
        })
        .expect("migrate");

        // Insert test memories
        db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO memories (id, searchable_text, memory_type, content_hash, record_json)
                 VALUES (1, 'auth error', 'semantic', 'h1', '{}')",
                [],
            )?;
            conn.execute(
                "INSERT INTO memories (id, searchable_text, memory_type, content_hash, record_json)
                 VALUES (2, 'db timeout', 'episodic', 'h2', '{}')",
                [],
            )?;
            conn.execute(
                "INSERT INTO memories (id, searchable_text, memory_type, content_hash, record_json)
                 VALUES (3, 'build fix', 'procedural', 'h3', '{}')",
                [],
            )?;
            Ok(())
        })
        .expect("insert");
        db
    }

    #[test]
    fn store_and_search_vectors() {
        let db = setup();

        // Store vectors (simple 3-dim for testing)
        let v1 = normalize_l2(&[1.0, 0.0, 0.0]);
        let v2 = normalize_l2(&[0.0, 1.0, 0.0]);
        let v3 = normalize_l2(&[0.9, 0.1, 0.0]);

        VectorSearch::store_vector(&db, 1, &v1, "test-model", "h1").expect("store 1");
        VectorSearch::store_vector(&db, 2, &v2, "test-model", "h2").expect("store 2");
        VectorSearch::store_vector(&db, 3, &v3, "test-model", "h3").expect("store 3");

        // Search with a query similar to v1
        let query = normalize_l2(&[1.0, 0.0, 0.0]);
        let results =
            VectorSearch::search(&db, &query, &[String::from("test-model")], 10).expect("search");

        assert_eq!(results.len(), 3);
        // v1 should be most similar (identical)
        assert_eq!(results[0].memory_id, 1);
        assert!((results[0].score - 1.0).abs() < 0.01);
        // v3 should be second (0.9 component)
        assert_eq!(results[1].memory_id, 3);
    }

    #[test]
    fn model_name_filter() {
        let db = setup();
        let v = normalize_l2(&[1.0, 0.0, 0.0]);

        VectorSearch::store_vector(&db, 1, &v, "model-a", "h1").expect("store");
        VectorSearch::store_vector(&db, 2, &v, "model-b", "h2").expect("store");

        let results =
            VectorSearch::search(&db, &v, &[String::from("model-a")], 10).expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, 1);
    }

    #[test]
    fn vector_exists_check() {
        let db = setup();
        let v = normalize_l2(&[1.0, 0.0, 0.0]);

        assert!(!VectorSearch::vector_exists(&db, "h1").expect("exists"));
        VectorSearch::store_vector(&db, 1, &v, "test", "h1").expect("store");
        assert!(VectorSearch::vector_exists(&db, "h1").expect("exists"));
    }

    #[test]
    fn limit_respected() {
        let db = setup();
        let v = normalize_l2(&[1.0, 0.0, 0.0]);
        for i in 1..=3 {
            VectorSearch::store_vector(&db, i, &v, "test", &format!("h{i}")).expect("store");
        }

        let results = VectorSearch::search(&db, &v, &[String::from("test")], 2).expect("search");
        assert_eq!(results.len(), 2);
    }
}
