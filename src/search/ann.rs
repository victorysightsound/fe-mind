//! ANN (Approximate Nearest Neighbor) vector search via HNSW.
//!
//! Uses instant-distance for pure Rust HNSW indexing.
//! Feature-gated behind `ann`.
//!
//! The HNSW index is built from vectors in the memory_vectors table
//! and stored in memory. It must be rebuilt when significant vectors are added.

#[cfg(feature = "ann")]
mod inner {
    use instant_distance::{Builder, HnswMap, Search};
    use std::sync::Mutex;

    use crate::embeddings::pooling::{bytes_to_vec, cosine_similarity};
    use crate::error::Result;
    use crate::search::fts5::FtsResult;
    use crate::storage::Database;

    /// HNSW point wrapper for our vectors.
    #[derive(Clone)]
    struct VecPoint(Vec<f32>);

    impl instant_distance::Point for VecPoint {
        fn distance(&self, other: &Self) -> f32 {
            // Cosine distance = 1 - cosine_similarity
            // Our vectors are L2-normalized, so dot product = cosine similarity
            let sim = cosine_similarity(&self.0, &other.0);
            1.0 - sim
        }
    }

    /// ANN index for fast approximate vector search.
    pub struct AnnIndex {
        /// HNSW map: maps points to memory IDs
        index: Mutex<Option<HnswMap<VecPoint, i64>>>,
        /// Number of vectors in the index
        count: std::sync::atomic::AtomicUsize,
    }

    impl AnnIndex {
        pub fn new() -> Self {
            Self {
                index: Mutex::new(None),
                count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        /// Build the HNSW index from all vectors in the database for a given model.
        pub fn build(&self, db: &Database, model_name: &str) -> Result<usize> {
            let vectors = db.with_reader(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT memory_id, embedding FROM memory_vectors WHERE model_name = ?1",
                )?;
                let rows: Vec<(i64, Vec<u8>)> = stmt.query_map(
                    [model_name],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?.filter_map(|r| r.ok()).collect();
                Ok::<_, crate::error::MindCoreError>(rows)
            })?;

            if vectors.is_empty() {
                return Ok(0);
            }

            let points: Vec<VecPoint> = vectors.iter()
                .map(|(_, blob)| VecPoint(bytes_to_vec(blob)))
                .collect();
            let values: Vec<i64> = vectors.iter().map(|(id, _)| *id).collect();

            let hnsw = Builder::default().build(points, values);

            let vec_count = vectors.len();
            let mut guard = self.index.lock().map_err(|e| {
                crate::error::MindCoreError::Embedding(format!("index lock: {e}"))
            })?;
            *guard = Some(hnsw);
            self.count.store(vec_count, std::sync::atomic::Ordering::Relaxed);

            tracing::debug!("ANN index built with {vec_count} vectors");
            Ok(vec_count)
        }

        /// Search the HNSW index for nearest neighbors.
        ///
        /// Returns results in the same format as VectorSearch::search().
        pub fn search(
            &self,
            query_vector: &[f32],
            limit: usize,
        ) -> Result<Vec<FtsResult>> {
            let guard = self.index.lock().map_err(|e| {
                crate::error::MindCoreError::Embedding(format!("index lock: {e}"))
            })?;

            let Some(ref hnsw) = *guard else {
                return Ok(Vec::new()); // Index not built
            };

            let query = VecPoint(query_vector.to_vec());
            let mut search = Search::default();
            let results: Vec<_> = hnsw.search(&query, &mut search)
                .take(limit)
                .map(|item| {
                    let similarity = 1.0 - item.distance;
                    FtsResult {
                        memory_id: *item.value,
                        score: similarity,
                    }
                })
                .collect();

            Ok(results)
        }

        /// Whether the index has been built.
        pub fn is_built(&self) -> bool {
            self.index.lock()
                .map(|guard| guard.is_some())
                .unwrap_or(false)
        }

        /// Number of vectors in the index.
        pub fn len(&self) -> usize {
            self.index.lock()
                .map(|guard| guard.as_ref().map(|_| self.count.load(std::sync::atomic::Ordering::Relaxed)).unwrap_or(0))
                .unwrap_or(0)
        }
    }
}

#[cfg(feature = "ann")]
pub use inner::AnnIndex;

#[cfg(test)]
#[cfg(feature = "ann")]
mod tests {
    use super::*;
    use crate::embeddings::pooling::{normalize_l2, vec_to_bytes};
    use crate::storage::{Database, migrations};

    fn setup() -> Database {
        let db = Database::open_in_memory().expect("open");
        db.with_writer(|conn| { migrations::migrate(conn)?; Ok(()) }).expect("migrate");

        // Insert test memories
        for i in 1..=5 {
            db.with_writer(|conn| {
                conn.execute(
                    "INSERT INTO memories (id, searchable_text, memory_type, content_hash, record_json)
                     VALUES (?1, ?2, 'semantic', ?3, '{}')",
                    rusqlite::params![i, format!("memory {i}"), format!("h{i}")],
                )?;
                Ok(())
            }).expect("insert");
        }

        // Insert vectors
        let vectors = vec![
            normalize_l2(&[1.0, 0.0, 0.0, 0.0]),
            normalize_l2(&[0.9, 0.1, 0.0, 0.0]),
            normalize_l2(&[0.0, 1.0, 0.0, 0.0]),
            normalize_l2(&[0.0, 0.0, 1.0, 0.0]),
            normalize_l2(&[0.0, 0.0, 0.0, 1.0]),
        ];

        for (i, v) in vectors.iter().enumerate() {
            let blob = vec_to_bytes(v);
            db.with_writer(|conn| {
                conn.execute(
                    "INSERT INTO memory_vectors (memory_id, embedding, model_name, dimensions, content_hash)
                     VALUES (?1, ?2, 'test', 4, ?3)",
                    rusqlite::params![i as i64 + 1, blob, format!("h{}", i + 1)],
                )?;
                Ok(())
            }).expect("insert vec");
        }

        db
    }

    #[test]
    fn build_and_search() {
        let db = setup();
        let index = AnnIndex::new();

        let count = index.build(&db, "test").expect("build");
        assert_eq!(count, 5);
        assert!(index.is_built());
        assert_eq!(index.len(), 5);

        // Search for vector similar to [1, 0, 0, 0]
        let query = normalize_l2(&[1.0, 0.0, 0.0, 0.0]);
        let results = index.search(&query, 3).expect("search");

        assert_eq!(results.len(), 3);
        // Memory 1 should be first (exact match)
        assert_eq!(results[0].memory_id, 1);
        // Memory 2 should be second (0.9 component)
        assert_eq!(results[1].memory_id, 2);
    }

    #[test]
    fn search_empty_index() {
        let index = AnnIndex::new();
        assert!(!index.is_built());

        let query = normalize_l2(&[1.0, 0.0, 0.0, 0.0]);
        let results = index.search(&query, 5).expect("search");
        assert!(results.is_empty());
    }
}
