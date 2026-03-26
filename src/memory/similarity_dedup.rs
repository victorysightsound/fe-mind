use crate::traits::{ConsolidationAction, ConsolidationStrategy, MemoryMeta, ScoredResult};

/// Near-duplicate detection via vector similarity threshold.
///
/// When a new memory is stored, if an existing memory has similarity
/// above the threshold, the action depends on the score:
/// - similarity > 0.95: Noop (near-exact duplicate)
/// - similarity > 0.85: Update the existing memory
/// - otherwise: Add as new
///
/// Requires vector search to be enabled (the engine must pass
/// vector-similarity results as the `existing` parameter).
pub struct SimilarityDedup {
    /// Threshold above which two memories are considered duplicates.
    noop_threshold: f32,
    /// Threshold above which an existing memory should be updated.
    update_threshold: f32,
}

impl SimilarityDedup {
    /// Create with a single threshold (used for both noop and update).
    pub fn new(threshold: f32) -> Self {
        Self {
            noop_threshold: threshold.max(0.90),
            update_threshold: threshold.max(0.80) - 0.10,
        }
    }

    /// Create with explicit noop and update thresholds.
    pub fn with_thresholds(noop: f32, update: f32) -> Self {
        Self {
            noop_threshold: noop,
            update_threshold: update,
        }
    }
}

impl Default for SimilarityDedup {
    fn default() -> Self {
        Self::new(0.95)
    }
}

impl ConsolidationStrategy for SimilarityDedup {
    fn consolidate(
        &self,
        _new: &MemoryMeta,
        existing: &[ScoredResult],
    ) -> Vec<ConsolidationAction> {
        if existing.is_empty() {
            return vec![ConsolidationAction::Add];
        }

        let top = &existing[0];

        if top.score >= self.noop_threshold {
            vec![ConsolidationAction::Noop]
        } else if top.score >= self.update_threshold {
            vec![ConsolidationAction::Update {
                target_id: top.memory_id,
            }]
        } else {
            vec![ConsolidationAction::Add]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;

    fn meta() -> MemoryMeta {
        MemoryMeta {
            id: None,
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    fn scored(id: i64, score: f32) -> ScoredResult {
        ScoredResult {
            memory_id: id,
            score,
            raw_score: score,
            score_multiplier: 1.0,
        }
    }

    #[test]
    fn no_existing_adds() {
        let dedup = SimilarityDedup::default();
        let actions = dedup.consolidate(&meta(), &[]);
        assert_eq!(actions, vec![ConsolidationAction::Add]);
    }

    #[test]
    fn high_similarity_noops() {
        let dedup = SimilarityDedup::default();
        let existing = vec![scored(42, 0.98)];
        let actions = dedup.consolidate(&meta(), &existing);
        assert_eq!(actions, vec![ConsolidationAction::Noop]);
    }

    #[test]
    fn medium_similarity_updates() {
        let dedup = SimilarityDedup::with_thresholds(0.95, 0.85);
        let existing = vec![scored(42, 0.90)];
        let actions = dedup.consolidate(&meta(), &existing);
        assert_eq!(actions, vec![ConsolidationAction::Update { target_id: 42 }]);
    }

    #[test]
    fn low_similarity_adds() {
        let dedup = SimilarityDedup::with_thresholds(0.95, 0.85);
        let existing = vec![scored(42, 0.50)];
        let actions = dedup.consolidate(&meta(), &existing);
        assert_eq!(actions, vec![ConsolidationAction::Add]);
    }
}
