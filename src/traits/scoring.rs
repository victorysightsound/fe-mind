use crate::search::SearchResult;
use crate::traits::MemoryMeta;

/// Post-search scoring adjustments.
///
/// Applied after FTS5/vector/RRF merge, before final ranking.
/// Implementations return a multiplier (1.0 = no change, >1 = boost, <1 = penalize).
pub trait ScoringStrategy: Send + Sync {
    /// Compute a score multiplier for a search result.
    ///
    /// - `record`: metadata about the memory being scored
    /// - `query`: the original search query text
    /// - `base_score`: the raw retrieval score (from FTS5, vector, or RRF)
    ///
    /// Returns a multiplier applied to the base score.
    fn score_multiplier(
        &self,
        record: &MemoryMeta,
        query: &str,
        base_score: f32,
    ) -> f32;
}

/// A scored search result containing the memory record and score breakdown.
#[derive(Debug, Clone)]
pub struct ScoredResult {
    /// Memory row ID.
    pub memory_id: i64,
    /// Final combined score (higher = more relevant).
    pub score: f32,
    /// Raw retrieval score before post-search scoring.
    pub raw_score: f32,
    /// Multiplier applied by scoring strategies.
    pub score_multiplier: f32,
}

impl ScoredResult {
    /// Create from a raw search result with no scoring applied yet.
    pub fn from_search_result(result: &SearchResult) -> Self {
        Self {
            memory_id: result.memory_id,
            score: result.score,
            raw_score: result.score,
            score_multiplier: 1.0,
        }
    }

    /// Apply a scoring strategy, updating the multiplier and final score.
    pub fn apply_scoring(&mut self, multiplier: f32) {
        self.score_multiplier *= multiplier;
        self.score = self.raw_score * self.score_multiplier;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;

    struct BoostScorer(f32);

    impl ScoringStrategy for BoostScorer {
        fn score_multiplier(&self, _record: &MemoryMeta, _query: &str, _base: f32) -> f32 {
            self.0
        }
    }

    fn test_meta() -> MemoryMeta {
        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn scored_result_from_search() {
        let sr = SearchResult {
            memory_id: 42,
            score: 0.75,
        };
        let scored = ScoredResult::from_search_result(&sr);
        assert_eq!(scored.memory_id, 42);
        assert!((scored.score - 0.75).abs() < f32::EPSILON);
        assert!((scored.raw_score - 0.75).abs() < f32::EPSILON);
        assert!((scored.score_multiplier - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_scoring_multiplier() {
        let sr = SearchResult {
            memory_id: 1,
            score: 1.0,
        };
        let mut scored = ScoredResult::from_search_result(&sr);

        scored.apply_scoring(2.0);
        assert!((scored.score - 2.0).abs() < f32::EPSILON);
        assert!((scored.score_multiplier - 2.0).abs() < f32::EPSILON);
        assert!((scored.raw_score - 1.0).abs() < f32::EPSILON);

        scored.apply_scoring(0.5);
        assert!((scored.score - 1.0).abs() < 0.01); // 1.0 * 2.0 * 0.5 = 1.0
        assert!((scored.score_multiplier - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn scoring_strategy_trait_object() {
        let scorer: Box<dyn ScoringStrategy> = Box::new(BoostScorer(1.5));
        let meta = test_meta();
        let m = scorer.score_multiplier(&meta, "test", 1.0);
        assert!((m - 1.5).abs() < f32::EPSILON);
    }
}
