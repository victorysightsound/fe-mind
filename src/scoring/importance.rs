use crate::traits::{MemoryMeta, ScoringStrategy};

/// Linear importance scorer.
///
/// Maps importance 1-10 to a multiplier range.
/// Default: importance 1 → 0.5x, importance 5 → 1.0x, importance 10 → 1.5x.
pub struct ImportanceScorer {
    min_multiplier: f32,
    max_multiplier: f32,
}

impl ImportanceScorer {
    /// Create with custom min/max multiplier range.
    pub fn new(min_multiplier: f32, max_multiplier: f32) -> Self {
        Self {
            min_multiplier,
            max_multiplier,
        }
    }
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self::new(0.5, 1.5)
    }
}

impl ScoringStrategy for ImportanceScorer {
    fn score_multiplier(&self, record: &MemoryMeta, _query: &str, _base_score: f32) -> f32 {
        let normalized = (record.importance as f32 - 1.0) / 9.0; // 0.0 to 1.0
        self.min_multiplier + normalized * (self.max_multiplier - self.min_multiplier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;

    fn meta_importance(imp: u8) -> MemoryMeta {
        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: imp,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn importance_1_gets_min() {
        let scorer = ImportanceScorer::default();
        let m = scorer.score_multiplier(&meta_importance(1), "q", 1.0);
        assert!((m - 0.5).abs() < 0.01);
    }

    #[test]
    fn importance_10_gets_max() {
        let scorer = ImportanceScorer::default();
        let m = scorer.score_multiplier(&meta_importance(10), "q", 1.0);
        assert!((m - 1.5).abs() < 0.01);
    }

    #[test]
    fn importance_5_is_middle() {
        let scorer = ImportanceScorer::default();
        let m = scorer.score_multiplier(&meta_importance(5), "q", 1.0);
        // (5-1)/9 = 0.444, 0.5 + 0.444*1.0 = 0.944
        assert!(
            m > 0.9 && m < 1.1,
            "importance 5 should be near 1.0, got {m}"
        );
    }

    #[test]
    fn higher_importance_higher_score() {
        let scorer = ImportanceScorer::default();
        let low = scorer.score_multiplier(&meta_importance(2), "q", 1.0);
        let high = scorer.score_multiplier(&meta_importance(9), "q", 1.0);
        assert!(high > low);
    }
}
