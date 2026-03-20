use crate::traits::{MemoryMeta, ScoringStrategy};

/// Combines multiple scoring strategies by multiplying their results.
///
/// Each strategy produces a multiplier, and the final multiplier is the
/// product of all individual multipliers.
pub struct CompositeScorer {
    strategies: Vec<Box<dyn ScoringStrategy>>,
}

impl CompositeScorer {
    /// Create from a list of scoring strategies.
    pub fn new(strategies: Vec<Box<dyn ScoringStrategy>>) -> Self {
        Self { strategies }
    }

    /// Create an empty composite (no-op, multiplier = 1.0).
    pub fn empty() -> Self {
        Self {
            strategies: Vec::new(),
        }
    }

    /// Add a scoring strategy.
    pub fn add(mut self, strategy: Box<dyn ScoringStrategy>) -> Self {
        self.strategies.push(strategy);
        self
    }
}

impl ScoringStrategy for CompositeScorer {
    fn score_multiplier(&self, record: &MemoryMeta, query: &str, base_score: f32) -> f32 {
        self.strategies
            .iter()
            .map(|s| s.score_multiplier(record, query, base_score))
            .product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::{ImportanceScorer, MemoryTypeScorer, RecencyScorer};
    use crate::traits::MemoryType;
    use chrono::Utc;

    fn test_meta() -> MemoryMeta {
        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 10,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn empty_composite_is_noop() {
        let scorer = CompositeScorer::empty();
        let m = scorer.score_multiplier(&test_meta(), "q", 1.0);
        assert!((m - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn multiplies_strategies() {
        // Two 2x strategies = 4x
        struct Double;
        impl ScoringStrategy for Double {
            fn score_multiplier(&self, _: &MemoryMeta, _: &str, _: f32) -> f32 {
                2.0
            }
        }

        let scorer = CompositeScorer::new(vec![Box::new(Double), Box::new(Double)]);
        let m = scorer.score_multiplier(&test_meta(), "q", 1.0);
        assert!((m - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn real_strategies_compose() {
        let scorer = CompositeScorer::new(vec![
            Box::new(RecencyScorer::default_half_life()),
            Box::new(ImportanceScorer::default()),
            Box::new(MemoryTypeScorer::default()),
        ]);

        let meta = test_meta(); // recent, importance 10, Semantic
        let m = scorer.score_multiplier(&meta, "q", 1.0);

        // Recent (~1.0) * importance 10 (~1.5) * semantic (1.0) ≈ 1.5
        assert!(m > 1.0, "composite should boost: {m}");
    }

    #[test]
    fn builder_pattern() {
        let scorer = CompositeScorer::empty()
            .add(Box::new(RecencyScorer::new(7.0)))
            .add(Box::new(ImportanceScorer::default()));

        let m = scorer.score_multiplier(&test_meta(), "q", 1.0);
        assert!(m > 0.0);
    }
}
