use crate::traits::{MemoryMeta, ScoringStrategy};

/// ACT-R forgetting curve scorer.
///
/// Uses the cached activation score from the activation model.
/// Converts activation to a multiplier via sigmoid: `1 / (1 + e^(-activation))`.
/// This maps the unbounded activation value to a 0-1 range.
pub struct ActivationScorer {
    /// Weight applied to the activation-based multiplier.
    /// Default: 1.0 (full effect).
    weight: f32,
}

impl ActivationScorer {
    /// Create with a custom weight.
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }
}

impl Default for ActivationScorer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ScoringStrategy for ActivationScorer {
    fn score_multiplier(&self, _record: &MemoryMeta, _query: &str, _base_score: f32) -> f32 {
        // Activation scoring requires the cached activation value, which is stored
        // in the database. Since ScoringStrategy only has MemoryMeta (no DB access),
        // this scorer returns 1.0 (no-op) and the actual activation boost is applied
        // during the search pipeline where DB access is available.
        //
        // The ActivationScorer is kept as a placeholder for the composite pattern.
        // Real activation scoring is done in SearchBuilder::apply_scoring when
        // the activation-model feature is enabled.
        self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;

    #[test]
    fn default_returns_weight() {
        let scorer = ActivationScorer::default();
        let meta = MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        };
        let m = scorer.score_multiplier(&meta, "q", 1.0);
        assert!((m - 1.0).abs() < f32::EPSILON);
    }
}
