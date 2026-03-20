use chrono::Utc;

use crate::traits::{MemoryMeta, ScoringStrategy};

/// Exponential decay scorer based on memory age.
///
/// Score multiplier decreases exponentially as the memory ages.
/// Formula: `2^(-age_days / half_life_days)`
///
/// At `half_life_days`, the multiplier is 0.5.
/// At `2 * half_life_days`, the multiplier is 0.25.
pub struct RecencyScorer {
    half_life_days: f64,
}

impl RecencyScorer {
    /// Create with the specified half-life in days.
    ///
    /// - 7 days: aggressive decay (episodic memories)
    /// - 30 days: moderate decay (procedural memories)
    /// - 90 days: slow decay (semantic memories)
    pub fn new(half_life_days: f64) -> Self {
        Self {
            half_life_days: half_life_days.max(0.1), // prevent division by zero
        }
    }

    /// Default half-life: 30 days.
    pub fn default_half_life() -> Self {
        Self::new(30.0)
    }
}

impl ScoringStrategy for RecencyScorer {
    fn score_multiplier(&self, record: &MemoryMeta, _query: &str, _base_score: f32) -> f32 {
        let age = Utc::now()
            .signed_duration_since(record.created_at)
            .num_seconds() as f64;
        let age_days = (age / 86400.0).max(0.0);

        let multiplier = 2.0_f64.powf(-age_days / self.half_life_days);
        multiplier as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use std::collections::HashMap;
    use crate::traits::MemoryType;

    fn meta_aged(days: i64) -> MemoryMeta {
        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: Utc::now() - Duration::days(days),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn recent_memory_high_score() {
        let scorer = RecencyScorer::new(30.0);
        let m = scorer.score_multiplier(&meta_aged(0), "q", 1.0);
        assert!(m > 0.99, "just-created memory should score ~1.0, got {m}");
    }

    #[test]
    fn half_life_gives_half_score() {
        let scorer = RecencyScorer::new(30.0);
        let m = scorer.score_multiplier(&meta_aged(30), "q", 1.0);
        assert!((m - 0.5).abs() < 0.05, "at half-life should be ~0.5, got {m}");
    }

    #[test]
    fn double_half_life_gives_quarter() {
        let scorer = RecencyScorer::new(30.0);
        let m = scorer.score_multiplier(&meta_aged(60), "q", 1.0);
        assert!((m - 0.25).abs() < 0.05, "at 2x half-life should be ~0.25, got {m}");
    }

    #[test]
    fn very_old_approaches_zero() {
        let scorer = RecencyScorer::new(7.0);
        let m = scorer.score_multiplier(&meta_aged(365), "q", 1.0);
        assert!(m < 0.001, "year-old memory with 7-day half-life should be ~0, got {m}");
    }

    #[test]
    fn different_half_lives() {
        let fast = RecencyScorer::new(7.0);
        let slow = RecencyScorer::new(90.0);
        let meta = meta_aged(30);

        let fast_score = fast.score_multiplier(&meta, "q", 1.0);
        let slow_score = slow.score_multiplier(&meta, "q", 1.0);

        assert!(
            slow_score > fast_score,
            "slow decay should score higher: slow={slow_score}, fast={fast_score}"
        );
    }
}
