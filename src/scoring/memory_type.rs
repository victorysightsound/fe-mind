use crate::traits::{MemoryMeta, MemoryType, ScoringStrategy};

/// Different base weights per cognitive memory type.
///
/// Semantic memories (stable facts) generally score higher than
/// episodic memories (transient events) in standard searches.
pub struct MemoryTypeScorer {
    episodic_weight: f32,
    semantic_weight: f32,
    procedural_weight: f32,
}

impl MemoryTypeScorer {
    /// Create with custom weights per type.
    pub fn new(episodic: f32, semantic: f32, procedural: f32) -> Self {
        Self {
            episodic_weight: episodic,
            semantic_weight: semantic,
            procedural_weight: procedural,
        }
    }
}

impl Default for MemoryTypeScorer {
    fn default() -> Self {
        Self::new(0.8, 1.0, 1.2)
    }
}

impl ScoringStrategy for MemoryTypeScorer {
    fn score_multiplier(&self, record: &MemoryMeta, _query: &str, _base_score: f32) -> f32 {
        match record.memory_type {
            MemoryType::Episodic => self.episodic_weight,
            MemoryType::Semantic => self.semantic_weight,
            MemoryType::Procedural => self.procedural_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn meta_type(t: MemoryType) -> MemoryMeta {
        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: t,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn default_weights() {
        let scorer = MemoryTypeScorer::default();

        let ep = scorer.score_multiplier(&meta_type(MemoryType::Episodic), "q", 1.0);
        let sem = scorer.score_multiplier(&meta_type(MemoryType::Semantic), "q", 1.0);
        let proc = scorer.score_multiplier(&meta_type(MemoryType::Procedural), "q", 1.0);

        assert!((ep - 0.8).abs() < 0.01);
        assert!((sem - 1.0).abs() < 0.01);
        assert!((proc - 1.2).abs() < 0.01);
        assert!(proc > sem && sem > ep);
    }

    #[test]
    fn custom_weights() {
        let scorer = MemoryTypeScorer::new(0.5, 1.0, 2.0);
        let m = scorer.score_multiplier(&meta_type(MemoryType::Procedural), "q", 1.0);
        assert!((m - 2.0).abs() < 0.01);
    }
}
