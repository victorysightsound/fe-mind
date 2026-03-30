use crate::traits::{MemoryMeta, ScoringStrategy};

/// Trust-weight retrieval using stable metadata carried on the memory record.
///
/// Expected metadata key:
/// - `source_trust`: `trusted` | `verified` | `high` | `normal` | `low` | `untrusted`
pub struct SourceTrustScorer {
    trusted_weight: f32,
    low_weight: f32,
    untrusted_weight: f32,
}

impl SourceTrustScorer {
    /// Create with explicit trust weights.
    pub fn new(trusted_weight: f32, low_weight: f32, untrusted_weight: f32) -> Self {
        Self {
            trusted_weight,
            low_weight,
            untrusted_weight,
        }
    }
}

impl Default for SourceTrustScorer {
    fn default() -> Self {
        Self::new(1.15, 0.78, 0.45)
    }
}

impl ScoringStrategy for SourceTrustScorer {
    fn score_multiplier(&self, record: &MemoryMeta, _query: &str, _base_score: f32) -> f32 {
        match source_trust_level(record) {
            SourceTrustLevel::Trusted => self.trusted_weight,
            SourceTrustLevel::Low => self.low_weight,
            SourceTrustLevel::Untrusted => self.untrusted_weight,
            SourceTrustLevel::Normal => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceTrustLevel {
    Trusted,
    Normal,
    Low,
    Untrusted,
}

pub(crate) fn source_trust_level(record: &MemoryMeta) -> SourceTrustLevel {
    let Some(value) = record.metadata.get("source_trust") else {
        return SourceTrustLevel::Normal;
    };

    match value.trim().to_lowercase().as_str() {
        "trusted" | "verified" | "maintainer" | "system" | "high" => SourceTrustLevel::Trusted,
        "low" | "weak" | "speculative" => SourceTrustLevel::Low,
        "untrusted" | "external" | "poisoned" | "unsafe" => SourceTrustLevel::Untrusted,
        _ => SourceTrustLevel::Normal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;
    use std::collections::HashMap;

    fn meta_with_trust(level: Option<&str>) -> MemoryMeta {
        let mut metadata = HashMap::new();
        if let Some(level) = level {
            metadata.insert("source_trust".to_string(), level.to_string());
        }

        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata,
        }
    }

    #[test]
    fn trusted_sources_get_boost() {
        let scorer = SourceTrustScorer::default();
        let m = scorer.score_multiplier(&meta_with_trust(Some("trusted")), "q", 1.0);
        assert!(m > 1.0);
    }

    #[test]
    fn untrusted_sources_are_penalized() {
        let scorer = SourceTrustScorer::default();
        let m = scorer.score_multiplier(&meta_with_trust(Some("untrusted")), "q", 1.0);
        assert!(m < 0.5);
    }

    #[test]
    fn missing_trust_is_neutral() {
        let scorer = SourceTrustScorer::default();
        let m = scorer.score_multiplier(&meta_with_trust(None), "q", 1.0);
        assert!((m - 1.0).abs() < 0.01);
    }
}
