use crate::scoring::source_trust::{SourceTrustLevel, source_trust_level};
use crate::traits::{MemoryMeta, MemoryType, ScoringStrategy};

/// Bias procedural guidance toward trusted procedural memories and demote
/// low-trust procedural instructions that could poison agent behavior.
pub struct ProceduralSafetyScorer {
    trusted_procedural_weight: f32,
    low_trust_procedural_weight: f32,
    untrusted_procedural_weight: f32,
    non_procedural_guidance_weight: f32,
    non_guidance_untrusted_procedural_weight: f32,
}

impl ProceduralSafetyScorer {
    pub fn new(
        trusted_procedural_weight: f32,
        low_trust_procedural_weight: f32,
        untrusted_procedural_weight: f32,
        non_procedural_guidance_weight: f32,
        non_guidance_untrusted_procedural_weight: f32,
    ) -> Self {
        Self {
            trusted_procedural_weight,
            low_trust_procedural_weight,
            untrusted_procedural_weight,
            non_procedural_guidance_weight,
            non_guidance_untrusted_procedural_weight,
        }
    }
}

impl Default for ProceduralSafetyScorer {
    fn default() -> Self {
        Self::new(1.28, 0.35, 0.12, 0.82, 0.5)
    }
}

impl ScoringStrategy for ProceduralSafetyScorer {
    fn score_multiplier(&self, record: &MemoryMeta, query: &str, _base_score: f32) -> f32 {
        let trust = source_trust_level(record);
        let is_procedural = record.memory_type == MemoryType::Procedural;
        let asks_guidance = query_requests_procedural_guidance(query);

        if asks_guidance {
            if is_procedural {
                return match trust {
                    SourceTrustLevel::Trusted => self.trusted_procedural_weight,
                    SourceTrustLevel::Low => self.low_trust_procedural_weight,
                    SourceTrustLevel::Untrusted => self.untrusted_procedural_weight,
                    SourceTrustLevel::Normal => 1.12,
                };
            }
            return self.non_procedural_guidance_weight;
        }

        if is_procedural {
            return match trust {
                SourceTrustLevel::Untrusted => self.non_guidance_untrusted_procedural_weight,
                SourceTrustLevel::Low => 0.72,
                SourceTrustLevel::Trusted | SourceTrustLevel::Normal => 1.0,
            };
        }

        1.0
    }
}

pub(crate) fn query_requests_procedural_guidance(query: &str) -> bool {
    let normalized = query.to_lowercase();
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();

    tokens.windows(2).any(|pair| {
        matches!(
            pair,
            ["how", "do"]
                | ["how", "should"]
                | ["what", "command"]
                | ["which", "command"]
                | ["what", "host"]
                | ["which", "host"]
                | ["what", "address"]
                | ["which", "address"]
                | ["what", "port"]
                | ["which", "port"]
                | ["what", "steps"]
                | ["which", "steps"]
        )
    }) || normalized.contains("restart")
        || normalized.contains("start ")
        || normalized.contains("stop ")
        || normalized.contains("run ")
        || normalized.contains("install")
        || normalized.contains("deploy")
        || normalized.contains("configure")
        || normalized.contains("expose")
        || normalized.contains("open ")
        || normalized.contains("opened ")
        || normalized.contains("bind ")
        || normalized.contains("still be used")
        || normalized.contains("should i use")
        || normalized.contains("should i run")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    fn meta(memory_type: MemoryType, trust: Option<&str>) -> MemoryMeta {
        let mut metadata = HashMap::new();
        if let Some(trust) = trust {
            metadata.insert("source_trust".to_string(), trust.to_string());
        }

        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata,
        }
    }

    #[test]
    fn trusted_procedural_guidance_gets_boosted() {
        let scorer = ProceduralSafetyScorer::default();
        let m = scorer.score_multiplier(
            &meta(MemoryType::Procedural, Some("trusted")),
            "What command should restart the tunnel?",
            1.0,
        );
        assert!(m > 1.2);
    }

    #[test]
    fn untrusted_procedural_guidance_is_heavily_demoted() {
        let scorer = ProceduralSafetyScorer::default();
        let m = scorer.score_multiplier(
            &meta(MemoryType::Procedural, Some("untrusted")),
            "What command should restart the tunnel?",
            1.0,
        );
        assert!(m < 0.2);
    }

    #[test]
    fn untrusted_procedural_memory_still_penalized_for_non_guidance_queries() {
        let scorer = ProceduralSafetyScorer::default();
        let m = scorer.score_multiplier(
            &meta(MemoryType::Procedural, Some("untrusted")),
            "What execution mode is the service using?",
            1.0,
        );
        assert!(m < 1.0);
    }
}
