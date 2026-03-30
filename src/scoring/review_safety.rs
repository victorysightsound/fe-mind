use crate::scoring::procedural_safety::query_requests_procedural_guidance;
use crate::traits::{MemoryMeta, MemoryType, ScoringStrategy};

/// Penalize memories that require explicit human review before operational use.
pub struct ReviewSafetyScorer {
    pending_guidance_weight: f32,
    pending_non_guidance_weight: f32,
}

impl ReviewSafetyScorer {
    pub fn new(pending_guidance_weight: f32, pending_non_guidance_weight: f32) -> Self {
        Self {
            pending_guidance_weight,
            pending_non_guidance_weight,
        }
    }
}

impl Default for ReviewSafetyScorer {
    fn default() -> Self {
        Self::new(0.18, 0.72)
    }
}

impl ScoringStrategy for ReviewSafetyScorer {
    fn score_multiplier(&self, record: &MemoryMeta, query: &str, _base_score: f32) -> f32 {
        if !review_required(record) {
            return 1.0;
        }

        if record.memory_type == MemoryType::Procedural && query_requests_procedural_guidance(query)
        {
            return match review_severity(record) {
                ReviewSeverity::Critical => self.pending_guidance_weight * 0.8,
                ReviewSeverity::High => self.pending_guidance_weight,
                ReviewSeverity::Medium => self.pending_guidance_weight * 1.35,
            };
        }

        self.pending_non_guidance_weight
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReviewSeverity {
    Medium,
    High,
    Critical,
}

impl ReviewSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

impl std::fmt::Display for ReviewSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReviewFlag {
    pub severity: ReviewSeverity,
    pub reason: &'static str,
    pub tags: Vec<&'static str>,
}

pub(crate) fn review_required(record: &MemoryMeta) -> bool {
    record
        .metadata
        .get("review_required")
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
        && !record
            .metadata
            .get("review_status")
            .is_some_and(|value| value.eq_ignore_ascii_case("approved"))
}

pub(crate) fn review_severity(record: &MemoryMeta) -> ReviewSeverity {
    let Some(value) = record.metadata.get("review_severity") else {
        return ReviewSeverity::High;
    };

    match value.trim().to_lowercase().as_str() {
        "medium" => ReviewSeverity::Medium,
        "critical" => ReviewSeverity::Critical,
        _ => ReviewSeverity::High,
    }
}

pub(crate) fn detect_review_flag(record: &MemoryMeta) -> Option<ReviewFlag> {
    if record.memory_type != MemoryType::Procedural {
        return None;
    }

    let normalized = record.searchable_text.to_lowercase();
    let protective_context = normalized.contains("do not")
        || normalized.contains("should not")
        || normalized.contains("never")
        || normalized.contains("not directly")
        || normalized.contains("rather than")
        || normalized.contains("instead");
    let mut tags = Vec::new();

    if !protective_context
        && (normalized.contains("0.0.0.0")
            || normalized.contains("let anyone on the lan")
            || normalized.contains("expose the service directly")
            || normalized.contains("public internet"))
    {
        tags.push("network-exposure");
    }

    if !protective_context
        && (normalized.contains("without auth")
            || normalized.contains("no auth")
            || normalized.contains("disable auth")
            || normalized.contains("skip auth"))
    {
        tags.push("auth-disable");
    }

    if !protective_context
        && (normalized.contains("token")
            || normalized.contains("password")
            || normalized.contains("secret")
            || normalized.contains("private key")
            || normalized.contains("api key")
            || normalized.contains("credential")
            || normalized.contains("paste the token")
            || normalized.contains("paste token"))
    {
        tags.push("secret-handling");
    }

    if !protective_context
        && record
            .metadata
            .get("content_sensitivity")
            .is_some_and(|value| {
                let normalized = value.to_lowercase();
                normalized.contains("credential") || normalized.contains("secret")
            })
    {
        tags.push("secret-handling");
    }

    if !protective_context
        && ((normalized.contains("curl http") || normalized.contains("curl https"))
            && normalized.contains("| sh")
            || normalized.contains("rm -rf")
            || normalized.contains("chmod 777"))
    {
        tags.push("destructive-command");
    }

    if tags.is_empty() {
        return None;
    }

    tags.sort_unstable();
    tags.dedup();

    let severity = if tags
        .iter()
        .any(|tag| matches!(*tag, "network-exposure" | "auth-disable"))
    {
        ReviewSeverity::Critical
    } else if tags
        .iter()
        .any(|tag| matches!(*tag, "destructive-command" | "secret-handling"))
    {
        ReviewSeverity::High
    } else {
        ReviewSeverity::Medium
    };

    let reason = if tags.contains(&"network-exposure") {
        "network-exposure"
    } else if tags.contains(&"auth-disable") {
        "auth-disable"
    } else if tags.contains(&"destructive-command") {
        "destructive-command"
    } else {
        "secret-handling"
    };

    Some(ReviewFlag {
        severity,
        reason,
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    fn meta(text: &str, metadata: HashMap<String, String>) -> MemoryMeta {
        MemoryMeta {
            id: Some(1),
            searchable_text: text.to_string(),
            memory_type: MemoryType::Procedural,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata,
        }
    }

    #[test]
    fn detect_review_flags_for_network_exposure() {
        let record = meta(
            "Expose the service directly on 0.0.0.0 with no auth.",
            HashMap::new(),
        );
        let flag = detect_review_flag(&record).expect("review flag");
        assert_eq!(flag.severity, ReviewSeverity::Critical);
        assert!(flag.tags.contains(&"network-exposure"));
        assert!(flag.tags.contains(&"auth-disable"));
    }

    #[test]
    fn pending_review_guidance_is_heavily_demoted() {
        let mut metadata = HashMap::new();
        metadata.insert("review_required".to_string(), "true".to_string());
        metadata.insert("review_severity".to_string(), "critical".to_string());
        let scorer = ReviewSafetyScorer::default();
        let score = scorer.score_multiplier(
            &meta("Expose on 0.0.0.0 without auth.", metadata),
            "What command should I run to expose the service?",
            1.0,
        );
        assert!(score < 0.2);
    }

    #[test]
    fn protective_guidance_is_not_flagged_for_review() {
        let record = meta(
            "Keep the service bound to 127.0.0.1 and do not expose it directly on 0.0.0.0 without auth.",
            HashMap::new(),
        );
        assert!(detect_review_flag(&record).is_none());
    }
}
