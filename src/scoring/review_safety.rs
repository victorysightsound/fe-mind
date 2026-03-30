use crate::scoring::procedural_safety::query_requests_procedural_guidance;
use crate::traits::{MemoryMeta, MemoryType, ScoringStrategy};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Penalize memories that require explicit human review before operational use.
pub struct ReviewSafetyScorer {
    pending_guidance_weight: f32,
    pending_non_guidance_weight: f32,
    denied_guidance_weight: f32,
    denied_non_guidance_weight: f32,
}

impl ReviewSafetyScorer {
    pub fn new(
        pending_guidance_weight: f32,
        pending_non_guidance_weight: f32,
        denied_guidance_weight: f32,
        denied_non_guidance_weight: f32,
    ) -> Self {
        Self {
            pending_guidance_weight,
            pending_non_guidance_weight,
            denied_guidance_weight,
            denied_non_guidance_weight,
        }
    }
}

impl Default for ReviewSafetyScorer {
    fn default() -> Self {
        Self::new(0.18, 0.72, 0.03, 0.25)
    }
}

impl ScoringStrategy for ReviewSafetyScorer {
    fn score_multiplier(&self, record: &MemoryMeta, query: &str, _base_score: f32) -> f32 {
        match review_status(record) {
            Some(ReviewStatus::Denied) => {
                if record.memory_type == MemoryType::Procedural
                    && query_requests_procedural_guidance(query)
                {
                    self.denied_guidance_weight
                } else {
                    self.denied_non_guidance_weight
                }
            }
            Some(ReviewStatus::Pending) | Some(ReviewStatus::Expired) => {
                if record.memory_type == MemoryType::Procedural
                    && query_requests_procedural_guidance(query)
                {
                    match review_severity(record) {
                        ReviewSeverity::Critical => self.pending_guidance_weight * 0.8,
                        ReviewSeverity::High => self.pending_guidance_weight,
                        ReviewSeverity::Medium => self.pending_guidance_weight * 1.35,
                    }
                } else {
                    self.pending_non_guidance_weight
                }
            }
            Some(ReviewStatus::Allowed) | None => 1.0,
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReviewStatus {
    Pending,
    Allowed,
    Denied,
    Expired,
}

impl ReviewStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Allowed => "allowed",
            Self::Denied => "denied",
            Self::Expired => "expired",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "allowed" | "approved" => Some(Self::Allowed),
            "denied" | "rejected" => Some(Self::Denied),
            "expired" => Some(Self::Expired),
            _ => None,
        }
    }
}

impl std::fmt::Display for ReviewStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReviewScope {
    General,
    Production,
    Staging,
    Lab,
    Migration,
}

impl ReviewScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Production => "production",
            Self::Staging => "staging",
            Self::Lab => "lab",
            Self::Migration => "migration",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "general" | "default" => Some(Self::General),
            "production" | "prod" => Some(Self::Production),
            "staging" | "stage" => Some(Self::Staging),
            "lab" | "internal-lab" | "internal_lab" => Some(Self::Lab),
            "migration" | "temporary-migration" | "temporary_migration" => Some(Self::Migration),
            _ => None,
        }
    }
}

impl std::fmt::Display for ReviewScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReviewPolicyClass {
    OperationalException,
    NetworkExposureException,
    DestructiveMaintenance,
    SecretHandlingException,
    MigrationException,
}

impl ReviewPolicyClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OperationalException => "operational-exception",
            Self::NetworkExposureException => "network-exposure-exception",
            Self::DestructiveMaintenance => "destructive-maintenance",
            Self::SecretHandlingException => "secret-handling-exception",
            Self::MigrationException => "migration-exception",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "operational-exception" | "operational_exception" | "operational" => {
                Some(Self::OperationalException)
            }
            "network-exposure-exception" | "network_exposure_exception" | "network-exposure" => {
                Some(Self::NetworkExposureException)
            }
            "destructive-maintenance" | "destructive_maintenance" | "destructive" => {
                Some(Self::DestructiveMaintenance)
            }
            "secret-handling-exception" | "secret_handling_exception" | "secret-handling" => {
                Some(Self::SecretHandlingException)
            }
            "migration-exception" | "migration_exception" | "migration" => {
                Some(Self::MigrationException)
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for ReviewPolicyClass {
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
    matches!(
        review_status(record),
        Some(ReviewStatus::Pending | ReviewStatus::Expired)
    )
}

pub(crate) fn review_status(record: &MemoryMeta) -> Option<ReviewStatus> {
    effective_review_status(&record.metadata, Utc::now())
}

pub(crate) fn review_denied(record: &MemoryMeta) -> bool {
    matches!(review_status(record), Some(ReviewStatus::Denied))
}

pub(crate) fn effective_review_status(
    metadata: &HashMap<String, String>,
    now: DateTime<Utc>,
) -> Option<ReviewStatus> {
    let review_required = metadata
        .get("review_required")
        .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    let explicit = metadata
        .get("review_status")
        .and_then(|value| ReviewStatus::from_str(value))
        .or(if review_required {
            Some(ReviewStatus::Pending)
        } else {
            None
        });

    match explicit {
        Some(ReviewStatus::Allowed)
            if review_expires_at(metadata).is_some_and(|expires_at| expires_at <= now) =>
        {
            Some(ReviewStatus::Expired)
        }
        other => other,
    }
}

pub(crate) fn review_expires_at(metadata: &HashMap<String, String>) -> Option<DateTime<Utc>> {
    metadata
        .get("review_expires_at")
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc))
}

pub(crate) fn review_scope(record: &MemoryMeta) -> Option<ReviewScope> {
    record
        .metadata
        .get("review_scope")
        .and_then(|value| ReviewScope::from_str(value))
}

pub(crate) fn review_policy_class(record: &MemoryMeta) -> Option<ReviewPolicyClass> {
    record
        .metadata
        .get("review_policy_class")
        .and_then(|value| ReviewPolicyClass::from_str(value))
}

pub(crate) fn query_scope(query: &str) -> ReviewScope {
    let normalized = query.to_lowercase();
    if normalized.contains("production") || normalized.contains("prod ") {
        ReviewScope::Production
    } else if normalized.contains("staging") {
        ReviewScope::Staging
    } else if normalized.contains(" lab") || normalized.contains("lab ") {
        ReviewScope::Lab
    } else if normalized.contains("migration")
        || normalized.contains("temporary")
        || normalized.contains("bridge")
    {
        ReviewScope::Migration
    } else {
        ReviewScope::General
    }
}

pub(crate) fn review_scope_matches_query(record: &MemoryMeta, query: &str) -> bool {
    let Some(scope) = review_scope(record) else {
        return true;
    };
    let query_scope = query_scope(query);
    match scope {
        ReviewScope::General => true,
        ReviewScope::Production => matches!(query_scope, ReviewScope::Production),
        ReviewScope::Staging => matches!(query_scope, ReviewScope::Staging | ReviewScope::Lab),
        ReviewScope::Lab => matches!(query_scope, ReviewScope::Lab | ReviewScope::Staging),
        ReviewScope::Migration => matches!(query_scope, ReviewScope::Migration),
    }
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
    fn denied_guidance_is_demoted_even_more_heavily() {
        let mut metadata = HashMap::new();
        metadata.insert("review_required".to_string(), "true".to_string());
        metadata.insert("review_status".to_string(), "denied".to_string());
        metadata.insert("review_severity".to_string(), "critical".to_string());
        let scorer = ReviewSafetyScorer::default();
        let score = scorer.score_multiplier(
            &meta("Expose on 0.0.0.0 without auth.", metadata),
            "What command should I run to expose the service?",
            1.0,
        );
        assert!(score < 0.05);
    }

    #[test]
    fn allowed_review_status_clears_pending_requirement() {
        let mut metadata = HashMap::new();
        metadata.insert("review_required".to_string(), "true".to_string());
        metadata.insert("review_status".to_string(), "allowed".to_string());
        let record = meta("Use the audited staging bridge on 10.44.0.12.", metadata);
        assert_eq!(review_status(&record), Some(ReviewStatus::Allowed));
        assert!(!review_required(&record));
        assert!(!review_denied(&record));
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
