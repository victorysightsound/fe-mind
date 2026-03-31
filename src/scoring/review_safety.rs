use crate::scoring::procedural_safety::query_requests_procedural_guidance;
use crate::scoring::secret_policy::{
    query_requests_private_infra_guidance, query_requests_secret_location_or_reference,
};
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
    Maintenance,
}

impl ReviewScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Production => "production",
            Self::Staging => "staging",
            Self::Lab => "lab",
            Self::Migration => "migration",
            Self::Maintenance => "maintenance",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "general" | "default" => Some(Self::General),
            "production" | "prod" => Some(Self::Production),
            "staging" | "stage" => Some(Self::Staging),
            "lab" | "internal-lab" | "internal_lab" => Some(Self::Lab),
            "migration" | "temporary-migration" | "temporary_migration" => Some(Self::Migration),
            "maintenance" | "maintenance-window" | "maintenance_window" => Some(Self::Maintenance),
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
    BreakglassException,
    PrivateInfrastructureException,
    AuthBypassException,
    DataResetException,
    TrafficCutoverException,
}

impl ReviewPolicyClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OperationalException => "operational-exception",
            Self::NetworkExposureException => "network-exposure-exception",
            Self::DestructiveMaintenance => "destructive-maintenance",
            Self::SecretHandlingException => "secret-handling-exception",
            Self::MigrationException => "migration-exception",
            Self::BreakglassException => "breakglass-exception",
            Self::PrivateInfrastructureException => "private-infrastructure-exception",
            Self::AuthBypassException => "auth-bypass-exception",
            Self::DataResetException => "data-reset-exception",
            Self::TrafficCutoverException => "traffic-cutover-exception",
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
            "breakglass-exception" | "breakglass_exception" | "breakglass" => {
                Some(Self::BreakglassException)
            }
            "private-infrastructure-exception"
            | "private_infrastructure_exception"
            | "private-infrastructure" => Some(Self::PrivateInfrastructureException),
            "auth-bypass-exception" | "auth_bypass_exception" | "auth-bypass" => {
                Some(Self::AuthBypassException)
            }
            "data-reset-exception" | "data_reset_exception" | "data-reset" => {
                Some(Self::DataResetException)
            }
            "traffic-cutover-exception"
            | "traffic_cutover_exception"
            | "traffic-cutover"
            | "cutover" => Some(Self::TrafficCutoverException),
            _ => None,
        }
    }
}

impl std::fmt::Display for ReviewPolicyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReviewApprovalTemplate {
    StagingBridge,
    MigrationBridge,
    LabException,
    BreakglassOps,
    PrivateEndpointBridge,
    LabAuthBypass,
    MaintenanceReset,
    TrafficCutover,
}

impl ReviewApprovalTemplate {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StagingBridge => "staging-bridge",
            Self::MigrationBridge => "migration-bridge",
            Self::LabException => "lab-exception",
            Self::BreakglassOps => "breakglass-ops",
            Self::PrivateEndpointBridge => "private-endpoint-bridge",
            Self::LabAuthBypass => "lab-auth-bypass",
            Self::MaintenanceReset => "maintenance-reset",
            Self::TrafficCutover => "traffic-cutover",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "staging-bridge" | "staging_bridge" | "staging" => Some(Self::StagingBridge),
            "migration-bridge" | "migration_bridge" | "migration" => Some(Self::MigrationBridge),
            "lab-exception" | "lab_exception" | "lab" => Some(Self::LabException),
            "breakglass-ops" | "breakglass_ops" | "breakglass" => Some(Self::BreakglassOps),
            "private-endpoint-bridge" | "private_endpoint_bridge" | "private-endpoint" => {
                Some(Self::PrivateEndpointBridge)
            }
            "lab-auth-bypass" | "lab_auth_bypass" | "auth-bypass" => Some(Self::LabAuthBypass),
            "maintenance-reset" | "maintenance_reset" | "reset" => Some(Self::MaintenanceReset),
            "traffic-cutover" | "traffic_cutover" | "cutover" => Some(Self::TrafficCutover),
            _ => None,
        }
    }

    pub fn default_scope(self) -> ReviewScope {
        match self {
            Self::StagingBridge => ReviewScope::Staging,
            Self::MigrationBridge => ReviewScope::Migration,
            Self::LabException => ReviewScope::Lab,
            Self::BreakglassOps => ReviewScope::Production,
            Self::PrivateEndpointBridge => ReviewScope::Production,
            Self::LabAuthBypass => ReviewScope::Lab,
            Self::MaintenanceReset => ReviewScope::Maintenance,
            Self::TrafficCutover => ReviewScope::Migration,
        }
    }

    pub fn default_policy_class(self) -> ReviewPolicyClass {
        match self {
            Self::StagingBridge => ReviewPolicyClass::NetworkExposureException,
            Self::MigrationBridge => ReviewPolicyClass::MigrationException,
            Self::LabException => ReviewPolicyClass::OperationalException,
            Self::BreakglassOps => ReviewPolicyClass::BreakglassException,
            Self::PrivateEndpointBridge => ReviewPolicyClass::PrivateInfrastructureException,
            Self::LabAuthBypass => ReviewPolicyClass::AuthBypassException,
            Self::MaintenanceReset => ReviewPolicyClass::DataResetException,
            Self::TrafficCutover => ReviewPolicyClass::TrafficCutoverException,
        }
    }

    pub fn default_expiry(self, now: DateTime<Utc>) -> DateTime<Utc> {
        let days = match self {
            Self::StagingBridge => 14,
            Self::MigrationBridge => 7,
            Self::LabException => 30,
            Self::BreakglassOps => 1,
            Self::PrivateEndpointBridge => 14,
            Self::LabAuthBypass => 1,
            Self::MaintenanceReset => 2,
            Self::TrafficCutover => 3,
        };
        now + chrono::Duration::days(days)
    }
}

impl std::fmt::Display for ReviewApprovalTemplate {
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
    } else if normalized.contains("maintenance")
        || normalized.contains("reset window")
        || normalized.contains("maintenance window")
        || normalized.contains("rebuild window")
    {
        ReviewScope::Maintenance
    } else if normalized.contains("breakglass")
        || normalized.contains("outage")
        || normalized.contains("emergency")
        || normalized.contains("restore")
        || normalized.contains("recovery")
    {
        ReviewScope::Production
    } else if normalized.contains("migration")
        || normalized.contains("cutover")
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
        ReviewScope::Maintenance => matches!(query_scope, ReviewScope::Maintenance),
    }
}

pub(crate) fn review_policy_class_matches_query(record: &MemoryMeta, query: &str) -> bool {
    let Some(policy_class) = review_policy_class(record) else {
        return true;
    };
    let normalized = query.to_lowercase();

    match policy_class {
        ReviewPolicyClass::OperationalException => true,
        ReviewPolicyClass::NetworkExposureException => {
            normalized.contains("host")
                || normalized.contains("bind")
                || normalized.contains("listen")
                || normalized.contains("service")
                || normalized.contains("port")
                || normalized.contains("0.0.0.0")
                || normalized.contains("localhost")
        }
        ReviewPolicyClass::DestructiveMaintenance => {
            normalized.contains("delete")
                || normalized.contains("drop")
                || normalized.contains("reset")
                || normalized.contains("rebuild")
                || normalized.contains("prune")
                || normalized.contains("cleanup")
                || normalized.contains("rollback")
                || normalized.contains("remove")
        }
        ReviewPolicyClass::SecretHandlingException => {
            query_requests_secret_location_or_reference(query)
                || normalized.contains("token")
                || normalized.contains("password")
                || normalized.contains("secret")
                || normalized.contains("credential")
                || normalized.contains("vault")
                || normalized.contains("env")
        }
        ReviewPolicyClass::MigrationException => {
            normalized.contains("migration")
                || normalized.contains("cutover")
                || normalized.contains("bridge")
                || normalized.contains("temporary")
        }
        ReviewPolicyClass::BreakglassException => {
            normalized.contains("breakglass")
                || normalized.contains("outage")
                || normalized.contains("emergency")
                || normalized.contains("temporary recovery")
                || normalized.contains("temporary procedure")
                || normalized.contains("restore")
        }
        ReviewPolicyClass::PrivateInfrastructureException => {
            query_requests_private_infra_guidance(query)
                || normalized.contains("relay")
                || normalized.contains("internal host")
                || normalized.contains("endpoint")
                || normalized.contains("subnet")
                || normalized.contains("cidr")
                || normalized.contains("share path")
                || normalized.contains("unc path")
                || normalized.contains("internal path")
                || normalized.contains("tunnel")
        }
        ReviewPolicyClass::AuthBypassException => {
            normalized.contains("without auth")
                || normalized.contains("no auth")
                || normalized.contains("disable auth")
                || normalized.contains("skip auth")
                || normalized.contains("bypass auth")
                || normalized.contains("packet capture")
        }
        ReviewPolicyClass::DataResetException => {
            normalized.contains("drop")
                || normalized.contains("reset")
                || normalized.contains("wipe")
                || normalized.contains("rebuild from scratch")
                || normalized.contains("truncate")
                || normalized.contains("erase")
        }
        ReviewPolicyClass::TrafficCutoverException => {
            normalized.contains("cutover")
                || normalized.contains("switch traffic")
                || normalized.contains("redirect traffic")
                || normalized.contains("promote")
                || normalized.contains("swap endpoint")
                || normalized.contains("new relay")
        }
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

    if !protective_context
        && (normalized.contains("drop the ann")
            || normalized.contains("drop the tables")
            || normalized.contains("rebuild from scratch")
            || normalized.contains("wipe the index")
            || normalized.contains("truncate"))
    {
        tags.push("data-reset");
    }

    if !protective_context
        && (normalized.contains("cutover")
            || normalized.contains("switch clients")
            || normalized.contains("redirect traffic")
            || normalized.contains("new relay endpoint")
            || normalized.contains("promote the new"))
    {
        tags.push("traffic-cutover");
    }

    if tags.is_empty() {
        return None;
    }

    tags.sort_unstable();
    tags.dedup();

    let severity = if tags.iter().any(|tag| {
        matches!(
            *tag,
            "network-exposure" | "auth-disable" | "traffic-cutover"
        )
    }) {
        ReviewSeverity::Critical
    } else if tags.iter().any(|tag| {
        matches!(
            *tag,
            "destructive-command" | "secret-handling" | "data-reset"
        )
    }) {
        ReviewSeverity::High
    } else {
        ReviewSeverity::Medium
    };

    let reason = if tags.contains(&"network-exposure") {
        "network-exposure"
    } else if tags.contains(&"auth-disable") {
        "auth-disable"
    } else if tags.contains(&"traffic-cutover") {
        "traffic-cutover"
    } else if tags.contains(&"data-reset") {
        "data-reset"
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
    fn review_policy_class_matches_breakglass_query_shape() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "review_policy_class".to_string(),
            "breakglass-exception".to_string(),
        );
        let record = meta("Temporary breakglass recovery path", metadata);

        assert!(review_policy_class_matches_query(
            &record,
            "What temporary procedure is approved during breakglass recovery?"
        ));
        assert!(!review_policy_class_matches_query(
            &record,
            "How should the production service normally run?"
        ));
    }

    #[test]
    fn breakglass_queries_resolve_to_production_scope() {
        assert_eq!(
            query_scope("What temporary procedure is approved during breakglass recovery?"),
            ReviewScope::Production
        );
    }

    #[test]
    fn maintenance_queries_resolve_to_maintenance_scope() {
        assert_eq!(
            query_scope(
                "What destructive procedure is approved during the maintenance reset window?"
            ),
            ReviewScope::Maintenance
        );
    }

    #[test]
    fn review_policy_class_matches_high_impact_query_shapes() {
        let mut auth_metadata = HashMap::new();
        auth_metadata.insert(
            "review_policy_class".to_string(),
            "auth-bypass-exception".to_string(),
        );
        let auth_record = meta("Temporary lab auth bypass procedure", auth_metadata);
        assert!(review_policy_class_matches_query(
            &auth_record,
            "What temporary lab procedure is approved for packet-capture debugging without auth?"
        ));

        let mut reset_metadata = HashMap::new();
        reset_metadata.insert(
            "review_policy_class".to_string(),
            "data-reset-exception".to_string(),
        );
        let reset_record = meta("Approved maintenance reset workflow", reset_metadata);
        assert!(review_policy_class_matches_query(
            &reset_record,
            "What destructive procedure is approved during the maintenance reset window?"
        ));

        let mut cutover_metadata = HashMap::new();
        cutover_metadata.insert(
            "review_policy_class".to_string(),
            "traffic-cutover-exception".to_string(),
        );
        let cutover_record = meta("Approved migration cutover workflow", cutover_metadata);
        assert!(review_policy_class_matches_query(
            &cutover_record,
            "What traffic cutover is approved during the migration window?"
        ));
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

    #[test]
    fn detect_review_flags_for_data_reset_and_cutover() {
        let reset = meta(
            "During the maintenance reset window, drop the ANN tables and rebuild from scratch.",
            HashMap::new(),
        );
        let reset_flag = detect_review_flag(&reset).expect("reset flag");
        assert_eq!(reset_flag.severity, ReviewSeverity::High);
        assert!(reset_flag.tags.contains(&"data-reset"));

        let cutover = meta(
            "Approved cutover: switch clients to the new relay endpoint after validation.",
            HashMap::new(),
        );
        let cutover_flag = detect_review_flag(&cutover).expect("cutover flag");
        assert_eq!(cutover_flag.severity, ReviewSeverity::Critical);
        assert!(cutover_flag.tags.contains(&"traffic-cutover"));
    }
}
