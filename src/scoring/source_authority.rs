use std::sync::Arc;

use crate::traits::{MemoryMeta, ScoringStrategy};

/// Query-aware source-chain authority scoring.
///
/// Expected metadata keys:
/// - `source_authority_domain`: single domain tag
/// - `source_authority_domains`: comma/semicolon/pipe separated domain tags
/// - `source_authority_level`: `authoritative` | `primary` | `delegated` | `reference`
/// - `source_chain`: stable source-chain identifier for arbitration
///
/// Applications can also provide a [`SourceAuthorityRegistry`] at engine build
/// time so authority can be defined centrally rather than only on individual
/// records.
pub struct SourceAuthorityScorer {
    authoritative_weight: f32,
    primary_weight: f32,
    delegated_weight: f32,
    reference_weight: f32,
    registry: Arc<SourceAuthorityRegistry>,
}

impl SourceAuthorityScorer {
    pub fn new(
        authoritative_weight: f32,
        primary_weight: f32,
        delegated_weight: f32,
        reference_weight: f32,
    ) -> Self {
        Self {
            authoritative_weight,
            primary_weight,
            delegated_weight,
            reference_weight,
            registry: Arc::new(SourceAuthorityRegistry::default()),
        }
    }

    pub fn with_registry(mut self, registry: Arc<SourceAuthorityRegistry>) -> Self {
        self.registry = registry;
        self
    }
}

impl Default for SourceAuthorityScorer {
    fn default() -> Self {
        Self::new(1.18, 1.12, 1.06, 1.02)
    }
}

impl ScoringStrategy for SourceAuthorityScorer {
    fn score_multiplier(&self, record: &MemoryMeta, query: &str, _base_score: f32) -> f32 {
        let Some(domain) = infer_authority_domain(query) else {
            return 1.0;
        };

        match source_authority_level_for_domain(record, Some(domain), Some(&self.registry)) {
            SourceAuthorityLevel::Authoritative => self.authoritative_weight,
            SourceAuthorityLevel::Primary => self.primary_weight,
            SourceAuthorityLevel::Delegated => self.delegated_weight,
            SourceAuthorityLevel::Reference => self.reference_weight,
            SourceAuthorityLevel::Unknown => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SourceAuthorityDomain {
    RuntimeOps,
    Deployment,
    Networking,
    Security,
    BuildToolchain,
    Maintenance,
}

impl SourceAuthorityDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeOps => "runtime-ops",
            Self::Deployment => "deployment",
            Self::Networking => "networking",
            Self::Security => "security",
            Self::BuildToolchain => "build-toolchain",
            Self::Maintenance => "maintenance",
        }
    }
}

impl std::fmt::Display for SourceAuthorityDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum SourceAuthorityLevel {
    Unknown,
    Reference,
    Delegated,
    Primary,
    Authoritative,
}

impl SourceAuthorityLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Reference => "reference",
            Self::Delegated => "delegated",
            Self::Primary => "primary",
            Self::Authoritative => "authoritative",
        }
    }
}

impl std::fmt::Display for SourceAuthorityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceAuthorityPolicy {
    pub domain: SourceAuthorityDomain,
    pub chain: String,
    pub level: SourceAuthorityLevel,
}

impl SourceAuthorityPolicy {
    pub fn new(
        domain: SourceAuthorityDomain,
        chain: impl Into<String>,
        level: SourceAuthorityLevel,
    ) -> Self {
        Self {
            domain,
            chain: normalize_tag(&chain.into()),
            level,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceAuthorityRegistry {
    policies: Vec<SourceAuthorityPolicy>,
}

impl SourceAuthorityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn policies(&self) -> &[SourceAuthorityPolicy] {
        &self.policies
    }

    pub fn with_policy(mut self, policy: SourceAuthorityPolicy) -> Self {
        self.add_policy(policy);
        self
    }

    pub fn add_policy(&mut self, policy: SourceAuthorityPolicy) {
        let normalized_chain = normalize_tag(&policy.chain);
        if let Some(existing) = self
            .policies
            .iter_mut()
            .find(|existing| existing.domain == policy.domain && existing.chain == normalized_chain)
        {
            existing.level = policy.level;
            return;
        }

        self.policies.push(SourceAuthorityPolicy {
            chain: normalized_chain,
            ..policy
        });
    }

    pub fn set_chain(
        &mut self,
        domain: SourceAuthorityDomain,
        chain: impl Into<String>,
        level: SourceAuthorityLevel,
    ) -> &mut Self {
        self.add_policy(SourceAuthorityPolicy::new(domain, chain, level));
        self
    }

    pub fn set_authoritative(
        &mut self,
        domain: SourceAuthorityDomain,
        chain: impl Into<String>,
    ) -> &mut Self {
        self.set_chain(domain, chain, SourceAuthorityLevel::Authoritative)
    }

    pub fn set_primary(
        &mut self,
        domain: SourceAuthorityDomain,
        chain: impl Into<String>,
    ) -> &mut Self {
        self.set_chain(domain, chain, SourceAuthorityLevel::Primary)
    }

    pub fn level_for_chain(
        &self,
        domain: SourceAuthorityDomain,
        chain: &str,
    ) -> SourceAuthorityLevel {
        let normalized = normalize_tag(chain);
        self.policies
            .iter()
            .filter(|policy| policy.domain == domain && policy.chain == normalized)
            .map(|policy| policy.level)
            .max()
            .unwrap_or(SourceAuthorityLevel::Unknown)
    }
}

pub(crate) fn infer_authority_domain(text: &str) -> Option<SourceAuthorityDomain> {
    let normalized = normalize_match_text(text);

    if contains_any(
        &normalized,
        &[
            "token",
            "credential",
            "secret",
            "api key",
            "private key",
            "certificate",
            "authentication",
            "authorization",
            "without auth",
            "bypass auth",
            "disable auth",
            "auth token",
        ],
    ) {
        return Some(SourceAuthorityDomain::Security);
    }

    if contains_any(
        &normalized,
        &[
            "network",
            "subnet",
            "private endpoint",
            "internal hostname",
            "internal share path",
            "share path",
            "relay endpoint",
            "hostname",
            "host name",
            "dns",
            "port",
            "tunnel",
            "bridge",
        ],
    ) {
        return Some(SourceAuthorityDomain::Networking);
    }

    if contains_any(
        &normalized,
        &[
            "maintenance",
            "breakglass",
            "recovery",
            "outage",
            "cutover",
            "reset window",
            "maintenance reset",
            "failover",
        ],
    ) {
        return Some(SourceAuthorityDomain::Maintenance);
    }

    if contains_any(
        &normalized,
        &[
            "build",
            "compile",
            "toolchain",
            "msvc",
            "cargo",
            "rustc",
            "cuda toolkit",
            "nvcc",
            "linker",
        ],
    ) {
        return Some(SourceAuthorityDomain::BuildToolchain);
    }

    if contains_any(
        &normalized,
        &[
            "runtime",
            "run on",
            "runs on",
            "gpu host",
            "cuda device",
            "embedding service",
            "rerank service",
            "native windows",
            "wsl host",
            "where should",
        ],
    ) {
        return Some(SourceAuthorityDomain::RuntimeOps);
    }

    if contains_any(
        &normalized,
        &[
            "startup",
            "scheduled task",
            "systemd",
            "autostart",
            "launch at logon",
            "logon",
            "startup path",
            "boot path",
            "service host path",
        ],
    ) {
        return Some(SourceAuthorityDomain::Deployment);
    }

    None
}

pub(crate) fn source_authority_rank(
    record: &MemoryMeta,
    domain: Option<SourceAuthorityDomain>,
    registry: Option<&SourceAuthorityRegistry>,
) -> u8 {
    match source_authority_level_for_domain(record, domain, registry) {
        SourceAuthorityLevel::Authoritative => 100,
        SourceAuthorityLevel::Primary => 80,
        SourceAuthorityLevel::Delegated => 55,
        SourceAuthorityLevel::Reference => 30,
        SourceAuthorityLevel::Unknown => 0,
    }
}

pub(crate) fn source_chain_for_domain(
    record: &MemoryMeta,
    domain: Option<SourceAuthorityDomain>,
    registry: Option<&SourceAuthorityRegistry>,
) -> Option<String> {
    if source_authority_rank(record, domain, registry) == 0 {
        return None;
    }

    record_source_chain(record)
}

pub(crate) fn source_authority_level_for_domain(
    record: &MemoryMeta,
    domain: Option<SourceAuthorityDomain>,
    registry: Option<&SourceAuthorityRegistry>,
) -> SourceAuthorityLevel {
    let Some(domain) = domain else {
        return SourceAuthorityLevel::Unknown;
    };

    let metadata_level = if source_authority_domains(record).contains(&domain) {
        source_authority_level(record)
    } else {
        SourceAuthorityLevel::Unknown
    };

    let registry_level = record_source_chain(record)
        .as_deref()
        .map(|chain| source_authority_level_from_registry(domain, chain, registry))
        .unwrap_or(SourceAuthorityLevel::Unknown);

    metadata_level.max(registry_level)
}

fn source_authority_domains(record: &MemoryMeta) -> Vec<SourceAuthorityDomain> {
    let raw = record
        .metadata
        .get("source_authority_domains")
        .or_else(|| record.metadata.get("source_authority_domain"))
        .or_else(|| record.metadata.get("authority_domain"));
    let Some(raw) = raw else {
        return Vec::new();
    };

    raw.split([',', ';', '|'])
        .filter_map(parse_domain_tag)
        .collect()
}

fn record_source_chain(record: &MemoryMeta) -> Option<String> {
    record
        .metadata
        .get("source_chain")
        .or_else(|| record.metadata.get("source_authority_chain"))
        .or_else(|| record.metadata.get("authority_chain"))
        .map(|value| normalize_tag(value))
        .filter(|value| !value.is_empty())
}

fn source_authority_level_from_registry(
    domain: SourceAuthorityDomain,
    chain: &str,
    registry: Option<&SourceAuthorityRegistry>,
) -> SourceAuthorityLevel {
    registry
        .map(|registry| registry.level_for_chain(domain, chain))
        .unwrap_or(SourceAuthorityLevel::Unknown)
}

fn source_authority_level(record: &MemoryMeta) -> SourceAuthorityLevel {
    let raw = record
        .metadata
        .get("source_authority_level")
        .or_else(|| record.metadata.get("authority_level"));
    let Some(raw) = raw else {
        return SourceAuthorityLevel::Unknown;
    };

    match normalize_tag(raw).as_str() {
        "authoritative" | "authority" | "owner" => SourceAuthorityLevel::Authoritative,
        "primary" | "preferred" => SourceAuthorityLevel::Primary,
        "delegated" | "delegate" => SourceAuthorityLevel::Delegated,
        "reference" | "advisory" | "fallback" => SourceAuthorityLevel::Reference,
        _ => SourceAuthorityLevel::Unknown,
    }
}

fn parse_domain_tag(raw: &str) -> Option<SourceAuthorityDomain> {
    match normalize_tag(raw).as_str() {
        "runtime" | "runtime-ops" | "runtime_ops" | "service-runtime" | "gpu-runtime" => {
            Some(SourceAuthorityDomain::RuntimeOps)
        }
        "deployment" | "startup" | "startup-path" | "service-hosting" | "host-placement" => {
            Some(SourceAuthorityDomain::Deployment)
        }
        "network" | "networking" | "network-ops" | "private-infra" | "infra-network" => {
            Some(SourceAuthorityDomain::Networking)
        }
        "security" | "auth" | "secrets" | "secret-management" => {
            Some(SourceAuthorityDomain::Security)
        }
        "build" | "toolchain" | "build-toolchain" | "ci-build" => {
            Some(SourceAuthorityDomain::BuildToolchain)
        }
        "maintenance" | "recovery" | "breakglass" | "cutover" => {
            Some(SourceAuthorityDomain::Maintenance)
        }
        _ => None,
    }
}

fn normalize_text(value: &str) -> String {
    value.trim().to_lowercase().replace('_', "-")
}

fn normalize_tag(value: &str) -> String {
    normalize_text(value)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
}

fn normalize_match_text(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| {
        let needle = normalize_match_text(needle);
        if needle.contains(' ') {
            haystack.contains(&needle)
        } else {
            haystack.split_whitespace().any(|token| token == needle)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;
    use std::collections::HashMap;

    fn meta(
        authority_domain: Option<&str>,
        authority_level: Option<&str>,
        source_chain: Option<&str>,
    ) -> MemoryMeta {
        let mut metadata = HashMap::new();
        if let Some(authority_domain) = authority_domain {
            metadata.insert(
                "source_authority_domain".to_string(),
                authority_domain.to_string(),
            );
        }
        if let Some(authority_level) = authority_level {
            metadata.insert(
                "source_authority_level".to_string(),
                authority_level.to_string(),
            );
        }
        if let Some(source_chain) = source_chain {
            metadata.insert("source_chain".to_string(), source_chain.to_string());
        }

        MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Procedural,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata,
        }
    }

    #[test]
    fn infers_runtime_domain_from_gpu_host_queries() {
        assert_eq!(
            infer_authority_domain("Where should the embedding service run on the GPU host?"),
            Some(SourceAuthorityDomain::RuntimeOps)
        );
    }

    #[test]
    fn source_authority_rank_requires_domain_match() {
        let record = meta(Some("runtime"), Some("authoritative"), Some("runtime-ops"));

        assert_eq!(
            source_authority_rank(&record, Some(SourceAuthorityDomain::RuntimeOps), None),
            100
        );
        assert_eq!(
            source_authority_rank(&record, Some(SourceAuthorityDomain::Deployment), None),
            0
        );
    }

    #[test]
    fn source_chain_is_returned_only_for_matching_domains() {
        let record = meta(
            Some("runtime"),
            Some("primary"),
            Some("platform-runtime-chain"),
        );

        assert_eq!(
            source_chain_for_domain(&record, Some(SourceAuthorityDomain::RuntimeOps), None),
            Some("platform-runtime-chain".to_string())
        );
        assert_eq!(
            source_chain_for_domain(&record, Some(SourceAuthorityDomain::Security), None),
            None
        );
    }

    #[test]
    fn registry_can_promote_chain_without_record_domain_metadata() {
        let record = meta(None, None, Some("runtime-ops"));
        let registry = SourceAuthorityRegistry::new().with_policy(SourceAuthorityPolicy::new(
            SourceAuthorityDomain::RuntimeOps,
            "runtime-ops",
            SourceAuthorityLevel::Authoritative,
        ));

        assert_eq!(
            source_authority_rank(
                &record,
                Some(SourceAuthorityDomain::RuntimeOps),
                Some(&registry),
            ),
            100
        );
    }

    #[test]
    fn registry_and_metadata_use_the_stronger_authority_level() {
        let record = meta(Some("runtime"), Some("primary"), Some("runtime-ops"));
        let registry = SourceAuthorityRegistry::new().with_policy(SourceAuthorityPolicy::new(
            SourceAuthorityDomain::RuntimeOps,
            "runtime-ops",
            SourceAuthorityLevel::Authoritative,
        ));

        assert_eq!(
            source_authority_level_for_domain(
                &record,
                Some(SourceAuthorityDomain::RuntimeOps),
                Some(&registry),
            ),
            SourceAuthorityLevel::Authoritative
        );
    }
}
