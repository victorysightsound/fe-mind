use std::sync::Arc;

use crate::traits::{MemoryMeta, ScoringStrategy};

/// Query-aware source-chain authority scoring.
///
/// Expected metadata keys:
/// - `source_authority_domain`: single domain tag
/// - `source_authority_domains`: comma/semicolon/pipe separated domain tags
/// - `source_authority_level`: `authoritative` | `primary` | `delegated` | `reference`
/// - `source_chain`: stable source-chain identifier for arbitration
/// - `source_kind`: stable source-kind identifier for app-facing domain defaults
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
        let domains = infer_authority_domains(query);
        if domains.is_empty() {
            return 1.0;
        }

        match source_authority_level_for_domains(record, &domains, Some(&self.registry)) {
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

/// App-facing policy for how contested reflected summaries should compose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ContestedSummaryPolicy {
    PreferContestedAnswer,
    WinnerWithConflictNote,
    AbstainUntilResolved,
}

impl ContestedSummaryPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PreferContestedAnswer => "prefer-contested-answer",
            Self::WinnerWithConflictNote => "winner-with-conflict-note",
            Self::AbstainUntilResolved => "abstain-until-resolved",
        }
    }
}

impl Default for ContestedSummaryPolicy {
    fn default() -> Self {
        Self::PreferContestedAnswer
    }
}

impl std::fmt::Display for ContestedSummaryPolicy {
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceAuthorityKindPolicy {
    pub domain: SourceAuthorityDomain,
    pub kind: String,
    pub level: SourceAuthorityLevel,
}

impl SourceAuthorityKindPolicy {
    pub fn new(
        domain: SourceAuthorityDomain,
        kind: impl Into<String>,
        level: SourceAuthorityLevel,
    ) -> Self {
        Self {
            domain,
            kind: normalize_tag(&kind.into()),
            level,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceAuthorityDomainPolicy {
    pub domain: SourceAuthorityDomain,
    pub authoritative_chains: Vec<String>,
    pub primary_chains: Vec<String>,
    pub delegated_chains: Vec<String>,
    pub reference_chains: Vec<String>,
    pub authoritative_kinds: Vec<String>,
    pub primary_kinds: Vec<String>,
    pub delegated_kinds: Vec<String>,
    pub reference_kinds: Vec<String>,
    pub contested_summary_policy: Option<ContestedSummaryPolicy>,
}

impl SourceAuthorityDomainPolicy {
    pub fn new(domain: SourceAuthorityDomain) -> Self {
        Self {
            domain,
            authoritative_chains: Vec::new(),
            primary_chains: Vec::new(),
            delegated_chains: Vec::new(),
            reference_chains: Vec::new(),
            authoritative_kinds: Vec::new(),
            primary_kinds: Vec::new(),
            delegated_kinds: Vec::new(),
            reference_kinds: Vec::new(),
            contested_summary_policy: None,
        }
    }

    pub fn with_authoritative_chain(mut self, chain: impl Into<String>) -> Self {
        self.authoritative_chains.push(normalize_tag(&chain.into()));
        self
    }

    pub fn with_primary_chain(mut self, chain: impl Into<String>) -> Self {
        self.primary_chains.push(normalize_tag(&chain.into()));
        self
    }

    pub fn with_delegated_chain(mut self, chain: impl Into<String>) -> Self {
        self.delegated_chains.push(normalize_tag(&chain.into()));
        self
    }

    pub fn with_reference_chain(mut self, chain: impl Into<String>) -> Self {
        self.reference_chains.push(normalize_tag(&chain.into()));
        self
    }

    pub fn with_authoritative_kind(mut self, kind: impl Into<String>) -> Self {
        self.authoritative_kinds.push(normalize_tag(&kind.into()));
        self
    }

    pub fn with_primary_kind(mut self, kind: impl Into<String>) -> Self {
        self.primary_kinds.push(normalize_tag(&kind.into()));
        self
    }

    pub fn with_delegated_kind(mut self, kind: impl Into<String>) -> Self {
        self.delegated_kinds.push(normalize_tag(&kind.into()));
        self
    }

    pub fn with_reference_kind(mut self, kind: impl Into<String>) -> Self {
        self.reference_kinds.push(normalize_tag(&kind.into()));
        self
    }

    pub fn with_contested_summary_policy(mut self, policy: ContestedSummaryPolicy) -> Self {
        self.contested_summary_policy = Some(policy);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ContestedSummaryDomainPolicy {
    pub domain: SourceAuthorityDomain,
    pub policy: ContestedSummaryPolicy,
}

impl ContestedSummaryDomainPolicy {
    pub fn new(domain: SourceAuthorityDomain, policy: ContestedSummaryPolicy) -> Self {
        Self { domain, policy }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SourceAuthorityRegistry {
    policies: Vec<SourceAuthorityPolicy>,
    kind_policies: Vec<SourceAuthorityKindPolicy>,
    contested_summary_policies: Vec<ContestedSummaryDomainPolicy>,
}

impl SourceAuthorityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn policies(&self) -> &[SourceAuthorityPolicy] {
        &self.policies
    }

    pub fn kind_policies(&self) -> &[SourceAuthorityKindPolicy] {
        &self.kind_policies
    }

    pub fn contested_summary_policies(&self) -> &[ContestedSummaryDomainPolicy] {
        &self.contested_summary_policies
    }

    pub fn with_policy(mut self, policy: SourceAuthorityPolicy) -> Self {
        self.add_policy(policy);
        self
    }

    pub fn with_kind_policy(mut self, policy: SourceAuthorityKindPolicy) -> Self {
        self.add_kind_policy(policy);
        self
    }

    pub fn with_domain_policy(mut self, policy: SourceAuthorityDomainPolicy) -> Self {
        self.add_domain_policy(policy);
        self
    }

    pub fn with_contested_summary_policy(
        mut self,
        domain: SourceAuthorityDomain,
        policy: ContestedSummaryPolicy,
    ) -> Self {
        self.set_contested_summary_policy(domain, policy);
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

    pub fn add_kind_policy(&mut self, policy: SourceAuthorityKindPolicy) {
        let normalized_kind = normalize_tag(&policy.kind);
        if let Some(existing) = self
            .kind_policies
            .iter_mut()
            .find(|existing| existing.domain == policy.domain && existing.kind == normalized_kind)
        {
            existing.level = policy.level;
            return;
        }

        self.kind_policies.push(SourceAuthorityKindPolicy {
            kind: normalized_kind,
            ..policy
        });
    }

    pub fn add_domain_policy(&mut self, policy: SourceAuthorityDomainPolicy) {
        let SourceAuthorityDomainPolicy {
            domain,
            authoritative_chains,
            primary_chains,
            delegated_chains,
            reference_chains,
            authoritative_kinds,
            primary_kinds,
            delegated_kinds,
            reference_kinds,
            contested_summary_policy,
        } = policy;

        for chain in authoritative_chains {
            self.add_policy(SourceAuthorityPolicy::new(
                domain,
                chain,
                SourceAuthorityLevel::Authoritative,
            ));
        }
        for chain in primary_chains {
            self.add_policy(SourceAuthorityPolicy::new(
                domain,
                chain,
                SourceAuthorityLevel::Primary,
            ));
        }
        for chain in delegated_chains {
            self.add_policy(SourceAuthorityPolicy::new(
                domain,
                chain,
                SourceAuthorityLevel::Delegated,
            ));
        }
        for chain in reference_chains {
            self.add_policy(SourceAuthorityPolicy::new(
                domain,
                chain,
                SourceAuthorityLevel::Reference,
            ));
        }

        for kind in authoritative_kinds {
            self.add_kind_policy(SourceAuthorityKindPolicy::new(
                domain,
                kind,
                SourceAuthorityLevel::Authoritative,
            ));
        }
        for kind in primary_kinds {
            self.add_kind_policy(SourceAuthorityKindPolicy::new(
                domain,
                kind,
                SourceAuthorityLevel::Primary,
            ));
        }
        for kind in delegated_kinds {
            self.add_kind_policy(SourceAuthorityKindPolicy::new(
                domain,
                kind,
                SourceAuthorityLevel::Delegated,
            ));
        }
        for kind in reference_kinds {
            self.add_kind_policy(SourceAuthorityKindPolicy::new(
                domain,
                kind,
                SourceAuthorityLevel::Reference,
            ));
        }

        if let Some(policy) = contested_summary_policy {
            self.set_contested_summary_policy(domain, policy);
        }
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

    pub fn set_kind(
        &mut self,
        domain: SourceAuthorityDomain,
        kind: impl Into<String>,
        level: SourceAuthorityLevel,
    ) -> &mut Self {
        self.add_kind_policy(SourceAuthorityKindPolicy::new(domain, kind, level));
        self
    }

    pub fn set_authoritative_kind(
        &mut self,
        domain: SourceAuthorityDomain,
        kind: impl Into<String>,
    ) -> &mut Self {
        self.set_kind(domain, kind, SourceAuthorityLevel::Authoritative)
    }

    pub fn set_primary_kind(
        &mut self,
        domain: SourceAuthorityDomain,
        kind: impl Into<String>,
    ) -> &mut Self {
        self.set_kind(domain, kind, SourceAuthorityLevel::Primary)
    }

    pub fn set_contested_summary_policy(
        &mut self,
        domain: SourceAuthorityDomain,
        policy: ContestedSummaryPolicy,
    ) -> &mut Self {
        if let Some(existing) = self
            .contested_summary_policies
            .iter_mut()
            .find(|existing| existing.domain == domain)
        {
            existing.policy = policy;
            return self;
        }

        self.contested_summary_policies
            .push(ContestedSummaryDomainPolicy::new(domain, policy));
        self
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

    pub fn level_for_kind(
        &self,
        domain: SourceAuthorityDomain,
        kind: &str,
    ) -> SourceAuthorityLevel {
        let normalized = normalize_tag(kind);
        self.kind_policies
            .iter()
            .filter(|policy| policy.domain == domain && policy.kind == normalized)
            .map(|policy| policy.level)
            .max()
            .unwrap_or(SourceAuthorityLevel::Unknown)
    }

    pub fn contested_summary_policy_for_domain(
        &self,
        domain: SourceAuthorityDomain,
    ) -> Option<ContestedSummaryPolicy> {
        self.contested_summary_policies
            .iter()
            .find(|entry| entry.domain == domain)
            .map(|entry| entry.policy)
    }

    pub fn contested_summary_policy_for_domains(
        &self,
        domains: &[SourceAuthorityDomain],
    ) -> ContestedSummaryPolicy {
        domains
            .iter()
            .filter_map(|domain| self.contested_summary_policy_for_domain(*domain))
            .max_by_key(|policy| match policy {
                ContestedSummaryPolicy::WinnerWithConflictNote => 1u8,
                ContestedSummaryPolicy::PreferContestedAnswer => 2u8,
                ContestedSummaryPolicy::AbstainUntilResolved => 3u8,
            })
            .unwrap_or_default()
    }
}

pub(crate) fn infer_authority_domain(text: &str) -> Option<SourceAuthorityDomain> {
    infer_authority_domains(text).into_iter().next()
}

pub(crate) fn infer_authority_domains(text: &str) -> Vec<SourceAuthorityDomain> {
    let normalized = normalize_match_text(text);
    let mut domains = Vec::new();

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
        domains.push(SourceAuthorityDomain::Security);
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
        domains.push(SourceAuthorityDomain::Networking);
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
        domains.push(SourceAuthorityDomain::Maintenance);
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
        domains.push(SourceAuthorityDomain::BuildToolchain);
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
        domains.push(SourceAuthorityDomain::RuntimeOps);
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
        domains.push(SourceAuthorityDomain::Deployment);
    }

    domains
}

pub(crate) fn source_authority_rank(
    record: &MemoryMeta,
    domain: Option<SourceAuthorityDomain>,
    registry: Option<&SourceAuthorityRegistry>,
) -> u8 {
    authority_rank_for_level(source_authority_level_for_domain(record, domain, registry))
}

pub(crate) fn source_authority_rank_for_domains(
    record: &MemoryMeta,
    domains: &[SourceAuthorityDomain],
    registry: Option<&SourceAuthorityRegistry>,
) -> u8 {
    authority_rank_for_level(source_authority_level_for_domains(
        record, domains, registry,
    ))
}

pub(crate) fn source_authority_score_sum_for_domains(
    record: &MemoryMeta,
    domains: &[SourceAuthorityDomain],
    registry: Option<&SourceAuthorityRegistry>,
) -> u16 {
    domains
        .iter()
        .copied()
        .map(|domain| {
            authority_rank_for_level(source_authority_level_for_domain(
                record,
                Some(domain),
                registry,
            )) as u16
        })
        .sum()
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

pub(crate) fn source_chain_for_domains(
    record: &MemoryMeta,
    domains: &[SourceAuthorityDomain],
    registry: Option<&SourceAuthorityRegistry>,
) -> Option<String> {
    for domain in domains {
        if let Some(chain) = source_chain_for_domain(record, Some(*domain), registry) {
            return Some(chain);
        }
    }

    None
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
    let registry_kind_level = record_source_kind(record)
        .as_deref()
        .map(|kind| source_authority_kind_level_from_registry(domain, kind, registry))
        .unwrap_or(SourceAuthorityLevel::Unknown);

    metadata_level.max(registry_level).max(registry_kind_level)
}

pub(crate) fn source_authority_level_for_domains(
    record: &MemoryMeta,
    domains: &[SourceAuthorityDomain],
    registry: Option<&SourceAuthorityRegistry>,
) -> SourceAuthorityLevel {
    domains
        .iter()
        .copied()
        .map(|domain| source_authority_level_for_domain(record, Some(domain), registry))
        .max()
        .unwrap_or(SourceAuthorityLevel::Unknown)
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

fn record_source_kind(record: &MemoryMeta) -> Option<String> {
    record
        .metadata
        .get("source_kind")
        .or_else(|| record.metadata.get("authority_kind"))
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

fn source_authority_kind_level_from_registry(
    domain: SourceAuthorityDomain,
    kind: &str,
    registry: Option<&SourceAuthorityRegistry>,
) -> SourceAuthorityLevel {
    registry
        .map(|registry| registry.level_for_kind(domain, kind))
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

fn authority_rank_for_level(level: SourceAuthorityLevel) -> u8 {
    match level {
        SourceAuthorityLevel::Authoritative => 100,
        SourceAuthorityLevel::Primary => 80,
        SourceAuthorityLevel::Delegated => 55,
        SourceAuthorityLevel::Reference => 30,
        SourceAuthorityLevel::Unknown => 0,
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
        source_kind: Option<&str>,
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
        if let Some(source_kind) = source_kind {
            metadata.insert("source_kind".to_string(), source_kind.to_string());
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
    fn infers_multiple_domains_for_cross_domain_queries() {
        let domains = infer_authority_domains(
            "Which runtime path should the service use through the private endpoint on the GPU host now?",
        );

        assert!(domains.contains(&SourceAuthorityDomain::Networking));
        assert!(domains.contains(&SourceAuthorityDomain::RuntimeOps));
    }

    #[test]
    fn source_authority_rank_requires_domain_match() {
        let record = meta(
            Some("runtime"),
            Some("authoritative"),
            Some("runtime-ops"),
            None,
        );

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
            None,
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
        let record = meta(None, None, Some("runtime-ops"), None);
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
        let record = meta(Some("runtime"), Some("primary"), Some("runtime-ops"), None);
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

    #[test]
    fn registry_can_promote_kind_without_chain_metadata() {
        let record = meta(None, None, None, Some("system"));
        let registry =
            SourceAuthorityRegistry::new().with_kind_policy(SourceAuthorityKindPolicy::new(
                SourceAuthorityDomain::RuntimeOps,
                "system",
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
    fn registry_prefers_stronger_of_kind_and_chain_policies() {
        let record = meta(None, None, Some("runtime-ops"), Some("maintainer"));
        let registry = SourceAuthorityRegistry::new()
            .with_policy(SourceAuthorityPolicy::new(
                SourceAuthorityDomain::RuntimeOps,
                "runtime-ops",
                SourceAuthorityLevel::Primary,
            ))
            .with_kind_policy(SourceAuthorityKindPolicy::new(
                SourceAuthorityDomain::RuntimeOps,
                "maintainer",
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

    #[test]
    fn domain_policy_expands_to_kind_and_chain_entries() {
        let registry = SourceAuthorityRegistry::new().with_domain_policy(
            SourceAuthorityDomainPolicy::new(SourceAuthorityDomain::RuntimeOps)
                .with_authoritative_kind("maintainer")
                .with_primary_kind("project-doc")
                .with_authoritative_chain("runtime-ops")
                .with_primary_chain("runtime-bootstrap"),
        );

        assert_eq!(
            registry.level_for_kind(SourceAuthorityDomain::RuntimeOps, "maintainer"),
            SourceAuthorityLevel::Authoritative
        );
        assert_eq!(
            registry.level_for_kind(SourceAuthorityDomain::RuntimeOps, "project-doc"),
            SourceAuthorityLevel::Primary
        );
        assert_eq!(
            registry.level_for_chain(SourceAuthorityDomain::RuntimeOps, "runtime-ops"),
            SourceAuthorityLevel::Authoritative
        );
        assert_eq!(
            registry.level_for_chain(SourceAuthorityDomain::RuntimeOps, "runtime-bootstrap"),
            SourceAuthorityLevel::Primary
        );
    }

    #[test]
    fn multi_domain_rank_uses_strongest_matching_domain() {
        let record = meta(None, None, None, Some("maintainer"));
        let registry = SourceAuthorityRegistry::new()
            .with_kind_policy(SourceAuthorityKindPolicy::new(
                SourceAuthorityDomain::RuntimeOps,
                "maintainer",
                SourceAuthorityLevel::Authoritative,
            ))
            .with_kind_policy(SourceAuthorityKindPolicy::new(
                SourceAuthorityDomain::Networking,
                "maintainer",
                SourceAuthorityLevel::Reference,
            ));

        let domains = infer_authority_domains(
            "Which runtime path should the service use through the private endpoint on the GPU host now?",
        );

        assert_eq!(
            source_authority_rank_for_domains(&record, &domains, Some(&registry)),
            100
        );
        assert_eq!(
            source_chain_for_domains(&record, &domains, Some(&registry)),
            None
        );
    }

    #[test]
    fn multi_domain_score_sum_accumulates_matching_domains() {
        let record = meta(None, None, None, Some("maintainer"));
        let registry = SourceAuthorityRegistry::new()
            .with_kind_policy(SourceAuthorityKindPolicy::new(
                SourceAuthorityDomain::RuntimeOps,
                "maintainer",
                SourceAuthorityLevel::Authoritative,
            ))
            .with_kind_policy(SourceAuthorityKindPolicy::new(
                SourceAuthorityDomain::Networking,
                "maintainer",
                SourceAuthorityLevel::Primary,
            ))
            .with_kind_policy(SourceAuthorityKindPolicy::new(
                SourceAuthorityDomain::Deployment,
                "maintainer",
                SourceAuthorityLevel::Reference,
            ));

        let domains = infer_authority_domains(
            "Which runtime startup path should the service use through the private endpoint on the GPU host now?",
        );

        assert_eq!(
            source_authority_rank_for_domains(&record, &domains, Some(&registry)),
            100
        );
        assert_eq!(
            source_authority_score_sum_for_domains(&record, &domains, Some(&registry)),
            210
        );
    }

    #[test]
    fn contested_summary_policy_uses_strictest_matching_domain_policy() {
        let registry = SourceAuthorityRegistry::new()
            .with_domain_policy(
                SourceAuthorityDomainPolicy::new(SourceAuthorityDomain::RuntimeOps)
                    .with_contested_summary_policy(ContestedSummaryPolicy::WinnerWithConflictNote),
            )
            .with_domain_policy(
                SourceAuthorityDomainPolicy::new(SourceAuthorityDomain::Security)
                    .with_contested_summary_policy(ContestedSummaryPolicy::AbstainUntilResolved),
            );

        assert_eq!(
            registry.contested_summary_policy_for_domains(&[
                SourceAuthorityDomain::RuntimeOps,
                SourceAuthorityDomain::Security,
            ]),
            ContestedSummaryPolicy::AbstainUntilResolved
        );
        assert_eq!(
            registry.contested_summary_policy_for_domains(&[SourceAuthorityDomain::RuntimeOps]),
            ContestedSummaryPolicy::WinnerWithConflictNote
        );
    }
}
