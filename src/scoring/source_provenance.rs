use crate::traits::{MemoryMeta, ScoringStrategy};

/// Provenance-aware scoring for richer source metadata carried on memory records.
///
/// Expected metadata keys:
/// - `source_kind`: `system` | `maintainer` | `project-doc` | `local-observation`
///   | `user-note` | `copied-chat` | `forum-post` | `external-web`
/// - `source_verification`: `verified` | `observed` | `partially-verified`
///   | `declared` | `relayed` | `copied` | `unverified`
pub struct SourceProvenanceScorer {
    internal_weight: f32,
    project_doc_weight: f32,
    local_observation_weight: f32,
    partial_verified_weight: f32,
    relayed_weight: f32,
    copied_chat_weight: f32,
    external_weight: f32,
    verified_weight: f32,
    copied_weight: f32,
    unverified_weight: f32,
}

impl SourceProvenanceScorer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        internal_weight: f32,
        project_doc_weight: f32,
        local_observation_weight: f32,
        partial_verified_weight: f32,
        relayed_weight: f32,
        copied_chat_weight: f32,
        external_weight: f32,
        verified_weight: f32,
        copied_weight: f32,
        unverified_weight: f32,
    ) -> Self {
        Self {
            internal_weight,
            project_doc_weight,
            local_observation_weight,
            partial_verified_weight,
            relayed_weight,
            copied_chat_weight,
            external_weight,
            verified_weight,
            copied_weight,
            unverified_weight,
        }
    }
}

impl Default for SourceProvenanceScorer {
    fn default() -> Self {
        Self::new(1.08, 1.06, 1.08, 1.03, 0.94, 0.88, 0.8, 1.08, 0.9, 0.78)
    }
}

impl ScoringStrategy for SourceProvenanceScorer {
    fn score_multiplier(&self, record: &MemoryMeta, _query: &str, _base_score: f32) -> f32 {
        let kind_weight = match source_kind(record) {
            SourceKind::System | SourceKind::Maintainer => self.internal_weight,
            SourceKind::ProjectDoc => self.project_doc_weight,
            SourceKind::LocalObservation | SourceKind::UserNote => self.local_observation_weight,
            SourceKind::CopiedChat => self.copied_chat_weight,
            SourceKind::ForumPost | SourceKind::ExternalWeb => self.external_weight,
            SourceKind::Unknown => 1.0,
        };

        let verification_weight = match source_verification(record) {
            SourceVerification::Verified | SourceVerification::Observed => self.verified_weight,
            SourceVerification::PartiallyVerified => self.partial_verified_weight,
            SourceVerification::Declared | SourceVerification::Unknown => 1.0,
            SourceVerification::Relayed => self.relayed_weight,
            SourceVerification::Copied => self.copied_weight,
            SourceVerification::Unverified => self.unverified_weight,
        };

        kind_weight * verification_weight
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceKind {
    System,
    Maintainer,
    ProjectDoc,
    LocalObservation,
    UserNote,
    CopiedChat,
    ForumPost,
    ExternalWeb,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceVerification {
    Verified,
    Observed,
    PartiallyVerified,
    Declared,
    Relayed,
    Copied,
    Unverified,
    Unknown,
}

pub(crate) fn source_kind(record: &MemoryMeta) -> SourceKind {
    let Some(value) = record.metadata.get("source_kind") else {
        return SourceKind::Unknown;
    };

    match normalize_tag(value).as_str() {
        "system" => SourceKind::System,
        "maintainer" | "maintainer-note" => SourceKind::Maintainer,
        "project-doc" | "project_doc" | "documentation" => SourceKind::ProjectDoc,
        "local-observation" | "local_observation" | "observed-local" => {
            SourceKind::LocalObservation
        }
        "user-note" | "user_note" => SourceKind::UserNote,
        "copied-chat" | "copied_chat" | "chat-copy" => SourceKind::CopiedChat,
        "forum-post" | "forum_post" => SourceKind::ForumPost,
        "external-web" | "external_web" | "web" => SourceKind::ExternalWeb,
        _ => SourceKind::Unknown,
    }
}

pub(crate) fn source_verification(record: &MemoryMeta) -> SourceVerification {
    let Some(value) = record.metadata.get("source_verification") else {
        return SourceVerification::Unknown;
    };

    match normalize_tag(value).as_str() {
        "verified" => SourceVerification::Verified,
        "observed" => SourceVerification::Observed,
        "partially-verified" | "partially_verified" | "partial" | "partially-observed" => {
            SourceVerification::PartiallyVerified
        }
        "declared" => SourceVerification::Declared,
        "relayed" | "second-hand" | "second_hand" => SourceVerification::Relayed,
        "copied" => SourceVerification::Copied,
        "unverified" => SourceVerification::Unverified,
        _ => SourceVerification::Unknown,
    }
}

pub(crate) fn source_provenance_rank(record: &MemoryMeta) -> u8 {
    let kind_rank = match source_kind(record) {
        SourceKind::System => 60,
        SourceKind::Maintainer => 55,
        SourceKind::ProjectDoc => 50,
        SourceKind::LocalObservation => 48,
        SourceKind::UserNote => 44,
        SourceKind::CopiedChat => 28,
        SourceKind::ExternalWeb => 18,
        SourceKind::ForumPost => 14,
        SourceKind::Unknown => 32,
    };
    let verification_rank = match source_verification(record) {
        SourceVerification::Verified => 20,
        SourceVerification::Observed => 18,
        SourceVerification::PartiallyVerified => 14,
        SourceVerification::Declared => 10,
        SourceVerification::Relayed => 7,
        SourceVerification::Copied => 5,
        SourceVerification::Unverified => 0,
        SourceVerification::Unknown => 8,
    };
    kind_rank + verification_rank
}

fn normalize_tag(value: &str) -> String {
    value.trim().to_lowercase().replace('_', "-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;
    use std::collections::HashMap;

    fn meta(kind: Option<&str>, verification: Option<&str>) -> MemoryMeta {
        let mut metadata = HashMap::new();
        if let Some(kind) = kind {
            metadata.insert("source_kind".to_string(), kind.to_string());
        }
        if let Some(verification) = verification {
            metadata.insert("source_verification".to_string(), verification.to_string());
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
    fn verified_internal_sources_get_boosted() {
        let scorer = SourceProvenanceScorer::default();
        let m = scorer.score_multiplier(&meta(Some("maintainer"), Some("verified")), "q", 1.0);
        assert!(m > 1.0);
    }

    #[test]
    fn copied_external_sources_are_penalized() {
        let scorer = SourceProvenanceScorer::default();
        let m = scorer.score_multiplier(&meta(Some("forum-post"), Some("unverified")), "q", 1.0);
        assert!(m < 1.0);
    }

    #[test]
    fn provenance_rank_prefers_verified_system_sources() {
        let system_verified = meta(Some("system"), Some("verified"));
        let maintainer_declared = meta(Some("maintainer"), Some("declared"));

        assert!(
            source_provenance_rank(&system_verified) > source_provenance_rank(&maintainer_declared)
        );
    }

    #[test]
    fn provenance_rank_prefers_partial_verification_over_declared() {
        let partial = meta(Some("maintainer"), Some("partially-verified"));
        let declared = meta(Some("maintainer"), Some("declared"));

        assert!(source_provenance_rank(&partial) > source_provenance_rank(&declared));
    }

    #[test]
    fn provenance_rank_prefers_declared_over_relayed_chain() {
        let declared = meta(Some("project-doc"), Some("declared"));
        let relayed = meta(Some("project-doc"), Some("relayed"));

        assert!(source_provenance_rank(&declared) > source_provenance_rank(&relayed));
    }
}
