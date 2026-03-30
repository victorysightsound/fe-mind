use crate::traits::{MemoryMeta, ScoringStrategy};

/// Provenance-aware scoring for richer source metadata carried on memory records.
///
/// Expected metadata keys:
/// - `source_kind`: `system` | `maintainer` | `project-doc` | `local-observation`
///   | `user-note` | `copied-chat` | `forum-post` | `external-web`
/// - `source_verification`: `verified` | `observed` | `declared` | `copied`
///   | `unverified`
pub struct SourceProvenanceScorer {
    internal_weight: f32,
    project_doc_weight: f32,
    local_observation_weight: f32,
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
        Self::new(1.08, 1.06, 1.08, 0.88, 0.8, 1.08, 0.9, 0.78)
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
            SourceVerification::Declared | SourceVerification::Unknown => 1.0,
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
    Declared,
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
        "declared" => SourceVerification::Declared,
        "copied" => SourceVerification::Copied,
        "unverified" => SourceVerification::Unverified,
        _ => SourceVerification::Unknown,
    }
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
}
