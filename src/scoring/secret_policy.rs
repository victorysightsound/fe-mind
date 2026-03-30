use std::collections::HashMap;

/// Secret-sensitivity classes used by deterministic composition and safety rules.
///
/// Expected metadata keys:
/// - `content_secret_class`
/// - `content_sensitivity`
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SecretClass {
    /// Raw credential or token material that must never be surfaced verbatim.
    CredentialMaterial,
    /// A safe storage location for secret material (env file, vault path, item name).
    CredentialLocation,
    /// A safe reference to a secret-handling mechanism (for example `op read ...`).
    SecretReference,
}

impl SecretClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CredentialMaterial => "credential-material",
            Self::CredentialLocation => "credential-location",
            Self::SecretReference => "secret-reference",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match normalize_tag(value).as_str() {
            "credential-material"
            | "credential"
            | "secret-material"
            | "token-material"
            | "api-key"
            | "private-key"
            | "password-material" => Some(Self::CredentialMaterial),
            "credential-location"
            | "credential-storage"
            | "secret-location"
            | "secret-storage"
            | "vault-location" => Some(Self::CredentialLocation),
            "secret-reference" | "credential-reference" | "secret-guidance" => {
                Some(Self::SecretReference)
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for SecretClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

pub fn secret_class_from_metadata(metadata: &HashMap<String, String>) -> Option<SecretClass> {
    metadata
        .get("content_secret_class")
        .or_else(|| metadata.get("content_sensitivity"))
        .and_then(|value| SecretClass::from_str(value))
}

pub fn query_requests_secret_location_or_reference(query: &str) -> bool {
    let normalized = normalize_query(query);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();

    let secret_signal = contains_secret_signal(&normalized);
    if !secret_signal {
        return false;
    }

    tokens.contains(&"where")
        || normalized.contains("stored")
        || normalized.contains("storage")
        || normalized.contains("which file")
        || normalized.contains("which vault")
        || normalized.contains("which item")
        || normalized.contains("how should")
        || normalized.contains("how do i load")
        || normalized.contains("how is")
        || normalized.contains("loaded from")
        || normalized.contains("loaded")
        || normalized.contains("reference")
        || normalized.contains("safe way")
}

pub fn query_requests_sensitive_secret_detail(query: &str) -> bool {
    let normalized = normalize_query(query);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();

    if !contains_secret_signal(&normalized) {
        return false;
    }

    let location_signal = query_requests_secret_location_or_reference(query);
    let reveal_signal = normalized.contains("exact")
        || normalized.contains("actual")
        || normalized.contains("full")
        || normalized.contains("raw")
        || normalized.contains("literal")
        || normalized.contains("complete")
        || normalized.contains("entire")
        || tokens.contains(&"value")
        || tokens.contains(&"show")
        || tokens.contains(&"print")
        || tokens.contains(&"paste")
        || tokens.contains(&"dump")
        || normalized.contains("what is the")
        || normalized.contains("give me the");

    reveal_signal || !location_signal
}

pub fn evidence_contains_secret_material(text: &str, metadata: &HashMap<String, String>) -> bool {
    matches!(
        secret_class_from_metadata(metadata),
        Some(SecretClass::CredentialMaterial)
    ) || text
        .split_whitespace()
        .any(token_contains_secret_assignment)
}

pub fn redact_secret_material(text: &str, metadata: &HashMap<String, String>) -> String {
    if !evidence_contains_secret_material(text, metadata) {
        return text.to_string();
    }

    let mut tokens = text
        .split_whitespace()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    for idx in 0..tokens.len() {
        if let Some((lhs, _rhs)) = tokens[idx].split_once('=') {
            if contains_secret_signal(&normalize_tag(lhs)) {
                tokens[idx] = format!("{lhs}=[REDACTED]");
                continue;
            }
        }

        if ends_with_secret_label(&tokens[idx]) {
            if let Some(next) = tokens.get_mut(idx + 1) {
                *next = "[REDACTED]".to_string();
            }
            continue;
        }

        let current = normalize_tag(&tokens[idx]);
        if contains_secret_signal(&current)
            && matches!(
                tokens.get(idx + 1).map(|value| normalize_tag(value)),
                Some(ref joiner)
                    if matches!(joiner.as_str(), "is" | "was" | "equals" | "equal" | "set" | "to")
            )
        {
            if let Some(next) = tokens.get_mut(idx + 2) {
                *next = "[REDACTED]".to_string();
            }
        }
    }

    tokens.join(" ")
}

fn contains_secret_signal(normalized: &str) -> bool {
    normalized.contains("password")
        || normalized.contains("passphrase")
        || normalized.contains("api-key")
        || normalized.contains("api key")
        || normalized.contains("access-key")
        || normalized.contains("access key")
        || normalized.contains("secret")
        || normalized.contains("token")
        || normalized.contains("credential")
        || normalized.contains("private-key")
        || normalized.contains("private key")
        || normalized.contains("ssh-key")
        || normalized.contains("ssh key")
        || normalized.contains("client-secret")
        || normalized.contains("client secret")
        || normalized.contains("bearer-token")
        || normalized.contains("bearer token")
}

fn token_contains_secret_assignment(token: &str) -> bool {
    let Some((lhs, rhs)) = token.split_once('=') else {
        return false;
    };
    contains_secret_signal(&normalize_tag(lhs))
        && !rhs.is_empty()
        && rhs != "[REDACTED]"
        && rhs != "<redacted>"
}

fn ends_with_secret_label(token: &str) -> bool {
    let normalized = normalize_tag(token);
    let trimmed = normalized.trim_end_matches([':', '=']);
    contains_secret_signal(trimmed) && (token.ends_with(':') || token.ends_with('='))
}

fn normalize_query(value: &str) -> String {
    value.trim().to_lowercase()
}

fn normalize_tag(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .replace('_', "-")
        .trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ';')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_detail_queries_are_detected() {
        assert!(query_requests_sensitive_secret_detail(
            "What is the exact FEMIND_REMOTE_EMBED_TOKEN value?"
        ));
        assert!(!query_requests_sensitive_secret_detail(
            "Where is the FeMind remote embed token loaded from?"
        ));
    }

    #[test]
    fn secret_assignment_tokens_are_redacted() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "content_secret_class".to_string(),
            "credential-material".to_string(),
        );

        let redacted = redact_secret_material(
            "The file contains FEMIND_REMOTE_EMBED_TOKEN=sk-prod-123 in ~/.config/recallbench/femind-remote.env.",
            &metadata,
        );

        assert!(redacted.contains("FEMIND_REMOTE_EMBED_TOKEN=[REDACTED]"));
        assert!(redacted.contains("~/.config/recallbench/femind-remote.env"));
        assert!(!redacted.contains("sk-prod-123"));
    }

    #[test]
    fn location_queries_are_detected() {
        assert!(query_requests_secret_location_or_reference(
            "Which file stores the remote embed token?"
        ));
        assert!(query_requests_secret_location_or_reference(
            "How should I load the token safely?"
        ));
    }
}
