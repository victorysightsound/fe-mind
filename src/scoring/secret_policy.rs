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
    /// Raw bearer or session token material.
    TokenMaterial,
    /// Raw private-key or signing-key material.
    KeyMaterial,
    /// A safe storage location for secret material (env file, vault path, item name).
    CredentialLocation,
    /// A safe reference to a secret-handling mechanism (for example `op read ...`).
    SecretReference,
    /// A private service endpoint that should not be surfaced verbatim.
    PrivateEndpoint,
    /// An internal-only hostname that should not be surfaced verbatim.
    InternalHostname,
}

impl SecretClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CredentialMaterial => "credential-material",
            Self::TokenMaterial => "token-material",
            Self::KeyMaterial => "key-material",
            Self::CredentialLocation => "credential-location",
            Self::SecretReference => "secret-reference",
            Self::PrivateEndpoint => "private-endpoint",
            Self::InternalHostname => "internal-hostname",
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
            "session-token" | "bearer-token" | "access-token" | "token-value" => {
                Some(Self::TokenMaterial)
            }
            "signing-key" | "ssh-key-material" | "key-value" => Some(Self::KeyMaterial),
            "credential-location"
            | "credential-storage"
            | "secret-location"
            | "secret-storage"
            | "vault-location" => Some(Self::CredentialLocation),
            "secret-reference" | "credential-reference" | "secret-guidance" => {
                Some(Self::SecretReference)
            }
            "private-endpoint" | "internal-endpoint" | "service-endpoint" => {
                Some(Self::PrivateEndpoint)
            }
            "internal-hostname" | "private-hostname" | "internal-host" => {
                Some(Self::InternalHostname)
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

    let sensitive_signal =
        contains_secret_signal(&normalized) || query_requests_private_infra_detail(query);
    if !sensitive_signal {
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

    reveal_signal || (!location_signal && contains_secret_signal(&normalized))
}

pub fn evidence_contains_secret_material(text: &str, metadata: &HashMap<String, String>) -> bool {
    matches!(
        secret_class_from_metadata(metadata),
        Some(
            SecretClass::CredentialMaterial
                | SecretClass::TokenMaterial
                | SecretClass::KeyMaterial
                | SecretClass::PrivateEndpoint
                | SecretClass::InternalHostname
        )
    ) || text
        .split_whitespace()
        .any(token_contains_secret_assignment)
}

pub fn redact_secret_material(text: &str, metadata: &HashMap<String, String>) -> String {
    let secret_class = secret_class_from_metadata(metadata);
    if !evidence_contains_secret_material(text, metadata)
        && !matches!(
            secret_class,
            Some(SecretClass::PrivateEndpoint | SecretClass::InternalHostname)
        )
    {
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

    let mut redacted = tokens.join(" ");
    if matches!(secret_class, Some(SecretClass::PrivateEndpoint)) {
        redacted = redact_private_endpoint_material(&redacted);
    }
    if matches!(secret_class, Some(SecretClass::InternalHostname)) {
        redacted = redact_internal_hostname_material(&redacted);
    }

    redacted
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

pub fn query_requests_private_infra_detail(query: &str) -> bool {
    let normalized = normalize_query(query);
    let exact_signal = normalized.contains("exact")
        || normalized.contains("actual")
        || normalized.contains("literal")
        || normalized.contains("full")
        || normalized.contains("complete");
    exact_signal && query_requests_private_infra_guidance(query)
}

pub fn query_requests_private_infra_guidance(query: &str) -> bool {
    let normalized = normalize_query(query);
    normalized.contains("private endpoint")
        || normalized.contains("internal endpoint")
        || normalized.contains("internal hostname")
        || normalized.contains("private hostname")
        || normalized.contains("internal host")
        || normalized.contains("private host")
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

fn redact_private_endpoint_material(text: &str) -> String {
    text.split_whitespace()
        .map(|token| {
            if token.contains("://") {
                "[REDACTED_ENDPOINT]".to_string()
            } else if looks_like_ipv4(token) {
                "[REDACTED_ENDPOINT]".to_string()
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn redact_internal_hostname_material(text: &str) -> String {
    text.split_whitespace()
        .map(|token| {
            let trimmed = token.trim_matches(|c: char| c == ',' || c == ';' || c == '.');
            if looks_like_internal_hostname(trimmed) {
                token.replace(trimmed, "[REDACTED_HOSTNAME]")
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn looks_like_ipv4(token: &str) -> bool {
    let token = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != ':');
    let host = token.split(':').next().unwrap_or(token);
    let parts = host.split('.').collect::<Vec<_>>();
    parts.len() == 4 && parts.iter().all(|part| part.parse::<u8>().is_ok())
}

fn looks_like_internal_hostname(token: &str) -> bool {
    let token = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '-');
    token.contains('.')
        && (token.contains(".internal")
            || token.contains(".local")
            || token.contains(".lan")
            || token.ends_with(".corp"))
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

    #[test]
    fn exact_private_infra_queries_are_sensitive() {
        assert!(query_requests_sensitive_secret_detail(
            "What is the exact private endpoint for the embed relay?"
        ));
        assert!(query_requests_private_infra_detail(
            "What is the exact internal hostname for the service?"
        ));
    }

    #[test]
    fn private_endpoint_and_internal_hostname_are_redacted() {
        let mut endpoint = HashMap::new();
        endpoint.insert(
            "content_secret_class".to_string(),
            "private-endpoint".to_string(),
        );
        let mut hostname = HashMap::new();
        hostname.insert(
            "content_secret_class".to_string(),
            "internal-hostname".to_string(),
        );

        assert!(
            redact_secret_material(
                "Use https://relay.calvaryav.internal:8899/embed behind the tunnel.",
                &endpoint,
            )
            .contains("[REDACTED_ENDPOINT]")
        );
        assert!(
            redact_secret_material(
                "The host is relay.calvaryav.internal for the audited path.",
                &hostname,
            )
            .contains("[REDACTED_HOSTNAME]")
        );
    }
}
