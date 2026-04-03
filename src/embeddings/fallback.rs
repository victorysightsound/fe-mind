use std::sync::Arc;
use std::time::Duration;

use crate::backend_policy::{BackendFailureClass, BackendMode, BackendPolicy};
use crate::embeddings::EmbeddingBackend;
use crate::error::{FemindError, Result};

/// Wraps embedding backends with graceful degradation.
///
/// Supports three modes:
/// - Single backend (API or local)
/// - API-first with local fallback (tries API, falls back to local on error)
/// - None (FTS5-only mode)
pub struct FallbackBackend {
    primary: Option<Box<dyn EmbeddingBackend>>,
    fallback: Option<Box<dyn EmbeddingBackend>>,
    dims: usize,
    policy: Arc<BackendPolicy>,
}

impl FallbackBackend {
    /// Create wrapping a single backend.
    pub fn new(backend: Box<dyn EmbeddingBackend>) -> Self {
        let dims = backend.dimensions();
        Self {
            primary: Some(backend),
            fallback: None,
            dims,
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        }
    }

    /// Create without a backend (FTS5-only mode).
    pub fn none(dims: usize) -> Self {
        Self {
            primary: None,
            fallback: None,
            dims,
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        }
    }

    /// Create with API as primary and local as fallback.
    ///
    /// Tries the primary (API) backend first. If it fails (network error,
    /// rate limit, etc.), falls back to the local backend transparently.
    pub fn api_with_local_fallback(
        api: Box<dyn EmbeddingBackend>,
        local: Box<dyn EmbeddingBackend>,
    ) -> Self {
        let dims = api.dimensions();
        Self {
            primary: Some(api),
            fallback: Some(local),
            dims,
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        }
    }

    /// Whether any embedding backend is available.
    pub fn has_backend(&self) -> bool {
        self.primary.as_ref().is_some_and(|b| b.is_available())
            || self.fallback.as_ref().is_some_and(|b| b.is_available())
    }

    pub fn backend_mode(&self) -> BackendMode {
        self.policy.mode()
    }

    pub fn last_failure_message(&self) -> Option<String> {
        self.policy.last_failure_message()
    }

    /// Try primary, fall back to secondary on error.
    fn try_with_fallback<F, T>(&self, op: F) -> Result<T>
    where
        F: Fn(&dyn EmbeddingBackend) -> Result<T>,
    {
        if let Some(ref primary) = self.primary {
            if self.policy.mode() == BackendMode::Offline {
                return Err(FemindError::ModelNotAvailable(
                    self.policy
                        .last_failure_message()
                        .unwrap_or_else(|| "remote embedding backend is offline".to_string()),
                ));
            }

            if primary.is_available() && self.policy.should_attempt_primary() {
                self.policy.begin_recovery_attempt();
                match op(primary.as_ref()) {
                    Ok(result) => {
                        self.policy.record_success();
                        return Ok(result);
                    }
                    Err(error) => {
                        let class = BackendPolicy::classify_error(&error);
                        self.policy.record_failure(class, error.to_string());
                        match class {
                            BackendFailureClass::Permanent => return Err(error),
                            BackendFailureClass::Transient => {
                                if self.fallback.is_some() {
                                    tracing::warn!(
                                        "Primary embedding failed, falling back to local: {error}"
                                    );
                                } else {
                                    return Err(error);
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(ref fallback) = self.fallback {
            if fallback.is_available() {
                return op(fallback.as_ref());
            }
        }

        Err(FemindError::ModelNotAvailable(
            "no embedding backend available".into(),
        ))
    }
}

impl EmbeddingBackend for FallbackBackend {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.try_with_fallback(|b| b.embed(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.try_with_fallback(|b| b.embed_batch(texts))
    }

    fn dimensions(&self) -> usize {
        self.primary
            .as_ref()
            .map(|b| b.dimensions())
            .or_else(|| self.fallback.as_ref().map(|b| b.dimensions()))
            .unwrap_or(self.dims)
    }

    fn is_available(&self) -> bool {
        self.has_backend()
    }

    fn model_name(&self) -> &str {
        self.primary
            .as_ref()
            .map(|b| b.model_name())
            .or_else(|| self.fallback.as_ref().map(|b| b.model_name()))
            .unwrap_or("none")
    }

    fn embedding_profile(&self) -> String {
        self.primary
            .as_ref()
            .map(|b| b.embedding_profile())
            .or_else(|| self.fallback.as_ref().map(|b| b.embedding_profile()))
            .unwrap_or_else(|| "none".to_string())
    }

    fn compatibility_model_names(&self) -> Vec<String> {
        self.primary
            .as_ref()
            .map(|b| b.compatibility_model_names())
            .or_else(|| {
                self.fallback
                    .as_ref()
                    .map(|b| b.compatibility_model_names())
            })
            .unwrap_or_else(|| vec!["none".to_string()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::NoopBackend;

    struct MismatchBackend;

    impl EmbeddingBackend for MismatchBackend {
        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Err(FemindError::RemoteProfileMismatch(
                "profile mismatch".into(),
            ))
        }

        fn dimensions(&self) -> usize {
            384
        }

        fn is_available(&self) -> bool {
            true
        }

        fn model_name(&self) -> &str {
            "remote"
        }
    }

    #[test]
    fn with_backend() {
        let backend = FallbackBackend::new(Box::new(NoopBackend::new(384)));
        assert!(backend.is_available());
        assert!(backend.has_backend());
        assert_eq!(backend.dimensions(), 384);

        let vec = backend.embed("test").expect("embed");
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn without_backend() {
        let backend = FallbackBackend::none(384);
        assert!(!backend.is_available());
        assert!(!backend.has_backend());

        let result = backend.embed("test");
        assert!(result.is_err());
    }

    #[test]
    fn model_name_with_backend() {
        let backend = FallbackBackend::new(Box::new(NoopBackend::new(384)));
        assert_eq!(backend.model_name(), "noop");
    }

    #[test]
    fn model_name_without_backend() {
        let backend = FallbackBackend::none(384);
        assert_eq!(backend.model_name(), "none");
    }

    #[test]
    fn api_with_local_fallback_uses_primary() {
        let primary = Box::new(NoopBackend::new(384));
        let local = Box::new(NoopBackend::new(384));
        let backend = FallbackBackend::api_with_local_fallback(primary, local);

        assert!(backend.is_available());
        let vec = backend.embed("test").expect("embed");
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn fallback_when_primary_unavailable() {
        // Primary is None, fallback is available
        let backend = FallbackBackend {
            primary: None,
            fallback: Some(Box::new(NoopBackend::new(384))),
            dims: 384,
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        };

        assert!(backend.is_available());
        let vec = backend.embed("test").expect("should use fallback");
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn permanent_primary_failure_is_not_masked() {
        let backend = FallbackBackend {
            primary: Some(Box::new(MismatchBackend)),
            fallback: Some(Box::new(NoopBackend::new(384))),
            dims: 384,
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        };

        let result = backend.embed("test");
        assert!(matches!(result, Err(FemindError::RemoteProfileMismatch(_))));
    }
}
