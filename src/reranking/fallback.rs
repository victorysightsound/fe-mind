use std::sync::Arc;
use std::time::Duration;

use crate::backend_policy::{BackendFailureClass, BackendMode, BackendPolicy};
use crate::error::{FemindError, Result};
use crate::traits::{RerankCandidate, RerankerBackend, ScoredResult};

/// Wraps reranker backends with graceful fallback.
pub struct FallbackRerankerBackend {
    primary: Option<Box<dyn RerankerBackend>>,
    fallback: Option<Box<dyn RerankerBackend>>,
    policy: Arc<BackendPolicy>,
}

impl FallbackRerankerBackend {
    pub fn new(backend: Box<dyn RerankerBackend>) -> Self {
        Self {
            primary: Some(backend),
            fallback: None,
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        }
    }

    pub fn none() -> Self {
        Self {
            primary: None,
            fallback: None,
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        }
    }

    pub fn remote_with_local_fallback(
        remote: Box<dyn RerankerBackend>,
        local: Box<dyn RerankerBackend>,
    ) -> Self {
        Self {
            primary: Some(remote),
            fallback: Some(local),
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        }
    }

    pub fn backend_mode(&self) -> BackendMode {
        self.policy.mode()
    }

    pub fn last_failure_message(&self) -> Option<String> {
        self.policy.last_failure_message()
    }

    fn try_with_fallback(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> Result<Vec<ScoredResult>> {
        if let Some(primary) = self.primary.as_ref() {
            if self.policy.mode() == BackendMode::Offline {
                return Err(FemindError::ModelNotAvailable(
                    self.policy
                        .last_failure_message()
                        .unwrap_or_else(|| "remote reranker backend is offline".to_string()),
                ));
            }

            if self.policy.should_attempt_primary() {
                self.policy.begin_recovery_attempt();
                match primary.rerank(query, candidates.clone()) {
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
                                        "primary reranker failed, falling back to local: {error}"
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

        if let Some(fallback) = self.fallback.as_ref() {
            return fallback.rerank(query, candidates);
        }

        Err(FemindError::ModelNotAvailable(
            "no reranker backend available".into(),
        ))
    }
}

impl RerankerBackend for FallbackRerankerBackend {
    fn rerank(&self, query: &str, candidates: Vec<RerankCandidate>) -> Result<Vec<ScoredResult>> {
        self.try_with_fallback(query, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IdentityReranker;

    struct MismatchReranker;

    impl RerankerBackend for IdentityReranker {
        fn rerank(
            &self,
            _query: &str,
            candidates: Vec<RerankCandidate>,
        ) -> Result<Vec<ScoredResult>> {
            Ok(candidates
                .into_iter()
                .map(|candidate| ScoredResult {
                    memory_id: candidate.memory_id,
                    score: candidate.score,
                    raw_score: candidate.raw_score,
                    score_multiplier: candidate.score_multiplier,
                })
                .collect())
        }
    }

    impl RerankerBackend for MismatchReranker {
        fn rerank(
            &self,
            _query: &str,
            _candidates: Vec<RerankCandidate>,
        ) -> Result<Vec<ScoredResult>> {
            Err(FemindError::RemoteProfileMismatch(
                "profile mismatch".into(),
            ))
        }
    }

    #[test]
    fn local_only_backend_works() {
        let backend = FallbackRerankerBackend::new(Box::new(IdentityReranker));
        let results = backend
            .rerank(
                "query",
                vec![RerankCandidate {
                    memory_id: 1,
                    text: "hello".into(),
                    score: 0.5,
                    raw_score: 0.5,
                    score_multiplier: 1.0,
                }],
            )
            .expect("rerank");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn permanent_primary_failure_is_not_masked() {
        let backend = FallbackRerankerBackend {
            primary: Some(Box::new(MismatchReranker)),
            fallback: Some(Box::new(IdentityReranker)),
            policy: Arc::new(BackendPolicy::new(Duration::from_secs(30))),
        };

        let result = backend.rerank(
            "query",
            vec![RerankCandidate {
                memory_id: 1,
                text: "hello".into(),
                score: 0.5,
                raw_score: 0.5,
                score_multiplier: 1.0,
            }],
        );

        assert!(matches!(result, Err(FemindError::RemoteProfileMismatch(_))));
    }
}
