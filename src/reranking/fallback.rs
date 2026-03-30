use crate::error::{FemindError, Result};
use crate::traits::{RerankCandidate, RerankerBackend, ScoredResult};

/// Wraps reranker backends with graceful fallback.
pub struct FallbackRerankerBackend {
    primary: Option<Box<dyn RerankerBackend>>,
    fallback: Option<Box<dyn RerankerBackend>>,
}

impl FallbackRerankerBackend {
    pub fn new(backend: Box<dyn RerankerBackend>) -> Self {
        Self {
            primary: Some(backend),
            fallback: None,
        }
    }

    pub fn none() -> Self {
        Self {
            primary: None,
            fallback: None,
        }
    }

    pub fn remote_with_local_fallback(
        remote: Box<dyn RerankerBackend>,
        local: Box<dyn RerankerBackend>,
    ) -> Self {
        Self {
            primary: Some(remote),
            fallback: Some(local),
        }
    }

    fn try_with_fallback(
        &self,
        query: &str,
        candidates: Vec<RerankCandidate>,
    ) -> Result<Vec<ScoredResult>> {
        if let Some(primary) = self.primary.as_ref() {
            match primary.rerank(query, candidates.clone()) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if self.fallback.is_some() {
                        tracing::warn!("primary reranker failed, falling back to local: {error}");
                    } else {
                        return Err(error);
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
}
