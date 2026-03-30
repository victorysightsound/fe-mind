#[cfg(feature = "remote-reranking")]
mod inner {
    use std::time::Duration;

    use crate::error::{FemindError, Result};
    use crate::traits::{RerankCandidate, RerankerBackend, ScoredResult};

    #[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
    pub struct RemoteRerankerStatus {
        pub model: String,
        pub reranker_profile: String,
        #[serde(default)]
        pub execution_mode: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RemoteRerankerVerificationReport {
        pub expected_model: String,
        pub expected_reranker_profile: String,
        pub remote_model: String,
        pub remote_reranker_profile: String,
        pub remote_execution_mode: Option<String>,
    }

    pub struct RemoteRerankerBackend {
        agent: ureq::Agent,
        base_url: String,
        auth_token: Option<String>,
        model: String,
        expected_profile: String,
        fallback: Option<Box<dyn RerankerBackend>>,
    }

    impl RemoteRerankerBackend {
        pub fn new(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            expected_profile: impl Into<String>,
        ) -> Result<Self> {
            Self::new_with_timeout(base_url, auth_token, model, expected_profile, None)
        }

        pub fn new_with_timeout(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            expected_profile: impl Into<String>,
            timeout: Option<Duration>,
        ) -> Result<Self> {
            Self::build(
                base_url.into(),
                auth_token,
                model.into(),
                expected_profile.into(),
                timeout,
                None,
                true,
            )
        }

        pub fn with_local_fallback(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            expected_profile: impl Into<String>,
            fallback: Box<dyn RerankerBackend>,
        ) -> Result<Self> {
            Self::with_local_fallback_and_timeout(
                base_url,
                auth_token,
                model,
                expected_profile,
                fallback,
                None,
            )
        }

        pub fn with_local_fallback_and_timeout(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            expected_profile: impl Into<String>,
            fallback: Box<dyn RerankerBackend>,
            timeout: Option<Duration>,
        ) -> Result<Self> {
            Self::build(
                base_url.into(),
                auth_token,
                model.into(),
                expected_profile.into(),
                timeout,
                Some(fallback),
                true,
            )
        }

        pub fn minilm(base_url: impl Into<String>, auth_token: Option<String>) -> Result<Self> {
            Self::new(
                base_url,
                auth_token,
                crate::reranking::RERANKER_CANONICAL_NAME,
                crate::reranking::RERANKER_PROFILE,
            )
        }

        pub fn minilm_with_local_fallback(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            fallback: Box<dyn RerankerBackend>,
        ) -> Result<Self> {
            Self::with_local_fallback(
                base_url,
                auth_token,
                crate::reranking::RERANKER_CANONICAL_NAME,
                crate::reranking::RERANKER_PROFILE,
                fallback,
            )
        }

        fn build(
            base_url: String,
            auth_token: Option<String>,
            model: String,
            expected_profile: String,
            timeout: Option<Duration>,
            fallback: Option<Box<dyn RerankerBackend>>,
            verify_remote: bool,
        ) -> Result<Self> {
            let backend = Self {
                agent: build_agent(timeout),
                base_url: normalize_base_url(&base_url),
                auth_token,
                model,
                expected_profile,
                fallback,
            };

            if verify_remote {
                match backend.verify_remote() {
                    Ok(_) => {}
                    Err(err) => {
                        if backend.fallback.is_some() {
                            tracing::warn!(
                                "remote reranker verification failed, starting with local fallback available: {err}"
                            );
                        } else {
                            return Err(err);
                        }
                    }
                }
            }

            Ok(backend)
        }

        pub fn verify_remote(&self) -> Result<RemoteRerankerVerificationReport> {
            let status = self.fetch_status()?;
            let accepted_models = crate::reranking::compatibility_reranker_names(&self.model);
            if !accepted_models.iter().any(|candidate| candidate == &status.model) {
                return Err(FemindError::Embedding(format!(
                    "remote reranker model mismatch: expected one of {:?} but got '{}'",
                    accepted_models, status.model
                )));
            }
            if status.reranker_profile != self.expected_profile {
                return Err(FemindError::Embedding(format!(
                    "remote reranker profile mismatch: expected '{}' but got '{}'",
                    self.expected_profile, status.reranker_profile
                )));
            }

            Ok(RemoteRerankerVerificationReport {
                expected_model: self.model.clone(),
                expected_reranker_profile: self.expected_profile.clone(),
                remote_model: status.model,
                remote_reranker_profile: status.reranker_profile,
                remote_execution_mode: status.execution_mode,
            })
        }

        pub fn status(&self) -> Result<RemoteRerankerStatus> {
            self.fetch_status()
        }

        fn fetch_status(&self) -> Result<RemoteRerankerStatus> {
            let url = format!("{}/status", self.base_url);
            let mut request = self.agent.get(&url);
            if let Some(token) = self.auth_token.as_deref() {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
            let response = request
                .call()
                .map_err(|e| FemindError::Embedding(format!("remote reranker status request failed: {e}")))?;

            response
                .into_json::<RemoteRerankerStatus>()
                .map_err(|e| FemindError::Embedding(format!("remote reranker status parse failed: {e}")))
        }

        fn call_rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            let url = format!("{}/rerank", self.base_url);
            let body = serde_json::json!({
                "model": self.model,
                "query": query,
                "documents": documents,
                "expected_reranker_profile": self.expected_profile,
            });

            let mut request = self.agent.post(&url).set("Content-Type", "application/json");
            if let Some(token) = self.auth_token.as_deref() {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
            let response = request.send_json(&body).map_err(|e| {
                FemindError::Embedding(format!("remote reranker request failed: {e}"))
            })?;

            let mut parsed = response.into_json::<RemoteRerankResponse>().map_err(|e| {
                FemindError::Embedding(format!("remote reranker parse failed: {e}"))
            })?;
            parsed.results.sort_by_key(|item| item.index);
            Ok(parsed
                .results
                .into_iter()
                .map(|item| item.relevance_score)
                .collect())
        }
    }

    #[derive(serde::Deserialize)]
    struct RemoteRerankResponse {
        results: Vec<RemoteRerankItem>,
    }

    #[derive(serde::Deserialize)]
    struct RemoteRerankItem {
        index: usize,
        relevance_score: f32,
    }

    impl RerankerBackend for RemoteRerankerBackend {
        fn rerank(&self, query: &str, candidates: Vec<RerankCandidate>) -> Result<Vec<ScoredResult>> {
            let documents = candidates
                .iter()
                .map(|candidate| candidate.text.as_str())
                .collect::<Vec<_>>();
            let scores = match self.call_rerank(query, &documents) {
                Ok(scores) => scores,
                Err(primary_error) => {
                    if let Some(fallback) = self.fallback.as_ref() {
                        tracing::warn!(
                            "remote reranker failed, falling back to local backend: {primary_error}"
                        );
                        return fallback.rerank(query, candidates);
                    }
                    return Err(primary_error);
                }
            };

            let mut reranked = candidates
                .into_iter()
                .zip(scores)
                .map(|(candidate, score)| ScoredResult {
                    memory_id: candidate.memory_id,
                    score,
                    raw_score: score,
                    score_multiplier: 1.0,
                })
                .collect::<Vec<_>>();

            reranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(reranked)
        }
    }

    fn normalize_base_url(base_url: &str) -> String {
        base_url.trim_end_matches('/').to_string()
    }

    fn build_agent(timeout: Option<Duration>) -> ureq::Agent {
        let mut builder = ureq::AgentBuilder::new();
        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }
        builder.build()
    }
}

#[cfg(feature = "remote-reranking")]
pub use inner::{RemoteRerankerBackend, RemoteRerankerStatus, RemoteRerankerVerificationReport};
