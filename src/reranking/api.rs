#[cfg(feature = "api-reranking")]
mod inner {
    use crate::error::{FemindError, Result};
    use crate::traits::{RerankCandidate, RerankerBackend, ScoredResult};

    pub struct ApiRerankerBackend {
        agent: ureq::Agent,
        base_url: String,
        api_key: String,
        model: String,
    }

    impl ApiRerankerBackend {
        pub fn new(
            base_url: impl Into<String>,
            api_key: impl Into<String>,
            model: impl Into<String>,
        ) -> Self {
            Self {
                agent: ureq::Agent::new(),
                base_url: base_url.into().trim_end_matches('/').to_string(),
                api_key: api_key.into(),
                model: model.into(),
            }
        }

        pub fn call_api(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            let url = format!("{}/rerank", self.base_url);
            let body = serde_json::json!({
                "model": self.model,
                "query": query,
                "documents": documents,
            });

            let response = self
                .agent
                .post(&url)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("Content-Type", "application/json")
                .send_json(&body)
                .map_err(|e| FemindError::Embedding(format!("rerank API request failed: {e}")))?;

            let mut parsed: ApiRerankResponse = response
                .into_json()
                .map_err(|e| FemindError::Embedding(format!("rerank API parse failed: {e}")))?;
            parsed.results.sort_by_key(|item| item.index);
            Ok(parsed
                .results
                .into_iter()
                .map(|item| item.relevance_score)
                .collect())
        }
    }

    #[derive(serde::Deserialize)]
    struct ApiRerankResponse {
        results: Vec<ApiRerankItem>,
    }

    #[derive(serde::Deserialize)]
    struct ApiRerankItem {
        index: usize,
        relevance_score: f32,
    }

    impl RerankerBackend for ApiRerankerBackend {
        fn rerank(&self, query: &str, candidates: Vec<RerankCandidate>) -> Result<Vec<ScoredResult>> {
            let documents = candidates
                .iter()
                .map(|candidate| candidate.text.as_str())
                .collect::<Vec<_>>();
            let scores = self.call_api(query, &documents)?;

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
}

#[cfg(feature = "api-reranking")]
pub use inner::ApiRerankerBackend;
