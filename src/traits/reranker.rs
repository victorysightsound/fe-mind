use crate::error::Result;
use crate::traits::ScoredResult;

/// Candidate passed to a reranker.
///
/// Carries the raw text alongside the current score so the reranker can
/// rescore without reaching back into storage itself.
#[derive(Debug, Clone)]
pub struct RerankCandidate {
    pub memory_id: i64,
    pub text: String,
    pub score: f32,
    pub raw_score: f32,
    pub score_multiplier: f32,
}

/// Cross-encoder reranking applied after RRF merge, before final scoring.
///
/// Re-scores (query, document) pairs jointly using a cross-encoder model,
/// which captures cross-attention patterns missed by bi-encoder embeddings.
pub trait RerankerBackend: Send + Sync {
    /// Rerank candidates by query-document relevance.
    ///
    /// Returns the same candidates with updated scores.
    fn rerank(&self, query: &str, candidates: Vec<RerankCandidate>) -> Result<Vec<ScoredResult>>;
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
    fn trait_object_works() {
        let reranker: Box<dyn RerankerBackend> = Box::new(IdentityReranker);
        let candidates = vec![RerankCandidate {
            memory_id: 1,
            text: "test".into(),
            score: 0.8,
            raw_score: 0.8,
            score_multiplier: 1.0,
        }];
        let result = reranker.rerank("query", candidates).expect("rerank");
        assert_eq!(result.len(), 1);
    }
}
