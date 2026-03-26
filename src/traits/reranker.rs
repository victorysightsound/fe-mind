use crate::error::Result;
use crate::traits::ScoredResult;

/// Cross-encoder reranking applied after RRF merge, before final scoring.
///
/// Re-scores (query, document) pairs jointly using a cross-encoder model,
/// which captures cross-attention patterns missed by bi-encoder embeddings.
pub trait RerankerBackend: Send + Sync {
    /// Rerank candidates by query-document relevance.
    ///
    /// Returns the same candidates with updated scores.
    fn rerank(&self, query: &str, candidates: Vec<ScoredResult>) -> Result<Vec<ScoredResult>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IdentityReranker;

    impl RerankerBackend for IdentityReranker {
        fn rerank(&self, _query: &str, candidates: Vec<ScoredResult>) -> Result<Vec<ScoredResult>> {
            Ok(candidates)
        }
    }

    #[test]
    fn trait_object_works() {
        let reranker: Box<dyn RerankerBackend> = Box::new(IdentityReranker);
        let candidates = vec![ScoredResult {
            memory_id: 1,
            score: 0.8,
            raw_score: 0.8,
            score_multiplier: 1.0,
        }];
        let result = reranker.rerank("query", candidates).expect("rerank");
        assert_eq!(result.len(), 1);
    }
}
