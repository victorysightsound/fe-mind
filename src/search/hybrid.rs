use std::collections::HashMap;

use crate::search::fts5::FtsResult;

/// Reciprocal Rank Fusion merge of keyword and vector search results.
///
/// Combines two ranked lists using: `score(d) = Σ 1/(k + rank + 1)`
///
/// Dynamic k-values adjust weighting based on query analysis:
/// - Quoted text → favor keyword search
/// - Question words → favor semantic/vector search
/// - Default → equal weight
pub fn rrf_merge(
    keyword_results: &[FtsResult],
    vector_results: &[FtsResult],
    query: &str,
    limit: usize,
) -> Vec<FtsResult> {
    let (keyword_k, vector_k) = analyze_query(query);

    let mut scores: HashMap<i64, f32> = HashMap::new();

    for (rank, result) in keyword_results.iter().enumerate() {
        *scores.entry(result.memory_id).or_default() +=
            1.0 / (keyword_k as f32 + rank as f32 + 1.0);
    }

    for (rank, result) in vector_results.iter().enumerate() {
        *scores.entry(result.memory_id).or_default() +=
            1.0 / (vector_k as f32 + rank as f32 + 1.0);
    }

    let mut merged: Vec<FtsResult> = scores
        .into_iter()
        .map(|(id, score)| FtsResult {
            memory_id: id,
            score,
        })
        .collect();

    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    merged.truncate(limit);
    merged
}

/// Analyze query to determine RRF k-values.
///
/// Returns (keyword_k, vector_k). Lower k means that source is favored.
fn analyze_query(query: &str) -> (u32, u32) {
    if query.contains('"') {
        (40, 60) // Quoted text → favor keyword
    } else {
        let lower = query.to_lowercase();
        let question_words = ["what", "how", "why", "when", "where", "explain", "describe"];
        if lower
            .split_whitespace()
            .any(|w| question_words.contains(&w))
        {
            (100, 20) // Question → heavily favor semantic
        } else {
            (60, 60) // Default → equal weight
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(id: i64, score: f32) -> FtsResult {
        FtsResult {
            memory_id: id,
            score,
        }
    }

    #[test]
    fn basic_merge() {
        let kw = vec![result(1, 1.0), result(2, 0.8), result(3, 0.5)];
        let vec = vec![result(2, 0.9), result(4, 0.7), result(1, 0.3)];

        let merged = rrf_merge(&kw, &vec, "test query", 10);

        // Memory 2 appears in both lists → should have highest RRF score
        assert_eq!(merged[0].memory_id, 2, "memory in both lists should rank first");
        // Memory 1 also in both
        assert_eq!(merged[1].memory_id, 1);
        assert!(merged[0].score > merged[1].score);
    }

    #[test]
    fn limit_applied() {
        let kw = vec![result(1, 1.0), result(2, 0.8)];
        let vec = vec![result(3, 0.9), result(4, 0.7)];

        let merged = rrf_merge(&kw, &vec, "q", 2);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn quoted_query_favors_keyword() {
        let (kw_k, vec_k) = analyze_query("\"exact phrase\" search");
        assert!(kw_k < vec_k, "quoted query should favor keyword (lower k)");
    }

    #[test]
    fn question_favors_vector() {
        let (kw_k, vec_k) = analyze_query("how does authentication work");
        assert!(vec_k < kw_k, "question should favor vector (lower k)");
    }

    #[test]
    fn default_equal_weight() {
        let (kw_k, vec_k) = analyze_query("authentication error JWT");
        assert_eq!(kw_k, vec_k, "default should be equal weight");
    }

    #[test]
    fn empty_inputs() {
        let merged = rrf_merge(&[], &[], "q", 10);
        assert!(merged.is_empty());
    }

    #[test]
    fn one_source_empty() {
        let kw = vec![result(1, 1.0), result(2, 0.5)];
        let merged = rrf_merge(&kw, &[], "q", 10);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].memory_id, 1);
    }
}
