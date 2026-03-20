use crate::traits::{ConsolidationAction, ConsolidationStrategy, MemoryMeta, ScoredResult};

/// Zero-cost deduplication via SHA-256 content hash.
///
/// If a memory with the same searchable text already exists (exact hash match),
/// returns `Noop`. Otherwise returns `Add`.
///
/// This is the default consolidation strategy — it prevents exact duplicates
/// with zero computational overhead beyond hashing.
pub struct HashDedup;

impl ConsolidationStrategy for HashDedup {
    fn consolidate(
        &self,
        _new: &MemoryMeta,
        existing: &[ScoredResult],
    ) -> Vec<ConsolidationAction> {
        // If we got existing results, the store layer already found a hash match
        // and passed them here. Any non-empty existing set means duplicate.
        if existing.is_empty() {
            vec![ConsolidationAction::Add]
        } else {
            vec![ConsolidationAction::Noop]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;

    fn meta() -> MemoryMeta {
        MemoryMeta {
            id: None,
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn no_existing_adds() {
        let dedup = HashDedup;
        let actions = dedup.consolidate(&meta(), &[]);
        assert_eq!(actions, vec![ConsolidationAction::Add]);
    }

    #[test]
    fn existing_match_noops() {
        let dedup = HashDedup;
        let existing = vec![ScoredResult {
            memory_id: 42,
            score: 1.0,
            raw_score: 1.0,
            score_multiplier: 1.0,
        }];
        let actions = dedup.consolidate(&meta(), &existing);
        assert_eq!(actions, vec![ConsolidationAction::Noop]);
    }
}
