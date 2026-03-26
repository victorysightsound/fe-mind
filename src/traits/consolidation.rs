use crate::traits::MemoryMeta;
use crate::traits::ScoredResult;

/// Determines what happens when a new memory is stored.
///
/// Prevents duplicates, updates existing, or merges memories.
/// The engine calls `consolidate()` before inserting, passing the new memory's
/// metadata and any similar existing memories found by search.
pub trait ConsolidationStrategy: Send + Sync {
    /// Given a new memory and existing similar memories, decide what to do.
    ///
    /// Returns a list of actions to perform. An empty list means "proceed with ADD".
    fn consolidate(&self, new: &MemoryMeta, existing: &[ScoredResult]) -> Vec<ConsolidationAction>;
}

/// Actions the consolidation pipeline can take.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsolidationAction {
    /// Store as a new memory.
    Add,
    /// Update an existing memory (replace content).
    Update {
        /// ID of the memory to update.
        target_id: i64,
    },
    /// Delete an existing memory (superseded or contradicted).
    Delete {
        /// ID of the memory to delete.
        target_id: i64,
    },
    /// Do nothing (duplicate or irrelevant).
    Noop,
    /// Link new memory to existing via relationship.
    #[cfg(feature = "graph-memory")]
    Relate {
        /// ID of the memory to link to.
        target_id: i64,
        /// Relationship type.
        relation: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysAdd;
    impl ConsolidationStrategy for AlwaysAdd {
        fn consolidate(
            &self,
            _new: &MemoryMeta,
            _existing: &[ScoredResult],
        ) -> Vec<ConsolidationAction> {
            vec![ConsolidationAction::Add]
        }
    }

    struct AlwaysNoop;
    impl ConsolidationStrategy for AlwaysNoop {
        fn consolidate(
            &self,
            _new: &MemoryMeta,
            _existing: &[ScoredResult],
        ) -> Vec<ConsolidationAction> {
            vec![ConsolidationAction::Noop]
        }
    }

    #[test]
    fn trait_object_works() {
        let strategy: Box<dyn ConsolidationStrategy> = Box::new(AlwaysAdd);
        let meta = MemoryMeta {
            id: None,
            searchable_text: "test".into(),
            memory_type: crate::traits::MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
        };
        let actions = strategy.consolidate(&meta, &[]);
        assert_eq!(actions, vec![ConsolidationAction::Add]);
    }

    #[test]
    fn noop_action() {
        let strategy: Box<dyn ConsolidationStrategy> = Box::new(AlwaysNoop);
        let meta = MemoryMeta {
            id: None,
            searchable_text: "test".into(),
            memory_type: crate::traits::MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
        };
        let actions = strategy.consolidate(&meta, &[]);
        assert_eq!(actions, vec![ConsolidationAction::Noop]);
    }
}
