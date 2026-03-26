use std::collections::HashMap;

use crate::error::Result;
use crate::memory::RelationType;
use crate::traits::{MemoryMeta, ScoredResult};

/// Post-write hook: when a new memory is stored, optionally update
/// related existing memories (their metadata, keywords, links).
pub trait EvolutionStrategy: Send + Sync {
    /// Given a newly stored memory and its top-k similar existing memories,
    /// return updates to apply to existing memories.
    fn evolve(
        &self,
        new_memory: &MemoryMeta,
        similar: &[ScoredResult],
    ) -> Result<Vec<EvolutionAction>>;
}

/// Actions that memory evolution can take on existing memories.
#[derive(Debug, Clone)]
pub enum EvolutionAction {
    /// Update metadata on an existing memory.
    UpdateMetadata {
        target_id: i64,
        metadata: HashMap<String, String>,
    },
    /// Create a relationship between memories.
    Relate {
        source_id: i64,
        target_id: i64,
        relation: RelationType,
    },
    /// No changes needed.
    Noop,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::MemoryType;
    use chrono::Utc;

    struct NoopEvolution;

    impl EvolutionStrategy for NoopEvolution {
        fn evolve(
            &self,
            _new: &MemoryMeta,
            _similar: &[ScoredResult],
        ) -> Result<Vec<EvolutionAction>> {
            Ok(vec![EvolutionAction::Noop])
        }
    }

    #[test]
    fn trait_object_works() {
        let strategy: Box<dyn EvolutionStrategy> = Box::new(NoopEvolution);
        let meta = MemoryMeta {
            id: Some(1),
            searchable_text: "test".into(),
            memory_type: MemoryType::Semantic,
            importance: 5,
            category: None,
            created_at: Utc::now(),
            metadata: Default::default(),
        };
        let result = strategy.evolve(&meta, &[]).expect("evolve");
        assert!(matches!(result[0], EvolutionAction::Noop));
    }
}
