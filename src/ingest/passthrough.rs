use crate::error::Result;
use crate::ingest::{ExtractedFact, IngestStrategy};
use crate::traits::MemoryType;

/// Default ingest strategy: store text as-is with no extraction.
///
/// Zero cost — just wraps the input as a single ExtractedFact.
pub struct PassthroughIngest;

impl IngestStrategy for PassthroughIngest {
    fn extract(&self, raw: &str) -> Result<Vec<ExtractedFact>> {
        Ok(vec![ExtractedFact {
            text: raw.to_string(),
            category: None,
            memory_type: MemoryType::Episodic,
            importance: 5,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_returns_input() {
        let ingest = PassthroughIngest;
        let facts = ingest.extract("hello world").expect("extract");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].text, "hello world");
        assert_eq!(facts[0].memory_type, MemoryType::Episodic);
    }

    #[test]
    fn trait_object_works() {
        let ingest: Box<dyn IngestStrategy> = Box::new(PassthroughIngest);
        let facts = ingest.extract("test").expect("extract");
        assert_eq!(facts.len(), 1);
    }
}
