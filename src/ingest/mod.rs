mod passthrough;

pub use passthrough::PassthroughIngest;

use crate::error::Result;
use crate::traits::MemoryType;

/// Controls how raw input is processed before storage.
pub trait IngestStrategy: Send + Sync {
    /// Extract indexable items from raw input.
    ///
    /// Default: store as-is (PassthroughIngest).
    /// LLM-assisted: extract atomic facts (LlmIngest).
    fn extract(&self, raw: &str) -> Result<Vec<ExtractedFact>>;
}

/// An atomic fact extracted from raw input.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    /// The extracted text.
    pub text: String,
    /// Optional category classification.
    pub category: Option<String>,
    /// Cognitive memory type.
    pub memory_type: MemoryType,
    /// Importance score (1-10).
    pub importance: u8,
}
