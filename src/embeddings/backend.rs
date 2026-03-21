use crate::error::Result;

/// Trait for generating vector embeddings from text.
///
/// Requires Rust 1.75+ (native async fn in traits).
///
/// Shipped implementations:
/// - `NoopBackend`: returns zero vectors (testing)
/// - `FallbackBackend`: wraps an optional backend, degrades to FTS5-only
/// - `CandleNativeBackend`: granite-small-r2 via ModernBERT (feature: `local-embeddings`)
pub trait EmbeddingBackend: Send + Sync {
    /// Generate embedding for a single text.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for a batch of texts.
    ///
    /// Default: sequential. Implementations can optimize for batch throughput.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text)?);
        }
        Ok(results)
    }

    /// Generate embedding for a search query.
    ///
    /// Many embedding models use instruction prefixes to align query and document
    /// embeddings (e.g., "Represent this sentence for searching relevant passages: ").
    /// Override this method to add the appropriate prefix for your model.
    ///
    /// Default: delegates to `embed()` (no prefix).
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed(query)
    }

    /// Number of dimensions in output vectors.
    fn dimensions(&self) -> usize;

    /// Whether the backend is ready to serve requests.
    fn is_available(&self) -> bool;

    /// Model identifier for tracking which model produced a vector.
    ///
    /// Used to filter stored vectors: only vectors from the same model
    /// are used in similarity search (Decision 020).
    fn model_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::NoopBackend;

    #[test]
    fn trait_object_works() {
        let backend: Box<dyn EmbeddingBackend> = Box::new(NoopBackend::new(384));
        assert_eq!(backend.dimensions(), 384);
        assert!(backend.is_available());
        assert_eq!(backend.model_name(), "noop");
    }
}
