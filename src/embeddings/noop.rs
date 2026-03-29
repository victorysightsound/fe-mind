use crate::embeddings::EmbeddingBackend;
use crate::error::Result;

/// No-op embedding backend that returns zero vectors.
///
/// Useful for testing and for consumers who only need FTS5 search.
pub struct NoopBackend {
    dims: usize,
}

impl NoopBackend {
    /// Create with the specified number of dimensions.
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl EmbeddingBackend for NoopBackend {
    fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0; self.dims])
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.dims]).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn is_available(&self) -> bool {
        true
    }

    fn model_name(&self) -> &str {
        "noop"
    }

    fn embedding_profile(&self) -> String {
        "noop".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_embed_dimensions() {
        let backend = NoopBackend::new(384);
        let vec = backend.embed("hello").expect("embed");
        assert_eq!(vec.len(), 384);
        assert!(vec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn noop_batch() {
        let backend = NoopBackend::new(128);
        let vecs = backend.embed_batch(&["a", "b", "c"]).expect("batch");
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 128));
    }
}
