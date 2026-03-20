use crate::embeddings::EmbeddingBackend;
use crate::error::{MindCoreError, Result};

/// Wraps an optional embedding backend with graceful degradation.
///
/// If the inner backend is `None` or unavailable, operations return
/// errors that the engine interprets as "skip vector search, use FTS5 only."
pub struct FallbackBackend {
    inner: Option<Box<dyn EmbeddingBackend>>,
    dims: usize,
}

impl FallbackBackend {
    /// Create wrapping an existing backend.
    pub fn new(backend: Box<dyn EmbeddingBackend>) -> Self {
        let dims = backend.dimensions();
        Self {
            inner: Some(backend),
            dims,
        }
    }

    /// Create without a backend (FTS5-only mode).
    pub fn none(dims: usize) -> Self {
        Self { inner: None, dims }
    }

    /// Whether an embedding backend is available.
    pub fn has_backend(&self) -> bool {
        self.inner
            .as_ref()
            .map(|b| b.is_available())
            .unwrap_or(false)
    }
}

impl EmbeddingBackend for FallbackBackend {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match &self.inner {
            Some(backend) if backend.is_available() => backend.embed(text),
            _ => Err(MindCoreError::ModelNotAvailable(
                "no embedding backend available".into(),
            )),
        }
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &self.inner {
            Some(backend) if backend.is_available() => backend.embed_batch(texts),
            _ => Err(MindCoreError::ModelNotAvailable(
                "no embedding backend available".into(),
            )),
        }
    }

    fn dimensions(&self) -> usize {
        self.inner
            .as_ref()
            .map(|b| b.dimensions())
            .unwrap_or(self.dims)
    }

    fn is_available(&self) -> bool {
        self.has_backend()
    }

    fn model_name(&self) -> &str {
        self.inner
            .as_ref()
            .map(|b| b.model_name())
            .unwrap_or("none")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::NoopBackend;

    #[test]
    fn with_backend() {
        let backend = FallbackBackend::new(Box::new(NoopBackend::new(384)));
        assert!(backend.is_available());
        assert!(backend.has_backend());
        assert_eq!(backend.dimensions(), 384);

        let vec = backend.embed("test").expect("embed");
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn without_backend() {
        let backend = FallbackBackend::none(384);
        assert!(!backend.is_available());
        assert!(!backend.has_backend());

        let result = backend.embed("test");
        assert!(result.is_err());
    }

    #[test]
    fn model_name_with_backend() {
        let backend = FallbackBackend::new(Box::new(NoopBackend::new(384)));
        assert_eq!(backend.model_name(), "noop");
    }

    #[test]
    fn model_name_without_backend() {
        let backend = FallbackBackend::none(384);
        assert_eq!(backend.model_name(), "none");
    }
}
