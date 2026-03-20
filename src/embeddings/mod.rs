mod backend;
mod noop;
mod fallback;

pub use backend::EmbeddingBackend;
pub use noop::NoopBackend;
pub use fallback::FallbackBackend;
