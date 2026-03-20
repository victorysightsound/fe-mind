mod backend;
mod fallback;
mod noop;
pub mod pooling;

pub use backend::EmbeddingBackend;
pub use fallback::FallbackBackend;
pub use noop::NoopBackend;
