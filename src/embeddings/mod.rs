mod backend;
mod candle_native;
mod fallback;
mod noop;
pub mod pooling;

pub use backend::EmbeddingBackend;
#[cfg(feature = "local-embeddings")]
pub use candle_native::CandleNativeBackend;
pub use fallback::FallbackBackend;
pub use noop::NoopBackend;
