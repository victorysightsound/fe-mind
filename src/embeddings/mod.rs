mod api;
mod backend;
mod candle_native;
mod fallback;
mod noop;
pub mod pooling;

#[cfg(feature = "api-embeddings")]
pub use api::ApiBackend;
pub use backend::EmbeddingBackend;
#[cfg(feature = "local-embeddings")]
pub use candle_native::CandleNativeBackend;
pub use fallback::FallbackBackend;
pub use noop::NoopBackend;
