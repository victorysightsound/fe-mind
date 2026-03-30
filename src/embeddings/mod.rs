mod api;
mod backend;
mod candle_native;
mod fallback;
mod noop;
pub mod pooling;
mod profile;
mod remote;
mod service;

#[cfg(feature = "api-embeddings")]
pub use api::ApiBackend;
pub use backend::EmbeddingBackend;
#[cfg(feature = "local-embeddings")]
pub use candle_native::{CandleNativeBackend, LocalEmbeddingDevice};
#[cfg(feature = "local-embeddings")]
pub(crate) use candle_native::{describe_device, execution_mode_from_label, select_device};
pub use fallback::FallbackBackend;
pub use noop::NoopBackend;
pub use profile::{
    MINILM_CANONICAL_NAME, MINILM_DIMENSIONS, MINILM_MODEL_REPO, MINILM_PROFILE, MINILM_SHORT_NAME,
    canonical_model_name, compatibility_model_names, embedding_profile_for_model,
};
#[cfg(feature = "remote-embeddings")]
pub use remote::{RemoteEmbeddingBackend, RemoteStatus, RemoteVerificationReport};
#[cfg(feature = "embed-service")]
pub use service::{EmbedServiceOptions, serve_remote_embedding_service_blocking};
