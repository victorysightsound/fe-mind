mod api;
mod backend;
mod candle_native;
mod fallback;
mod profile;
mod remote;

#[cfg(feature = "api-reranking")]
pub use api::ApiRerankerBackend;
pub use backend::RerankerRuntime;
#[cfg(feature = "reranking")]
pub use candle_native::CandleReranker;
pub use fallback::FallbackRerankerBackend;
pub use profile::{
    RERANKER_CANONICAL_NAME, RERANKER_MODEL_REPO, RERANKER_PROFILE, RERANKER_SHORT_NAME,
    canonical_reranker_name, compatibility_reranker_names, reranker_profile_for_model,
};
#[cfg(feature = "remote-reranking")]
pub use remote::{RemoteRerankerBackend, RemoteRerankerStatus, RemoteRerankerVerificationReport};
