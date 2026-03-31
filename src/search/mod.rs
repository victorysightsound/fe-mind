pub mod ann;
pub mod builder;
pub mod fts5;
pub mod hybrid;
pub mod query_expand;
pub mod vector;

#[cfg(feature = "ann")]
pub use ann::AnnIndex;
pub use builder::{
    QueryIntent, QueryRoute, ReflectionSearchPreference, SearchBuilder, SearchDepth, SearchMode,
    SearchResult, StableSummaryPolicy, StateConflictPolicy, TemporalPolicy,
};
pub use fts5::{FtsResult, FtsSearch};
pub use hybrid::rrf_merge;
pub use vector::VectorSearch;
