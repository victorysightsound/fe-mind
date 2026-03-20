pub mod builder;
mod fts5;

pub use builder::{SearchBuilder, SearchDepth, SearchMode, SearchResult};
pub use fts5::{FtsResult, FtsSearch};
