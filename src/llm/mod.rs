mod api;
#[cfg(feature = "cli-llm")]
mod cli;

#[cfg(feature = "api-llm")]
pub use api::ApiLlmCallback;
#[cfg(feature = "cli-llm")]
pub use cli::CliLlmCallback;
