//! LLM callback trait — pluggable interface for any language model provider.
//!
//! Consumers implement this trait to provide LLM capabilities to mindcore.
//! Used for fact extraction, consolidation, evolution, and query decomposition.
//!
//! MindCore defines the prompts and logic; consumers provide the transport.

use crate::error::Result;

/// Callback trait for LLM text generation.
///
/// Any LLM provider can implement this — local models, API endpoints, CLI tools.
/// MindCore calls `generate()` with a crafted prompt and expects a text response.
///
/// # Implementations
/// - `ApiLlmCallback` — OpenAI-compatible API (DeepInfra, OpenAI, Together, Ollama)
/// - `CliLlmCallback` — Claude/ChatGPT/Gemini CLI tools
///
/// # Example
/// ```ignore
/// use mindcore::traits::LlmCallback;
///
/// struct MyLlm;
/// impl LlmCallback for MyLlm {
///     fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
///         // Call your LLM here
///         Ok("response".to_string())
///     }
/// }
/// ```
pub trait LlmCallback: Send + Sync {
    /// Generate a text response from a prompt.
    ///
    /// - `prompt`: The full prompt text (mindcore crafts this internally)
    /// - `max_tokens`: Maximum response length hint (provider may ignore)
    ///
    /// Returns the generated text, or an error if the LLM call fails.
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;

    /// Name of the LLM provider/model for logging and diagnostics.
    fn model_name(&self) -> &str {
        "unknown"
    }

    /// Whether the LLM is available and ready to serve requests.
    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockLlm {
        response: String,
    }

    impl MockLlm {
        fn new(response: &str) -> Self {
            Self { response: response.to_string() }
        }
    }

    impl LlmCallback for MockLlm {
        fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
            Ok(self.response.clone())
        }

        fn model_name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn trait_object_works() {
        let llm: Box<dyn LlmCallback> = Box::new(MockLlm::new("hello"));
        assert_eq!(llm.generate("test", 100).unwrap(), "hello");
        assert_eq!(llm.model_name(), "mock");
        assert!(llm.is_available());
    }

    #[test]
    fn default_methods() {
        struct MinimalLlm;
        impl LlmCallback for MinimalLlm {
            fn generate(&self, _prompt: &str, _max_tokens: usize) -> Result<String> {
                Ok("ok".into())
            }
        }

        let llm = MinimalLlm;
        assert_eq!(llm.model_name(), "unknown");
        assert!(llm.is_available());
    }
}
