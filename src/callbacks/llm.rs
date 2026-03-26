use crate::error::Result;

/// Consumer-provided LLM access for all LLM-assisted operations.
///
/// femind never calls an LLM directly — the consumer controls model choice,
/// cost, and retry behavior by providing this trait implementation.
///
/// Used by: `LLMConsolidation`, `LlmIngest`, `EvolutionStrategy`, `reflect()`.
/// All LLM-dependent features work (degraded) when no callback is provided.
pub trait LlmCallback: Send + Sync {
    /// Given a prompt, return the LLM's response.
    ///
    /// The consumer is responsible for:
    /// - Model selection (Claude, GPT, local Llama, etc.)
    /// - Token budget management
    /// - Retry logic and error handling
    /// - Rate limiting
    fn complete(&self, prompt: &str) -> Result<String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockLlm {
        response: String,
    }

    impl LlmCallback for MockLlm {
        fn complete(&self, _prompt: &str) -> Result<String> {
            Ok(self.response.clone())
        }
    }

    #[test]
    fn mock_llm_works() {
        let llm = MockLlm {
            response: "This is a test response".into(),
        };
        let result = llm.complete("test prompt").expect("complete");
        assert_eq!(result, "This is a test response");
    }

    #[test]
    fn trait_object_works() {
        let llm: Box<dyn LlmCallback> = Box::new(MockLlm {
            response: "trait object response".into(),
        });
        let result = llm.complete("test").expect("complete");
        assert_eq!(result, "trait object response");
    }
}
