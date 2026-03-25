//! ApiLlmCallback: OpenAI-compatible /v1/chat/completions endpoint.
//!
//! Works with DeepInfra, OpenAI, Together AI, local vLLM, Ollama, or any
//! provider using the OpenAI chat completions API format.
//!
//! Feature-gated behind `api-llm`. Uses ureq (sync, no tokio conflicts).

#[cfg(feature = "api-llm")]
mod inner {
    use crate::error::{MindCoreError, Result};
    use crate::traits::LlmCallback;

    /// LLM callback using an OpenAI-compatible chat completions API.
    pub struct ApiLlmCallback {
        agent: ureq::Agent,
        base_url: String,
        api_key: String,
        model: String,
    }

    impl ApiLlmCallback {
        /// Create a new API LLM callback.
        pub fn new(
            base_url: impl Into<String>,
            api_key: impl Into<String>,
            model: impl Into<String>,
        ) -> Self {
            Self {
                agent: ureq::Agent::new(),
                base_url: base_url.into().trim_end_matches('/').to_string(),
                api_key: api_key.into(),
                model: model.into(),
            }
        }

        /// Create from a command that produces the API key.
        pub fn with_key_cmd(
            base_url: impl Into<String>,
            key_cmd: &str,
            model: impl Into<String>,
        ) -> Result<Self> {
            let output = std::process::Command::new("sh")
                .args(["-c", key_cmd])
                .output()
                .map_err(|e| MindCoreError::Embedding(format!("key_cmd failed: {e}")))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(MindCoreError::Embedding(format!("key_cmd error: {stderr}")));
            }

            let api_key = String::from_utf8_lossy(&output.stdout).trim().to_string();
            Ok(Self::new(base_url, api_key, model))
        }

        /// DeepInfra with Haiku-equivalent model (convenience constructor).
        pub fn deepinfra_haiku(api_key: impl Into<String>) -> Self {
            Self::new(
                "https://api.deepinfra.com/v1/openai",
                api_key,
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )
        }
    }

    impl LlmCallback for ApiLlmCallback {
        fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
            let url = format!("{}/chat/completions", self.base_url);

            let body = serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            });

            let response = self.agent
                .post(&url)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("Content-Type", "application/json")
                .send_json(&body)
                .map_err(|e| MindCoreError::Embedding(format!("LLM API request failed: {e}")))?;

            let resp: ChatResponse = response.into_json()
                .map_err(|e| MindCoreError::Embedding(format!("LLM API response parse: {e}")))?;

            resp.choices
                .into_iter()
                .next()
                .map(|c| c.message.content)
                .ok_or_else(|| MindCoreError::Embedding("Empty LLM response".into()))
        }

        fn model_name(&self) -> &str {
            &self.model
        }

        fn is_available(&self) -> bool {
            !self.api_key.is_empty()
        }
    }

    #[derive(serde::Deserialize)]
    struct ChatResponse {
        choices: Vec<ChatChoice>,
    }

    #[derive(serde::Deserialize)]
    struct ChatChoice {
        message: ChatMessage,
    }

    #[derive(serde::Deserialize)]
    struct ChatMessage {
        content: String,
    }
}

#[cfg(feature = "api-llm")]
pub use inner::ApiLlmCallback;
