//! ApiBackend: OpenAI-compatible /v1/embeddings endpoint.
//!
//! Works with DeepInfra, OpenAI, Together AI, or any provider using the
//! OpenAI embeddings API format. Feature-gated behind `api-embeddings`.
//!
//! Uses ureq (sync HTTP) to avoid tokio runtime conflicts.

#[cfg(feature = "api-embeddings")]
mod inner {
    use crate::embeddings::EmbeddingBackend;
    use crate::error::{FemindError, Result};

    /// Embedding backend using an OpenAI-compatible API endpoint.
    ///
    /// Sends POST requests to `{base_url}/embeddings` with the standard format.
    /// Uses ureq (sync HTTP client) — no tokio runtime conflicts.
    pub struct ApiBackend {
        agent: ureq::Agent,
        base_url: String,
        api_key: String,
        model: String,
        dimensions: usize,
    }

    impl ApiBackend {
        /// Create a new API embedding backend.
        pub fn new(
            base_url: impl Into<String>,
            api_key: impl Into<String>,
            model: impl Into<String>,
            dimensions: usize,
        ) -> Self {
            Self {
                agent: ureq::Agent::new(),
                base_url: base_url.into().trim_end_matches('/').to_string(),
                api_key: api_key.into(),
                model: model.into(),
                dimensions,
            }
        }

        /// Create from a base URL and a command that produces the API key.
        pub fn with_key_cmd(
            base_url: impl Into<String>,
            key_cmd: &str,
            model: impl Into<String>,
            dimensions: usize,
        ) -> Result<Self> {
            let output = std::process::Command::new("sh")
                .args(["-c", key_cmd])
                .output()
                .map_err(|e| FemindError::Embedding(format!("key_cmd failed: {e}")))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(FemindError::Embedding(format!("key_cmd error: {stderr}")));
            }

            let api_key = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if api_key.is_empty() {
                return Err(FemindError::Embedding("key_cmd returned empty key".into()));
            }

            Ok(Self::new(base_url, api_key, model, dimensions))
        }

        /// DeepInfra with all-MiniLM-L6-v2 (convenience constructor).
        pub fn deepinfra_minilm(api_key: impl Into<String>) -> Self {
            Self::new(
                "https://api.deepinfra.com/v1/openai",
                api_key,
                "sentence-transformers/all-MiniLM-L6-v2",
                384,
            )
        }

        /// Call the API for a batch of texts.
        fn call_api(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let url = format!("{}/embeddings", self.base_url);

            let body = serde_json::json!({
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            });

            let response = self
                .agent
                .post(&url)
                .set("Authorization", &format!("Bearer {}", self.api_key))
                .set("Content-Type", "application/json")
                .send_json(&body)
                .map_err(|e| FemindError::Embedding(format!("API request failed: {e}")))?;

            let resp: ApiResponse = response
                .into_json()
                .map_err(|e| FemindError::Embedding(format!("API response parse: {e}")))?;

            // Sort by index to maintain input order
            let mut data = resp.data;
            data.sort_by_key(|d| d.index);

            Ok(data.into_iter().map(|d| d.embedding).collect())
        }
    }

    /// OpenAI-compatible embeddings API response.
    #[derive(serde::Deserialize)]
    struct ApiResponse {
        data: Vec<EmbeddingData>,
    }

    #[derive(serde::Deserialize)]
    struct EmbeddingData {
        embedding: Vec<f32>,
        index: usize,
    }

    impl EmbeddingBackend for ApiBackend {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let results = self.call_api(&[text])?;
            results
                .into_iter()
                .next()
                .ok_or_else(|| FemindError::Embedding("empty API response".into()))
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            const MAX_BATCH: usize = 256;

            let mut all_results = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(MAX_BATCH) {
                let results = self.call_api(chunk)?;
                all_results.extend(results);
            }
            Ok(all_results)
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn is_available(&self) -> bool {
            !self.api_key.is_empty()
        }

        fn model_name(&self) -> &str {
            &self.model
        }
    }
}

#[cfg(feature = "api-embeddings")]
pub use inner::ApiBackend;
