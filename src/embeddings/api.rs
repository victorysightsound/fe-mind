//! ApiBackend: OpenAI-compatible /v1/embeddings endpoint.
//!
//! Works with DeepInfra, OpenAI, Together AI, or any provider using the
//! OpenAI embeddings API format. Feature-gated behind `api-embeddings`.

#[cfg(feature = "api-embeddings")]
mod inner {
    use crate::embeddings::EmbeddingBackend;
    use crate::error::{MindCoreError, Result};

    /// Embedding backend using an OpenAI-compatible API endpoint.
    ///
    /// Sends POST requests to `{base_url}/embeddings` with the standard format.
    /// Uses reqwest blocking client (sync, matches EmbeddingBackend trait).
    pub struct ApiBackend {
        client: reqwest::blocking::Client,
        base_url: String,
        api_key: String,
        model: String,
        dimensions: usize,
    }

    impl ApiBackend {
        /// Create a new API embedding backend.
        ///
        /// - `base_url`: API base URL (e.g., "https://api.deepinfra.com/v1/openai")
        /// - `api_key`: Bearer token for authentication
        /// - `model`: Model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        /// - `dimensions`: Expected embedding dimensions (384 for MiniLM)
        pub fn new(
            base_url: impl Into<String>,
            api_key: impl Into<String>,
            model: impl Into<String>,
            dimensions: usize,
        ) -> Self {
            Self {
                client: reqwest::blocking::Client::new(),
                base_url: base_url.into().trim_end_matches('/').to_string(),
                api_key: api_key.into(),
                model: model.into(),
                dimensions,
            }
        }

        /// Create from a base URL and a command that produces the API key.
        ///
        /// Runs the command at construction time and captures stdout as the key.
        /// Useful for fetching keys from 1Password, macOS Keychain, etc.
        pub fn with_key_cmd(
            base_url: impl Into<String>,
            key_cmd: &str,
            model: impl Into<String>,
            dimensions: usize,
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
            if api_key.is_empty() {
                return Err(MindCoreError::Embedding("key_cmd returned empty key".into()));
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

            let response = self.client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .map_err(|e| MindCoreError::Embedding(format!("API request failed: {e}")))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().unwrap_or_default();
                return Err(MindCoreError::Embedding(format!("API error {status}: {body}")));
            }

            let resp: ApiResponse = response.json()
                .map_err(|e| MindCoreError::Embedding(format!("API response parse: {e}")))?;

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
                .ok_or_else(|| MindCoreError::Embedding("empty API response".into()))
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            // API endpoints typically handle batching well, but cap at 256 per request
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
