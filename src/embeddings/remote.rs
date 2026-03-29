//! RemoteEmbeddingBackend: local-network MiniLM embedding service.
//!
//! Feature-gated behind `remote-embeddings`.

#[cfg(feature = "remote-embeddings")]
mod inner {
    use std::time::Duration;

    use crate::embeddings::EmbeddingBackend;
    use crate::error::{FemindError, Result};

    #[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
    pub struct RemoteStatus {
        pub model: String,
        pub dimensions: usize,
        pub embedding_profile: String,
        #[serde(default)]
        pub execution_mode: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RemoteVerificationReport {
        pub expected_model: String,
        pub expected_dimensions: usize,
        pub expected_embedding_profile: String,
        pub remote_model: String,
        pub remote_dimensions: usize,
        pub remote_embedding_profile: String,
        pub remote_execution_mode: Option<String>,
    }

    pub struct RemoteEmbeddingBackend {
        agent: ureq::Agent,
        base_url: String,
        auth_token: Option<String>,
        model: String,
        canonical_model_name: String,
        dimensions: usize,
        expected_profile: String,
        fallback: Option<Box<dyn EmbeddingBackend>>,
    }

    impl RemoteEmbeddingBackend {
        pub fn new(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            dimensions: usize,
            expected_profile: impl Into<String>,
        ) -> Result<Self> {
            Self::new_with_timeout(
                base_url,
                auth_token,
                model,
                dimensions,
                expected_profile,
                None,
            )
        }

        pub fn new_with_timeout(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            dimensions: usize,
            expected_profile: impl Into<String>,
            timeout: Option<Duration>,
        ) -> Result<Self> {
            Self::build(
                base_url.into(),
                auth_token,
                model.into(),
                dimensions,
                expected_profile.into(),
                timeout,
                None,
                true,
            )
        }

        pub fn with_local_fallback(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            dimensions: usize,
            expected_profile: impl Into<String>,
            fallback: Box<dyn EmbeddingBackend>,
        ) -> Result<Self> {
            Self::with_local_fallback_and_timeout(
                base_url,
                auth_token,
                model,
                dimensions,
                expected_profile,
                fallback,
                None,
            )
        }

        pub fn with_local_fallback_and_timeout(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            model: impl Into<String>,
            dimensions: usize,
            expected_profile: impl Into<String>,
            fallback: Box<dyn EmbeddingBackend>,
            timeout: Option<Duration>,
        ) -> Result<Self> {
            Self::build(
                base_url.into(),
                auth_token,
                model.into(),
                dimensions,
                expected_profile.into(),
                timeout,
                Some(fallback),
                true,
            )
        }

        pub fn minilm(base_url: impl Into<String>, auth_token: Option<String>) -> Result<Self> {
            Self::minilm_with_timeout(base_url, auth_token, None)
        }

        pub fn minilm_with_timeout(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            timeout: Option<Duration>,
        ) -> Result<Self> {
            Self::new_with_timeout(
                base_url,
                auth_token,
                crate::embeddings::MINILM_CANONICAL_NAME,
                crate::embeddings::MINILM_DIMENSIONS,
                crate::embeddings::MINILM_PROFILE,
                timeout,
            )
        }

        pub fn minilm_with_local_fallback(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            fallback: Box<dyn EmbeddingBackend>,
        ) -> Result<Self> {
            Self::minilm_with_local_fallback_and_timeout(base_url, auth_token, fallback, None)
        }

        pub fn minilm_with_local_fallback_and_timeout(
            base_url: impl Into<String>,
            auth_token: Option<String>,
            fallback: Box<dyn EmbeddingBackend>,
            timeout: Option<Duration>,
        ) -> Result<Self> {
            Self::with_local_fallback_and_timeout(
                base_url,
                auth_token,
                crate::embeddings::MINILM_CANONICAL_NAME,
                crate::embeddings::MINILM_DIMENSIONS,
                crate::embeddings::MINILM_PROFILE,
                fallback,
                timeout,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn build(
            base_url: String,
            auth_token: Option<String>,
            model: String,
            dimensions: usize,
            expected_profile: String,
            timeout: Option<Duration>,
            fallback: Option<Box<dyn EmbeddingBackend>>,
            verify_remote: bool,
        ) -> Result<Self> {
            let backend = Self {
                agent: build_agent(timeout),
                base_url: normalize_base_url(&base_url),
                auth_token,
                canonical_model_name: crate::embeddings::canonical_model_name(&model),
                model,
                dimensions,
                expected_profile,
                fallback,
            };

            if verify_remote {
                match backend.verify_remote() {
                    Ok(_) => {}
                    Err(err) => {
                        if let Some(fallback) = backend.fallback.as_ref()
                            && fallback.is_available()
                        {
                            tracing::warn!(
                                "remote embedding backend verification failed, starting with local fallback available: {err}"
                            );
                        } else {
                            return Err(err);
                        }
                    }
                }
            }

            Ok(backend)
        }

        pub fn verify_remote(&self) -> Result<RemoteVerificationReport> {
            let status = self.fetch_status()?;
            let accepted_models = crate::embeddings::compatibility_model_names(&self.model);
            if !accepted_models.iter().any(|candidate| candidate == &status.model) {
                return Err(FemindError::Embedding(format!(
                    "remote embedding service model mismatch: expected one of {:?} but got '{}'",
                    accepted_models, status.model
                )));
            }
            if status.dimensions != self.dimensions {
                return Err(FemindError::Embedding(format!(
                    "remote embedding service dimension mismatch: expected {} but got {}",
                    self.dimensions, status.dimensions
                )));
            }
            if status.embedding_profile != self.expected_profile {
                return Err(FemindError::Embedding(format!(
                    "remote embedding service profile mismatch: expected '{}' but got '{}'",
                    self.expected_profile, status.embedding_profile
                )));
            }

            Ok(RemoteVerificationReport {
                expected_model: self.model.clone(),
                expected_dimensions: self.dimensions,
                expected_embedding_profile: self.expected_profile.clone(),
                remote_model: status.model,
                remote_dimensions: status.dimensions,
                remote_embedding_profile: status.embedding_profile,
                remote_execution_mode: status.execution_mode,
            })
        }

        pub fn status(&self) -> Result<RemoteStatus> {
            self.fetch_status()
        }

        fn fetch_status(&self) -> Result<RemoteStatus> {
            let url = format!("{}/status", self.base_url);
            let mut request = self.agent.get(&url);
            if let Some(token) = self.auth_token.as_deref() {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
            let response = request
                .call()
                .map_err(|e| FemindError::Embedding(format!("remote status request failed: {e}")))?;

            response
                .into_json::<RemoteStatus>()
                .map_err(|e| FemindError::Embedding(format!("remote status parse failed: {e}")))
        }

        fn call_embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let url = format!("{}/embed", self.base_url);
            let body = serde_json::json!({
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            });

            let mut request = self.agent.post(&url).set("Content-Type", "application/json");
            if let Some(token) = self.auth_token.as_deref() {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
            let response = request.send_json(&body).map_err(|e| {
                FemindError::Embedding(format!("remote embedding request failed: {e}"))
            })?;

            let mut parsed = response
                .into_json::<RemoteEmbedResponse>()
                .map_err(|e| FemindError::Embedding(format!("remote embed parse failed: {e}")))?;

            parsed.data.sort_by_key(|item| item.index);
            Ok(parsed.data.into_iter().map(|item| item.embedding).collect())
        }

        fn fallback_embed(&self, text: &str, primary_error: FemindError) -> Result<Vec<f32>> {
            if let Some(fallback) = self.fallback.as_ref()
                && fallback.is_available()
            {
                tracing::warn!(
                    "remote embedding failed, falling back to local backend: {primary_error}"
                );
                return fallback.embed(text);
            }
            Err(primary_error)
        }

        fn fallback_embed_batch(
            &self,
            texts: &[&str],
            primary_error: FemindError,
        ) -> Result<Vec<Vec<f32>>> {
            if let Some(fallback) = self.fallback.as_ref()
                && fallback.is_available()
            {
                tracing::warn!(
                    "remote embedding failed, falling back to local backend: {primary_error}"
                );
                return fallback.embed_batch(texts);
            }
            Err(primary_error)
        }
    }

    #[derive(serde::Deserialize)]
    struct RemoteEmbedResponse {
        data: Vec<RemoteEmbedData>,
    }

    #[derive(serde::Deserialize)]
    struct RemoteEmbedData {
        embedding: Vec<f32>,
        index: usize,
    }

    impl EmbeddingBackend for RemoteEmbeddingBackend {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            match self.call_embed(&[text]) {
                Ok(mut batch) => batch.pop().ok_or_else(|| {
                    FemindError::Embedding("remote embedding response was empty".into())
                }),
                Err(err) => self.fallback_embed(text, err),
            }
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            match self.call_embed(texts) {
                Ok(batch) => Ok(batch),
                Err(err) => self.fallback_embed_batch(texts, err),
            }
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn is_available(&self) -> bool {
            true
        }

        fn model_name(&self) -> &str {
            &self.canonical_model_name
        }

        fn embedding_profile(&self) -> String {
            self.expected_profile.clone()
        }

        fn compatibility_model_names(&self) -> Vec<String> {
            crate::embeddings::compatibility_model_names(&self.model)
        }
    }

    fn normalize_base_url(value: &str) -> String {
        value.trim_end_matches('/').to_string()
    }

    fn build_agent(timeout: Option<Duration>) -> ureq::Agent {
        let mut builder = ureq::AgentBuilder::new();
        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }
        builder.build()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::embeddings::NoopBackend;
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        use std::thread;

        fn serve_once(
            status_payload: serde_json::Value,
            embed_payload: serde_json::Value,
        ) -> std::result::Result<String, Box<dyn std::error::Error>> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let address = listener.local_addr()?;
            thread::spawn(move || {
                for _ in 0..2 {
                    let result = listener.accept();
                    if let Ok((mut stream, _)) = result {
                        let _ = respond(&mut stream, &status_payload, &embed_payload);
                    }
                }
            });
            Ok(format!("http://{address}"))
        }

        fn respond(
            stream: &mut TcpStream,
            status_payload: &serde_json::Value,
            embed_payload: &serde_json::Value,
        ) -> std::result::Result<(), Box<dyn std::error::Error>> {
            let mut buffer = [0_u8; 8192];
            let bytes = stream.read(&mut buffer)?;
            let request = String::from_utf8_lossy(&buffer[..bytes]);
            let body = if request.starts_with("GET /status") {
                serde_json::to_string(status_payload)?
            } else if request.starts_with("POST /embed") {
                serde_json::to_string(embed_payload)?
            } else {
                serde_json::to_string(&serde_json::json!({"ok": true}))?
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes())?;
            stream.flush()?;
            Ok(())
        }

        #[test]
        fn remote_backend_embeds_when_profile_matches()
        -> std::result::Result<(), Box<dyn std::error::Error>> {
            let status = serde_json::json!({
                "model": crate::embeddings::MINILM_MODEL_REPO,
                "dimensions": crate::embeddings::MINILM_DIMENSIONS,
                "embedding_profile": crate::embeddings::MINILM_PROFILE,
                "execution_mode": "local-gpu"
            });
            let embed = serde_json::json!({
                "data": [
                    { "index": 0, "embedding": vec![0.5_f32; crate::embeddings::MINILM_DIMENSIONS] }
                ]
            });
            let base_url = serve_once(status, embed)?;
            let backend = RemoteEmbeddingBackend::minilm(base_url, None)?;
            let vector = backend.embed("hello world")?;
            assert_eq!(vector.len(), crate::embeddings::MINILM_DIMENSIONS);
            assert_eq!(backend.model_name(), crate::embeddings::MINILM_CANONICAL_NAME);
            Ok(())
        }

        #[test]
        fn verify_remote_reports_profile_mismatch()
        -> std::result::Result<(), Box<dyn std::error::Error>> {
            let status = serde_json::json!({
                "model": crate::embeddings::MINILM_MODEL_REPO,
                "dimensions": crate::embeddings::MINILM_DIMENSIONS,
                "embedding_profile": "local|sentence-transformers/all-MiniLM-L6-v2|384|v1|chars:max:100",
                "execution_mode": "local-gpu"
            });
            let embed = serde_json::json!({ "data": [] });
            let base_url = serve_once(status, embed)?;
            let result = RemoteEmbeddingBackend::minilm(base_url, None);
            assert!(result.is_err());
            let message = match result {
                Ok(_) => String::new(),
                Err(err) => err.to_string(),
            };
            assert!(message.contains("profile mismatch"));
            Ok(())
        }

        #[test]
        fn remote_backend_falls_back_to_local_on_transport_failure()
        -> std::result::Result<(), Box<dyn std::error::Error>> {
            let backend = RemoteEmbeddingBackend::build(
                "http://127.0.0.1:9".to_string(),
                None,
                crate::embeddings::MINILM_MODEL_REPO.to_string(),
                crate::embeddings::MINILM_DIMENSIONS,
                crate::embeddings::MINILM_PROFILE.to_string(),
                None,
                Some(Box::new(NoopBackend::new(crate::embeddings::MINILM_DIMENSIONS))),
                false,
            )?;

            let vector = backend.embed("fallback")?;
            assert_eq!(vector.len(), crate::embeddings::MINILM_DIMENSIONS);
            Ok(())
        }

        #[test]
        fn remote_backend_can_initialize_with_fallback_when_remote_is_down()
        -> std::result::Result<(), Box<dyn std::error::Error>> {
            let backend = RemoteEmbeddingBackend::minilm_with_local_fallback(
                "http://127.0.0.1:9",
                None,
                Box::new(NoopBackend::new(crate::embeddings::MINILM_DIMENSIONS)),
            )?;

            let vector = backend.embed("fallback-on-startup")?;
            assert_eq!(vector.len(), crate::embeddings::MINILM_DIMENSIONS);
            Ok(())
        }
    }
}

#[cfg(feature = "remote-embeddings")]
pub use inner::{RemoteEmbeddingBackend, RemoteStatus, RemoteVerificationReport};
