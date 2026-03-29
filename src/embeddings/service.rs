#[cfg(feature = "embed-service")]
mod inner {
    use std::{sync::Arc, time::Duration};

    use axum::{
        Json, Router,
        extract::State,
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::{get, post},
    };
    use tokio::net::TcpListener;

    use crate::embeddings::{
        CandleNativeBackend, EmbeddingBackend, LocalEmbeddingDevice, MINILM_CANONICAL_NAME,
        MINILM_DIMENSIONS, MINILM_MODEL_REPO, MINILM_PROFILE, MINILM_SHORT_NAME,
    };

    #[derive(Clone, Debug)]
    pub struct EmbedServiceOptions {
        pub host: String,
        pub port: u16,
        pub prefix: String,
        pub auth_token: Option<String>,
        pub device_mode: LocalEmbeddingDevice,
        pub cuda_ordinal: usize,
        pub request_timeout_secs: Option<u64>,
        pub max_batch_texts: usize,
    }

    impl Default for EmbedServiceOptions {
        fn default() -> Self {
            Self {
                host: "127.0.0.1".to_string(),
                port: 8899,
                prefix: "/embed".to_string(),
                auth_token: None,
                device_mode: LocalEmbeddingDevice::Auto,
                cuda_ordinal: 0,
                request_timeout_secs: None,
                max_batch_texts: 32,
            }
        }
    }

    #[derive(Clone)]
    struct ServiceState {
        backend: Arc<CandleNativeBackend>,
        auth_token: Option<String>,
        status: ServiceStatusResponse,
        request_timeout: Option<Duration>,
        max_batch_texts: usize,
    }

    #[derive(Clone, serde::Serialize)]
    struct HealthResponse {
        ok: bool,
        service: String,
    }

    #[derive(Clone, serde::Serialize)]
    struct ServiceStatusResponse {
        ok: bool,
        service: String,
        provider: String,
        model: String,
        model_repo: String,
        dimensions: usize,
        embedding_profile: String,
        execution_mode: String,
        device_label: String,
        request_timeout_secs: Option<u64>,
        max_batch_texts: usize,
    }

    #[derive(serde::Deserialize)]
    struct EmbedRequest {
        model: String,
        input: Option<Vec<String>>,
        texts: Option<Vec<String>>,
        #[allow(dead_code)]
        encoding_format: Option<String>,
        expected_embedding_profile: Option<String>,
    }

    #[derive(serde::Serialize)]
    struct EmbedItem {
        index: usize,
        embedding: Vec<f32>,
    }

    #[derive(serde::Serialize)]
    struct EmbedResponse {
        data: Vec<EmbedItem>,
        model: String,
        dimensions: usize,
        embedding_profile: String,
        execution_mode: String,
    }

    pub fn serve_remote_embedding_service_blocking(options: EmbedServiceOptions) -> Result<(), String> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;

        runtime.block_on(async move {
            let backend = Arc::new(
                CandleNativeBackend::new_with_device(options.device_mode, options.cuda_ordinal)
                    .map_err(|error| error.to_string())?,
            );
            let status = ServiceStatusResponse {
                ok: true,
                service: "femind-embed-service".to_string(),
                provider: "local".to_string(),
                model: MINILM_CANONICAL_NAME.to_string(),
                model_repo: MINILM_MODEL_REPO.to_string(),
                dimensions: MINILM_DIMENSIONS,
                embedding_profile: MINILM_PROFILE.to_string(),
                execution_mode: backend.execution_mode().to_string(),
                device_label: backend.device_label().to_string(),
                request_timeout_secs: options.request_timeout_secs,
                max_batch_texts: options.max_batch_texts,
            };

            let state = ServiceState {
                backend,
                auth_token: options.auth_token,
                status,
                request_timeout: options.request_timeout_secs.map(Duration::from_secs),
                max_batch_texts: options.max_batch_texts,
            };

            let routes = Router::new()
                .route("/health", get(health))
                .route("/status", get(status_handler))
                .route("/embed", post(embed))
                .with_state(state);
            let router = if normalize_prefix(&options.prefix) == "/" {
                routes
            } else {
                Router::new().nest(&normalize_prefix(&options.prefix), routes)
            };

            let bind_addr = format!("{}:{}", options.host, options.port);
            let listener = TcpListener::bind(&bind_addr)
                .await
                .map_err(|error| format!("bind {bind_addr}: {error}"))?;
            eprintln!(
                "femind-embed-service: listening on http://{bind_addr}{}",
                normalize_prefix(&options.prefix)
            );
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
                .map_err(|error| error.to_string())
        })
    }

    async fn health(State(state): State<ServiceState>) -> Json<HealthResponse> {
        Json(HealthResponse {
            ok: true,
            service: state.status.service.clone(),
        })
    }

    async fn status_handler(
        State(state): State<ServiceState>,
        headers: HeaderMap,
    ) -> impl IntoResponse {
        if let Some(response) = authorize(&headers, state.auth_token.as_deref()) {
            return response;
        }
        (StatusCode::OK, Json(state.status)).into_response()
    }

    async fn embed(
        State(state): State<ServiceState>,
        headers: HeaderMap,
        Json(request): Json<EmbedRequest>,
    ) -> impl IntoResponse {
        if let Some(response) = authorize(&headers, state.auth_token.as_deref()) {
            return response;
        }
        if request.model != MINILM_CANONICAL_NAME
            && request.model != MINILM_MODEL_REPO
            && request.model != MINILM_SHORT_NAME
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "unsupported model '{}'; this service only serves {}",
                        request.model,
                        MINILM_CANONICAL_NAME
                    )
                })),
            )
                .into_response();
        }

        if let Some(expected_profile) = request.expected_embedding_profile.as_deref()
            && expected_profile != MINILM_PROFILE
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "embedding profile mismatch: expected '{}' but service provides '{}'",
                        expected_profile,
                        MINILM_PROFILE
                    )
                })),
            )
                .into_response();
        }

        let Some(text_inputs) = request
            .input
            .filter(|items| !items.is_empty())
            .or_else(|| request.texts.filter(|items| !items.is_empty()))
        else {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "embed request requires a non-empty 'input' or 'texts' array"
                })),
            )
                .into_response();
        };

        if text_inputs.len() > state.max_batch_texts {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!(
                        "embed request exceeds max_batch_texts {}; got {} inputs",
                        state.max_batch_texts,
                        text_inputs.len()
                    )
                })),
            )
                .into_response();
        }

        let backend = Arc::clone(&state.backend);
        let embed_future = tokio::task::spawn_blocking(move || {
            let texts: Vec<&str> = text_inputs.iter().map(String::as_str).collect();
            backend.embed_batch(&texts)
        });

        let embed_result = if let Some(timeout) = state.request_timeout {
            match tokio::time::timeout(timeout, embed_future).await {
                Ok(join_result) => join_result,
                Err(_) => {
                    return (
                        StatusCode::REQUEST_TIMEOUT,
                        Json(serde_json::json!({
                            "error": format!(
                                "embedding request exceeded timeout of {} seconds",
                                timeout.as_secs()
                            )
                        })),
                    )
                        .into_response();
                }
            }
        } else {
            embed_future.await
        };

        match embed_result {
            Err(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("embedding task join failed: {error}")
                })),
            )
                .into_response(),
            Ok(Err(error)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error.to_string() })),
            )
                .into_response(),
            Ok(Ok(vectors)) => {
                let data = vectors
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| EmbedItem { index, embedding })
                    .collect::<Vec<_>>();
                (StatusCode::OK, Json(serde_json::json!(EmbedResponse {
                    data,
                    model: MINILM_CANONICAL_NAME.to_string(),
                    dimensions: MINILM_DIMENSIONS,
                    embedding_profile: MINILM_PROFILE.to_string(),
                    execution_mode: state.status.execution_mode.clone(),
                }))).into_response()
            }
        }
    }

    fn authorize(
        headers: &HeaderMap,
        expected_token: Option<&str>,
    ) -> Option<axum::response::Response> {
        let expected_token = expected_token?;
        let header = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if header == format!("Bearer {expected_token}") {
            None
        } else {
            Some(
                (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({
                        "error": "missing or invalid bearer token"
                    })),
                )
                    .into_response(),
            )
        }
    }

    fn normalize_prefix(value: &str) -> String {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed == "/" {
            "/".to_string()
        } else if trimmed.starts_with('/') {
            trimmed.trim_end_matches('/').to_string()
        } else {
            format!("/{}", trimmed.trim_end_matches('/'))
        }
    }
}

#[cfg(feature = "embed-service")]
pub use inner::{EmbedServiceOptions, serve_remote_embedding_service_blocking};
