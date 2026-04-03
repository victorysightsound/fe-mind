use std::{fs, path::PathBuf, time::Duration};

use femind::embeddings::{
    EmbedServiceOptions, LocalEmbeddingDevice, MINILM_CANONICAL_NAME, MINILM_DIMENSIONS,
    MINILM_PROFILE, serve_remote_embedding_service_blocking,
};
use femind::reranking::{RERANKER_CANONICAL_NAME, RERANKER_PROFILE};
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), String> {
    init_logging();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let Some(first) = args.first().cloned() else {
        return run_serve(&[]);
    };

    match first.as_str() {
        "serve" => run_serve(&args[1..]),
        "status" => run_status(&args[1..]),
        "verify-remote" => run_verify_remote(&args[1..]),
        "verify-remote-reranker" => run_verify_remote_reranker(&args[1..]),
        "--help" | "-h" => {
            print_help();
            Ok(())
        }
        _ => run_serve(&args),
    }
}

fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Human,
    Json,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct FileConfig {
    #[serde(default)]
    embedding_service: Option<FileEmbeddingServiceConfig>,
    #[serde(default)]
    embeddings: Option<FileEmbeddingsConfig>,
    #[serde(default)]
    reranking: Option<FileRerankingConfig>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct FileEmbeddingServiceConfig {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    bind_host: Option<String>,
    #[serde(default)]
    bind_port: Option<u16>,
    #[serde(default)]
    path_prefix: Option<String>,
    #[serde(default)]
    auth_token_env: Option<String>,
    #[serde(default)]
    auth_token_env_file: Option<PathBuf>,
    #[serde(default)]
    request_timeout_secs: Option<u64>,
    #[serde(default)]
    max_batch_texts: Option<usize>,
    #[serde(default)]
    max_batch_documents: Option<usize>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    cuda_ordinal: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct FileEmbeddingsConfig {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    local: Option<FileLocalEmbeddingsConfig>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct FileLocalEmbeddingsConfig {
    #[serde(default)]
    execution_mode: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    remote_service: Option<FileRemoteServiceConfig>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct FileRemoteServiceConfig {
    base_url: String,
    #[serde(default)]
    auth_token_env: Option<String>,
    #[serde(default)]
    auth_token_env_file: Option<PathBuf>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    fallback_to_local: Option<bool>,
    #[serde(default)]
    verify_profile: Option<bool>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct FileRerankingConfig {
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    local: Option<FileLocalRerankingConfig>,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct FileLocalRerankingConfig {
    #[serde(default)]
    execution_mode: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    remote_service: Option<FileRemoteServiceConfig>,
}

#[derive(Debug, Clone)]
struct RemoteProbeConfig {
    base_url: String,
    auth_token: Option<String>,
    timeout_secs: Option<u64>,
    verify_profile: bool,
    fallback_to_local: bool,
    model: String,
    dimensions: usize,
    embedding_profile: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct RemoteProbeStatus {
    model: String,
    dimensions: usize,
    embedding_profile: String,
    #[serde(default)]
    execution_mode: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model_repo: Option<String>,
    #[serde(default)]
    device_label: Option<String>,
    #[serde(default)]
    request_timeout_secs: Option<u64>,
    #[serde(default)]
    max_batch_texts: Option<usize>,
}

#[derive(Debug, Clone)]
struct RemoteRerankProbeConfig {
    base_url: String,
    auth_token: Option<String>,
    timeout_secs: Option<u64>,
    verify_profile: bool,
    fallback_to_local: bool,
    model: String,
    reranker_profile: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct RemoteRerankProbeStatus {
    model: String,
    reranker_profile: String,
    #[serde(default)]
    execution_mode: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model_repo: Option<String>,
    #[serde(default)]
    device_label: Option<String>,
    #[serde(default)]
    request_timeout_secs: Option<u64>,
    #[serde(default)]
    max_batch_documents: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct RemoteVerifyOutput {
    ok: bool,
    profile_match: bool,
    remote_service_base_url: String,
    remote_service_auth_configured: bool,
    expected_model: String,
    expected_dimensions: usize,
    expected_embedding_profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_embedding_profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_execution_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct RemoteRerankVerifyOutput {
    ok: bool,
    profile_match: bool,
    remote_service_base_url: String,
    remote_service_auth_configured: bool,
    expected_model: String,
    expected_reranker_profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_reranker_profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_execution_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct StatusOutput {
    execution_mode: String,
    model: String,
    dimensions: usize,
    embedding_profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_fallback_to_local: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_verify_profile: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_auth_configured: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_status: Option<RemoteProbeStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_verification: Option<RemoteVerifyOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reranking: Option<RerankStatusOutput>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct RerankStatusOutput {
    execution_mode: String,
    model: String,
    reranker_profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_fallback_to_local: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_verify_profile: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_auth_configured: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_status: Option<RemoteRerankProbeStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote_service_verification: Option<RemoteRerankVerifyOutput>,
}

fn run_serve(args: &[String]) -> Result<(), String> {
    let mut options = EmbedServiceOptions::default();
    let mut config_path: Option<PathBuf> = None;
    let mut auth_token_env: Option<String> = None;
    let mut auth_token_env_file: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                config_path = Some(PathBuf::from(required_value(args, i, "--config")?));
            }
            "--host" => {
                i += 1;
                options.host = required_value(args, i, "--host")?.to_string();
            }
            "--port" => {
                i += 1;
                let value = required_value(args, i, "--port")?;
                options.port = value
                    .parse::<u16>()
                    .map_err(|error| format!("invalid --port value '{value}': {error}"))?;
            }
            "--prefix" => {
                i += 1;
                options.prefix = required_value(args, i, "--prefix")?.to_string();
            }
            "--auth-token-env" => {
                i += 1;
                auth_token_env = Some(required_value(args, i, "--auth-token-env")?.to_string());
            }
            "--auth-token-env-file" => {
                i += 1;
                auth_token_env_file = Some(PathBuf::from(required_value(
                    args,
                    i,
                    "--auth-token-env-file",
                )?));
            }
            "--device" => {
                i += 1;
                options.device_mode = parse_device(required_value(args, i, "--device")?)?;
            }
            "--cuda-ordinal" => {
                i += 1;
                let value = required_value(args, i, "--cuda-ordinal")?;
                options.cuda_ordinal = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --cuda-ordinal value '{value}': {error}"))?;
            }
            "--request-timeout-secs" => {
                i += 1;
                let value = required_value(args, i, "--request-timeout-secs")?;
                options.request_timeout_secs = Some(value.parse::<u64>().map_err(|error| {
                    format!("invalid --request-timeout-secs value '{value}': {error}")
                })?);
            }
            "--max-batch-texts" => {
                i += 1;
                let value = required_value(args, i, "--max-batch-texts")?;
                options.max_batch_texts = value.parse::<usize>().map_err(|error| {
                    format!("invalid --max-batch-texts value '{value}': {error}")
                })?;
            }
            "--max-batch-documents" => {
                i += 1;
                let value = required_value(args, i, "--max-batch-documents")?;
                options.max_batch_documents = value.parse::<usize>().map_err(|error| {
                    format!("invalid --max-batch-documents value '{value}': {error}")
                })?;
            }
            "--help" | "-h" => {
                print_serve_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    if let Some(path) = config_path {
        let config = load_config(&path)?;
        if let Some(service) = config.embedding_service {
            if service.enabled == Some(false) {
                return Err("embedding_service.enabled is false in config".to_string());
            }
            if let Some(host) = service.bind_host {
                options.host = host;
            }
            if let Some(port) = service.bind_port {
                options.port = port;
            }
            if let Some(prefix) = service.path_prefix {
                options.prefix = prefix;
            }
            if let Some(timeout) = service.request_timeout_secs {
                options.request_timeout_secs = Some(timeout);
            }
            if let Some(max_batch) = service.max_batch_texts {
                options.max_batch_texts = max_batch;
            }
            if let Some(max_batch) = service.max_batch_documents {
                options.max_batch_documents = max_batch;
            }
            if let Some(device) = service.device {
                options.device_mode = parse_device(&device)?;
            }
            if let Some(cuda_ordinal) = service.cuda_ordinal {
                options.cuda_ordinal = cuda_ordinal;
            }
            auth_token_env = auth_token_env.or(service.auth_token_env);
            auth_token_env_file = auth_token_env_file.or(service.auth_token_env_file);
        }
    }

    options.auth_token = resolve_secret(auth_token_env.as_deref(), auth_token_env_file.as_deref())?;
    serve_remote_embedding_service_blocking(options)
}

fn run_status(args: &[String]) -> Result<(), String> {
    let (config, format) = match load_config_and_format(args, "status") {
        Ok(value) => value,
        Err(error) if error.is_empty() => return Ok(()),
        Err(error) => return Err(error),
    };
    let output = build_status_output(&config)?;
    print_output(&output, format)
}

fn run_verify_remote(args: &[String]) -> Result<(), String> {
    let (config, format) = match load_config_and_format(args, "verify-remote") {
        Ok(value) => value,
        Err(error) if error.is_empty() => return Ok(()),
        Err(error) => return Err(error),
    };
    let probe = remote_probe_from_config(&config)?;
    let result = verify_remote_probe(&probe);
    let exit_ok = result.ok;
    print_output(&result, format)?;
    if exit_ok {
        Ok(())
    } else {
        Err(result
            .error
            .unwrap_or_else(|| "remote verification failed".to_string()))
    }
}

fn run_verify_remote_reranker(args: &[String]) -> Result<(), String> {
    let (config, format) = match load_config_and_format(args, "verify-remote-reranker") {
        Ok(value) => value,
        Err(error) if error.is_empty() => return Ok(()),
        Err(error) => return Err(error),
    };
    let probe = remote_rerank_probe_from_config(&config)?;
    let result = verify_remote_rerank_probe(&probe);
    let exit_ok = result.ok;
    print_output(&result, format)?;
    if exit_ok {
        Ok(())
    } else {
        Err(result
            .error
            .unwrap_or_else(|| "remote reranker verification failed".to_string()))
    }
}

fn load_config_and_format(
    args: &[String],
    command: &str,
) -> Result<(FileConfig, OutputFormat), String> {
    let mut config_path: Option<PathBuf> = None;
    let mut format = OutputFormat::Human;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                config_path = Some(PathBuf::from(required_value(args, i, "--config")?));
            }
            "--format" => {
                i += 1;
                format = parse_format(required_value(args, i, "--format")?)?;
            }
            "--help" | "-h" => {
                print_command_help(command);
                return Err(String::new());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    let Some(config_path) = config_path else {
        return Err(format!("{command} requires --config <path>"));
    };
    Ok((load_config(&config_path)?, format))
}

fn build_status_output(config: &FileConfig) -> Result<StatusOutput, String> {
    let embeddings = config
        .embeddings
        .as_ref()
        .ok_or_else(|| "missing [embeddings] config".to_string())?;
    if embeddings.enabled == Some(false) {
        return Err("embeddings.enabled is false in config".to_string());
    }
    let execution_mode = embeddings
        .local
        .as_ref()
        .and_then(|local| local.execution_mode.clone())
        .unwrap_or_else(|| "embedded".to_string());
    let model = embeddings
        .model
        .clone()
        .unwrap_or_else(|| MINILM_CANONICAL_NAME.to_string());
    let dimensions = embeddings.dimensions.unwrap_or(MINILM_DIMENSIONS);
    let embedding_profile = if model == MINILM_CANONICAL_NAME
        || model == "sentence-transformers/all-MiniLM-L6-v2"
        || model == "all-MiniLM-L6-v2"
    {
        MINILM_PROFILE.to_string()
    } else {
        format!("custom|{model}|{dimensions}")
    };

    if execution_mode == "remote_service" {
        let probe = remote_probe_from_config(config)?;
        let auth_configured = probe.auth_token.is_some();
        let status = fetch_remote_status(&probe)?;
        let verification = if probe.verify_profile {
            Some(verify_remote_probe(&probe))
        } else {
            None
        };
        Ok(StatusOutput {
            execution_mode,
            model: MINILM_CANONICAL_NAME.to_string(),
            dimensions,
            embedding_profile,
            local_device: embeddings
                .local
                .as_ref()
                .and_then(|local| local.device.clone()),
            remote_service_base_url: Some(probe.base_url),
            remote_service_timeout_secs: probe.timeout_secs,
            remote_service_fallback_to_local: Some(probe.fallback_to_local),
            remote_service_verify_profile: Some(probe.verify_profile),
            remote_service_auth_configured: Some(auth_configured),
            remote_service_status: Some(status),
            remote_service_verification: verification,
            reranking: build_rerank_status_output(config)?,
        })
    } else {
        Ok(StatusOutput {
            execution_mode,
            model: MINILM_CANONICAL_NAME.to_string(),
            dimensions,
            embedding_profile,
            local_device: embeddings
                .local
                .as_ref()
                .and_then(|local| local.device.clone()),
            remote_service_base_url: None,
            remote_service_timeout_secs: None,
            remote_service_fallback_to_local: None,
            remote_service_verify_profile: None,
            remote_service_auth_configured: None,
            remote_service_status: None,
            remote_service_verification: None,
            reranking: build_rerank_status_output(config)?,
        })
    }
}

fn build_rerank_status_output(config: &FileConfig) -> Result<Option<RerankStatusOutput>, String> {
    let Some(reranking) = config.reranking.as_ref() else {
        return Ok(None);
    };
    if reranking.enabled == Some(false) {
        return Ok(None);
    }

    let execution_mode = reranking
        .local
        .as_ref()
        .and_then(|local| local.execution_mode.clone())
        .unwrap_or_else(|| "embedded".to_string());
    let model = reranking
        .model
        .clone()
        .unwrap_or_else(|| RERANKER_CANONICAL_NAME.to_string());

    if execution_mode == "remote_service" {
        let probe = remote_rerank_probe_from_config(config)?;
        let auth_configured = probe.auth_token.is_some();
        let status = fetch_remote_rerank_status(&probe)?;
        let verification = if probe.verify_profile {
            Some(verify_remote_rerank_probe(&probe))
        } else {
            None
        };
        Ok(Some(RerankStatusOutput {
            execution_mode,
            model,
            reranker_profile: RERANKER_PROFILE.to_string(),
            local_device: reranking
                .local
                .as_ref()
                .and_then(|local| local.device.clone()),
            remote_service_base_url: Some(probe.base_url),
            remote_service_timeout_secs: probe.timeout_secs,
            remote_service_fallback_to_local: Some(probe.fallback_to_local),
            remote_service_verify_profile: Some(probe.verify_profile),
            remote_service_auth_configured: Some(auth_configured),
            remote_service_status: Some(status),
            remote_service_verification: verification,
        }))
    } else {
        Ok(Some(RerankStatusOutput {
            execution_mode,
            model,
            reranker_profile: RERANKER_PROFILE.to_string(),
            local_device: reranking
                .local
                .as_ref()
                .and_then(|local| local.device.clone()),
            remote_service_base_url: None,
            remote_service_timeout_secs: None,
            remote_service_fallback_to_local: None,
            remote_service_verify_profile: None,
            remote_service_auth_configured: None,
            remote_service_status: None,
            remote_service_verification: None,
        }))
    }
}

fn remote_probe_from_config(config: &FileConfig) -> Result<RemoteProbeConfig, String> {
    let embeddings = config
        .embeddings
        .as_ref()
        .ok_or_else(|| "missing [embeddings] config".to_string())?;
    if embeddings.enabled == Some(false) {
        return Err("embeddings.enabled is false in config".to_string());
    }
    let local = embeddings
        .local
        .as_ref()
        .ok_or_else(|| "missing [embeddings.local] config".to_string())?;
    let remote = local
        .remote_service
        .as_ref()
        .ok_or_else(|| "missing [embeddings.local.remote_service] config".to_string())?;
    let model = embeddings
        .model
        .clone()
        .unwrap_or_else(|| MINILM_CANONICAL_NAME.to_string());
    let dimensions = embeddings.dimensions.unwrap_or(MINILM_DIMENSIONS);
    let auth_token = resolve_secret(
        remote.auth_token_env.as_deref(),
        remote.auth_token_env_file.as_deref(),
    )?;

    Ok(RemoteProbeConfig {
        base_url: remote.base_url.clone(),
        auth_token,
        timeout_secs: remote.timeout_secs,
        verify_profile: remote.verify_profile.unwrap_or(true),
        fallback_to_local: remote.fallback_to_local.unwrap_or(false),
        model,
        dimensions,
        embedding_profile: MINILM_PROFILE.to_string(),
    })
}

fn remote_rerank_probe_from_config(config: &FileConfig) -> Result<RemoteRerankProbeConfig, String> {
    let reranking = config
        .reranking
        .as_ref()
        .ok_or_else(|| "missing [reranking] config".to_string())?;
    if reranking.enabled == Some(false) {
        return Err("reranking.enabled is false in config".to_string());
    }
    let local = reranking
        .local
        .as_ref()
        .ok_or_else(|| "missing [reranking.local] config".to_string())?;
    let remote = local
        .remote_service
        .as_ref()
        .ok_or_else(|| "missing [reranking.local.remote_service] config".to_string())?;
    let auth_token = resolve_secret(
        remote.auth_token_env.as_deref(),
        remote.auth_token_env_file.as_deref(),
    )?;

    Ok(RemoteRerankProbeConfig {
        base_url: remote.base_url.clone(),
        auth_token,
        timeout_secs: remote.timeout_secs,
        verify_profile: remote.verify_profile.unwrap_or(true),
        fallback_to_local: remote.fallback_to_local.unwrap_or(false),
        model: reranking
            .model
            .clone()
            .unwrap_or_else(|| RERANKER_CANONICAL_NAME.to_string()),
        reranker_profile: RERANKER_PROFILE.to_string(),
    })
}

fn verify_remote_probe(probe: &RemoteProbeConfig) -> RemoteVerifyOutput {
    match fetch_remote_status(probe) {
        Ok(status) => {
            let accepted_models = [
                MINILM_CANONICAL_NAME,
                "sentence-transformers/all-MiniLM-L6-v2",
                "all-MiniLM-L6-v2",
            ];
            let profile_match = accepted_models.contains(&status.model.as_str())
                && status.dimensions == probe.dimensions
                && status.embedding_profile == probe.embedding_profile;
            RemoteVerifyOutput {
                ok: profile_match,
                profile_match,
                remote_service_base_url: probe.base_url.clone(),
                remote_service_auth_configured: probe.auth_token.is_some(),
                expected_model: probe.model.clone(),
                expected_dimensions: probe.dimensions,
                expected_embedding_profile: probe.embedding_profile.clone(),
                remote_model: Some(status.model),
                remote_dimensions: Some(status.dimensions),
                remote_embedding_profile: Some(status.embedding_profile),
                remote_execution_mode: status.execution_mode,
                error: if profile_match {
                    None
                } else {
                    Some(
                        "remote embedding profile does not match expected FeMind MiniLM profile"
                            .to_string(),
                    )
                },
            }
        }
        Err(error) => RemoteVerifyOutput {
            ok: false,
            profile_match: false,
            remote_service_base_url: probe.base_url.clone(),
            remote_service_auth_configured: probe.auth_token.is_some(),
            expected_model: probe.model.clone(),
            expected_dimensions: probe.dimensions,
            expected_embedding_profile: probe.embedding_profile.clone(),
            remote_model: None,
            remote_dimensions: None,
            remote_embedding_profile: None,
            remote_execution_mode: None,
            error: Some(error),
        },
    }
}

fn verify_remote_rerank_probe(probe: &RemoteRerankProbeConfig) -> RemoteRerankVerifyOutput {
    match fetch_remote_rerank_status(probe) {
        Ok(status) => {
            let accepted_models = [
                RERANKER_CANONICAL_NAME,
                "cross-encoder/ms-marco-MiniLM-L6-v2",
                "minilm-reranker",
            ];
            let profile_match = accepted_models.contains(&status.model.as_str())
                && status.reranker_profile == probe.reranker_profile;
            RemoteRerankVerifyOutput {
                ok: profile_match,
                profile_match,
                remote_service_base_url: probe.base_url.clone(),
                remote_service_auth_configured: probe.auth_token.is_some(),
                expected_model: probe.model.clone(),
                expected_reranker_profile: probe.reranker_profile.clone(),
                remote_model: Some(status.model),
                remote_reranker_profile: Some(status.reranker_profile),
                remote_execution_mode: status.execution_mode,
                error: if profile_match {
                    None
                } else {
                    Some("remote reranker profile does not match expected FeMind MiniLM reranker profile".to_string())
                },
            }
        }
        Err(error) => RemoteRerankVerifyOutput {
            ok: false,
            profile_match: false,
            remote_service_base_url: probe.base_url.clone(),
            remote_service_auth_configured: probe.auth_token.is_some(),
            expected_model: probe.model.clone(),
            expected_reranker_profile: probe.reranker_profile.clone(),
            remote_model: None,
            remote_reranker_profile: None,
            remote_execution_mode: None,
            error: Some(error),
        },
    }
}

fn fetch_remote_status(probe: &RemoteProbeConfig) -> Result<RemoteProbeStatus, String> {
    let url = format!("{}/status", probe.base_url.trim_end_matches('/'));
    let mut builder = ureq::AgentBuilder::new();
    if let Some(timeout_secs) = probe.timeout_secs {
        builder = builder.timeout(Duration::from_secs(timeout_secs));
    }
    let agent = builder.build();
    let mut request = agent.get(&url);
    if let Some(token) = probe.auth_token.as_deref() {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request
        .call()
        .map_err(|error| format!("remote status request failed: {error}"))?;
    response
        .into_json::<RemoteProbeStatus>()
        .map_err(|error| format!("remote status parse failed: {error}"))
}

fn fetch_remote_rerank_status(
    probe: &RemoteRerankProbeConfig,
) -> Result<RemoteRerankProbeStatus, String> {
    let url = format!("{}/status", probe.base_url.trim_end_matches('/'));
    let mut builder = ureq::AgentBuilder::new();
    if let Some(timeout_secs) = probe.timeout_secs {
        builder = builder.timeout(Duration::from_secs(timeout_secs));
    }
    let agent = builder.build();
    let mut request = agent.get(&url);
    if let Some(token) = probe.auth_token.as_deref() {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request
        .call()
        .map_err(|error| format!("remote rerank status request failed: {error}"))?;
    response
        .into_json::<RemoteRerankProbeStatus>()
        .map_err(|error| format!("remote rerank status parse failed: {error}"))
}

fn load_config(path: &PathBuf) -> Result<FileConfig, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read config '{}': {error}", path.display()))?;
    toml::from_str::<FileConfig>(&text)
        .map_err(|error| format!("failed to parse config '{}': {error}", path.display()))
}

fn resolve_secret(
    env_name: Option<&str>,
    env_file: Option<&std::path::Path>,
) -> Result<Option<String>, String> {
    if let Some(env_name) = env_name
        && let Ok(value) = std::env::var(env_name)
        && !value.trim().is_empty()
    {
        return Ok(Some(value));
    }
    if let (Some(env_name), Some(env_file)) = (env_name, env_file) {
        return read_env_file_value(env_file, env_name).map(Some);
    }
    Ok(None)
}

fn read_env_file_value(path: &std::path::Path, key: &str) -> Result<String, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read env file '{}': {error}", path.display()))?;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((candidate_key, candidate_value)) = line.split_once('=') else {
            continue;
        };
        if candidate_key.trim() == key {
            return Ok(unquote(candidate_value.trim()).to_string());
        }
    }
    Err(format!(
        "env file '{}' does not define {}",
        path.display(),
        key
    ))
}

fn unquote(value: &str) -> &str {
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        &value[1..value.len() - 1]
    } else {
        value
    }
}

fn required_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, String> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_device(value: &str) -> Result<LocalEmbeddingDevice, String> {
    match value {
        "auto" => Ok(LocalEmbeddingDevice::Auto),
        "cpu" => Ok(LocalEmbeddingDevice::Cpu),
        "cuda" => Ok(LocalEmbeddingDevice::Cuda),
        other => Err(format!(
            "invalid --device value '{other}'; expected auto, cpu, or cuda"
        )),
    }
}

fn parse_format(value: &str) -> Result<OutputFormat, String> {
    match value {
        "human" => Ok(OutputFormat::Human),
        "json" => Ok(OutputFormat::Json),
        other => Err(format!(
            "invalid --format value '{other}'; expected human or json"
        )),
    }
}

fn print_output<T: serde::Serialize + std::fmt::Debug>(
    value: &T,
    format: OutputFormat,
) -> Result<(), String> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(value)
                .map_err(|error| format!("failed to serialize json output: {error}"))?;
            println!("{json}");
        }
        OutputFormat::Human => {
            let json = serde_json::to_value(value)
                .map_err(|error| format!("failed to serialize output: {error}"))?;
            if let Some(object) = json.as_object() {
                for (key, value) in object {
                    if !value.is_null() {
                        println!("{key}: {}", human_value(value));
                    }
                }
            } else {
                println!("{json}");
            }
        }
    }
    Ok(())
}

fn human_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        _ => value.to_string(),
    }
}

fn print_help() {
    println!("femind-embed-service");
    println!();
    println!("Commands:");
    println!("  serve           Run the embedding service");
    println!("  status          Show embedding runtime status from config");
    println!("  verify-remote   Verify a configured remote embedding service");
    println!("  verify-remote-reranker   Verify a configured remote reranker service");
    println!();
    println!("Run `femind-embed-service <command> --help` for command-specific options.");
}

fn print_command_help(command: &str) {
    match command {
        "serve" => print_serve_help(),
        "status" => {
            println!("femind-embed-service status --config <path> [--format human|json]");
        }
        "verify-remote" => {
            println!("femind-embed-service verify-remote --config <path> [--format human|json]");
        }
        "verify-remote-reranker" => {
            println!(
                "femind-embed-service verify-remote-reranker --config <path> [--format human|json]"
            );
        }
        _ => print_help(),
    }
}

fn print_serve_help() {
    println!("femind-embed-service serve");
    println!();
    println!("Options:");
    println!("  --config <path>             Load service settings from TOML");
    println!("  --host <addr>               Bind host (default: 127.0.0.1)");
    println!("  --port <port>               Bind port (default: 8899)");
    println!("  --prefix <path>             Path prefix (default: /embed)");
    println!("  --device <mode>             Runtime device: auto, cpu, cuda (default: auto)");
    println!("  --cuda-ordinal <n>          CUDA device ordinal when --device=cuda (default: 0)");
    println!("  --auth-token-env <name>     Resolve bearer token from environment");
    println!("  --auth-token-env-file <p>   Resolve bearer token from env file");
    println!("  --request-timeout-secs <n>  Max seconds for one embed request");
    println!("  --max-batch-texts <n>       Max texts accepted in one embed request");
    println!("  --max-batch-documents <n>   Max documents accepted in one rerank request");
}
