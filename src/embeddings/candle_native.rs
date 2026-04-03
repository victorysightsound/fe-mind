//! CandleNativeBackend: all-MiniLM-L6-v2 via BERT
//!
//! Only compiled when `local-embeddings` feature is enabled.
//! Provides 384-dimensional embeddings. WASM-compatible (standard BERT).

#[cfg(feature = "local-embeddings")]
mod inner {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config};
    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};
    use std::path::Path;
    use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

    use crate::embeddings::EmbeddingBackend;
    use crate::error::{FemindError, Result};

    const MODEL_REPO: &str = crate::embeddings::MINILM_MODEL_REPO;
    const DIMENSIONS: usize = crate::embeddings::MINILM_DIMENSIONS;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LocalEmbeddingDevice {
        Auto,
        Cpu,
        Cuda,
    }

    /// Embedding backend using all-MiniLM-L6-v2 via candle's BERT.
    ///
    /// Auto-downloads model files (~80MB) from HuggingFace on first use.
    /// Cached at the HuggingFace cache directory (`~/.cache/huggingface/hub/`).
    ///
    /// Same model works native and WASM — standard BERT architecture.
    /// Same vectors as the DeepInfra API endpoint — interchangeable.
    pub struct CandleNativeBackend {
        model: BertModel,
        tokenizer: Tokenizer,
        device: Device,
        device_label: String,
        dimensions_override: Option<usize>,
    }

    impl CandleNativeBackend {
        /// Create with auto-downloaded model from HuggingFace.
        pub fn new() -> Result<Self> {
            Self::new_with_device(LocalEmbeddingDevice::Cpu, 0)
        }

        /// Create with explicit runtime device selection.
        pub fn new_with_device(
            device_mode: LocalEmbeddingDevice,
            cuda_ordinal: usize,
        ) -> Result<Self> {
            let device = select_device(device_mode, cuda_ordinal)?;
            let device_label = describe_device(&device);

            let repo =
                Repo::with_revision(MODEL_REPO.to_string(), RepoType::Model, "main".to_string());
            let api =
                Api::new().map_err(|e| FemindError::Embedding(format!("HF API init: {e}")))?;
            let api = api.repo(repo);

            let config_path = api
                .get("config.json")
                .map_err(|e| FemindError::ModelNotAvailable(format!("config.json: {e}")))?;
            let tokenizer_path = api
                .get("tokenizer.json")
                .map_err(|e| FemindError::ModelNotAvailable(format!("tokenizer.json: {e}")))?;
            let weights_path = api
                .get("model.safetensors")
                .map_err(|e| FemindError::ModelNotAvailable(format!("model.safetensors: {e}")))?;

            Self::from_paths(
                &config_path,
                &tokenizer_path,
                &weights_path,
                device,
                device_label,
            )
        }

        /// Create from pre-downloaded model files.
        ///
        /// Expected files in `model_dir`: `config.json`, `tokenizer.json`, `model.safetensors`
        pub fn from_path(model_dir: impl AsRef<Path>) -> Result<Self> {
            let dir = model_dir.as_ref();
            let config_path = dir.join("config.json");
            let tokenizer_path = dir.join("tokenizer.json");
            let weights_path = dir.join("model.safetensors");

            for path in [&config_path, &tokenizer_path, &weights_path] {
                if !path.exists() {
                    return Err(FemindError::ModelNotAvailable(format!(
                        "missing model file: {}",
                        path.display()
                    )));
                }
            }

            Self::from_paths(
                &config_path,
                &tokenizer_path,
                &weights_path,
                Device::Cpu,
                "cpu".to_string(),
            )
        }

        /// Set Matryoshka dimension override (truncate vectors after embedding).
        pub fn with_dimensions_override(mut self, dims: usize) -> Self {
            self.dimensions_override = Some(dims);
            self
        }

        fn from_paths(
            config_path: &Path,
            tokenizer_path: &Path,
            weights_path: &Path,
            device: Device,
            device_label: String,
        ) -> Result<Self> {
            let config_str = std::fs::read_to_string(config_path)?;
            let config: Config = serde_json::from_str(&config_str)
                .map_err(|e| FemindError::Embedding(format!("config parse: {e}")))?;

            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| FemindError::Embedding(format!("tokenizer load: {e}")))?;

            // Safety: mmap is the standard way to load large model files.
            // The file must not be modified while mapped.
            #[allow(unsafe_code)]
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| FemindError::Embedding(format!("weights load: {e}")))?
            };

            let model = BertModel::load(vb, &config)
                .map_err(|e| FemindError::Embedding(format!("model load: {e}")))?;

            Ok(Self {
                model,
                tokenizer,
                device,
                device_label,
                dimensions_override: None,
            })
        }

        pub fn device_label(&self) -> &str {
            &self.device_label
        }

        pub fn execution_mode(&self) -> &'static str {
            execution_mode_from_label(&self.device_label)
        }

        /// Mean pooling with attention mask weighting.
        fn mean_pool(
            hidden_states: &Tensor,
            attention_mask: &Tensor,
        ) -> std::result::Result<Tensor, candle_core::Error> {
            let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
            let sum_embeddings = hidden_states.broadcast_mul(&mask)?.sum(1)?;
            let sum_mask = mask.sum(1)?;
            sum_embeddings.broadcast_div(&sum_mask)
        }

        /// L2 normalization.
        fn normalize(v: &Tensor) -> std::result::Result<Tensor, candle_core::Error> {
            v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
        }

        /// Process a single sub-batch through the model.
        fn embed_sub_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            let mut tokenizer = self.tokenizer.clone();
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));

            let encodings = tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| FemindError::Embedding(format!("tokenize: {e}")))?;

            let token_ids: Vec<Tensor> = encodings
                .iter()
                .map(|enc| {
                    Tensor::new(enc.get_ids(), &self.device)
                        .map_err(|e| FemindError::Embedding(format!("tensor: {e}")))
                })
                .collect::<Result<Vec<_>>>()?;

            let attention_masks: Vec<Tensor> = encodings
                .iter()
                .map(|enc| {
                    Tensor::new(enc.get_attention_mask(), &self.device)
                        .map_err(|e| FemindError::Embedding(format!("mask tensor: {e}")))
                })
                .collect::<Result<Vec<_>>>()?;

            let token_ids = Tensor::stack(&token_ids, 0)
                .map_err(|e| FemindError::Embedding(format!("stack ids: {e}")))?;
            let attention_mask = Tensor::stack(&attention_masks, 0)
                .map_err(|e| FemindError::Embedding(format!("stack masks: {e}")))?;

            // BERT requires token_type_ids (zeros for single-segment input)
            let token_type_ids = token_ids
                .zeros_like()
                .map_err(|e| FemindError::Embedding(format!("token_type_ids: {e}")))?;

            let hidden_states = self
                .model
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))
                .map_err(|e| FemindError::Embedding(format!("forward: {e}")))?;

            let pooled = Self::mean_pool(&hidden_states, &attention_mask)
                .map_err(|e| FemindError::Embedding(format!("pool: {e}")))?;

            let normalized = Self::normalize(&pooled)
                .map_err(|e| FemindError::Embedding(format!("normalize: {e}")))?;

            let batch_size = texts.len();
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let mut vec: Vec<f32> = normalized
                    .get(i)
                    .map_err(|e| FemindError::Embedding(format!("get vec: {e}")))?
                    .to_vec1::<f32>()
                    .map_err(|e| FemindError::Embedding(format!("to_vec1: {e}")))?;

                if let Some(dims) = self.dimensions_override {
                    if dims < vec.len() {
                        vec.truncate(dims);
                        crate::embeddings::pooling::normalize_l2_inplace(&mut vec);
                    }
                }

                results.push(vec);
            }

            Ok(results)
        }
    }

    impl EmbeddingBackend for CandleNativeBackend {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            let results = self.embed_batch(&[text])?;
            results
                .into_iter()
                .next()
                .ok_or_else(|| FemindError::Embedding("empty batch result".into()))
        }

        // all-MiniLM-L6-v2 is a symmetric bi-encoder —
        // queries and documents are encoded identically, no prefix needed.
        // embed_query() falls through to the default (delegates to embed()).

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            const SUB_BATCH_SIZE: usize = 16;

            let mut all_results = Vec::with_capacity(texts.len());

            for chunk in texts.chunks(SUB_BATCH_SIZE) {
                let sub_results = self.embed_sub_batch(chunk)?;
                all_results.extend(sub_results);
            }

            Ok(all_results)
        }

        fn dimensions(&self) -> usize {
            self.dimensions_override.unwrap_or(DIMENSIONS)
        }

        fn is_available(&self) -> bool {
            true
        }

        fn model_name(&self) -> &str {
            crate::embeddings::MINILM_CANONICAL_NAME
        }

        fn embedding_profile(&self) -> String {
            crate::embeddings::embedding_profile_for_model(self.model_name(), self.dimensions())
        }

        fn compatibility_model_names(&self) -> Vec<String> {
            crate::embeddings::compatibility_model_names(self.model_name())
        }
    }

    pub(crate) fn select_device(
        device_mode: LocalEmbeddingDevice,
        cuda_ordinal: usize,
    ) -> Result<Device> {
        match device_mode {
            LocalEmbeddingDevice::Cpu => Ok(Device::Cpu),
            LocalEmbeddingDevice::Auto => select_cuda_device(cuda_ordinal)
                .map_err(|error| {
                    FemindError::Embedding(format!(
                        "failed to probe CUDA device {cuda_ordinal}: {error}"
                    ))
                })
                .or(Ok(Device::Cpu)),
            LocalEmbeddingDevice::Cuda => select_cuda_device(cuda_ordinal).map_err(|error| {
                FemindError::ModelNotAvailable(cuda_request_error(cuda_ordinal, &error.to_string()))
            }),
        }
    }

    #[cfg(feature = "cuda")]
    fn select_cuda_device(ordinal: usize) -> std::result::Result<Device, candle_core::Error> {
        Device::new_cuda(ordinal)
    }

    #[cfg(not(feature = "cuda"))]
    fn select_cuda_device(_ordinal: usize) -> std::result::Result<Device, String> {
        Err("this build was compiled without the `cuda` feature".to_string())
    }

    fn cuda_request_error(cuda_ordinal: usize, error: &str) -> String {
        format!(
            "CUDA device {cuda_ordinal} requested but unavailable: {error}. Rebuild with `--features cuda` or use `--device cpu`."
        )
    }

    pub(crate) fn describe_device(device: &Device) -> String {
        format!("{device:?}").to_ascii_lowercase()
    }

    pub(crate) fn execution_mode_from_label(device_label: &str) -> &'static str {
        if device_label.contains("cuda") {
            "local-gpu"
        } else {
            "local-cpu"
        }
    }

    #[cfg(test)]
    mod tests {
        use super::execution_mode_from_label;

        #[cfg(not(feature = "cuda"))]
        use super::select_device;
        #[cfg(not(feature = "cuda"))]
        use crate::embeddings::LocalEmbeddingDevice;

        #[test]
        fn cpu_like_labels_report_local_cpu() {
            assert_eq!(execution_mode_from_label("cpu"), "local-cpu");
            assert_eq!(execution_mode_from_label("metal"), "local-cpu");
        }

        #[test]
        fn cuda_like_labels_report_local_gpu() {
            assert_eq!(execution_mode_from_label("cuda"), "local-gpu");
            assert_eq!(execution_mode_from_label("cuda:0"), "local-gpu");
            assert_eq!(execution_mode_from_label("device(cuda,0)"), "local-gpu");
        }

        #[cfg(not(feature = "cuda"))]
        #[test]
        fn explicit_cuda_request_reports_actionable_error() {
            let err = select_device(LocalEmbeddingDevice::Cuda, 0).unwrap_err();
            let message = err.to_string();
            assert!(message.contains("compiled without the `cuda` feature"));
            assert!(message.contains("--features cuda"));
            assert!(message.contains("--device cpu"));
        }
    }
}

#[cfg(feature = "local-embeddings")]
pub use inner::{CandleNativeBackend, LocalEmbeddingDevice};
#[cfg(feature = "local-embeddings")]
pub(crate) use inner::{describe_device, execution_mode_from_label, select_device};
