#[cfg(feature = "reranking")]
mod inner {
    use candle_core::{DType, IndexOp, Tensor};
    use candle_nn::{Module, VarBuilder, linear};
    use candle_transformers::models::bert::{BertModel, Config};
    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};
    use std::path::Path;
    use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
    use tokenizers::utils::truncation::TruncationParams;

    use crate::embeddings::{
        LocalEmbeddingDevice, describe_device, execution_mode_from_label, select_device,
    };
    use crate::error::{FemindError, Result};
    use crate::traits::{RerankCandidate, RerankerBackend, ScoredResult};

    const MODEL_REPO: &str = crate::reranking::RERANKER_MODEL_REPO;

    pub struct CandleReranker {
        model: BertModel,
        classifier: candle_nn::Linear,
        tokenizer: Tokenizer,
        _device: candle_core::Device,
        device_label: String,
    }

    impl CandleReranker {
        pub fn new() -> Result<Self> {
            Self::new_with_device(LocalEmbeddingDevice::Cpu, 0)
        }

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

            Self::from_paths(&config_path, &tokenizer_path, &weights_path, device, device_label)
        }

        pub fn from_path(model_dir: impl AsRef<Path>) -> Result<Self> {
            let dir = model_dir.as_ref();
            let config_path = dir.join("config.json");
            let tokenizer_path = dir.join("tokenizer.json");
            let weights_path = dir.join("model.safetensors");

            for path in [&config_path, &tokenizer_path, &weights_path] {
                if !path.exists() {
                    return Err(FemindError::ModelNotAvailable(format!(
                        "missing reranker model file: {}",
                        path.display()
                    )));
                }
            }

            Self::from_paths(
                &config_path,
                &tokenizer_path,
                &weights_path,
                candle_core::Device::Cpu,
                "cpu".to_string(),
            )
        }

        fn from_paths(
            config_path: &Path,
            tokenizer_path: &Path,
            weights_path: &Path,
            device: candle_core::Device,
            device_label: String,
        ) -> Result<Self> {
            let config_str = std::fs::read_to_string(config_path)?;
            let config: Config = serde_json::from_str(&config_str)
                .map_err(|e| FemindError::Embedding(format!("reranker config parse: {e}")))?;

            let tokenizer = Tokenizer::from_file(tokenizer_path)
                .map_err(|e| FemindError::Embedding(format!("reranker tokenizer load: {e}")))?;

            #[allow(unsafe_code)]
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .map_err(|e| FemindError::Embedding(format!("reranker weights load: {e}")))?
            };

            let model = BertModel::load(vb.clone(), &config)
                .map_err(|e| FemindError::Embedding(format!("reranker model load: {e}")))?;
            let classifier = linear(config.hidden_size, 1, vb.pp("classifier"))
                .map_err(|e| FemindError::Embedding(format!("reranker classifier load: {e}")))?;

            Ok(Self {
                model,
                classifier,
                tokenizer,
                _device: device,
                device_label,
            })
        }

        pub fn device_label(&self) -> &str {
            &self.device_label
        }

        pub fn execution_mode(&self) -> &'static str {
            execution_mode_from_label(&self.device_label)
        }

        pub fn rerank_scores(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
            if documents.is_empty() {
                return Ok(Vec::new());
            }

            let mut tokenizer = self.tokenizer.clone();
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: 512,
                    ..Default::default()
                }))
                .map_err(|e| FemindError::Embedding(format!("reranker truncation config: {e}")))?;

            let pair_inputs = documents
                .iter()
                .map(|document| (query, *document))
                .collect::<Vec<_>>();
            let encodings = tokenizer
                .encode_batch(pair_inputs, true)
                .map_err(|e| FemindError::Embedding(format!("reranker tokenize: {e}")))?;

            let token_ids: Vec<Tensor> = encodings
                .iter()
                .map(|enc| Tensor::new(enc.get_ids(), &self.model.device).map_err(map_tensor_err))
                .collect::<Result<Vec<_>>>()?;
            let token_type_ids: Vec<Tensor> = encodings
                .iter()
                .map(|enc| {
                    Tensor::new(enc.get_type_ids(), &self.model.device).map_err(map_tensor_err)
                })
                .collect::<Result<Vec<_>>>()?;
            let attention_masks: Vec<Tensor> = encodings
                .iter()
                .map(|enc| {
                    Tensor::new(enc.get_attention_mask(), &self.model.device)
                        .map_err(map_tensor_err)
                })
                .collect::<Result<Vec<_>>>()?;

            let token_ids =
                Tensor::stack(&token_ids, 0).map_err(|e| FemindError::Embedding(format!("reranker stack ids: {e}")))?;
            let token_type_ids = Tensor::stack(&token_type_ids, 0)
                .map_err(|e| FemindError::Embedding(format!("reranker stack type ids: {e}")))?;
            let attention_mask = Tensor::stack(&attention_masks, 0)
                .map_err(|e| FemindError::Embedding(format!("reranker stack masks: {e}")))?;

            let hidden_states = self
                .model
                .forward(&token_ids, &token_type_ids, Some(&attention_mask))
                .map_err(|e| FemindError::Embedding(format!("reranker forward: {e}")))?;
            let cls = hidden_states
                .i((.., 0, ..))
                .map_err(|e| FemindError::Embedding(format!("reranker cls slice: {e}")))?;
            let cls = cls
                .contiguous()
                .map_err(|e| FemindError::Embedding(format!("reranker cls contiguous: {e}")))?;
            let logits = self
                .classifier
                .forward(&cls)
                .map_err(|e| FemindError::Embedding(format!("reranker classifier: {e}")))?;
            let logits = logits
                .squeeze(1)
                .map_err(|e| FemindError::Embedding(format!("reranker squeeze: {e}")))?;
            let logits = logits
                .to_vec1::<f32>()
                .map_err(|e| FemindError::Embedding(format!("reranker logits: {e}")))?;

            Ok(logits.into_iter().map(sigmoid).collect())
        }
    }

    impl RerankerBackend for CandleReranker {
        fn rerank(&self, query: &str, candidates: Vec<RerankCandidate>) -> Result<Vec<ScoredResult>> {
            let documents = candidates
                .iter()
                .map(|candidate| candidate.text.as_str())
                .collect::<Vec<_>>();
            let scores = self.rerank_scores(query, &documents)?;

            let mut reranked = candidates
                .into_iter()
                .zip(scores)
                .map(|(candidate, score)| ScoredResult {
                    memory_id: candidate.memory_id,
                    score,
                    raw_score: score,
                    score_multiplier: 1.0,
                })
                .collect::<Vec<_>>();

            reranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(reranked)
        }
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn map_tensor_err(error: candle_core::Error) -> FemindError {
        FemindError::Embedding(format!("reranker tensor: {error}"))
    }
}

#[cfg(feature = "reranking")]
pub use inner::CandleReranker;
