#!/bin/zsh
set -euo pipefail

MODE="${FEMIND_EVAL_MODE:-retrieval}"
VECTOR_MODE="${FEMIND_VECTOR_MODE:-exact}"
GRAPH_DEPTH="${FEMIND_GRAPH_DEPTH:-0}"
SCENARIOS="${FEMIND_EVAL_SCENARIOS:-eval/practical/scenarios.json}"
FEATURES="${FEMIND_EVAL_FEATURES:-local-embeddings,remote-embeddings,reranking,remote-reranking,api-embeddings,api-reranking,api-llm,cli-llm,ann}"
BASE_URL="${FEMIND_API_BASE_URL:-https://api.deepinfra.com/v1/openai}"
API_KEY_ENV="${FEMIND_API_KEY_ENV:-FEMIND_API_KEY}"
KEY_CMD="${FEMIND_EVAL_KEY_CMD:-security find-generic-password -w -s 'DeepInfra API Key' -a deepinfra 2>/dev/null}"
EMBED_RUNTIME="${FEMIND_EMBED_RUNTIME:-local-cpu}"
EMBED_MODEL="${FEMIND_EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
EMBED_REMOTE_BASE_URL="${FEMIND_EMBED_REMOTE_BASE_URL:-http://127.0.0.1:18899/embed}"
EMBED_REMOTE_AUTH_ENV="${FEMIND_EMBED_REMOTE_AUTH_ENV:-FEMIND_REMOTE_EMBED_TOKEN}"
EMBED_REMOTE_TIMEOUT_SECS="${FEMIND_EMBED_REMOTE_TIMEOUT_SECS:-15}"
EXTRACT_BACKEND="${FEMIND_EXTRACT_BACKEND:-api}"
EXTRACT_MODEL="${FEMIND_EXTRACT_MODEL:-openai/gpt-oss-120b}"
RETRIEVAL_INGEST="${FEMIND_RETRIEVAL_INGEST:-records}"
RERANK_RUNTIME="${FEMIND_RERANK_RUNTIME:-off}"
RERANK_MODEL="${FEMIND_RERANK_MODEL:-local-minilm-reranker}"
RERANK_REMOTE_BASE_URL="${FEMIND_RERANK_REMOTE_BASE_URL:-http://127.0.0.1:18899/rerank}"
RERANK_REMOTE_AUTH_ENV="${FEMIND_RERANK_REMOTE_AUTH_ENV:-FEMIND_REMOTE_EMBED_TOKEN}"
RERANK_REMOTE_TIMEOUT_SECS="${FEMIND_RERANK_REMOTE_TIMEOUT_SECS:-15}"
RERANK_LIMIT="${FEMIND_RERANK_LIMIT:-20}"
SUMMARY="${FEMIND_EVAL_SUMMARY:-target/practical-eval/${MODE}-${VECTOR_MODE}.json}"

exec cargo run --example practical_eval --features "${FEATURES}" -- \
  --scenarios "${SCENARIOS}" \
  --mode "${MODE}" \
  --vector-mode "${VECTOR_MODE}" \
  --graph-depth "${GRAPH_DEPTH}" \
  --base-url "${BASE_URL}" \
  --api-key-env "${API_KEY_ENV}" \
  --embedding-runtime "${EMBED_RUNTIME}" \
  --embedding-model "${EMBED_MODEL}" \
  --embed-remote-base-url "${EMBED_REMOTE_BASE_URL}" \
  --embed-remote-auth-env "${EMBED_REMOTE_AUTH_ENV}" \
  --embed-remote-timeout-secs "${EMBED_REMOTE_TIMEOUT_SECS}" \
  --extract-backend "${EXTRACT_BACKEND}" \
  --extraction-model "${EXTRACT_MODEL}" \
  --retrieval-ingest "${RETRIEVAL_INGEST}" \
  --rerank-runtime "${RERANK_RUNTIME}" \
  --rerank-model "${RERANK_MODEL}" \
  --rerank-remote-base-url "${RERANK_REMOTE_BASE_URL}" \
  --rerank-remote-auth-env "${RERANK_REMOTE_AUTH_ENV}" \
  --rerank-remote-timeout-secs "${RERANK_REMOTE_TIMEOUT_SECS}" \
  --rerank-limit "${RERANK_LIMIT}" \
  --key-cmd "${KEY_CMD}" \
  --summary "${SUMMARY}" \
  "$@"
