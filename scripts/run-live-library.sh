#!/bin/zsh
set -euo pipefail

MODE="${FEMIND_EVAL_MODE:-all}"
VECTOR_MODE="${FEMIND_VECTOR_MODE:-exact}"
GRAPH_DEPTH="${FEMIND_GRAPH_DEPTH:-0}"
SCENARIOS="${FEMIND_EVAL_SCENARIOS:-eval/live-library/scenarios.json}"
FEATURES="${FEMIND_EVAL_FEATURES:-api-embeddings,api-llm,ann}"
EXTRACT_BACKEND="${FEMIND_EXTRACT_BACKEND:-api}"
EXTRACT_MODEL="${FEMIND_EXTRACT_MODEL:-openai/gpt-oss-120b}"
RETRIEVAL_INGEST="${FEMIND_RETRIEVAL_INGEST:-records}"
KEY_CMD="${FEMIND_EVAL_KEY_CMD:-security find-generic-password -w -s 'DeepInfra API Key' -a deepinfra 2>/dev/null}"
SUMMARY="${FEMIND_EVAL_SUMMARY:-target/live-library/live-library-${VECTOR_MODE}.json}"

exec cargo run --example practical_eval --features "${FEATURES}" -- \
  --scenarios "${SCENARIOS}" \
  --mode "${MODE}" \
  --vector-mode "${VECTOR_MODE}" \
  --graph-depth "${GRAPH_DEPTH}" \
  --extract-backend "${EXTRACT_BACKEND}" \
  --extraction-model "${EXTRACT_MODEL}" \
  --retrieval-ingest "${RETRIEVAL_INGEST}" \
  --key-cmd "${KEY_CMD}" \
  --summary "${SUMMARY}" \
  "$@"
