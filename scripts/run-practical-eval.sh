#!/bin/zsh
set -euo pipefail

MODE="${FEMIND_EVAL_MODE:-retrieval}"
VECTOR_MODE="${FEMIND_VECTOR_MODE:-exact}"
SCENARIOS="${FEMIND_EVAL_SCENARIOS:-eval/practical/scenarios.json}"
FEATURES="${FEMIND_EVAL_FEATURES:-api-embeddings,api-llm,ann}"
EXTRACT_BACKEND="${FEMIND_EXTRACT_BACKEND:-api}"
EXTRACT_MODEL="${FEMIND_EXTRACT_MODEL:-openai/gpt-oss-120b}"
KEY_CMD="${FEMIND_EVAL_KEY_CMD:-security find-generic-password -w -s 'DeepInfra API Key' -a deepinfra 2>/dev/null}"
SUMMARY="${FEMIND_EVAL_SUMMARY:-target/practical-eval/${MODE}-${VECTOR_MODE}.json}"

exec cargo run --example practical_eval --features "${FEATURES}" -- \
  --scenarios "${SCENARIOS}" \
  --mode "${MODE}" \
  --vector-mode "${VECTOR_MODE}" \
  --extract-backend "${EXTRACT_BACKEND}" \
  --extraction-model "${EXTRACT_MODEL}" \
  --key-cmd "${KEY_CMD}" \
  --summary "${SUMMARY}" \
  "$@"
