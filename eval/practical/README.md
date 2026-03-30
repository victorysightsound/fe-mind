# Practical Eval Set

This directory holds a small hand-curated evaluation set for real-world
`femind` validation.

It is intentionally small enough for manual review.

## Goals

- validate real extraction behavior
- validate retrieval quality on realistic working-memory workflows
- catch stale-fact and supersession failures early
- provide a repeatable regression set without needing a full benchmark run

## File Format

`scenarios.json` contains an array of scenarios.

Each scenario includes:

- `id`
- `title`
- `category`
- `goal`
- `records`
- optional `records[].key` for explicit graph-linked scenarios
- `records[].memory_type` (`episodic`, `semantic`, or `procedural`)
- optional `records[].metadata`, including trust/provenance markers such as
  `source_trust`, `source_kind`, `source_verification`, and
  `content_sensitivity` / `content_secret_class`
- optional `records[].valid_from` / `records[].valid_until` (RFC3339)
- optional `relations` for explicit source-record graph links
- `relations[].from` / `relations[].to` refer to `records[].key`
- `relations[].relation` uses the stored relation name such as `superseded_by` or `conflicts_with`
- optional `graph_depth` to request graph-assisted retrieval for the scenario
- `retrieval_checks`
- optional `retrieval_checks[].required_fragments`
- optional `retrieval_checks[].forbidden_fragments`
- optional `retrieval_checks[].min_observed_hits`
- optional `retrieval_checks[].graph_depth` to override scenario/default graph depth
- optional `extraction_checks`
- optional `abstention_checks`
- optional `abstention_checks[].graph_depth`
- optional `review_checks`
- optional `review_checks[].min_pending_items`
- optional `review_checks[].max_pending_items`
- optional `review_checks[].required_tags`
- optional `review_checks[].required_fragments`
- optional `review_checks[].forbidden_fragments`

Useful review metadata keys for scenario records:

- `review_required`
- `review_status`
- `review_severity`
- `review_reason`
- `review_tags`
- `review_note`
- `review_scope`
- `review_policy_class`
- `review_reviewer`
- `review_expires_at`

## Review Workflow

1. ingest the scenario records using the selected provider path
2. run the retrieval checks
3. compare the actual answer to the expected answer
4. note whether the failure is extraction, storage, ranking, or abstention

## Repeatable Runner

The curated set is wired to the `practical_eval` example.

Standard local run:

```bash
scripts/run-practical-eval.sh
```

Equivalent direct command:

```bash
cargo run --example practical_eval --features local-embeddings,remote-embeddings,reranking,remote-reranking,api-embeddings,api-reranking,api-llm,cli-llm,ann -- \
  --scenarios eval/practical/scenarios.json \
  --mode retrieval \
  --vector-mode exact \
  --embedding-runtime local-cpu \
  --rerank-runtime off \
  --summary target/practical-eval/retrieval-exact.json
```

Useful variants:

```bash
# Retrieval only
cargo run --example practical_eval --features local-embeddings,remote-embeddings,reranking,remote-reranking,api-embeddings,api-reranking,api-llm,cli-llm,ann -- \
  --mode retrieval --vector-mode exact

# Extraction only
cargo run --example practical_eval --features local-embeddings,remote-embeddings,reranking,remote-reranking,api-embeddings,api-reranking,api-llm,cli-llm,ann -- \
  --mode extraction --vector-mode exact

# ANN retrieval path
cargo run --example practical_eval --features local-embeddings,remote-embeddings,reranking,remote-reranking,api-embeddings,api-reranking,api-llm,cli-llm,ann -- \
  --mode retrieval --vector-mode ann

# Windows GPU remote path with local fallback
FEMIND_EMBED_RUNTIME=remote-fallback \
FEMIND_RERANK_RUNTIME=remote-fallback \
scripts/run-practical-eval.sh

# Include raw keyword/vector/hybrid traces for failed retrieval checks
FEMIND_EVAL_EXPLAIN_FAILURES=1 \
scripts/run-practical-eval.sh
```

The summary artifact now includes:

- per-check routed plans
- per-check retrieval criteria diagnostics
- per-check review-queue diagnostics when present
- review-item `updated_at`, `expires_at`, and `note` when review checks are present
- review-item `scope`, `policy_class`, and `reviewer` when scoped review policy is present
- pass-rate breakdowns by:
  - check type
  - scenario category
  - query intent
  - routed mode
  - temporal policy
  - state/conflict policy
  - graph depth
  - composed-answer confidence and abstention rationale when available

The graph-connected scenario now relies on routed graph depth rather than a
scenario-level forced graph override, so the practical set checks that the
engine can infer when multi-hop expansion is needed.

## Scope

This is the primary real-world validation set for `femind`.

Current local baseline:

- retrieval-only `exact` mode is the standard regression path
- latest fully green summary: `target/practical-eval/retrieval-exact.json`
- the practical set now includes explicit linked supersession/history,
  aggregation, graph-connected, provenance/abstention, and
  trust/procedural-safety, provenance/secret-guardrail, and
  review-policy-transition scenarios
- reranker-aware remote-fallback regression is currently green at `25/25`
- reranker-aware remote-fallback ANN regression is also green at `25/25`
- latest ANN summary: `target/practical-eval/retrieval-ann.json`

LongMemEval and MemoryAgentBench remain useful, but only as secondary
comparison or regression tools after this set is behaving well.
