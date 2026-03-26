# femind Live Validation Plan

## Purpose

This document defines the first approval-gated live validation pass for `femind`.
It is intentionally limited to real CLI/API model calls and benchmark paths that
were not exercised during the local non-LLM stabilization pass.

The practical validation set in `eval/practical/` is the primary real-world
check for this phase. Benchmark datasets are secondary and should only be used
after the practical eval set is directionally strong.

## Preconditions

- Local rename to `fe-mind` / `femind` is complete
- `cargo test` passes
- `cargo test --features full` passes
- `cargo clippy --all-targets --all-features -- -D warnings` passes
- User explicitly approves live CLI/API model usage

## What This Validation Covers

1. Real embedding generation through the configured embedding backend
2. Real LLM-assisted extraction through the configured CLI or API callback
3. End-to-end retrieval quality on a small approved sample before any large run
4. Failure behavior when the provider is unavailable or misconfigured

## What This Validation Does Not Cover

- crates.io publication
- GitHub repo/remote rename
- large benchmark sweeps without an explicit second approval

## Phase 1: Smoke Test

Run one approved provider path only:

- CLI path: `cli-llm`
- API path: `api-llm`

Goals:

- confirm credentials/tooling are configured correctly
- confirm a single extraction call succeeds
- confirm extracted facts can be stored and retrieved

## Phase 2: Small Real Sample

Use a very small approved sample set from `eval/practical/scenarios.json`.

Goals:

- verify extraction quality is directionally correct
- verify `store_with_extraction()` metrics are sensible
- verify retrieved context is coherent for follow-up questioning

Repeatable command surface:

```bash
cargo run --example practical_eval --features api-embeddings,api-llm,ann -- \
  --scenarios eval/practical/scenarios.json \
  --mode all \
  --vector-mode exact \
  --summary target/practical-eval/summary.json
```

## Phase 3: Provider Comparison

Only after Phase 2 succeeds:

- compare CLI and API providers on the same small sample
- compare cost, latency, and extraction quality

## Phase 4: Larger Benchmark Approval Gate

Stop after the small-sample pass and report:

- provider used
- success/failure status
- extraction quality observations
- retrieval quality observations
- approximate latency/cost

Any LongMemEval or larger benchmark run requires a separate explicit approval.

## Output Expectations

For each approved live run, record:

- provider and model
- feature flags used
- sample size
- extraction metrics
- retrieval observations
- any provider-specific failures

## Current State

As of 2026-03-25:

- local non-LLM verification is complete
- ANN/exact/off runtime behavior is implemented and tested
- live CLI/API validation has not been run yet
- practical real-world eval design is defined in `PRACTICAL_EVAL.md`
