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
5. Provider parity checks between the OpenAI-compatible API path and local CLI callbacks

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
scripts/run-practical-eval.sh
```

Default standard path:

- `retrieval` mode
- `exact` vector mode
- `api` extraction backend
- DeepInfra key resolution through Keychain
- summary output at `target/practical-eval/retrieval-exact.json`

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

As of 2026-03-26:

- local non-LLM verification is complete
- ANN/exact/off runtime behavior is implemented and tested
- extraction-only practical validation is directionally correct
- extraction-only practical validation with Codex CLI `gpt-5.4-mini` passes `4/4`
- retrieval-only practical validation in `exact` mode passes `9/9`
- retrieval-only practical validation in `ann` mode passes `9/9`
- broader live-usage validation from actual project docs passes `11/11`
- the standard local live-validation path is `scripts/run-practical-eval.sh`
- practical real-world eval design is defined in `PRACTICAL_EVAL.md`
