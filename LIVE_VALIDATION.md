# femind Live Validation Plan

## Purpose

This document defines the first approval-gated live validation pass for `femind`.
It is intentionally limited to real CLI/API model calls and benchmark paths that
were not exercised during the local non-LLM stabilization pass.

The practical validation set in `eval/practical/` is the primary real-world
check for this phase. The larger real-world library in `eval/live-library/`
is the next follow-up layer after the practical set stabilizes. Benchmark
datasets are secondary and should only be used after the real-world sets are
directionally strong.

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
- `graph_depth=0`
- `api` extraction backend
- `openai/gpt-oss-120b` extraction model
- `retrieval_ingest=records`
- DeepInfra key resolution through Keychain
- summary output at `target/practical-eval/retrieval-exact.json`

## Phase 3: Provider Comparison

Only after Phase 2 succeeds:

- compare CLI and API providers on the same small sample
- compare cost, latency, and extraction quality
- lock one default and one fallback per provider lane

## Phase 4: Larger Benchmark Approval Gate

Stop after the small-sample pass and report:

- provider used
- success/failure status
- extraction quality observations
- retrieval quality observations
- approximate latency/cost

Any LongMemEval or larger benchmark run requires a separate explicit approval.

## Phase 5: Larger Real-World Library

Before larger benchmark work, run the broader real-world library:

```bash
scripts/run-live-library.sh
```

Goals:

- verify retrieval quality on a larger realistic corpus
- compare exact and ANN behavior on the same broader set
- track quality and latency over time with stable summary metadata
- validate provider defaults on material that is larger than the curated
  practical sample but still easy to inspect by hand
- keep graph-specific validation separate from the simpler seeded-record path

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
- recommended API extraction default is `openai/gpt-oss-120b`
- recommended CLI extraction default is `gpt-5.4-mini`
- lower-cost CLI fallback is `gpt-5.1-codex-mini`
- extraction-only practical validation with Codex CLI `gpt-5.4-mini` passes `4/4`
- extraction-only practical validation with Codex CLI `gpt-5.1-codex-mini` passes `4/4`
- retrieval-only practical validation in `exact` mode passes `9/9`
- retrieval-only practical validation in `ann` mode passes `9/9`
- broader live-usage validation from actual project docs passes `11/11` across all four tested extraction models
- the larger real-world library under `eval/live-library/` passes `66/66` in both `all` + `exact` and `all` + `ann`
- the larger real-world library now covers 18 scenarios and 66 checks
- the memloft-derived real-data slice under `eval/memloft-slice/` is the current highest-value follow-up validation layer before benchmark work
- the memloft-derived real-data slice now covers 18 scenarios and 90 checks
- the memloft-derived real-data slice currently passes `90/90` in both `all` + `exact` and `all` + `ann`
- explicit graph-backed extraction retrieval is now available through
  `FEMIND_GRAPH_DEPTH=2` and `FEMIND_RETRIEVAL_INGEST=extraction`
- the first graph-backed run scores are `59/66` on the larger live library and
  `75/90` on the memloft-derived slice, so graph-expanded retrieval still needs
  targeted tuning before benchmark confirmation
- the standard local live-validation path is `scripts/run-practical-eval.sh`
- practical real-world eval design is defined in `PRACTICAL_EVAL.md`
