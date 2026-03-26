# femind Practical Evaluation

## Purpose

This document defines the real-world validation layer for `femind`.
It exists to keep production readiness tied to practical memory behavior rather
than benchmark scores.

The practical eval set should answer one question first:

Can `femind` extract, store, retrieve, and update real working memory in ways
that are useful to an actual application or developer workflow?

## Validation Order

1. Practical eval scenarios
2. Small approved live validation pass
3. Fixes for real-world failures
4. Repeatable local regression checks
5. Secondary benchmark comparison if still useful

Benchmark datasets remain useful for regression and comparison, but they are
not the primary design target.

## Practical Eval Categories

The curated eval set should cover these categories:

1. Current vs superseded facts
   The system should prefer the latest accepted fact and avoid surfacing stale
   answers as current.

2. Preferences and decisions
   The system should preserve stable preferences and explicit decisions without
   losing the reason behind them.

3. Temporal and recency reasoning
   The system should answer questions about what changed, when it changed, and
   what is current now.

4. Distractor resistance
   The system should still retrieve the correct information when unrelated but
   semantically similar text is present.

5. Messy source extraction
   The system should extract useful facts from rough notes, meetings, and
   transcripts instead of only from clean synthetic prompts.

6. Abstention
   The system should avoid confident fabrication when the answer is missing.

## Eval Artifact Layout

The curated eval set lives under `eval/practical/`.

- `eval/practical/README.md`
  Maintainer notes and review workflow
- `eval/practical/scenarios.json`
  Curated practical scenarios and expected behavior

The larger real-world library lives under `eval/live-library/`.

- `eval/live-library/README.md`
  Maintainer notes for the broader real-world corpus
- `eval/live-library/scenarios.json`
  Larger project-like scenarios for follow-up live validation

## Review Standard

Every practical scenario should be easy to inspect by hand.

Each scenario should include:

- source records or session text
- the main retrieval questions
- expected current answers
- expected abstentions where relevant
- optional extraction expectations for messy source material

## Release Use

Before larger live runs or benchmark comparisons:

- run a small approved sample from `eval/practical/scenarios.json`
- inspect extraction quality manually
- inspect retrieval answers manually
- log failures by category

`femind` should be treated as production-ready only when the practical eval
set is directionally strong, repeatable, and free of obvious category failures.

## Repeatable Command

The primary live-validation entry point is:

```bash
scripts/run-practical-eval.sh
```

Default behavior:

- `FEMIND_EVAL_MODE=retrieval`
- `FEMIND_VECTOR_MODE=exact`
- `FEMIND_EXTRACT_BACKEND=api`
- `FEMIND_EXTRACT_MODEL=openai/gpt-oss-120b`
- summary output at `target/practical-eval/retrieval-exact.json`
- runtime key resolution through macOS Keychain unless overridden with `FEMIND_EVAL_KEY_CMD`

Equivalent direct command:

```bash
cargo run --example practical_eval --features api-embeddings,api-llm,ann -- \
  --scenarios eval/practical/scenarios.json \
  --mode retrieval \
  --vector-mode exact \
  --summary target/practical-eval/retrieval-exact.json
```

The example uses a runtime key command and does not require secrets to be
written into source files or shell history.

Extraction backend options:

- `api`
  Uses the OpenAI-compatible API callback. Recommended default:
  `openai/gpt-oss-120b`
- `codex-cli`
  Uses the local Codex CLI callback. Recommended default:
  `gpt-5.4-mini`
  Lower-cost fallback:
  `gpt-5.1-codex-mini`

## Current Practical Baseline

Current validated baseline:

- extraction-only practical eval with DeepInfra `openai/gpt-oss-120b` passes `4/4`
- extraction-only practical eval with Codex CLI `gpt-5.4-mini` passes `4/4`
- extraction-only practical eval with Codex CLI `gpt-5.1-codex-mini` passes `4/4`
- retrieval-only practical eval with `vector_mode=exact` currently passes `9/9`
- retrieval-only practical eval with `vector_mode=ann` currently passes `9/9`
- summary artifact: `target/practical-eval/retrieval-exact.json`
- broader live-usage sample from actual project docs currently passes `11/11` for all four tested extraction models
- live-usage summary artifact: `target/practical-eval/live-usage-exact.json`

This exact-mode practical run is the standard local regression check before
trying wider live usage samples or ANN comparisons.

## Larger Real-World Library

After the small practical set is stable, the next real-world check is:

```bash
scripts/run-live-library.sh
```

Default behavior:

- `all` mode
- `exact` vector mode
- `api` extraction backend
- `openai/gpt-oss-120b` extraction model
- summary output at `target/live-library/live-library-exact.json`

This larger library is the preferred follow-up step before any broader
benchmark sweep through RecallBench.

Current larger-library baseline:

- the live-library corpus currently includes 18 scenarios and 66 checks
- `all` + `exact` currently passes `66/66`
- `all` + `ann` currently passes `66/66`
- the exact and ANN results currently match on the full larger corpus
- summary artifacts now include stable run metadata for backend, model, vector
  mode, duration, pass counts, and pass rate
