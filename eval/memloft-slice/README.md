# Memloft-Derived Validation Slice

This corpus is a real-data validation layer for `femind`.

It is built from technical project memories already stored in the live memloft
database, not from synthetic hand-written notes. Each record source points back
to a concrete memloft row ID such as `memloft:27162`.

## Purpose

This slice exists to test `femind` against a naturally accumulated corpus of:

- current versus superseded project state
- maintainer decisions and release sequencing
- provider defaults and fallback policy
- benchmark limits versus real-world validation policy
- operational memloft maintenance knowledge
- unsupported exact-detail questions that should abstain

That makes it a better bridge between the hand-authored `eval/live-library/`
corpus and larger benchmark-oriented work.

## Scope

This slice is intentionally technical-only.

It excludes personal and unrelated business memories and focuses on real
software-maintenance content already present in memloft.

Current size:

- 14 scenarios
- 70 checks

## Run

```bash
scripts/run-memloft-slice.sh
```

Default behavior:

- mode: `all`
- vector mode: `exact`
- scenarios: `eval/memloft-slice/scenarios.json`
- features: `api-embeddings,api-llm,ann`
- extraction backend: `api`
- extraction model: `openai/gpt-oss-120b`
- summary output: `target/memloft-slice/memloft-slice-exact.json`

To compare ANN:

```bash
FEMIND_VECTOR_MODE=ann scripts/run-memloft-slice.sh
```

## Notes

- This corpus uses the existing `practical_eval` runner and summary format.
- The expected answers are concise summaries of the memloft records, not a
  separate source of truth.
- Unsupported low-level details such as secret material, raw hashes, exact
  request headers, and exact cache metadata should abstain.
