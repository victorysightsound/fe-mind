# Live Library

This directory holds the larger real-world validation library for `femind`.

It sits between the small curated practical set and any benchmark-oriented
work. The goal is to stress the real memory behavior you care about with a
broader body of project-like material before spending time on leaderboard or
benchmark regression work.

## Purpose

Use this library to answer questions like:

- Does `femind` still retrieve the right current fact when the corpus grows?
- Does extraction stay useful on rough maintainer notes and mixed document
  styles?
- Do exact and ANN retrieval still behave the same on a larger realistic set?
- Do provider defaults stay competitive as the real-world sample grows?

## Scope

The live library is larger than `eval/practical/` and `eval/live-usage/`, but
it is still hand-reviewable. It is not meant to replace benchmark datasets.

This corpus should stay focused on:

- current versus superseded facts
- stable preferences and defaults
- procedural guidance and maintainer commands
- release and migration status
- extraction from rough notes
- abstention on unsupported or unknown details

Current size and baseline:

- 18 scenarios
- 66 total checks
- `all` + `exact`: `66/66`
- `all` + `ann`: `66/66`

## Standard Run

```bash
scripts/run-live-library.sh
```

Default behavior:

- `all` mode
- `exact` vector mode
- `api` extraction backend
- `openai/gpt-oss-120b` extraction model
- summary output at `target/live-library/live-library-exact.json`

## Tracking Over Time

The summary output now includes stable run metadata:

- scenario path
- scenario count
- retrieval mode
- vector mode
- embedding model
- extraction backend
- extraction model
- duration
- passed checks
- total checks
- pass rate

That gives you direct quality and latency tracking over time.

Provider cost tracking can be layered on top of this same run history, but the
current runner does not capture token-usage telemetry from providers yet, so
cost should be tracked as a separate provider-price snapshot rather than as an
exact per-run bill.

## When To Use This

Use the live library before:

- broad benchmark sweeps
- changing the default extraction model
- changing retrieval ranking defaults
- promoting a new ANN implementation or scoring policy
