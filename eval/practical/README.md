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
- `retrieval_checks`
- optional `extraction_checks`
- optional `abstention_checks`

## Review Workflow

1. ingest the scenario records using the selected provider path
2. run the retrieval checks
3. compare the actual answer to the expected answer
4. note whether the failure is extraction, storage, ranking, or abstention

## Scope

This is the primary real-world validation set for `femind`.

LongMemEval and MemoryAgentBench remain useful, but only as secondary
comparison or regression tools after this set is behaving well.
