# Live Usage Sample

This directory holds a small broader live-validation sample built from actual
`femind` project documents rather than the hand-authored practical scenarios.

The goal is to check whether the same provider-backed path still behaves well
when the source material comes straight from real project docs and decision
records.

Current standard run:

```bash
FEMIND_EVAL_SCENARIOS=eval/live-usage/scenarios.json \
FEMIND_EVAL_MODE=all \
scripts/run-practical-eval.sh
```

Current baseline:

- exact-mode live-usage sample currently passes `11/11`
- summary artifact: `target/practical-eval/live-usage-exact.json`
