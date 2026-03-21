# mindcore-bench (Deprecated)

**This tool has been superseded by [RecallBench](https://github.com/victorysightsound/recallbench).**

All prompt engineering, judge logic, self-verification, and the MindCore adapter have been ported to RecallBench. Use RecallBench for all future benchmark work:

```bash
cargo install recallbench
recallbench run --system mindcore --dataset longmemeval --variant oracle
```

## Why RecallBench

- Same prompts and scoring as mindcore-bench v3 (95.6% result parity)
- Supports 6 datasets (not just LongMemEval)
- Quick mode, stress test, budget sweep, longevity testing
- Web UI dashboard with Core UI
- Multi-provider LLM support (Claude, ChatGPT, Gemini, Codex, local inference)
- Compare multiple memory systems side-by-side

## Historical Results

This directory contains the benchmark results from the v1/v2/v3 iteration cycle that brought MindCore from 87.0% to 95.6% on LongMemEval Oracle. See `../BENCHMARK_PROGRESS.md` for the full history.

| Version | Score | Key Change |
|---------|-------|------------|
| v1 | 87.0% | Generic prompts, haiku judge |
| v2 | 94.8% | Type-specific prompts, sonnet judge, unlimited context |
| v3 | 95.6% | Self-verification, preference few-shots, lenient abstention |
