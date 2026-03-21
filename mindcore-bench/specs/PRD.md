# MindCore-Bench — LongMemEval Benchmark Harness PRD

**Version:** 1.0
**Date:** 2026-03-20
**Status:** Ready for Implementation

---

## 1. Overview

MindCore-Bench is a separate binary crate within the mindcore workspace that runs the LongMemEval benchmark against MindCore's memory engine. It downloads the dataset, ingests conversation histories as memories, retrieves answers to 500 questions, and uses an LLM judge (Claude) to score accuracy.

**Target:** 93-96% task-averaged accuracy (MindCore achieved 95.6%, surpassing OMEGA's verified 76.8%).

---

## 2. LongMemEval Benchmark Summary

- **500 questions** testing 5 memory abilities across 7 question types
- **Dataset variants:** oracle (15MB, evidence only), S (277MB, ~40 sessions), M (2.7GB, ~500 sessions)
- **Evaluation:** Binary correct/incorrect via LLM judge (GPT-4o standard, we'll use Claude)
- **Metrics:** Task-averaged accuracy (mean of 6 per-type accuracies), overall accuracy, abstention accuracy

### 5 Memory Abilities

| Ability | Abbrev | Description |
|---------|--------|-------------|
| Information Extraction | IE | Recall specific info from chat histories |
| Multi-Session Reasoning | MR | Synthesize across multiple sessions |
| Knowledge Updates | KU | Recognize and apply updated information |
| Temporal Reasoning | TR | Reason about time references and timestamps |
| Abstention | ABS | Correctly refuse unanswerable questions |

### 7 Question Types

| Type | Ability | Field Value |
|------|---------|-------------|
| Single-session user info | IE | `single-session-user` |
| Single-session assistant info | IE | `single-session-assistant` |
| Single-session preference | IE | `single-session-preference` |
| Multi-session | MR | `multi-session` |
| Knowledge update | KU | `knowledge-update` |
| Temporal reasoning | TR | `temporal-reasoning` |
| Abstention | ABS | Any type with `_abs` in question_id |

---

## 3. Architecture

```
mindcore-bench/
├── Cargo.toml          # Binary crate, depends on mindcore
├── src/
│   ├── main.rs         # CLI entry point
│   ├── dataset.rs      # Download and parse LongMemEval JSON
│   ├── ingest.rs       # Feed conversations into MindCore as memories
│   ├── retrieval.rs    # Search MindCore for each question, build context
│   ├── generation.rs   # Send context + question to LLM for answer
│   ├── judge.rs        # Send hypothesis to LLM judge for scoring
│   └── metrics.rs      # Compute per-type and overall accuracy
├── specs/
│   └── PRD.md
└── results/            # Output directory for benchmark runs
```

---

## 4. Dataset Format

Each entry in the JSON array:
```json
{
  "question_id": "string",
  "question_type": "single-session-user|...",
  "question": "the question text",
  "answer": "ground truth answer (string or array)",
  "question_date": "2024/01/15 (Mon) 14:30",
  "haystack_sessions": [
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    ...
  ],
  "answer_session_ids": ["session_id", ...]
}
```

---

## 5. Pipeline

### Step 1: Download Dataset
- Download `longmemeval_oracle.json` from HuggingFace
- Parse into typed Rust structs
- Start with oracle (evidence-only) for fast iteration, then test with S variant

### Step 2: Ingest Sessions into MindCore
- For each question entry, create a fresh MindCore engine
- Ingest all `haystack_sessions` as memories:
  - Each conversation turn becomes a memory
  - Mark with session metadata (date, session ID)
  - Use `MemoryType::Episodic` for raw turns
  - Run fact extraction if IngestStrategy is configured

### Step 3: Retrieve Context
- For each question, search MindCore with the question text
- Use hybrid search (FTS5 + vector if available)
- Assemble context within a token budget
- Include question date for temporal reasoning

### Step 4: Generate Answer
- Send the retrieved context + question to an LLM (Claude)
- Use the LongMemEval generation prompt template
- Collect the hypothesis (model's answer)

### Step 5: Judge Answer
- Send (question, ground truth, hypothesis) to the judge LLM
- Use question-type-specific judge prompts (from LongMemEval)
- Binary scoring: correct or incorrect

### Step 6: Compute Metrics
- Per-type accuracy (6 types)
- Task-averaged accuracy (mean of per-type)
- Overall accuracy (mean of all 500)
- Abstention accuracy (30 questions)

---

## 6. Phases

### Phase 1: Dataset + Parsing (Tasks 1-4)
Download, parse, and validate the LongMemEval dataset.

### Phase 2: Ingestion + Retrieval (Tasks 5-9)
Feed conversations into MindCore, retrieve context for questions.

### Phase 3: Generation + Judging (Tasks 10-14)
LLM answer generation and judge evaluation.

### Phase 4: Metrics + Reporting (Tasks 15-18)
Compute scores, generate reports, iterate on retrieval strategy.

---

## 7. Dependencies

```toml
[dependencies]
mindcore = { path = "..", features = ["full"] }
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
indicatif = "0.17"  # Progress bars
anyhow = "1"        # Binary crate, anyhow is fine
```

For the LLM judge/generation, use the Anthropic API via direct HTTP (reqwest) to avoid extra SDK deps.

---

## 8. LLM Integration

**Generation LLM:** Claude Sonnet (fast, cheap for 500 questions)
**Judge LLM:** Claude Sonnet (binary yes/no, max 10 tokens)

API key via environment variable `ANTHROPIC_API_KEY` or `op read` inline.

---

## 9. Constraints

- Start with oracle dataset (15MB) for fast iteration
- Run serially first, then add concurrency
- Save all results to JSONL for reproducibility
- Track token costs per run
- Binary crate — `anyhow` is acceptable here (not a library)
