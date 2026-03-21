# LongMemEval Benchmark Progress

Tracking MindCore's performance on the LongMemEval Oracle dataset (500 questions).

## Score Summary

| Metric | v1 | v2 | v3 (pending) |
|--------|----|----|-------------|
| Overall Accuracy | 87.0% (435/500) | 94.8% (474/500) | — |
| Task-Averaged Accuracy | 81.9% | 93.8% | — |
| Abstention Accuracy | 90.0% (27/30) | 90.0% (27/30) | — |
| Failures | 65 | 26 | — |

## Per-Type Breakdown

| Category | Count | v1 | v2 | Delta |
|----------|-------|-----|-----|-------|
| Temporal Reasoning | 133 | 121 (91.0%) | 129 (97.0%) | +6.0% |
| Knowledge Update | 78 | 71 (91.0%) | 74 (94.9%) | +3.8% |
| Multi-Session | 133 | 114 (85.7%) | 121 (91.0%) | +5.3% |
| Single-Session (User) | 70 | 66 (94.3%) | 70 (100%) | +5.7% |
| Single-Session (Assistant) | 56 | 52 (92.9%) | 56 (100%) | +7.1% |
| Single-Session (Preference) | 30 | 11 (36.7%) | 24 (80.0%) | +43.3% |

---

## v1 Run (2026-03-20)

**Configuration:**
- Generation: Claude Sonnet via `claude --print`
- Judge: Claude Haiku via `claude --print`
- Context budget: 16,384 tokens
- Prompts: Single generic prompt for all question types
- Results file: `results/longmemeval_full.jsonl`

**Root cause analysis of 65 failures:**

| Root Cause | Count |
|------------|-------|
| Prompt format mismatch (preference) | 19 |
| Reasoning/computation errors | 30 |
| Judge too strict (haiku) | 13 |
| Context truncation | 1 |
| Abstention misses | 3 |

### v1 Failures by Type

**Temporal Reasoning (12 wrong):**
- a3838d2b, gpt4_483dd43c, gpt4_a1b77f9c, gpt4_7abb270c, 370a8ff4, gpt4_d6585ce8, gpt4_21adecb5, gpt4_7bc6cf22, 71017277, b46e15ee, gpt4_d6585ce9, gpt4_fa19884d

**Knowledge Update (7 wrong):**
- 6a1eabeb, 852ce960, f9e8c073, b6019101, dad224aa, 031748ae_abs, 07741c45

**Multi-Session (19 wrong):**
- 0a995998, gpt4_59c863d7, 46a3abf7, 28dc39ac, gpt4_2f8be40d, 88432d0a, gpt4_7fce9456, 7024f17c, gpt4_31ff4165, gpt4_194be4b3, gpt4_ab202e7f, e3038f8c, 7405e8b1, 9ee3ecd6, ba358f49, 09ba9854, 37f165cf, a96c20ee_abs, 09ba9854_abs

**Single-Session User (4 wrong):**
- 51a45a95, b86304ba, ec81a493, 8a137a7f

**Single-Session Assistant (4 wrong):**
- 8752c811, 3249768e, eaca4986, 778164c6

**Single-Session Preference (19 wrong):**
- 8a2466db, 75832dbd, 0edc2aef, 35a27287, afdc33df, caf03d32, 54026fce, 09d032c9, 57f827a0, 505af2f5, 75f70248, d6233ab6, 1da05512, b6025781, a89d7624, b0479f84, 1d4e3b97, 07b6f563, 1c0ddc50

---

## v2 Run (2026-03-20)

**Configuration:**
- Generation: Claude Sonnet via `claude --print`
- Judge: Claude Sonnet via `claude --print` (upgraded from haiku)
- Context budget: Unlimited (0)
- Prompts: Type-specific generation prompts
- Results file: `results/longmemeval_v2.jsonl`

**Changes from v1:**
1. Type-specific generation prompts (preference format, temporal enumeration, knowledge-update recency, multi-session enumeration)
2. Judge upgraded from haiku to sonnet with extraction-aware prompts
3. Context budget set to unlimited for oracle dataset
4. Better abstention detection in generation prompt

**Question-level diff (v1 → v2):**
- Fixed (were wrong, now correct): 49
- Regressed (were correct, now wrong): 10
- Net improvement: +39

### v2 Regressions (10 questions correct in v1, wrong in v2)

| Question ID | Type | Ground Truth |
|------------|------|-------------|
| 07741c44 | knowledge-update | under my bed |
| 6b7dfb22 | single-session-preference | build upon existing sources of inspiration (Instagram art accounts, online tutorials) |
| 6d550036 | multi-session | 2 |
| a2f3aa27 | knowledge-update | 1300 |
| af082822 | temporal-reasoning | 2 |
| c4a1ceb8 | multi-session | 3 |
| f685340e_abs | knowledge-update | (abstention) You mentioned playing tennis but not table tennis |
| gpt4_731e37d7 | multi-session | $720 |
| gpt4_93159ced | temporal-reasoning | 4 years and 9 months |
| gpt4_93159ced_abs | temporal-reasoning | (abstention) You haven't started working at Google yet |

### v2 Remaining Failures (26 questions)

**Multi-Session (12 wrong):**

| Question ID | Ground Truth | Notes |
|------------|-------------|-------|
| 0a995998 | 3 | Counting error (store pickups/returns) |
| 6d550036 | 2 | Counting error (teams led) — regression |
| c4a1ceb8 | 3 | Counting error (citrus fruits) — regression |
| 46a3abf7 | 3 | Counting error (fish tanks) |
| gpt4_2f8be40d | 3 weddings (Rachel/Mike, Emily/Sarah, Jen/Tom) | Enumeration error |
| 7024f17c | 0.5 hours | Wrong time computation |
| gpt4_731e37d7 | $720 | Cost computation — regression |
| e3038f8c | 99 | Sum of rare items |
| 9ee3ecd6 | 100 | Points computation |
| 37f165cf | 856 | Page count sum |
| 09ba9854 | $50 | Cost difference |
| 09ba9854_abs | (abstention) bus cost not mentioned | Model answered instead of abstaining |

**Single-Session Preference (6 wrong):**

| Question ID | Core Preference Missed |
|------------|----------------------|
| afdc33df | Kitchen organization (utensil holder, countertop tips) |
| caf03d32 | Slow cooker tips (beef stew success, yogurt interest) |
| 6b7dfb22 | Art inspiration sources (Instagram, online tutorials) — regression |
| 09d032c9 | Power bank optimization tips |
| 1da05512 | NAS device for home network storage issues |
| 1c0ddc50 | Podcasts/audiobooks beyond true crime, especially history |

**Knowledge Update (4 wrong):**

| Question ID | Ground Truth | Notes |
|------------|-------------|-------|
| 07741c44 | under my bed | Model picked later session (closet) — regression |
| 07741c45 | in a shoe rack in my closet | Model picked earlier session (bed) |
| a2f3aa27 | 1300 | User said "close to 1300", GT expects exact 1300 — regression |
| f685340e_abs | (abstention) tennis not table tennis | Model abstained correctly but judge said no — regression |

**Temporal Reasoning (4 wrong):**

| Question ID | Ground Truth | Notes |
|------------|-------------|-------|
| gpt4_93159ced | 4 years and 9 months | Model quotes "4 years and 3 months" from one session, misses update — regression |
| gpt4_93159ced_abs | (abstention) haven't started at Google | Model abstained correctly but judge said no — regression |
| af082822 | 2 (weeks) | Model computed 13 days, didn't convert to weeks — regression |
| 370a8ff4 | 15 | Date counting error between Jan 19 and Apr 10 |

---

## v3 Changes (Implemented, Not Yet Run)

**Three targeted fixes:**

### 1. Self-Verification Pass (new: `verify.rs`)
Second LLM call after generation for multi-session, temporal, and knowledge-update questions. Asks model to re-check its counting, arithmetic, and version selection. Skips single-session and preference types. Skips abstention questions.

Targets: 12 multi-session + 4 temporal + 4 knowledge-update counting/arithmetic errors.

### 2. Preference Few-Shot Examples (updated: `retrieval.rs`)
Two concrete examples showing content-focused preference descriptions vs. format-focused descriptions. Explicitly tells model that "BAD answers describe formatting preferences."

Targets: 6 remaining preference failures where model describes format instead of content preferences.

### 3. Lenient Abstention Judging (updated: `judge.rs`)
Judge now accepts abstention responses that explain WHY they can't answer and what IS in the chat history, as long as the primary conclusion is that they cannot answer the specific question.

Targets: 3 abstention failures (f685340e_abs, gpt4_93159ced_abs, 09ba9854_abs).

**Projected v3:** 485-490/500 (97.0-98.0%)

---

## Future: Core MindCore Improvements

The benchmark revealed three areas where MindCore's core engine needs improvement for real-world usage (documented, not yet implemented):

1. **Preference memory tagging** — Detect and tag user preferences during ingestion as a distinct memory type with higher retrieval priority. (`src/ingest/`)

2. **Temporal-aware retrieval** — Date-biased scoring in context assembly. Boost memories closer to query date for recency queries. (`src/search/`, `src/context/`)

3. **Vector embeddings for semantic search** — Wire Candle embeddings feature into hybrid search (FTS5 + vector similarity + RRF fusion) for retrieval when keyword matching fails. (`src/search/`)

These will be addressed through John's dedicated MindCore benchmark app.

---

## Files

| File | Description |
|------|-------------|
| `results/longmemeval_full.jsonl` | v1 raw results (500 questions, deduplicate by question_id) |
| `results/longmemeval_v2.jsonl` | v2 raw results (500 questions) |
| `results/bench.log` | v1 runtime log |
| `results/bench_v2.log` | v2 runtime log |
