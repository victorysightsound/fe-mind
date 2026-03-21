# Task: Fix stop-word list: remove content words ('spend', 'participating', 'activities' and any others). Audit full list — keep ONLY function words (determiners, prepositions, pronouns, auxiliaries, conjunctions). Update tests.

## ⚠️ SIGNS (Critical Rules)


- **ONE TASK ONLY: Complete exactly this task. No scope creep.**

- **SEARCH BEFORE CREATE: Always search for existing files/functions before creating new ones.**

- **NO PLACEHOLDERS: Every implementation must be complete. No TODO, FIXME, or stub code.**

- **VALIDATE BEFORE DONE: Run `dial validate` after implementing. Don't mark complete without testing.**

- **RECORD LEARNINGS: After success, capture what you learned with `dial learn "..." -c category`.**

- **FAIL FAST: If blocked or confused, stop and ask rather than guessing.**



## Recent Unresolved Failures (avoid these)


- **RustCompileError**:    Compiling mindcore v0.2.0 (/Users/johndeaton/projects/mindcore)

thread 'rustc' (6011498) panicked at /rustc-dev/4a4ef493e3a1488c6e321570238084b38948f6db/compiler/rustc_query_system/src/dep_graph/s

- **UnknownError**:    Compiling mindcore v0.2.0 (/Users/johndeaton/projects/mindcore)
warning: unused import: `vec_to_bytes`
  --> examples/benchmark.rs:15:51
   |
15 | use mindcore::embeddings::pooling::{normalize_l2, 

- **UnknownError**: warning: struct `Mem` is never constructed
  --> tests/vector_search.rs:17:8
   |
17 | struct Mem {
   |        ^^^
   |
   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warn

## Project Learnings (apply these patterns)


- [gotcha] Module visibility: when engine.rs references types from other modules, those modules must be pub mod not mod. Fixed store and builder visibility.

- [pattern] Mutex<Vec<Connection>> makes Database auto-Sync without unsafe impl. Connection is Send, Mutex provides Sync. No need for unsafe.

- [gotcha] Tier filtering: default SearchDepth must be Deep (include tier 0) until consolidation promotes memories to higher tiers. Standard (tiers 1+2 only) breaks all tests when memories default to tier 0.

- [gotcha] ACT-R activation: t.max(1.0) gives ln(1.0)=0 for recent accesses. Use t.max(0.1) so sub-second accesses still contribute positively.

- [gotcha] candle-transformers modernbert: struct is ModernBert not ModernBertModel. Check pub struct names with grep before coding.

- [gotcha] granite-small-r2 uses sentence-transformers naming (no 'model.' prefix) but candle ModernBert expects HF transformers naming. Fix: vb.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string())

- [gotcha] LongMemEval answer field can be string, number (int like 3), or array. Use serde_json::Number for numeric answers.

- [build] Candle inference in debug mode is ~100x slower than optimized. Add [profile.dev.package.candle-core] opt-level = 2 (and similar for candle-nn, tokenizers, gemm, half) to make dev builds usable.

- [gotcha] FTS5 interprets hyphens in queries as column filters — 'faith-related' becomes 'faith:related' causing 'no such column' errors. Must replace ALL hyphens with spaces in sanitizer.