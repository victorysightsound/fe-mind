# Task: FTS5 OR-mode search: add search_or_mode() to FtsSearch that joins query terms with OR instead of AND. Use OR mode in hybrid search path (execute_hybrid). Keep AND available. Test with long natural language queries that previously returned 0 results.

## ⚠️ SIGNS (Critical Rules)


- **ONE TASK ONLY: Complete exactly this task. No scope creep.**

- **SEARCH BEFORE CREATE: Always search for existing files/functions before creating new ones.**

- **NO PLACEHOLDERS: Every implementation must be complete. No TODO, FIXME, or stub code.**

- **VALIDATE BEFORE DONE: Run `dial validate` after implementing. Don't mark complete without testing.**

- **RECORD LEARNINGS: After success, capture what you learned with `dial learn "..." -c category`.**

- **FAIL FAST: If blocked or confused, stop and ask rather than guessing.**



## Previous Failed Attempt

PREVIOUS ATTEMPT (failed):
Error: Compiling mindcore v0.2.0 (/Users/johndeaton/projects/mindcore)

thread 'rustc' (6011498) panicked at /rustc-dev/4a4ef493e3a1488c6e321570238084b38948f6db/compiler/rustc_query_system/src/dep_graph/serialized.rs:245:13:
assertion failed: node_header.node().kind != D::DEP_KIND_NULL && node.kind == D::DEP_KIND_NULL
stack backtrace:
   0:        0x11d9980af - <std::sys::backtrace::BacktraceLock::print::DisplayBacktrace as core::fmt::Display>::fmt::h6c1071e5bd23b3af
   1:        0x11a396e67 - core:
Changes attempted:
.dial/current_context.md | 70 ++++++++++++++++++++++++++++++++++++++++++++++--
 1 file changed, 68 insertions(+), 2 deletions(-)
diff --git a/.dial/current_context.md b/.dial/current_context.md
index 5525bb4..657ad1f 100644
--- a/.dial/current_context.md
+++ b/.dial/current_context.md
@@ -1,4 +1,4 @@
-# Task: Run full test suite with --features vector-search and --features full, fix any regressions.
+# Task: FTS5 OR-mode search: add search_or_mode() to FtsSearch that joins query terms with OR instead of AND. Use OR mode in hybrid search path (execute_hybrid). Keep AND available. Test with long natural language queries that previously returned 0 results.
 
 ## ⚠️ SIGNS (Critical Rules)
 
@@ -17,9 +17,71 @@
 
 
 
+## Previous Failed Attempt
+
+PREVIOUS ATTEMPT (failed):
+Error: Compiling mindcore v0.2.0 (/Users/johndeaton/projects/mindcore)
+warning: unused import: `vec_to_bytes`
+  --> examples/benchmark.rs:15:51
+   |
+15 | use mindcore::embeddings::pooling::{normalize_l2, vec_to_bytes};
+   |                                                   ^^^^^^^^^^^^
+   |
+   = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default
+
+warning: unused import: `SearchMode`
+  --> examples/benchmark.rs:20:24
+   |
+20 | use mindcore::search::{SearchMode, VectorSearch};
+Changes attempted:
+.dial/current_context.md |   8 +++-
+ src/search/builder.rs    |   4 +-
+ src/search/fts5.rs       | 111 +++++++++++++++++++++++++++++++++++++++++++++--
+ 3 files changed, 116 insertions(+), 7 deletions(-)
+diff --git a/.dial/current_context.md b/.dial/current_context.md
+index 5525bb4..13607f1 100644
+--- a/.dial/current_context.md
++++ b/.dial/current_context.md
+@@ -1,4 +1,4 @@
+-# Task: Run full test suite with --features vector-search and --features full, fix any regressions.
++# Task: FTS5 OR-mode search: add search_or_mode() to FtsSearch that joins query terms with OR instead of AND. Use OR mode in hybrid search path (execute_hybrid). Keep AND available. Test with long natural language queries that previously returned 0 results.
+ 
+ ## ⚠️ SIGNS (Critical Rules)
+ 
+@@ -45,4 +45,8 @@ war
DO NOT repeat this approach.

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