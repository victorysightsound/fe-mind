# Task: Full pipeline integration test with NoopBackend: create engine, store 20+ memories, search with Auto mode, verify hybrid path returns results, verify search ranking.

## ⚠️ SIGNS (Critical Rules)


- **ONE TASK ONLY: Complete exactly this task. No scope creep.**

- **SEARCH BEFORE CREATE: Always search for existing files/functions before creating new ones.**

- **NO PLACEHOLDERS: Every implementation must be complete. No TODO, FIXME, or stub code.**

- **VALIDATE BEFORE DONE: Run `dial validate` after implementing. Don't mark complete without testing.**

- **RECORD LEARNINGS: After success, capture what you learned with `dial learn "..." -c category`.**

- **FAIL FAST: If blocked or confused, stop and ask rather than guessing.**



## Previous Failed Attempt

PREVIOUS ATTEMPT (failed):
Error: warning: struct `Mem` is never constructed
  --> tests/vector_search.rs:17:8
   |
17 | struct Mem {
   |        ^^^
   |
   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default

warning: unused import: `vec_to_bytes`
  --> examples/benchmark.rs:15:51
   |
15 | use mindcore::embeddings::pooling::{normalize_l2, vec_to_bytes};
   |                                                   ^^^^^^^^^^^^
   |
   = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default

war
Changes attempted:
.dial/current_context.md |   2 +-
 tests/full_pipeline.rs   | 140 +++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 141 insertions(+), 1 deletion(-)
diff --git a/.dial/current_context.md b/.dial/current_context.md
index ab57ac4..fc962af 100644
--- a/.dial/current_context.md
+++ b/.dial/current_context.md
@@ -1,4 +1,4 @@
-# Task: Add edge case handling in engine.store() embedding path: skip embedding for empty/whitespace-only text, truncate text >8192 tokens before embedding, handle zero-length embedding results.
+# Task: Full pipeline integration test with NoopBackend: create engine, store 20+ memories, search with Auto mode, verify hybrid path returns results, verify search ranking.
 
 ## ⚠️ SIGNS (Critical Rules)
 
diff --git a/tests/full_pipeline.rs b/tests/full_pipeline.rs
index 8364bff..d804247 100644
--- a/tests/full_pipeline.rs
+++ b/tests/full_pipeline.rs
@@ -234,3 +234,143 @@ fn dedup_at_scale() {
 
     assert_eq!(engine.count().expect("count"), 50);
 }
+
+/// Full pipeline with NoopBackend: store → embed → hybrid search → context assembly.
+///
+/// Uses NoopBackend (zero vectors), so vector search won't add signal, but it verifies
+/// the entire embedding-at-store + hybrid-search path executes without errors.
+#[test]
+fn full_pipeline_with_noop_embedding() {
+    use mindcore::embeddings::NoopBackend;
+
+    let backend = NoopBackend::new(384);
+    let engine = MemoryEngine::<Mem>::builder()
+        .embedding_backend(backend)
+        .build()
+        .expect("build");
+
+    // Store 25 diverse memories
+    let topics = [
+        "The authentication system uses JWT tokens with 15-minute expiry",
+        "Database queries timeout after 30 seconds of inactivity",
+        "The CI pipeline runs tests in parallel across 4 workers",
+        "User preferences are stored in a PostgreSQL JSONB column",
+        "The caching layer uses Redis with a 5-minute TTL",
+        "Error monitoring is handled by Sentry with PII scrubbing",
+        "The API rate limiter allows 100 requests per minute per user",
+        "Deployments use blue-green strategy with health checks",
+        "The search
DO NOT repeat this approach.

## Recent Unresolved Failures (avoid these)


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