# Task: Define Rust data structures for LongMemEval dataset (Question, Session, Turn, QuestionType)

## ⚠️ SIGNS (Critical Rules)


- **ONE TASK ONLY: Complete exactly this task. No scope creep.**

- **SEARCH BEFORE CREATE: Always search for existing files/functions before creating new ones.**

- **NO PLACEHOLDERS: Every implementation must be complete. No TODO, FIXME, or stub code.**

- **VALIDATE BEFORE DONE: Run `dial validate` after implementing. Don't mark complete without testing.**

- **RECORD LEARNINGS: After success, capture what you learned with `dial learn "..." -c category`.**

- **FAIL FAST: If blocked or confused, stop and ask rather than guessing.**



## Project Learnings (apply these patterns)


- [gotcha] Module visibility: when engine.rs references types from other modules, those modules must be pub mod not mod. Fixed store and builder visibility.

- [pattern] Mutex<Vec<Connection>> makes Database auto-Sync without unsafe impl. Connection is Send, Mutex provides Sync. No need for unsafe.

- [gotcha] Tier filtering: default SearchDepth must be Deep (include tier 0) until consolidation promotes memories to higher tiers. Standard (tiers 1+2 only) breaks all tests when memories default to tier 0.

- [gotcha] ACT-R activation: t.max(1.0) gives ln(1.0)=0 for recent accesses. Use t.max(0.1) so sub-second accesses still contribute positively.

- [gotcha] candle-transformers modernbert: struct is ModernBert not ModernBertModel. Check pub struct names with grep before coding.

- [gotcha] granite-small-r2 uses sentence-transformers naming (no 'model.' prefix) but candle ModernBert expects HF transformers naming. Fix: vb.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string())