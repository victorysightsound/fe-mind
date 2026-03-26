# femind

A standalone Rust crate providing a pluggable, feature-gated memory engine for AI agent applications.

Handles persistent storage, keyword search (FTS5), vector search (candle), hybrid retrieval (RRF), graph relationships, memory consolidation, cognitive decay modeling, and token-budget-aware context assembly.

## Design Principles

- **Library, not framework** — projects call into femind, not the other way around
- **Feature-gated everything** — heavy dependencies behind compile-time flags
- **Local-first** — SQLite-backed, single-file databases, no cloud dependency
- **Pure Rust where possible** — candle over ort, SQLite over Postgres
- **Proven patterns only** — every component backed by research or established practice

## Status

The local crate and repo are now `femind` / `fe-mind`, and `femind 0.2.0`
has been published as the active crate line. The earlier `mindcore` package
line remains legacy only. Non-LLM verification is currently green:

- `cargo test`
- `cargo test --features full`
- `cargo clippy --all-targets --all-features -- -D warnings`

The practical live-validation path is now established and repeatable:

- recommended API extraction default: DeepInfra `openai/gpt-oss-120b`
- recommended CLI extraction default: Codex CLI `gpt-5.4-mini`
- lower-cost CLI fallback: Codex CLI `gpt-5.1-codex-mini`
- retrieval-only practical eval in `exact` mode currently passes `9/9`
- retrieval-only practical eval in `ann` mode currently passes `9/9`
- broader live-usage sample built from actual project docs currently passes `11/11`
- larger real-world follow-up library now lives under `eval/live-library/`
- larger real-world library currently passes `36/44` in both `all` + `exact` and `all` + `ann`
- standard local runner: `scripts/run-practical-eval.sh`
- larger real-world runner: `scripts/run-live-library.sh`

## Migration

`femind` is the successor to the earlier `mindcore` crate and repository.
The published `mindcore` crate remains the legacy package line; new work and
future releases should target `femind`.

Key maintainer references:

- `ARCHITECTURE.md` — full crate structure and API design
- `RESEARCH.md` — research, landscape analysis, and specification
- `DECISIONS.md` — architectural decisions log
- `PRACTICAL_EVAL.md` — real-world validation plan and practical eval categories
- `eval/practical/` — curated practical validation scenarios
- `eval/live-library/` — larger real-world validation library
- `research/` — competitive landscape analysis
