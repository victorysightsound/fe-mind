# FeMind

A standalone Rust crate providing a pluggable, feature-gated memory engine for AI agent applications.

Handles persistent storage, keyword search (FTS5), vector search (candle), hybrid retrieval (RRF), graph relationships, memory consolidation, cognitive decay modeling, and token-budget-aware context assembly.

## Design Principles

- **Library, not framework** — projects call into FeMind, not the other way around
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
- larger real-world library now covers 18 scenarios and 66 checks
- larger real-world library currently passes `66/66` in both `all` + `exact` and `all` + `ann`
- memloft-derived real-data slice now lives under `eval/memloft-slice/`
- memloft-derived real-data slice now covers 18 scenarios and 90 checks
- memloft-derived real-data slice currently passes `90/90` in both `all` + `exact` and `all` + `ann`
- standard local runner: `scripts/run-practical-eval.sh`
- larger real-world runner: `scripts/run-live-library.sh`
- memloft-derived real-data runner: `scripts/run-memloft-slice.sh`
- the practical runner now supports local, remote, and fallback embedding plus
  reranking runtimes without changing scenario files
- the recommended high-precision retrieval path is remote MiniLM plus remote
  MiniLM reranking with local fallback when the Windows GPU service is
  available
- current remote-fallback retrieval baselines are `9/9` practical, `58/58`
  live-library, and `90/90` memloft-slice

## Migration

`femind` is the successor to the earlier `mindcore` crate and repository.
The published `mindcore` crate remains the legacy package line; new work and
future releases should target `femind`.

Key maintainer references:

- `ARCHITECTURE.md` — full crate structure and API design
- `RESEARCH.md` — research, landscape analysis, and specification
- `DECISIONS.md` — architectural decisions log
- `specs/REMOTE_MINILM_BACKEND.md` — remote/local-network MiniLM backend contract
- `PRACTICAL_EVAL.md` — real-world validation plan and practical eval categories
- `eval/practical/` — curated practical validation scenarios
- `eval/live-library/` — larger real-world validation library
- `eval/memloft-slice/` — memloft-derived technical real-data validation slice
- `research/` — competitive landscape analysis

Current MiniLM direction:

- canonical logical model label: `local-minilm`
- strict compatibility identity lives in the embedding profile
- canonical reranker label is `local-minilm-reranker`
- reranker compatibility identity lives in the reranker profile
- remote execution is treated as a runtime mode for the same MiniLM profile, not
  as a different model family
- supported runtime targets are `local-cpu`, `local-gpu`, `remote-cpu`,
  `remote-gpu`, and `off`
- `femind-embed-service` can now host MiniLM in `auto`, `cpu`, or `cuda` mode
  and, when built with `reranking`, can also host the MiniLM cross-encoder
  reranker on the same process under `/rerank/*`
- `--device cuda` requires a FeMind build with the `cuda` feature on a host
  that actually has CUDA available

Remote deployment helpers:

- `scripts/remote/install-femind-embed-systemd.sh`
- `scripts/remote/configure-windows-wsl-autostart.ps1`
- `scripts/remote/configure-windows-native-autostart.ps1`
- example host config: `examples/config/remote-embed-service.toml`
- example client config: `examples/config/remote-embedding-client.toml`

Recommended host pattern:

- keep `femind-embed-service` bound to `127.0.0.1` on the remote host
- run it under `systemd` on Linux or inside WSL
- use the Windows WSL helper only to start the WSL service at logon/startup
- native Windows CUDA hosts can run `femind-embed-service.exe` directly under a
  scheduled task using the native helper
- reach the remote host through SSH over ZeroTier instead of exposing the port
  directly on the LAN

Remote service operator surface:

- `femind-embed-service serve`
  - runs the FeMind-owned embedding host
  - accepts direct flags or `--config examples/config/remote-embed-service.toml`
- `femind-embed-service status --config <path>`
  - resolves the configured embedding mode and reports remote-service status when
    `execution_mode = "remote_service"`
  - also reports configured reranker status when `[reranking]` is present
- `femind-embed-service verify-remote --config <path>`
  - checks auth, model identity, dimensions, and embedding profile against a
    configured remote MiniLM service
- `femind-embed-service verify-remote-reranker --config <path>`
  - checks auth, model identity, and reranker profile against a configured
    remote MiniLM reranker service

Lifecycle defaults:

- the remote host should run `femind-embed-service` warm under `systemd`
- the provided unit uses `Restart=always` and `RestartSec=2`
- the WSL helper supports `off`, `status`, `logon`, and `startup` modes
- the native Windows helper supports `off`, `status`, `logon`, and `startup`
  modes and prepares the MSVC/CUDA environment before launch
- idle CPU should stay low because the service only responds to requests; the
  tradeoff is that loaded MiniLM models stay resident in memory for fast warm
  responses
- native Windows CUDA hosts should keep toolkit and driver lines aligned
  (for example toolkit `12.9` with a `12.9` driver line)

Reranking notes:

- local reranking uses `cross-encoder/ms-marco-MiniLM-L6-v2` through candle BERT
- remote reranking uses the same shared host process under `/rerank/status` and
  `/rerank/rerank`
- API reranking is supported through a generic HTTPS `/rerank` endpoint; unlike
  embeddings, there is no broadly adopted OpenAI-native rerank schema today
