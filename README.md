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
- retrieval-only practical eval in `exact` mode currently passes `20/20`
- retrieval-only practical eval in `ann` mode currently passes `20/20`
- practical eval now includes explicit graph-linked state-history, aggregation,
  graph-connected, provenance/abstention, trust/procedural safety, and
  provenance/review-guardrail coverage, not just text-only changed-fact
  scenarios
- larger real-world follow-up library now lives under `eval/live-library/`
- larger real-world library now covers 18 scenarios and 58 retrieval checks
- larger real-world library currently passes `58/58` in the standard
  retrieval-only `exact` path
- memloft-derived real-data slice now lives under `eval/memloft-slice/`
- memloft-derived real-data slice now covers 18 scenarios and 90 checks
- memloft-derived real-data slice currently passes `90/90` in the standard
  retrieval-only `exact` path
- standard local runner: `scripts/run-practical-eval.sh`
- larger real-world runner: `scripts/run-live-library.sh`
- memloft-derived real-data runner: `scripts/run-memloft-slice.sh`
- the practical runner now supports local, remote, and fallback embedding plus
  reranking runtimes without changing scenario files
- the recommended high-precision retrieval path is remote MiniLM plus remote
  MiniLM reranking with local fallback when the Windows GPU service is
  available
- this pass revalidated the engine-first suites on remote GPU fallback at
  `20/20` practical (`exact` and `ann`), `58/58` live-library, and `90/90`
  memloft-slice
- FeMind is currently using an engine-first validation loop: `eval/practical`,
  `eval/live-library`, and `eval/memloft-slice` are the active tuning path, and
  benchmark-style evaluation is deferred to milestone checkpoints after
  meaningful engine changes
- practical evaluation summaries now include pass-rate breakdowns by check type,
  scenario category, inferred query intent, routed mode, temporal policy,
  state/conflict policy, and graph depth, along with the routed search plan
  used for each retrieval-style check
- retrieval-style checks can now declare required fragments, forbidden
  fragments, and minimum observed-hit counts so aggregation and provenance
  regressions fail for the right reason instead of only by loose token overlap
- aggregation-style retrieval now uses an engine-level composition path that
  preserves distinct supporting memories, records total/distinct match counts,
  and emits a composed evidence summary for coverage-sensitive rollup questions
- practical eval now also records a deterministic composed answer for each
  retrieval-style check, so yes/no, state, and aggregation behavior can be
  tuned at the engine level instead of only by inspecting raw hits
- composed answers now also record confidence, abstention, and rationale so
  maintainers can see when FeMind answered confidently, when it abstained, and
  why
- routed retrieval now includes an explicit temporal policy:
  current-state queries mildly favor newer evidence, historical-state queries
  mildly favor older evidence, and exact-detail / abstention routes stay
  temporally neutral unless the caller overrides them
- routed retrieval now also carries graph depth:
  graph-connected queries can trigger graph expansion through the engine path
  even when the global assembly config leaves `graph_depth` at `0`
- routed retrieval now also includes an explicit state/conflict policy:
  current-state routes demote superseded memories and can walk forward to the
  replacement fact, while historical-state routes can walk backward to prior
  states through supersession links
- linked conflict sets now get pairwise demotion inside the retrieved result
  set, so current/historical routes can prefer the right state even when both
  competing records remain textually relevant enough to surface together
- `SearchBuilder::valid_at(...)` is now enforced against stored `valid_from` /
  `valid_until` windows instead of being a no-op
- exact-detail composition now performs a broader evidence fallback when strict
  grounding filters everything out, which lets FeMind distinguish:
  - no evidence at all
  - related evidence exists but the exact detail was never recorded
  - related evidence exists but the surfaced detail still is not grounded
- retrieval scoring now honors stable `metadata.source_trust` values carried on
  memories, with the current contract:
  - `trusted` / `verified` / `maintainer` / `system` / `high`
  - `normal`
  - `low` / `speculative`
  - `untrusted` / `external` / `poisoned` / `unsafe`
- retrieval scoring now also honors richer provenance metadata:
  - `metadata.source_kind`
  - `metadata.source_verification`
- the engine now records pending-review metadata for high-impact procedural
  memories, and maintainers can inspect them through the review queue
- deterministic composition now abstains on sensitive secret/credential value
  requests even when storage guidance for those values remains retrievable
- procedural guidance queries now isolate low-trust procedural instructions when
  a safer procedural alternative is present, so unsafe command-like memories do
  not remain in the surfaced result set just because they are semantically close

## Migration

`femind` is the successor to the earlier `mindcore` crate and repository.
The published `mindcore` crate remains the legacy package line; new work and
future releases should target `femind`.

Key maintainer references:

- `ARCHITECTURE.md` — full crate structure and API design
- `RESEARCH.md` — research, landscape analysis, and specification
- `specs/TOP_TIER_MEMORY_ROADMAP_2026-03-30.md` — next-step research roadmap for routed, temporal, and high-precision retrieval
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
