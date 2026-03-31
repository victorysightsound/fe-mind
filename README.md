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
- retrieval-only practical eval in `exact` mode currently passes `56/56`
- retrieval-only practical eval in `ann` mode currently passes `56/56`
- practical eval now includes explicit graph-linked state-history, aggregation,
  graph-connected, provenance/abstention, trust/procedural safety, and
  provenance/review-guardrail plus review-policy-transition coverage, not just
  text-only changed-fact scenarios
- practical eval now also includes deterministic reflection coverage for:
  - stable supported procedures
  - stable current decisions synthesized from repeated trusted evidence
- practical eval now also covers higher-impact approval classes for:
  - auth bypass
  - destructive maintenance resets
  - traffic cutovers
  including both explicit routing and pending-review detection
- practical eval now also covers richer trusted-source provenance conflicts for:
  - partially verified chains
  - relayed trusted guidance
  - sensitive internal network-range and share-path answers with redaction
- practical eval now also covers source-chain authority conflicts for:
  - runtime-vs-deployment procedural guidance
  - reflected runtime guidance backed by authoritative chains
- practical eval now also covers mixed-authority multi-hop retrieval:
  - graph-linked client/runtime questions that must route through graph
    expansion and still prefer the authoritative runtime chain
  - graph-linked private endpoint and private subnet questions that must route
    as exact-detail retrieval instead of falling back to generic graph hits
- practical eval now also proves application-facing source-kind authority
  defaults:
  - runtime guidance can win without `source_chain` metadata when app policy
    marks the source kind authoritative
  - provenance still matters within an authoritative source kind, so
    `partially-verified` guidance can beat `relayed` guidance instead of being
    flattened by app policy
- FeMind now also supports an application-facing source authority registry, so
  apps can declare authoritative chains per domain once and let records
  participate with `source_chain` metadata instead of repeating full authority
  metadata on every row
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
  `50/50` practical (`exact` and `ann`), `58/58` live-library, and `90/90`
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
  fragments, required sources, forbidden sources, explicit observed depth, and
  minimum observed-hit counts so aggregation, provenance, and reflection
  regressions fail for the right reason instead of only by loose token overlap
- aggregation-style retrieval now uses an engine-level composition path that
  preserves distinct supporting memories, records total/distinct match counts,
  and emits a composed evidence summary for coverage-sensitive rollup questions
- practical eval now also records a deterministic composed answer for each
  retrieval-style check, so yes/no, state, and aggregation behavior can be
  tuned at the engine level instead of only by inspecting raw hits
- the engine now also exposes deterministic, metadata-assisted reflection via
  `reflect_knowledge_objects()`, which synthesizes stable knowledge objects
  from repeated trusted evidence without forcing internal derived rows into
  consumer-defined record storage
- reflected knowledge objects can now also be persisted safely through a
  consumer-supplied record builder via
  `persist_reflected_knowledge_objects_with(...)`, which lets FeMind annotate
  provenance, `source_ids`, and tier `2` without inventing its own opaque
  internal record format
- persisted reflected knowledge now has a lifecycle:
  - older reflected rows with the same `knowledge_key` are superseded when the
    derived summary changes
  - current reflected rows can now be retired when the derived summary no
    longer qualifies under the active reflection thresholds
  - current reflected rows now preserve contested-summary metadata when another
    qualified trusted summary competes for the same `knowledge_key`
  - persisted reflection rows get `validated_by` graph links back to their
    source memories
  - the practical suite now queries persisted reflection rows directly instead
    of validating only runtime reflection output
- applications can now query and manage reflected knowledge explicitly through:
  - `search_stable_knowledge(...)`
  - `search_stable_knowledge_with_policy(...)`
  - `search_stable_knowledge_only(...)`
  - `persisted_reflected_knowledge()`
  - `reflected_knowledge_for_key(...)`
  - `reflection_refresh_plan(...)`
  - `refresh_reflected_knowledge_objects_with_policy(...)`
- stable-knowledge search now over-fetches and prefers current reflected rows
  deliberately, so “stable summary” style queries do not get trapped in the
  ordinary keyword pool
- reflection refresh planning is now application-facing:
  stale, changed, or newly reinforced derived knowledge can be recomputed using
  an explicit `ReflectionRefreshPolicy` instead of ad hoc timer logic
- reflection refresh planning now also handles the negative side of memory
  quality:
  support weakening, trusted-summary conflicts, and retirement when a current
  reflected row no longer qualifies
- routed retrieval now also has an explicit `stable-summary` intent:
  questions that clearly ask for the supported, preferred, recommended, or
  current durable summary can automatically prefer current reflected knowledge
  instead of relying on ordinary retrieval to surface the right row by chance
- the practical suite now asserts route-level reflection behavior for those
  scenarios, including:
  - expected routed intent
  - expected reflection preference
- composed answers now also record confidence, abstention, and rationale so
  maintainers can see when FeMind answered confidently, when it abstained, and
  why
- composed answers now also record their evidence basis:
  - `source`
  - `reflected`
  - `blended`
- stable-summary composition can now deliberately:
  - answer from reflected knowledge
  - fall back to raw source evidence
  - blend reflected summaries with supporting source evidence for
    provenance-sensitive questions
- stable-summary promotion is now application-facing too:
  callers can choose `auto`, `prefer-reflection`, or `prefer-source` for
  stable-summary retrieval and composition instead of accepting only the
  engine default
- the practical suite now asserts stable-summary policy explicitly, not just
  routed reflection preference
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
- exact-detail routes now preserve current/historical bias when the query asks
  for a precise detail about the current or earlier state, instead of dropping
  back to a temporally neutral exact-detail path
- linked conflict sets now get pairwise demotion inside the retrieved result
  set, so current/historical routes can prefer the right state even when both
  competing records remain textually relevant enough to surface together
- linked supersession can now inject the newer or older side of a state pair
  into the candidate set when only one side was initially retrieved, so
  current-state and historical-state answers do not get trapped on the first
  matching record
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
- retrieval scoring now also honors source-chain authority metadata:
  - `metadata.source_authority_domain`
  - `metadata.source_authority_level`
  - `metadata.source_chain`
- applications can now also configure authority centrally through the engine
  builder:
  - `authority_registry(...)`
  - `authority_registry_arc(...)`
  - `authority_policy(...)`
  - `authority_domain_policy(...)`
  - `authority_kind_policy(...)`
  - `authoritative_source_chain(...)`
  - `primary_source_chain(...)`
  - `authoritative_source_kind(...)`
  - `primary_source_kind(...)`
- authority can now be supplied centrally by either:
  - grouped domain policy objects
  - `source_chain` policy
  - `source_kind` policy
  so apps do not have to encode full authority metadata onto every record
- when both record metadata and the application authority registry apply,
  FeMind uses the stronger authority level for that query domain
- when a query spans multiple authority domains, FeMind now scores against all
  relevant domains instead of collapsing to the first matched domain keyword
- mixed-domain policy coverage now also includes:
  - runtime + security overlap for secret guardrails
  - runtime + maintenance overlap for breakglass routing
- query routing now infers an authority domain for high-stakes procedural and
  stable-knowledge conflicts, so authoritative chains can win over stronger
  generic provenance when the domain is explicit
- the engine now records pending-review metadata for high-impact procedural
  memories, and maintainers can inspect them through the review queue
- review-state policy now supports:
  - `pending`
  - `allowed`
  - `denied`
  - `expired`
- review resolutions can now also carry:
  - `review_scope`
  - `review_policy_class`
  - `review_reviewer`
- review lifecycle metadata now also supports:
  - `review_template`
  - `review_replaced_by`
- review scopes now also include:
  - `maintenance`
- review policy classes now also include:
  - `auth-bypass-exception`
  - `data-reset-exception`
  - `traffic-cutover-exception`
- review templates now also include:
  - `lab-auth-bypass`
  - `maintenance-reset`
  - `traffic-cutover`
- temporary review allowances can now carry `review_expires_at` timestamps and
  be normalized back to `expired` automatically when the allowance window ends
- `femind-review` now provides an operator CLI for:
  - listing review items by status
  - resolving them with notes, reviewer, scope, policy class, and optional
    expiry timestamps
  - renewing temporary allowances while preserving template/scope/class defaults
  - revoking reviewed guidance explicitly
  - replacing dangerous guidance with a successor memory reference
  - expiring due temporary allowances
- retrieval now enforces review scope for allowed procedural exceptions, so a
  `staging` or `migration` exception does not surface as general production
  guidance
- procedural guidance queries now keep review-state semantics through the
  original routed query instead of broadening into stripped variants that can
  reintroduce denied or expired guidance
- deterministic composition now abstains on sensitive secret/credential value
  requests even when storage guidance for those values remains retrievable
- surfaced safe-location answers and observed retrieval evidence now redact raw
  credential material instead of only relying on abstention for exact-value
  requests
- secret-policy classes now also cover:
  - `token-material`
  - `key-material`
  - `private-endpoint`
  - `internal-hostname`
  - `internal-share-path`
  - `private-network-range`
- trusted sensitive guidance now resolves conflicts by provenance rank, so a
  stronger verified internal source can suppress weaker trusted-but-declared
  or partially verified / relayed alternatives for private endpoint, internal
  hostname, network-range, and share-path questions
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
- `femind-review list --database <path>`
  - lists review items from a FeMind database, optionally filtered by status
- `femind-review resolve --database <path> --memory-id <id> --status <state>`
  - resolves a review item with optional note, reviewer, scope, policy class,
    template, replacement memory ID, and expiry timestamp
- `femind-review renew --database <path> --memory-id <id>`
  - renews a temporary allowance while preserving the current template/scope
    and policy-class defaults
- `femind-review revoke --database <path> --memory-id <id>`
  - explicitly denies a previously reviewed item
- `femind-review replace --database <path> --memory-id <id> --replacement-id <id>`
  - denies the current item and records the successor memory ID
- `femind-review expire-due --database <path>`
  - marks temporary allowances as expired when their expiry timestamp has passed

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
