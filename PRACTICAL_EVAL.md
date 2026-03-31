# FeMind Practical Evaluation

## Purpose

This document defines the real-world validation layer for `femind`.
It exists to keep production readiness tied to practical memory behavior rather
than benchmark scores.

The practical eval set should answer one question first:

Can `femind` extract, store, retrieve, and update real working memory in ways
that are useful to an actual application or developer workflow?

## Validation Order

1. Practical eval scenarios
2. Small approved live validation pass
3. Fixes for real-world failures
4. Repeatable local regression checks
5. Secondary benchmark comparison if still useful

Benchmark datasets remain useful for regression and comparison, but they are
not the primary design target.

Current policy is explicit:

- FeMind tuning is engine-first
- `eval/practical/`, `eval/live-library/`, and `eval/memloft-slice` are the
  active development loop
- benchmark-style runs are milestone scoreboards only, not the daily tuning path

## Practical Eval Categories

The curated eval set should cover these categories:

1. Current vs superseded facts
   The system should prefer the latest accepted fact and avoid surfacing stale
   answers as current.

2. Preferences and decisions
   The system should preserve stable preferences and explicit decisions without
   losing the reason behind them.

3. Temporal and recency reasoning
   The system should answer questions about what changed, when it changed, and
   what is current now.

4. Distractor resistance
   The system should still retrieve the correct information when unrelated but
   semantically similar text is present.

5. Messy source extraction
   The system should extract useful facts from rough notes, meetings, and
   transcripts instead of only from clean synthetic prompts.

6. Abstention
   The system should avoid confident fabrication when the answer is missing.

7. Aggregation
   The system should preserve broad coverage for rollup questions instead of
   collapsing onto one top hit.

8. Multi-hop / graph retrieval
   The system should answer some questions because linked records can be
   traversed together, not only because one record happens to contain every
   keyword.

9. Provenance and exact artifact detail
   The system should ground exact paths, ports, filenames, and identifiers when
   they are documented, then abstain when that level of detail was never
   recorded.

10. Trust and procedural safety
    The system should prefer trusted procedural guidance, demote low-trust or
    malicious operational instructions, and keep unsafe command-like memories
    out of the surfaced result set when safe procedural alternatives exist.

11. Reflection / stable knowledge synthesis
    The system should be able to synthesize stable higher-order knowledge
    objects from repeated trusted evidence without forcing benchmark-oriented
    summarization into the tuning loop.

## Eval Artifact Layout

The curated eval set lives under `eval/practical/`.

- `eval/practical/README.md`
  Maintainer notes and review workflow
- `eval/practical/scenarios.json`
  Curated practical scenarios and expected behavior

The larger real-world library lives under `eval/live-library/`.

- `eval/live-library/README.md`
  Maintainer notes for the broader real-world corpus
- `eval/live-library/scenarios.json`
  Larger project-like scenarios for follow-up live validation

## Review Standard

Every practical scenario should be easy to inspect by hand.

Each scenario should include:

- source records or session text
- the main retrieval questions
- expected current answers
- expected abstentions where relevant
- optional extraction expectations for messy source material
- optional scenario-level `graph_depth` for graph-assisted retrieval checks
- optional `records[].metadata`, including stable trust markers such as
  `source_trust`, `source_kind`, `source_verification`, and
  `content_sensitivity`
- optional explicit retrieval criteria such as:
  - `expected_intent`
  - `expected_stable_summary_policy`
  - `expected_reflection_preference`
  - `required_fragments`
  - `forbidden_fragments`
  - `required_sources`
  - `forbidden_sources`
  - `top_k`
  - `min_observed_hits`
- optional `review_checks` for pending-review queue validation, including:
  - `min_pending_items`
  - `max_pending_items`
  - `required_tags`
  - `required_fragments`
  - `forbidden_fragments`
- optional `reflection` config for deterministic knowledge synthesis,
  including:
  - `min_support_count`
  - `min_trusted_support_count`
  - `max_objects`
  - `persist`
- optional `reflection_checks`, including:
  - `key`
  - `expected_summary`
  - `expected_kind`
  - `expected_contested`
  - `min_support_count`
  - `min_trusted_support_count`
  - `min_confidence`
  - `required_fragments`
  - `forbidden_fragments`
- review metadata on records such as `review_scope`, `review_policy_class`,
  `review_reviewer`, `review_template`, `review_expires_at`, and
  `review_replaced_by`
- reflection metadata on records such as:
  - `knowledge_key`
  - `knowledge_summary`
  - `knowledge_kind`
- source authority metadata on records such as:
  - `source_authority_domain`
  - `source_authority_level`
  - `source_chain`
  - `source_kind`
- application-facing authority policy can also be supplied once at engine
  build time through `SourceAuthorityRegistry`, so records that only declare
  `source_chain` or `source_kind` can still participate in authority
  arbitration
- app-facing `source_kind` defaults are now validated in two ways:
  - runtime guidance can win without any `source_chain` metadata
  - provenance still matters inside an authoritative source kind, so
    `partially-verified` and `relayed` guidance do not collapse into the same
    answer quality
- high-impact review policy examples now include:
  - auth bypass exceptions
  - destructive reset windows
  - traffic cutovers

## Release Use

Before larger live runs or benchmark comparisons:

- run a small approved sample from `eval/practical/scenarios.json`
- inspect extraction quality manually
- inspect retrieval answers manually
- log failures by category

`femind` should be treated as production-ready only when the practical eval
set is directionally strong, repeatable, and free of obvious category failures.

## Repeatable Command

The primary deterministic validation entry point is:

```bash
scripts/run-practical-eval.sh
```

Default behavior:

- `FEMIND_EVAL_MODE=retrieval`
- `FEMIND_VECTOR_MODE=exact`
- `FEMIND_GRAPH_DEPTH=0`
- `FEMIND_EMBED_RUNTIME=local-cpu`
- `FEMIND_RERANK_RUNTIME=off`
- `FEMIND_RERANK_LIMIT=20`
- `FEMIND_EVAL_EXPLAIN_FAILURES=0`
- `FEMIND_EXTRACT_BACKEND=api`
- `FEMIND_EXTRACT_MODEL=openai/gpt-oss-120b`
- `FEMIND_RETRIEVAL_INGEST=records`
- summary output at `target/practical-eval/retrieval-exact.json`
- runtime key resolution through macOS Keychain unless overridden with `FEMIND_EVAL_KEY_CMD`

Equivalent direct command:

```bash
cargo run --example practical_eval --features local-embeddings,remote-embeddings,reranking,remote-reranking,api-embeddings,api-reranking,api-llm,cli-llm,ann -- \
  --scenarios eval/practical/scenarios.json \
  --mode retrieval \
  --vector-mode exact \
  --embedding-runtime local-cpu \
  --rerank-runtime off \
  --summary target/practical-eval/retrieval-exact.json
```

The example uses runtime key resolution and optional remote auth env vars. It
does not require secrets to be written into source files or shell history.

Recommended remote-GPU regression path when the Windows FeMind service is
available:

```bash
/Users/johndeaton/bin/femind-remote-on
set -a && source /Users/johndeaton/.config/recallbench/femind-remote.env && set +a
FEMIND_EMBED_RUNTIME=remote-fallback \
FEMIND_RERANK_RUNTIME=remote-fallback \
scripts/run-practical-eval.sh
```

That path uses the remote GPU service first and falls back to the local Candle
backends if the remote service is unavailable.

When a retrieval check fails and you want raw keyword/vector/hybrid traces in
the summary artifact, set:

```bash
FEMIND_EVAL_EXPLAIN_FAILURES=1
```

The summary now also records:

- check-type pass-rate breakdowns
- scenario-category pass-rate breakdowns
- query-intent pass-rate breakdowns
- routed-mode pass-rate breakdowns
- temporal-policy pass-rate breakdowns
- state/conflict-policy pass-rate breakdowns
- graph-depth pass-rate breakdowns
- per-check routed search plans showing inferred intent, mode, depth,
  graph depth, temporal policy, state/conflict policy, grounding,
  query-alignment, and rerank settings
- per-check retrieval criteria reports showing whether the expected answer
  matched, whether required fragments were present, whether forbidden fragments
  leaked in, whether required or forbidden sources appeared, and whether enough
  hits surfaced for coverage-sensitive checks
- per-check aggregation reports showing total matches, distinct supporting
  matches, and the composed evidence text used for rollup-style validation
- per-check composed-answer reports showing the deterministic answer text and
  composition kind (`direct`, `stateful`, `yes-no`, or `aggregation`)
- per-check composed-answer reports now also include the evidence basis:
  `source`, `reflected`, or `blended`
- per-check composed-answer confidence, abstention state, and rationale so
  exact-detail and provenance failures can be diagnosed without guessing

That routed plan is now part of the actual retrieval behavior, not just a
diagnostic label:

- current-state routes mildly favor newer evidence by `created_at`
- historical-state routes mildly favor older evidence by `created_at`
- aggregation routes now use an engine-level composition path that preserves
  broad coverage instead of only surfacing a narrow top-k view
- retrieval checks now also exercise a deterministic engine-side answer
  composer so yes/no and stateful questions are validated against the composed
  answer, not only the raw retrieved snippets
- abstention checks now validate the engine’s own abstain decision instead of
  treating “no hits surfaced” as the only acceptable refusal path
- current-state routes explicitly demote superseded memories and can follow
  supersession links forward to the replacement fact
- historical-state routes can follow supersession links backward to earlier
  states instead of only favoring older timestamps
- exact-detail and abstention-focused routes stay temporally neutral unless a
  caller overrides the search settings directly
- private endpoint, private subnet, and similar internal-infra guidance
  questions now route as exact-detail procedural retrieval when the query asks
  for the approved or supported operational target, even if the scope and
  target words are separated (`private ... endpoint`, `internal ... subnet`)
- `valid_at(...)` searches now respect stored `valid_from` / `valid_until`
  windows, so time-scoped retrieval is part of the real engine path
- exact-detail composition now does a broader fallback evidence pass when
  strict grounding filters out all hits, so the engine can distinguish
  unsupported details from total absence of evidence

Graph-focused follow-up validation can be enabled explicitly:

```bash
FEMIND_GRAPH_DEPTH=2 FEMIND_RETRIEVAL_INGEST=hybrid scripts/run-live-library.sh
FEMIND_GRAPH_DEPTH=2 FEMIND_RETRIEVAL_INGEST=hybrid scripts/run-memloft-slice.sh
```

That path exists to test extraction-backed graph retrieval directly rather than
the simpler seeded-record retrieval path while still preserving exact raw
details alongside extracted facts.

Extraction backend options:

- `api`
  Uses the OpenAI-compatible API callback. Recommended default:
  `openai/gpt-oss-120b`
- `codex-cli`
  Uses the local Codex CLI callback. Recommended default:
  `gpt-5.4-mini`
  Lower-cost fallback:
  `gpt-5.1-codex-mini`

## Current Practical Baseline

Current validated baseline:

- extraction-only practical eval with DeepInfra `openai/gpt-oss-120b` passes `4/4`
- extraction-only practical eval with Codex CLI `gpt-5.4-mini` passes `4/4`
- extraction-only practical eval with Codex CLI `gpt-5.1-codex-mini` passes `4/4`
- retrieval-only practical eval with `vector_mode=exact` currently passes `50/50`
- retrieval-only practical eval with `vector_mode=ann` currently passes `50/50`
- summary artifact: `target/practical-eval/retrieval-exact.json`
- practical coverage now includes explicit linked supersession/history,
  aggregation, graph-connected, provenance/abstention, and trust/procedural
  safety scenarios
- the provenance/abstention scenario now proves FeMind can abstain on an exact
  Windows task GUID even when nearby Windows task evidence is present
- the provenance/secret-guardrail scenario now proves FeMind can return
  grounded token-storage guidance while refusing to surface the token value
- the trusted private-endpoint scenario now proves FeMind will prefer the
  strongest trusted source for sensitive infrastructure guidance and abstain on
  exact private-endpoint values
- the practical set now also proves that trusted-but-partially-verified or
  relayed infrastructure notes do not beat fully verified guidance for:
  - private network ranges
  - internal share paths
- exact subnet and internal-path questions now abstain while safe guidance
  answers return redacted references such as `[REDACTED_NETWORK]` and
  `[REDACTED_PATH]`
- safe secret-location retrieval now also redacts raw credential-bearing hits in
  the surfaced evidence list, not only in the composed answer
- practical scenarios can now validate pending-review queue behavior for
  dangerous procedural memories alongside ordinary retrieval/abstention checks
- review-state policy coverage now includes explicit `allowed`, `denied`, and
  `expired` transitions for high-impact procedural memories
- review-state policy coverage now also includes scoped allowed exceptions, so
  environment-specific overrides do not leak into general production answers
- review-policy coverage now also includes:
  - breakglass exceptions that surface only for explicit outage-recovery queries
  - scoped trusted-source conflict resolution for supported-path questions
- higher-impact approval-class coverage now also includes:
  - lab-only auth-bypass exceptions
  - maintenance reset windows
  - migration traffic cutovers
  - pending-review detection for unreviewed auth bypass, data reset, and cutover notes
- temporary review allowances can now be modeled with `review_expires_at`
  timestamps and inspected through the operator review CLI
- review operations now also support template-driven approvals plus
  renew/revoke/replace transitions for high-impact procedural guidance
- the trust/procedural-safety scenario now proves FeMind will prefer the
  trusted `femind-remote-on` recovery command over a malicious copied-chat
  instruction, and will answer `No` to exposing the remote service directly on
  `0.0.0.0` without auth
- the practical set now also proves deterministic reflection can synthesize:
  - the stable supported Windows startup path
  - the stable engine-first real-world evaluation strategy
  from repeated trusted evidence with explicit support counts and confidence
- reflected knowledge objects can now also be persisted through a
  consumer-supplied record builder, so higher-order stable knowledge can move
  into application memory stores without FeMind forcing an internal record
  schema
- persisted reflected knowledge now has an explicit lifecycle:
  - older reflected rows with the same `knowledge_key` are superseded when the
    derived summary changes
  - current reflected rows can be retired when they no longer qualify under
    the active support thresholds
  - current reflected rows can carry contested-summary metadata when another
    qualified trusted summary competes for the same key
  - persisted reflection rows get `validated_by` graph links back to their
    supporting source memories
  - the practical suite now queries persisted reflection rows directly through
    retrieval checks using source-aware criteria instead of validating only the
    runtime reflection object list
- applications can now use explicit stable-knowledge helpers instead of
  treating reflection as an internal-only mechanism:
  - `search_stable_knowledge(...)`
  - `search_stable_knowledge_with_policy(...)`
  - `search_stable_knowledge_only(...)`
  - `persisted_reflected_knowledge()`
  - `reflected_knowledge_for_key(...)`
  - `reflection_refresh_plan(...)`
- reflection refresh is now driven by `ReflectionRefreshPolicy`, so apps can
  recompute persisted knowledge when it is stale, materially changed, or backed
  by stronger support instead of relying on an opaque background cadence
- the same refresh policy now also covers quality regressions:
  - support weakening
  - trusted-summary contention
  - retirement when a current reflected row no longer qualifies
- the engine now also routes a first-class `stable-summary` intent:
  supported/preferred/recommended/current durable-summary questions can
  automatically prefer current reflected rows without forcing callers to opt
  into a separate helper
- stable-summary composition can now deliberately answer from:
  - reflected knowledge
  - raw source evidence
  - a blend of reflected summary plus supporting source evidence
- callers can now set an explicit stable-summary promotion policy:
  - `auto`
  - `prefer-reflection`
  - `prefer-source`
- the composer and the retrieval route both honor that policy through:
  - `search_stable_knowledge_with_policy(...)`
  - `compose_answer_with_config_and_summary_policy(...)`
- practical retrieval checks can now assert that with:
  - `expected_stable_summary_policy`
- practical retrieval checks can now assert that with
  `expected_composed_basis`
- practical retrieval checks can now fail explicitly when the routed intent or
  routed reflection preference is wrong, even if the answer still happens to
  pass by accident
- practical retrieval checks can now also assert routed graph depth through
  `expected_graph_depth`
- graph-connected practical coverage now passes with routed graph expansion
  even when the global CLI graph depth stays at `0`
- reranker-aware `remote-fallback` retrieval is now wired into the same runner
- latest reranker-aware `remote-fallback` exact run passes `51/51`
- latest reranker-aware `remote-fallback` ANN run passes `51/51`
- reranker-aware summary artifact: `target/practical-eval/retrieval-exact.json`
- broader live-library retrieval sample from actual project docs currently
  passes `58/58`
- live-library summary artifact: `target/live-library/live-library-exact.json`
- retrieval-only memloft-slice exact run passes `90/90`
- authority-chain coverage now proves two distinct behaviors:
  - top-answer selection can prefer an authoritative runtime chain over a
    stronger wrong-domain trusted source
  - deterministic reflection can prefer the authoritative chain for a
    knowledge key even when competing trusted chains have broader generic
    support
- mixed-authority multi-hop coverage now also proves:
  - graph-linked client questions can route with `graph_depth = 2`
  - the authoritative runtime chain can still win after graph expansion pulls
    both runtime and deployment guidance into the evidence set
  - app-facing authority policy can also promote the runtime answer from
    `source_kind` defaults alone, without per-record `source_chain` metadata
- targeted engine tests now also prove the application-facing authority
  registry works when records carry:
  - `source_chain`
  - `source_kind`
- the practical suite now also proves that source-kind defaults respect
  provenance ordering inside the same authoritative kind, instead of treating
  all authoritative-kind records as interchangeable

This exact-mode practical run is the standard local regression check before
trying wider live usage samples or ANN comparisons.

## Larger Real-World Library

After the small practical set is stable, the next deterministic real-world check is:

```bash
scripts/run-live-library.sh
```

Default behavior:

- `retrieval` mode
- `exact` vector mode
- `local-cpu` embedding runtime
- `off` reranker runtime
- `api` extraction backend
- `openai/gpt-oss-120b` extraction model
- summary output at `target/live-library/live-library-exact.json`

This larger library is the preferred follow-up step before any broader
benchmark sweep through RecallBench.

Current larger-library baseline:

- the live-library corpus currently includes 18 scenarios and 58 checks
- `all` + `exact` currently passes `58/58`
- summary artifacts now include stable run metadata for backend, model, vector
  mode, duration, pass counts, and pass rate
- retrieval-only `remote-fallback` exact run now passes `58/58`
- retrieval-only `remote-fallback` summary artifact:
  `target/live-library/live-library-exact.json`

## Memloft-Derived Real-Data Slice

After the larger real-world library is stable, the next validation layer is a
technical corpus sampled from the live memloft database:

```bash
scripts/run-memloft-slice.sh
```

Default behavior:

- `retrieval` mode
- `exact` vector mode
- `local-cpu` embedding runtime
- `off` reranker runtime
- `api` extraction backend
- `openai/gpt-oss-120b` extraction model
- summary output at `target/memloft-slice/memloft-slice-exact.json`

This slice exists to pressure-test `femind` against real maintainer memory
content instead of additional synthetic scenarios.

Current memloft-slice baseline:

- the memloft-derived slice currently includes 18 scenarios and 90 checks
- `all` + `exact` currently passes `90/90`
- sources are real technical memloft records, not hand-written synthetic notes
- retrieval-only `remote-fallback` exact run now passes `90/90`
- retrieval-only `remote-fallback` summary artifact:
  `target/memloft-slice/memloft-slice-exact.json`
