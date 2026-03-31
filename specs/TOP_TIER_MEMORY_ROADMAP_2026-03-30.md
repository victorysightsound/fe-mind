# FeMind Top-Tier Memory Roadmap

**Date:** 2026-03-30  
**Status:** Research addendum after the v0.2 retrieval/reranking stabilization pass  
**Scope:** What to build next if FeMind is meant to become a top-tier production memory engine

---

## Why This Note Exists

The current FeMind stack is in a good local optimum:

- hybrid retrieval is stable
- MiniLM embeddings are running locally and remotely
- MiniLM cross-encoder reranking is now working locally and remotely
- practical real-world regression suites are green

That is the right time to stop chasing benchmark loops and decide what the next architectural gains should be.

The research signal from 2025-2026 is consistent: top memory systems are not just "better chunk retrieval". They are:

- routed instead of fixed-path
- explicitly temporal instead of only recency-biased
- more knowledge-centric than chunk-centric
- evaluated with retrieval diagnostics, not only end scores
- hardened against persistent memory poisoning

As of the current FeMind pass, the first layer of that roadmap is already in place:

- `QueryIntent` routing is live
- route-level temporal policy is live
- route-level state/conflict policy is live
- route-owned graph depth is now live for graph-connected questions
- aggregation routes now use an explicit engine composition path with
  distinct-match accounting instead of relying only on plain top-k retrieval
- a deterministic answer-composition layer is now live for yes/no, stateful,
  and aggregation-style questions
- deterministic composition now also records confidence, abstention, and
  rationale, and exact-detail routes can distinguish unsupported details from
  total absence of evidence
- `valid_at(...)` now enforces validity windows during retrieval
- practical scenarios can now seed explicit supersession/conflict links, so
  linked state-history behavior is part of the real regression loop
- practical evaluation now includes explicit aggregation, graph-connected, and
  provenance/abstention scenarios with coverage-sensitive pass criteria
- high-impact procedural review policy now distinguishes `pending`, `allowed`,
  `denied`, and `expired` states, and the practical suite validates those state
  transitions directly

That means the next temporal work should focus on deeper state modeling, not on
re-adding the basic routing layer.

---

## Recommended Next Priorities

### 1. Add Query Routing

FeMind should not use one retrieval recipe for every question.

The first high-leverage change is a `QueryIntent` or retrieval-plan router that can classify at least:

- exact-detail lookup
- current-state / supersession lookup
- historical / earlier-state lookup
- aggregation / counting
- graph or multi-hop lookup
- abstention-risk lookup

That router should choose retrieval behavior per query, for example:

- FTS-heavy vs dense-heavy vs balanced hybrid
- reranker enabled vs bypassed
- strict grounding enabled vs relaxed
- graph traversal enabled vs disabled
- current-state filtering vs historical recall

This is the cleanest way to improve accuracy across different scenarios without overfitting every heuristic globally.

### 2. Strengthen Temporal State Modeling

FeMind already has recency and graph concepts, but top memory quality needs more explicit temporal semantics.

The next temporal layer should center on:

- validity windows
- `supersedes` / `superseded_by` chains
- explicit "current" vs "historical" search intent
- conflict sets for changed facts
- retrieval policies that can prefer the latest answer or intentionally retrieve older state

This matters especially for:

- changed preferences
- updated project status
- corrections
- progress tracking
- "what was true before?" queries

### 3. Keep MiniLM as Stage One, But Plan a Hard-Query Late-Interaction Lane

The current MiniLM + reranker stack is still the right default path.

If FeMind needs a stronger precision ceiling later, the likely next step is not replacing MiniLM. It is adding an optional late-interaction lane such as ColBERTv2 for hard queries only.

Recommended role:

- MiniLM remains first-stage retrieval
- MiniLM cross-encoder remains the default precision layer
- ColBERTv2 becomes an optional high-precision path for routed hard queries

That preserves efficiency for everyday retrieval while opening a stronger precision ceiling for difficult cases.

### 4. Move Enhanced Memory Toward Knowledge Objects, Not Just Better Chunk Graphs

The more ambitious long-term direction is to organize memory around compact, decision-relevant knowledge units rather than only raw chunks or chunk graphs.

That includes:

- explicit propositions
- explicit procedural guidance
- update/supersession links
- knowledge-centric retrieval units

This should come after routing and temporal work, not before. Otherwise FeMind risks paying large ingest cost without enough retrieval discipline to benefit from it.

### 5. Add Memory-Safety Defenses

Persistent memory is an attack surface.

Before FeMind is treated as production-grade agent memory, it should gain:

- source trust weighting
- stronger isolation for procedural memories
- high-impact update review hooks
- poisoning-oriented regression cases
- explicit tests for malicious retrieval triggers

Current status after the 2026-03-30 trust/safety pass:

- source-trust weighting is now implemented in the default scorer
- richer provenance classes are now implemented through
  `source_kind` and `source_verification`
- procedural guidance isolation is now implemented for routed procedural
  queries when safe alternatives exist
- secret-detail abstention is now implemented for credential/material-value
  queries while grounded storage guidance remains answerable
- poisoning-oriented practical coverage now includes trusted vs untrusted
  operational guidance
- high-impact procedural memories now enter a pending-review queue with
  severity and tag metadata
- review policy transitions are now implemented, including allow/deny/expire
  state handling during retrieval
- review operations now include:
  - `review_expires_at` timestamps for temporary allowances
  - a FeMind-native `femind-review` operator CLI
  - surfaced-evidence redaction for safe secret-location queries
- scoped procedural approvals are now part of the policy surface, so
  environment-specific exceptions stay confined to the matching query context
- review lifecycle automation now also includes:
  - approval templates for staged exceptions
  - renew/revoke/replace workflows through the FeMind operator CLI
  - replacement tracking for denied procedural memories
- approval policy classes now also include breakglass and private-infrastructure
  exceptions, and those classes actively shape retrieval rather than staying as
  passive metadata
- higher-impact approval classes now also cover:
  - auth bypass
  - destructive reset windows
  - traffic cutovers
  with template defaults and scope-aware retrieval behavior
- secret-policy classes now also cover:
  - private endpoints
  - internal hostnames
  - internal share paths
  - private network ranges
- trusted sensitive guidance now resolves conflicts by provenance strength, so
  higher-confidence internal sources can suppress weaker trusted alternatives
  for sensitive infrastructure guidance
- richer provenance states now include partially verified and relayed chains so
  trusted-but-indirect guidance does not outrank fully verified internal
  sources
- trusted procedural guidance now also resolves conflicts by scope, support
  state, policy class, and provenance so scoped supported-path answers can
  suppress generic defaults or weaker workaround notes when the query is
  explicit
- the current remote-GPU engine-first baseline is:
  - practical `46/46` exact
  - practical `46/46` ann
  - live-library `58/58` exact
  - memloft-slice `90/90` exact
- the practical suite now also validates deterministic reflection over repeated
  trusted evidence, which is the current low-risk path toward higher-order
  stable knowledge synthesis without reviving benchmark-driven iteration
- FeMind now also has a consumer-safe persistence contract for those reflected
  objects, so applications can opt into storing stable derived knowledge
  without breaking generic `MemoryRecord` deserialization
- persisted reflection now also has lifecycle behavior:
  - older reflected rows with the same `knowledge_key` are superseded when the
    derived summary changes
  - persisted reflection rows get graph links back to supporting source
    memories
  - the practical suite now queries persisted reflection rows directly, not
    only runtime reflection outputs
- reflection is now application-facing too:
  - stable-summary queries can deliberately prefer current reflected rows
  - applications can inspect persisted reflected knowledge by key
  - applications can compute an explicit reflection refresh plan instead of
    treating recomputation as an internal-only concern
- FeMind now also routes `stable-summary` queries automatically when the query
  shape clearly asks for the supported, preferred, or recommended durable
  summary of a fact family, so reflection can participate in ordinary engine
  behavior instead of staying only an application opt-in
- stable-summary composition now has an explicit evidence-basis policy:
  it can answer from reflected knowledge, raw source evidence, or a deliberate
  blend of both when the query is provenance-sensitive

### 6. Keep Evaluation Engine-Centric

Daily tuning should continue to happen in FeMind's practical and real-world suites, not inside benchmark harnesses.

Recommended evaluation stack:

- deterministic practical regressions
- live-library follow-up regressions
- memloft-derived real-data slice
- benchmark checkpoints only after meaningful engine changes

---

## What ColBERTv2 Would Actually Do for FeMind

ColBERTv2 is not a replacement for the current MiniLM embedding backend.

Its best fit inside FeMind is:

- a routed late-interaction retrieval stage
- used only when the query looks hard enough to justify the extra cost

Good candidates for a ColBERTv2 path:

- exact-but-ambiguous detail questions
- queries with several constraints that must all survive ranking
- cases where chunk overlap is semantically close but only one hit is truly correct
- difficult current-vs-stale selection
- evaluation or audit mode when FeMind wants the highest retrieval precision available

Bad candidates for a ColBERTv2 path:

- ordinary "good enough" semantic retrieval
- every query by default
- hot ingest paths

### Recommended Placement

If FeMind adopts ColBERTv2 later, it should sit after the current first-stage candidate generation.

Suggested flow:

1. FTS + MiniLM vector + hybrid retrieval collect a candidate pool
2. MiniLM cross-encoder reranker trims and orders that pool
3. Only for routed hard queries, ColBERTv2 rescoring or retrieval runs on the shortlisted set or a ColBERT index
4. Context assembly uses the highest-confidence results

This keeps ColBERTv2 in the role it is best at: high-precision late interaction when normal retrieval is not enough.

---

## ColBERTv2 Resource Profile

This section is the practical answer to "what would ColBERTv2 cost and when would it cost it?"

### 1. Storage Cost

ColBERTv2 stores token-level vectors, not one vector per chunk.

The NAACL 2022 paper compresses each token vector to:

- `20` bytes with 1-bit residual compression
- `36` bytes with 2-bit residual compression

This is far smaller than earlier ColBERT storage, but it is still much heavier than single-vector retrieval because the unit of storage is every retained token, not every chunk.

Practical FeMind implications:

- A `180`-token memory chunk would need about `3.5 KB` at 1-bit or `6.3 KB` at 2-bit for raw compressed token vectors alone.
- A `300`-token memory chunk would need about `5.9 KB` at 1-bit or `10.5 KB` at 2-bit for raw compressed token vectors alone.
- Real index size will be higher after metadata, centroid structures, postings, document maps, and filesystem overhead.

Those byte totals are an inference from the paper's per-token compression numbers, not a direct benchmark on FeMind corpora.

### 2. Index Build Cost

This is where ColBERTv2 is most expensive.

Indexing means:

- encode every passage token through the model
- write compressed token vectors to disk
- build centroid-oriented search structures

The official ColBERT repository is explicit that:

- GPU is required for training
- GPU is also required for indexing

The same repository gives one practical anchor point:

- indexing `10,000` passages on a free Colab T4 GPU takes about `6` minutes

So for FeMind, ColBERTv2 is not a "recompute this on every write" feature. It is a background or batch index job.

### 3. Query-Time Cost

ColBERTv2 query cost happens in two places:

- encode the query into token vectors
- score query tokens against many candidate document-token representations with MaxSim-style late interaction

The PLAID paper is the key operational source here:

- up to `7x` faster on GPU than vanilla ColBERTv2
- up to `45x` faster on CPU than vanilla ColBERTv2
- "tens of milliseconds" on GPU at large scale
- "tens or just few hundreds of milliseconds" on CPU at large scale, even at `140M` passages

That makes ColBERTv2 operationally plausible, but only with a proper engine such as PLAID and only as a selective path. It is still much heavier than single-vector retrieval.

### 4. Idle Resource Use

If FeMind hosts ColBERTv2 as a warm local or remote service:

- CPU usage when idle should stay low
- the model weights stay resident in RAM or VRAM
- the larger ongoing cost is often the index footprint and page cache behavior, not active compute

For small corpora, that is manageable.
For large corpora, the important operational question becomes:

- how much of the ColBERT index fits in RAM
- whether the index is memory-mapped
- how much disk IO the service pays under concurrency

This is where ColBERT-serve becomes relevant. Its reported result is a roughly `90%` RAM reduction for ColBERT serving via memory-mapped scoring, specifically to make late-interaction retrieval viable on cheaper servers.

### 5. When FeMind Would Spend Those Resources

If FeMind adds a ColBERTv2 lane, the resource profile should look like this:

#### Idle

- low CPU
- non-trivial RAM/VRAM for model weights
- disk plus page-cache pressure from the token index

#### Ingest / Reindex

- highest sustained cost
- GPU preferred
- CPU, RAM, and disk all active
- should run asynchronously or on a dedicated host

#### Ordinary Query

- no ColBERTv2 work unless the router says the query is hard enough
- current MiniLM path remains dominant

#### Hard Query

- query encoding plus late-interaction scoring activates
- GPU is ideal if FeMind wants low latency
- CPU is plausible but slower unless paired with a PLAID-style engine

This is why ColBERTv2 should be routed, not universal.

---

## Recommended FeMind Position on ColBERTv2

FeMind should not adopt ColBERTv2 as the default retrieval backbone.

It should treat ColBERTv2 as:

- an optional precision tier
- probably remote-first
- likely GPU-preferred
- only worth enabling for routed hard queries or audit-grade retrieval

If FeMind reaches the point where this becomes necessary, the better implementation target is not raw ColBERT alone. It is a ColBERTv2 + PLAID- or ColBERT-serve-style operational lane.

---

## Sources

- ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction  
  https://cs.stanford.edu/~matei/papers/2022/naacl_colbert_v2.pdf
- PLAID: An Efficient Engine for Late Interaction Retrieval  
  https://arxiv.org/abs/2205.09707
- ColBERT official repository  
  https://github.com/stanford-futuredata/ColBERT
- ColBERT-serve: Efficient Multi-Stage Memory-Mapped Scoring  
  https://arxiv.org/abs/2504.14903
- RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing  
  https://arxiv.org/abs/2602.00296
- PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents  
  https://arxiv.org/abs/2603.03296
- MIRIX: Multi-Agent Memory System for LLM-Based Agents  
  https://arxiv.org/abs/2507.07957
- MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval  
  https://arxiv.org/abs/2512.16962
- RAGChecker  
  https://arxiv.org/abs/2408.08067
