# MindCore

A standalone Rust crate providing a pluggable, feature-gated memory engine for AI agent applications.

Handles persistent storage, keyword search (FTS5), vector search (candle), hybrid retrieval (RRF), graph relationships, memory consolidation, cognitive decay modeling, and token-budget-aware context assembly.

## Design Principles

- **Library, not framework** — projects call into MindCore, not the other way around
- **Feature-gated everything** — heavy dependencies behind compile-time flags
- **Local-first** — SQLite-backed, single-file databases, no cloud dependency
- **Pure Rust where possible** — candle over ort, SQLite over Postgres
- **Proven patterns only** — every component is battle-tested in production

## Status

Architecture and research phase. See:

- `MINDCORE_ARCHITECTURE.md` — full crate structure and API design
- `MINDCORE_RESEARCH.md` — research, landscape analysis, and specification
- `DECISIONS.md` — architectural decisions log
- `research/` — competitive landscape analysis

## Origin

MindCore extracts and unifies patterns from three projects:

| Source | Contribution |
|--------|-------------|
| **Memloft** | Hybrid search (RRF), candle embeddings, FallbackBackend, background indexing |
| **Dial** | FTS5 + Porter stemming, trust scoring, token-budget context assembly |
| **PIRDLY** | Two-tier memory (global + project), error classification, MCP server interface |
