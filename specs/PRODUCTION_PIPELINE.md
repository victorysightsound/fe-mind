# MindCore Production Pipeline — Complete Memory System

## Goal

Build the complete production-ready memory engine pipeline: LLM extraction
→ individual fact storage → graph edges → hybrid search with graph filtering.
Every feature independently toggleable for testing and configuration.

## Why This Matters

Previous work optimized for benchmarks by chunking raw text without LLM
extraction. This doesn't match production usage where:
- An LLM processes each conversation and extracts individual facts
- Each fact becomes its own memory with its own embedding
- Graph edges connect related and superseded facts
- Search uses graph to filter outdated facts and expand to related ones

Graph doesn't work at chunk level (1000-char blocks mix old+new facts).
It only works at individual fact level. MAB Conflict Resolution proved
this — graph expansion at chunk level hurt accuracy (50%→30%).

## Architecture

### Ingest Flow
```
Raw text (conversation, document, note)
  → LlmIngest: LLM extracts individual facts via LlmCallback trait
    → Each fact stored as individual memory with embedding
    → Graph edges: SupersededBy (contradictions), RelatedTo (same entity)
    → Deduplication via content hash against existing memories
```

### Search Flow
```
Query
  → Multi-query: original + key-phrase variant
  → Hybrid: FTS5 OR-mode + vector similarity + RRF fusion
  → Graph filtering: demote superseded results, optionally expand via RelatedTo
  → Recency weighting (configurable)
  → Diversification (configurable per use case)
  → Context assembly within token budget
```

### Feature Toggles (EngineConfig)
All independently configurable:
- `llm_extraction`: on/off (falls back to passthrough/chunking when off)
- `graph_edges`: on/off (skip graph creation/querying when off)
- `embedding`: on/off (FTS5-only when off)
- `recency_weight`: 0.0 to 1.0 (0.0 = disabled)
- `diversification_limit`: 0 = unlimited, 1+ = max per session
- `deduplication`: on/off
- `vector_search_mode`: "ann" (default), "exact" (brute-force), "off"

## LlmCallback Trait

```rust
/// Trait for LLM providers — any model, any provider.
/// Consumer implements this and passes to mindcore.
pub trait LlmCallback: Send + Sync {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
}
```

Existing trait in src/traits/evolution.rs — refactor into standalone.

### Implementations
- **ApiLlmCallback**: OpenAI-compatible /v1/chat/completions via ureq (sync)
  - Works with DeepInfra, OpenAI, Together, local vLLM, Ollama
  - Feature-gated: `api-llm`
- **CliLlmCallback**: Claude/ChatGPT/Gemini via CLI
  - Pipes prompt to stdin, reads stdout
  - Feature-gated: `cli-llm`

## LlmIngest Extraction

### Extraction Prompt
The prompt asks the LLM to extract from raw text:
- **Facts**: concrete statements (X is Y, X has Z)
- **Decisions**: choices made (chose X over Y because Z)
- **Preferences**: likes/dislikes (prefers X, dislikes Y)
- **Entities**: named things (people, places, projects, tools)
- **Relationships**: how entities relate (married_to, works_at, created_by)

### Output Format
Structured response parsed into ExtractedFact items:
```rust
pub struct ExtractedFact {
    pub text: String,           // The fact statement
    pub category: String,       // fact, decision, preference, etc.
    pub importance: u8,         // 1-10
    pub entities: Vec<String>,  // Named entities mentioned
    pub relationships: Vec<(String, String, String)>,  // (subject, relation, object)
}
```

### Graph Edge Creation
During extraction, for each relationship triple:
1. Search existing memories for same subject + relation
2. If found with different object → create SupersededBy edge (old → new)
3. If found with same subject, different relation → create RelatedTo edge
4. Store edges in memory_relations table (already exists)

## store_with_extraction() API

```rust
impl<T: MemoryRecord> MemoryEngine<T> {
    /// Store raw text with LLM extraction.
    /// Extracts facts, stores each as individual memory,
    /// creates graph edges, deduplicates.
    pub fn store_with_extraction(
        &self,
        raw_text: &str,
        llm: &dyn LlmCallback,
    ) -> Result<ExtractionResult>;
}

pub struct ExtractionResult {
    pub facts_extracted: usize,
    pub memories_stored: usize,
    pub duplicates_skipped: usize,
    pub graph_edges_created: usize,
    pub superseded_count: usize,
}
```

## ANN Vector Search

Three modes, configurable:
- `"ann"` — Approximate Nearest Neighbor (default, fast at all scales)
  - Uses sqlite-vec or HNSW index
  - ~1-5ms regardless of database size
- `"exact"` — Brute-force scan (for validation/debugging)
  - Current implementation, O(n)
  - Use to verify ANN accuracy
- `"off"` — FTS5 keyword search only, no vector search

## Test Harnesses (RecallBench)

### Extraction Test
- Feed raw text through LlmIngest
- Check: facts extracted, entities found, relationships created
- No search — just extraction quality
- Zero search cost, LLM cost for extraction only

### Retrieval Test (existing)
- Search pre-stored memories
- Check: Recall@K, MRR, Hit Rate
- Zero LLM cost
- Supports configurable chunk size, budget, system

### Full Pipeline Test
- Extraction → storage → graph → search → context assembly
- Modular: can test any subset by toggling features
- Reports both extraction metrics and retrieval metrics
- The real-world test

## Testing Plan

1. **MAB Conflict Resolution with LLM extraction**: Each fact → one memory → graph edges.
   Expected: single-hop maintains 100%, multi-hop improves from 0%.

2. **LongMemEval with LLM extraction (20q sample)**: Conversations → extracted facts → search.
   Compare to chunk-based baseline (95.3% Recall@10).

3. **Feature toggle testing**: Run same questions with features on/off to measure impact of
   each component (graph, recency, diversification, ANN).

## Existing Infrastructure to Reuse

- `memory_relations` table + GraphMemory CRUD + recursive CTE traversal
- `EmbeddingBackend` trait pattern (reuse for LlmCallback)
- `AssemblyConfig` (diversification, recency, graph_depth, search_limit)
- Retrieval test harness with result saving
- Session cache (raw dataset text)
- Embedding cache system

## Dependencies

### New Cargo features
- `llm-ingest` = ["dep:ureq"] (for API LLM callback, reuse existing ureq)
- `api-llm` = ["dep:ureq"]
- `cli-llm` = []
- `ann` = ["dep:sqlite-vec"] (or similar)

### Existing features used
- `vector-search`, `api-embeddings`, `local-embeddings`, `graph-memory`
