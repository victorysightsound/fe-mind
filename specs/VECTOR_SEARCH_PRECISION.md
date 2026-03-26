# Vector Search Precision — Phase 3

## Goal

Close the remaining accuracy gaps in retrieval quality. Phase 1 made the
pipeline work. Phase 2 added OR-mode and stop-words. Phase 3 fixes mistakes
from Phase 2 and adds the techniques that separate good retrieval from great.

## Gaps

### Bug Fix
1. Stop-word list contains content words ("spend", "participating",
   "activities") — these carry semantic meaning and must be removed from
   the stop list

### Retrieval Precision
2. No query/passage prefixes for all-MiniLM-L6-v2 — the model supports
   instruction prefixes that align the embedding space for retrieval
3. Chunks hard-cut at 500 chars — information at boundaries gets split;
   overlapping sliding window prevents this
4. Search results cluster from the same session — no diversification
   across sessions wastes context budget on redundant info
5. Single query formulation — multiple query variants merged would
   dramatically improve recall
6. Lightweight reranker uses crude word overlap — n-gram matching or
   TF-IDF overlap would be more precise
7. Near-duplicate results waste context budget — high-cosine-similarity
   results should be deduplicated

---

## Phase 7: Precision Improvements (femind)

### 7.1 — Fix stop-word list: remove content words

Remove "spend", "participating", "activities" and any other content-bearing
words from the STOP_WORDS list. The list should ONLY contain function words
(determiners, prepositions, pronouns, auxiliary verbs, conjunctions).

Audit the full list and remove anything that could be a meaningful search term.

### 7.2 — Query/passage prefixes for CandleNativeBackend

Granite-small-r2 (and most modern embedding models) use instruction prefixes:
- Query: "Represent this sentence for searching relevant passages: <text>"
- Passage/document: stored as-is (no prefix)

Update CandleNativeBackend:
- Add `embed_query(&str)` method that prepends the query instruction prefix
- Keep `embed(&str)` for document/passage embedding (used at store time)
- Update EmbeddingBackend trait to add `embed_query()` with a default that
  falls back to `embed()`
- Update SearchBuilder to use `embed_query()` for search queries

Check all-MiniLM-L6-v2 documentation for the exact prefix format.

### 7.3 — Overlapping chunk windows

Update chunk_session() to use sliding window with overlap:
- Window size: ~500 chars (same as current)
- Overlap: ~100 chars (20%) — enough to capture information at boundaries
- Each chunk includes the last ~100 chars of the previous chunk
- This ensures no fact gets split across chunks without appearing in at
  least one complete chunk

### 7.4 — Result diversification across sessions

After search results are ranked, apply maximal marginal relevance (MMR)
or a simpler session-based diversification:
- Track which session each result came from (via metadata)
- Limit to max 3 results per session
- Backfill with results from underrepresented sessions

This prevents a single verbose session from dominating the context.

### 7.5 — Multi-query retrieval

For assemble_context, generate 2-3 query variants:
1. Original query (as-is)
2. Key-phrase extraction: strip to just nouns and verbs
3. For temporal queries: extract the time reference separately

Run search for each variant, merge and deduplicate results by memory_id,
keep the highest score for each.

### 7.6 — Improved reranker: n-gram overlap scoring

Replace simple word-overlap reranker with bigram/trigram overlap:
- Extract bigrams from query and candidate text
- Score by bigram overlap ratio (captures phrase-level matching)
- This catches "golden retriever" as a phrase, not just "golden" + "retriever"
- Keep the short-memory penalty

### 7.7 — Near-duplicate result filtering

After ranking, filter results with very high cosine similarity to each
other (>0.95). Keep only the highest-ranked of each near-duplicate cluster.
This maximizes information density in the context budget.

Load stored vectors for top-N results and compute pairwise similarity.
Remove later-ranked items that are >0.95 similar to any higher-ranked item.

---

## Success Criteria

1. Stop-word list contains ONLY function words
2. Query prefix used for search embeddings (if all-MiniLM-L6-v2 supports it)
3. Overlapping chunks prevent boundary information loss
4. No more than 3 results from any single session in top-20
5. Multi-query retrieval finds items that single query misses
6. All 223+ tests still pass
7. Distractor scenario test results improve or maintain quality
