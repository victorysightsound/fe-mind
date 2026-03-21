# Vector Search Accuracy & Efficiency — Phase 2

## Goal

Maximize retrieval accuracy for LongMemEval-S conditions (800+ memories,
~40 sessions with distractors, semantic queries). Phase 1 made the pipeline
work. Phase 2 makes it work *well*.

## Current Gaps

### Accuracy
1. FTS5 uses AND mode — long queries with 10+ words return near-zero results
2. No stop-word removal — common words ("how", "many", "did", "the") dilute queries
3. No query reformulation — raw question text is suboptimal for both FTS5 and vector
4. assemble_context fetches only top 50 candidates — too few for 800+ memory pools
5. Reranking not connected — cross-encoder infrastructure exists but isn't wired in
6. Session metadata (dates) not used in search ranking

### Efficiency
7. No session-aware chunking — individual turns like "Thanks!" waste embeddings
8. recallbench adapter uses store() in a loop instead of store_batch()

---

## Phase 5: Search Quality (mindcore)

### 5.1 — FTS5 OR-mode search

Add a new search method or modify FtsSearch to use OR-based queries instead
of AND. FTS5 supports OR: `"term1 OR term2 OR term3"`. This returns partial
matches ranked by BM25, which is far more useful for long natural language queries.

Implementation:
- Add `search_or_mode()` to FtsSearch that joins query terms with OR
- Use OR mode in hybrid search (the RRF merge handles ranking)
- Keep AND mode available for exact-match scenarios

### 5.2 — Stop-word removal for search queries

Strip common English stop words before passing to FTS5. Keep content words
that carry semantic meaning. Standard stop list: "the", "a", "an", "is",
"are", "was", "were", "do", "did", "does", "how", "many", "much", "in",
"on", "at", "to", "for", "of", "and", "or", "but", "not", "my", "i", "me",
"we", "you", "he", "she", "it", "they", "this", "that", "with", "from",
"be", "been", "being", "have", "has", "had", "will", "would", "could",
"should", "can", "may", "might", "shall", "about", "what", "when", "where",
"which", "who", "whom", "why", "some", "any", "all", "each", "every",
"no", "so", "if", "then", "than", "very", "just", "also", "there", "here",
"up", "out", "its".

Apply to both FTS5 queries and (optionally) before embedding for tighter vectors.

### 5.3 — Increase candidate pool in assemble_context

Change `self.search(query).limit(50)` to `self.search(query).limit(200)`.
For large memory pools (800+), 50 candidates is insufficient for RRF to
surface buried relevant content. The context budget already constrains
final output size, so over-fetching candidates is cheap.

### 5.4 — Wire reranking into hybrid search pipeline

Connect the existing reranking infrastructure into the SearchBuilder.
After RRF merge produces candidates, run cross-encoder reranking on the
top-N to refine ordering. This is the highest-precision step.

If cross-encoder is too heavy, implement a lightweight reranker:
- Score by query-term overlap ratio (what % of query content words appear)
- Boost recent memories for temporal queries
- Penalize very short memories (<20 chars)

### 5.5 — Session-aware memory chunking

Add a utility for concatenating conversation turns within a session into
larger chunks before storage. Instead of storing "User: Hi" and
"Assistant: Hello!" as separate memories, concatenate into session segments:

Strategy:
- Concatenate consecutive turns into chunks of ~500-1000 chars
- Include role prefixes ("User:", "Assistant:") in concatenated text
- Include session date in the chunk text for temporal grounding
- Filter out turns shorter than 10 chars before concatenation

This reduces the number of embeddings needed and improves retrieval quality
by giving each embedding more semantic context.

### 5.6 — Date-aware context for temporal queries

When assembling context, prepend session date metadata to each context item.
This helps the LLM answer temporal questions ("How many days in December...")
by making dates visible in the retrieved context.

---

## Phase 6: Adapter Efficiency (recallbench)

### 6.1 — Use store_batch() in MindCoreAdapter

Update `ingest_session()` in recallbench's mindcore_adapter.rs to:
1. Collect all ConversationMemory records from the session
2. Call `engine.store_batch()` instead of individual `engine.store()` calls
3. This batches embedding inference for ~20 turns per session

### 6.2 — Session chunking in adapter

Implement the chunking strategy from 5.5 in the adapter layer:
1. Concatenate turns within each session into ~500-char chunks
2. Include session date and role prefixes
3. Filter noise turns (<10 chars)
4. Store chunks via store_batch()

---

## Success Criteria

1. FTS5 OR-mode returns results for long natural language queries
2. Stop words stripped — "How many days did I spend" → "days spend"
3. Candidate pool increased to 200
4. Reranking improves top-5 precision in distractor test
5. Session chunking reduces memory count by 3-5x with better quality
6. Adapter uses store_batch() for efficiency
7. All existing tests still pass
8. Distractor scenario test still passes (or improves)
