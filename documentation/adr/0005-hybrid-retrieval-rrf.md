# ADR 0005: Hybrid retrieval (dense + BM25) fused with RRF, on ONNX embeddings

- **Status:** Accepted (2026-09-26)
- **Code:** `src/storage/vector_db.py` (`search`, collection layout, schema guard), `src/storage/embedder.py`, `src/agents/graph.py` (`select_sources`)
- **Tests:** `tests/test_vector_store.py`, `tests/test_chat_graph.py`

## Context

The plan (§2, §4) says "search-first": hybrid retrieval with BM25 + vector search, fused before generation. ML questions mix two kinds of matching:

- **Semantic:** "why is attention expensive" should find "quadratic complexity in sequence length". Dense embeddings handle this.
- **Exact tokens:** "LoRA", "NF4", "KV-cache", "GPT-4o", "Eq. 3", and author or method names. Dense models blur rare tokens, so "LoRA" can come back as generic fine-tuning text. BM25 matches these exactly.

Neither retriever alone covers both kinds.

## Decision

1. **Two named vectors per point, in one collection.** `dense` holds bge-small (384-d, cosine). `bm25` holds sparse term weights, with `Modifier.IDF`: Qdrant keeps the corpus statistics and applies IDF at query time, so document vectors never need recomputing as the corpus grows.
2. **Server-side Reciprocal Rank Fusion in one request.** Two `Prefetch` branches (dense and BM25, `RETRIEVAL_PREFETCH` candidates each) are fused with `FusionQuery(RRF)`. RRF scores a document as Σ 1/(k + rank). It uses only ranks, so the two branches' incompatible score scales (cosine in [-1, 1] versus unbounded BM25) never need calibrating.
3. **Diversity cap.** `RETRIEVAL_MAX_PER_PAPER` limits how many chunks one paper contributes, after over-fetching 2 × top-k. One long paper cannot crowd out all others, which matters most for "compare" questions.
4. **Embeddings on fastembed (ONNX Runtime) instead of sentence-transformers (PyTorch).** Same bge-small weights, no torch. PyTorch is by far the largest dependency, and dropping it is what makes a 512 MB free host plausible. The image size and RSS before and after are to be measured in Phase 6. The BM25 encoder comes from the same library.
5. **The query instruction is config.** bge recommends a retrieval instruction prefix for short queries. It is `EMBEDDING_QUERY_PREFIX`, so the Phase 6 eval can measure it instead of us assuming.
6. **Schema guard.** On startup, a collection whose layout differs (the old unnamed dense-only v1, no BM25, or a different dimension) is refused with the fix in the error message: use a new collection name and run `reindex --all`, which ADR 0002 made cheap.
7. **Denormalised payload.** `title` and `published` are stored on each point, so a result can be shown and cited without a Postgres round trip on the hot path.

## Alternatives considered

| Option | Why not |
|---|---|
| Dense only | Loses exact matches on names and acronyms, the queries researchers type most. |
| BM25 only (Postgres full-text) | No semantic matching. It stays as the plan's "full-text fallback" option. |
| Weighted score sum (α·dense + β·bm25) | Needs score normalisation and tuned weights per corpus. RRF is parameter-light and robust. |
| Client-side fusion (two queries, merge in Python) | Two round trips and more code. Qdrant does it in one request. |
| SPLADE / BM42 learned sparse vectors | Better recall on paper, but bigger models and slower on CPU. BM25 is the strong, cheap baseline, and it's one config value (`SPARSE_MODEL`) to try later against the eval set. |
| Cross-encoder re-ranker | The largest quality gain per unit of effort, but about 100–300 ms of CPU per query on a free host. Deferred to Phase 4 or 6 behind a flag, and measured first. |

## Consequences

- Good: exact-token and semantic queries both work, fusion needs no tuning, and it's one Qdrant call.
- Cost: sparse vectors add storage and a second encoder at ingest (BM25 is cheap: no neural model).
- Migration: v1 collections must be rebuilt (`reindex --all`). No re-download or re-parse is needed.
- RRF discards score magnitudes, so "how relevant is the best hit" cannot be read from the fused score. A relevance threshold, if needed, should come from the dense branch or a re-ranker.
