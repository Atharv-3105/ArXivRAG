# ADR 0015: A curated foundation corpus, embedded with all-MiniLM-L6-v2

- **Status:** Accepted (2026-10-10). Supersedes the embedding-model choice in [ADR 0005](0005-hybrid-retrieval-rrf.md) (bge-small); the hybrid + RRF design in that ADR is unchanged.
- **Code:** `corpus/foundational.csv`, `src/pipeline/build_corpus.py`, `src/core/config.py` (`EMBEDDING_MODEL`, `QDRANT_COLLECTION`, `CHUNK_SIZE`), `src/storage/vector_db.py` (BM25 fallback)
- **Tests:** `tests/test_build_corpus.py`, `tests/test_vector_store.py`

## Context

The product's differentiator is the agents (tutor, critic, synthesizer, advisor), and an agent is only as good as what it can ground on. Until now the corpus was whatever an arXiv search query returned (3 to 5 papers per run), so most questions had nothing to retrieve and answers fell back to the model's memory, which is exactly what a grounded tutor must not do.

Two decisions follow from that:

1. **Choose the corpus, don't sample it.** A tutor for ML papers should know the papers the field is built on.
2. **Use a small, well-known embedding model that cannot become a hosting constraint.**

## Decision

### 1. A curated list of 236 papers, verified before ingest
`corpus/foundational.csv` holds `arxiv_id, topic, title_match`, grouped into 20 topics (transformers, NLP pre-training, scaling and LLMs, alignment, efficiency, reasoning, retrieval, **agents**, evaluation, safety, data, vision, multimodal, generative models, reinforcement learning, graphs, interpretability, theory).

`python -m src.pipeline.build_corpus verify` fetches arXiv's metadata for every id and compares titles. A wrong id is reported and skipped, never ingested under a famous name. `ingest` then runs the normal pipeline in batches, so it inherits idempotency (indexed = skipped, half-written = repaired, parse failure = DLQ), and re-running after a failure resumes.

### 2. all-MiniLM-L6-v2 through fastembed (ONNX), not the sentence-transformers package
- Same model weights, **no PyTorch** at runtime (the reason the API fits a free host, [ADR 0005](0005-hybrid-retrieval-rrf.md)).
- 384 dimensions, so the vector layout and storage budget are unchanged.
- 87 MB on disk, 1.5 s warm load.
- It is *symmetric*, so `EMBEDDING_QUERY_PREFIX` is empty (bge needed an instruction prefix).
- A **new collection name** (`arxiv_papers_minilm`): vectors from two models must never share a collection. Postgres still holds the text, so a model swap stays `reindex --all`.

### 3. Chunks resized to fit the model
MiniLM reads 256 word-pieces and drops the rest without an error. **Characters cannot guarantee a fit**: the first version of this decision cut 1000 → 800 characters, and measuring the stored chunks showed 32% still overflowed the window (LaTeX packs 2-3x more tokens per character than prose), losing 124 tokens each on average, nearly half the chunk.

The chunker is therefore **token-aware** (`CHUNK_MAX_TOKENS=200`, overlap 40): it counts tokens with the embedding model's own tokenizer (exact for WordPiece, which splits on whitespace first, so a running per-word sum equals the chunk's count), closes a chunk *before* the word that would overflow, and reserves the section path's tokens (the path is embedded with every chunk). Formulas are still never split. If the tokenizer is unavailable it falls back to characters with a warning. Result on 2,735 real chunks: median 198, p90 200 tokens, 0.1% over 256 (two chunks holding single oversized formulas), down from 32%.

### 4. "The project must not die on the embedding model": BM25 fallback
The dense model is needed at **query** time too, so a precomputed index cannot remove that dependency. What it can do is make it non-fatal. BM25 needs no neural model, so if the dense encoder fails to load or run, hybrid search degrades to BM25-only and logs a warning, instead of failing the turn. The explicit `mode="dense"` (used by the retrieval ablation) still raises, so a measurement is never silently taken on a different path.

## Alternatives considered

| Option | Why not |
|---|---|
| `sentence-transformers` package | Same weights as the fastembed ONNX export, plus PyTorch (hundreds of MB). It would not fit the free host's memory. |
| Keep bge-small | Stronger on public retrieval benchmarks, but the requirement here is a model that is ubiquitous, tiny and trivially replaceable. We keep the *ability* to swap (config + `reindex`) and will measure the gap on our own questions rather than assume it. |
| An embedding API (OpenAI, Cohere...) | A key, a quota and a data-egress path in the hot loop, which is the constraint we are removing. |
| Ship precomputed vectors, no runtime model | Queries still have to be embedded. Only BM25 works without a model, hence decision 4. |
| Search query as the corpus (status quo) | Non-reproducible and sparse. A fixed id list gives the same corpus everywhere, which also makes retrieval evaluation meaningful. |
| 100–200 papers (the original ask) | The list has 236. Fewer topics would be cheaper to ingest; the extra breadth is vision, generative models, RL and graphs. `--topic` and `--limit` select any subset, and the list is the only thing to edit. |

## Consequences

- Good: grounded answers for the questions a learner actually asks; reproducible corpus; a model failure degrades quality instead of availability.
- Good: ids are verified, so the corpus can be audited by reading one CSV.
- Cost: MiniLM is generally weaker than bge-small at retrieval, and its 256-token window forces smaller chunks (more chunks per paper). The hybrid BM25 branch compensates for exact terms (model names, acronyms).
- **Measured** (21 papers, 2,735 chunks, 14 hand-written questions about specific passages; bge-small embeds the same `path + content` text; small sample): MiniLM + BM25 hybrid R@1 0.86 / R@3 1.00 / MRR 0.92; MiniLM dense alone MRR 0.76; bge-small dense MRR 0.77; BM25 alone R@1 0.93 / MRR 0.95. After token-aware chunking **MiniLM and bge-small are equivalent**; an earlier 7-point gap was the truncation confound. Dense alone is the weakest branch on fact-seeking questions (names, numbers), which justifies the hybrid. Re-run on the 236-paper corpus before trusting it further.
- Open: licences. arXiv papers carry different licences, and some do not permit redistribution. Storing text for private retrieval is fine; a public demo should show snippets with attribution and link out, and the corpus list should gain a licence column.
