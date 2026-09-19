## Retrieval Experiments

Evaluation is performed on the training split of 50 FinanceBench questions.

Relevant items are defined at the page level:
`(document_id, page_idx)`.

Retrieved chunks are deduplicated into ranked unique pages before metric calculation.

### Results

| Experiment | Retrieval | Chunk size | Overlap | Reranker | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | p50 latency | p95 latency |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | BM25 | 400 | 60 | — | 0.1000 | 0.1400 | 0.1800 | 0.1800 | 0.1199 | 3.1 ms | 6.1 ms |
| E2 | Dense (`BAAI/bge-base-en-v1.5`) | 400 | 60 | — | — | — | — | — | — | — | — |
| E3 | Hybrid (BM25 + Dense + RRF) | 400 | 60 | — | — | — | — | — | — | — | — |
| E4 | Hybrid + Reranker | 400 | 60 | `BAAI/bge-reranker-base` | — | — | — | — | — | — | — |

### E1 — BM25 baseline

- Vector database: Qdrant
- Sparse retrieval: BM25
- Chunk size: 400 tokens
- Chunk overlap: 60 tokens
- Retrieval limit before page deduplication: 50 chunks
- Evaluation cutoff: 10 unique pages
- Evaluation set: 50 questions
- Metrics implementation verified against `ranx`
