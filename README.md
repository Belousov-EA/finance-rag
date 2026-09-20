## Retrieval Experiments

Evaluation is performed on the training split of 50 FinanceBench questions.

Relevant items are defined at the page level:
`(document_id, page_idx)`.

Retrieved chunks are deduplicated into ranked unique pages before metric calculation.

### Results
| Experiment | Retrieval | Chunk size | Overlap | Reranker | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 | p50 latency | p95 latency |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | BM25 | 400 | 60 | — | 0.1000 | 0.1400 | 0.1800 | 0.1800 | 0.1199 | 3.1 ms | 6.1 ms |
| E2 | Dense (`BAAI/bge-m3`, Cloud.ru) | 400 | 60 | — | 0.1400 | 0.3000 | 0.3200 | 0.2900 | 0.1973 | 275.0 ms | 543.3 ms |
| E3 | Hybrid (BM25 + Dense + RRF) | 400 | 60 | — | 0.1200 | 0.2200 | 0.3000 | 0.2800 | 0.1616 | 295.7 ms | 502.3 ms |
| E4 | Hybrid + Reranker | 400 | 60 | `BAAI/bge-reranker-v2-m3` | 0.2000 | 0.3800 | 0.4600 | 0.4400 | 0.2811 | 1028.3 ms | 1228.0 ms |

### E1 — BM25 baseline

- Vector database: Qdrant
- Sparse retrieval: BM25
- Chunk size: 400 tokens
- Chunk overlap: 60 tokens
- Retrieval limit before page deduplication: 50 chunks
- Evaluation cutoff: 10 unique pages
- Evaluation set: 50 questions
- Metrics implementation verified against `ranx`

### E2 — Dense retrieval baseline

- Vector database: Qdrant
- Dense retrieval model: `BAAI/bge-m3`
- Embedding inference: Cloud.ru API
- Embedding dimension: 1024
- Similarity metric: cosine similarity
- Chunk size: 400 tokens
- Chunk overlap: 60 tokens
- Retrieval limit before page deduplication: 50 chunks
- Evaluation cutoff: 10 unique pages
- Evaluation set: 50 questions
- Query embedding latency is included in retrieval latency
- Metrics calculated with `ranx`

### E3 — Hybrid retrieval with RRF

- Vector database: Qdrant
- Sparse retrieval: BM25
- Dense retrieval model: `BAAI/bge-m3`
- Embedding inference: Cloud.ru API
- Fusion method: Reciprocal Rank Fusion (RRF)
- RRF constant: 60
- Fusion level: chunk-level
- Chunk size: 400 tokens
- Chunk overlap: 60 tokens
- Retrieval limit per retriever: 50 chunks
- Page deduplication is applied after RRF
- Evaluation cutoff: 10 unique pages
- Evaluation set: 50 questions
- Query embedding latency is included in retrieval latency
- Metrics calculated with `ranx`

### E4 — Hybrid retrieval with reranking

- Vector database: Qdrant
- Sparse retrieval: BM25
- Dense retrieval model: `BAAI/bge-m3`
- Dense embedding inference: Cloud.ru API
- Fusion method: Reciprocal Rank Fusion (RRF)
- RRF constant: 60
- Reranker: `BAAI/bge-reranker-v2-m3`
- Reranker inference: Cloud.ru API
- Reranker candidate pool: 50 hybrid chunks
- Chunk size: 400 tokens
- Chunk overlap: 60 tokens
- Page deduplication is applied after reranking
- Evaluation cutoff: 10 unique pages
- Evaluation set: 50 questions
- End-to-end retrieval latency includes embedding generation, Qdrant retrieval, RRF fusion, and reranking
- Metrics calculated with `ranx`
