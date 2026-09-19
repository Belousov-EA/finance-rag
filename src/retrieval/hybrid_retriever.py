from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever

RRF_K = 60


class HybridRetriever:
    def __init__(self) -> None:
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

    def search(
        self,
        query: str,
        limit: int = 50,
    ) -> list[dict]:
        bm25_results = self.bm25.search(
            query=query,
            limit=limit,
        )

        dense_results = self.dense.search(
            query=query,
            limit=limit,
        )

        fused = {}

        for results in (bm25_results, dense_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **result,
                        "rrf_score": 0.0,
                    }

                fused[chunk_id]["rrf_score"] += 1.0 / (RRF_K + rank)

        return sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:limit]
