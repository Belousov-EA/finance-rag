from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.types import HybridSearchResult


class HybridRetriever:
    def __init__(
        self,
        rrf_k=60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        source_limit: int = 50,
    ) -> None:
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.source_limit = source_limit

    def search(self, query: str, limit: int) -> list[HybridSearchResult]:
        bm25_results = self.bm25.search(
            query=query,
            limit=self.source_limit,
        )

        dense_results = self.dense.search(
            query=query,
            limit=self.source_limit,
        )

        fused: dict[str, HybridSearchResult] = {}

        for results, weight in (
            (bm25_results, self.bm25_weight),
            (dense_results, self.dense_weight),
        ):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **result,
                        "rrf_score": 0.0,
                    }

                fused[chunk_id]["rrf_score"] += weight / (self.rrf_k + rank)

        return sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:limit]
