from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.types import RerankedSearchResult


class RerankedRetriever:
    def __init__(
        self,
        candidate_limit: int = 50,
        source_limit: int = 50,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self.hybrid = HybridRetriever(
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            source_limit=source_limit,
        )
        self.reranker = Reranker()
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        limit: int = 50,
    ) -> list[RerankedSearchResult]:
        candidates = self.hybrid.search(
            query=query,
            limit=self.candidate_limit,
        )

        return self.reranker.rerank(
            query=query,
            results=candidates,
            limit=limit,
        )
