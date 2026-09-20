from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.types import RerankedSearchResult


class RerankedRetriever:
    def __init__(self, candidate_limit=50) -> None:
        self.hybrid = HybridRetriever()
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
