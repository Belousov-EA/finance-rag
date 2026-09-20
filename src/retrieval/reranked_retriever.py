from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

CANDIDATE_LIMIT = 50


class RerankedRetriever:
    def __init__(self) -> None:
        self.hybrid = HybridRetriever()
        self.reranker = Reranker()

    def search(
        self,
        query: str,
        limit: int = 50,
    ) -> list[dict]:
        candidates = self.hybrid.search(
            query=query,
            limit=CANDIDATE_LIMIT,
        )

        return self.reranker.rerank(
            query=query,
            results=candidates,
            limit=limit,
        )
