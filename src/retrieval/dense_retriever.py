from qdrant_client import QdrantClient

from src.config import DENSE_COLLECTION, QDRANT_URL
from src.retrieval.embedder import Embedder
from src.types import SearchResult


class DenseRetriever:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)
        self.embedder = Embedder()

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        query_vector = self.embedder.encode([query])[0]

        response = self.client.query_points(
            collection_name=DENSE_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        results = []

        for point in response.points:
            results.append(
                {
                    "score": point.score,
                    "chunk_id": point.payload["chunk_id"],
                    "document_id": point.payload["document_id"],
                    "page_idx": point.payload["page_idx"],
                    "page_number": point.payload["page_number"],
                    "chunk_idx": point.payload["chunk_idx"],
                    "text": point.payload["text"],
                }
            )

        return results
