from qdrant_client import QdrantClient, models

from src.config import BM25_COLLECTION, BM25_VECTOR_NAME, QDRANT_URL


class BM25Retriever:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        response = self.client.query_points(
            collection_name=BM25_COLLECTION,
            query=models.Document(
                text=query,
                model="Qdrant/bm25",
            ),
            using=BM25_VECTOR_NAME,
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
