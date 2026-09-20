from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.config import DENSE_COLLECTION as COLLECTION_NAME
from src.config import QDRANT_URL
from src.retrieval.embedder import Embedder

VECTOR_SIZE = 1024


class DenseIndexer:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)
        self.embedder = Embedder()

    def recreate_collection(self) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            print(f"Deleting collection: {COLLECTION_NAME}")
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        print(f"Created collection: {COLLECTION_NAME}")

    def index_batch(self, batch: list[dict]) -> None:
        texts = [chunk["text"] for chunk in batch]

        vectors = self.embedder.encode(texts)

        points = []

        for chunk, vector in zip(batch, vectors):
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    chunk["chunk_id"],
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "page_idx": chunk["page_idx"],
                        "page_number": chunk["page_number"],
                        "chunk_idx": chunk["chunk_idx"],
                        "text": chunk["text"],
                    },
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
