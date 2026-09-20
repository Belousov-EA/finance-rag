from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.config import BM25_COLLECTION as COLLECTION_NAME
from src.config import BM25_VECTOR_NAME as VECTOR_NAME
from src.config import QDRANT_URL
from src.types import Chunk


class BM25Indexer:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)

    def recreate_collection(self) -> None:
        if self.client.collection_exists(COLLECTION_NAME):
            print(f"Deleting collection: {COLLECTION_NAME}")
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            sparse_vectors_config={
                VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        print(f"Created collection: {COLLECTION_NAME}")

    def index_batch(
        self,
        batch: list[Chunk],
        avg_len: float,
    ) -> None:
        points = []

        for chunk in batch:
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    chunk["chunk_id"],
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        VECTOR_NAME: models.Document(
                            text=chunk["text"],
                            model="Qdrant/bm25",
                            options={
                                "avg_len": avg_len,
                            },
                        )
                    },
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
