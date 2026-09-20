import os

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

BM25_COLLECTION = "financebench_bm25_400_60"
DENSE_COLLECTION = "financebench_dense_bge_m3_400_60"

BM25_VECTOR_NAME = "bm25"

BASE_URL = "https://foundation-models.api.cloud.ru"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

EMBEDDING_SIZE = 1024

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
