from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.chunker import iter_chunks
from src.ingestion.dense_indexer import DenseIndexer
from src.ingestion.loader import iter_corpus_pages
from src.utils.batching import batched

PDF_DIR = Path("data/raw/financebench/pdfs")


# Начнём консервативно для внешнего API.
BATCH_SIZE = 512


def create_chunk_stream():
    pages = iter_corpus_pages(PDF_DIR)

    return iter_chunks(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )


def main() -> None:
    indexer = DenseIndexer()
    indexer.recreate_collection()

    chunks = create_chunk_stream()

    indexed = 0

    for batch in batched(chunks, BATCH_SIZE):
        indexer.index_batch(batch)

        indexed += len(batch)

        print(f"Indexed {indexed} chunks")

    print("\nDone.")


if __name__ == "__main__":
    main()
