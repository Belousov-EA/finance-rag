from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.ingestion.bm25_indexer import BM25Indexer
from src.ingestion.chunker import iter_chunks
from src.ingestion.loader import iter_corpus_pages
from src.utils.batching import batched

PDF_DIR = Path("data/raw/financebench/pdfs")


BATCH_SIZE = 128


def create_chunk_stream():
    pages = iter_corpus_pages(PDF_DIR)

    return iter_chunks(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )


def calculate_avg_chunk_length() -> tuple[int, float]:
    total_words = 0
    chunk_count = 0

    for chunk in create_chunk_stream():
        total_words += len(chunk["text"].split())
        chunk_count += 1

    if chunk_count == 0:
        raise RuntimeError("No chunks found")

    avg_len = total_words / chunk_count

    return chunk_count, avg_len


def main() -> None:
    print("Pass 1: calculating corpus statistics...")

    chunk_count, avg_len = calculate_avg_chunk_length()

    print(f"\nChunks: {chunk_count}")
    print(f"Average chunk length: {avg_len:.2f} words")

    print("\nCreating Qdrant collection...")

    indexer = BM25Indexer()
    indexer.recreate_collection()

    print("\nPass 2: indexing...")

    chunks = create_chunk_stream()

    indexed = 0

    for batch in batched(chunks, BATCH_SIZE):
        indexer.index_batch(
            batch=batch,
            avg_len=avg_len,
        )

        indexed += len(batch)

        print(f"Indexed {indexed}/{chunk_count} ({indexed / chunk_count:.1%})")

    print("\nDone.")


if __name__ == "__main__":
    main()
