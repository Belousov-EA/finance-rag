from pathlib import Path

from src.ingestion.chunker import chunk_pages
from src.ingestion.loader import load_pdf

PDF_DIR = Path("data/raw/financebench/pdfs")


def main() -> None:
    pdf_path = next(PDF_DIR.glob("*.pdf"))

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Document: {pdf_path.stem}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    print()

    for chunk in chunks[:5]:
        print("=" * 80)
        print(chunk["chunk_id"])
        print(f"page_idx={chunk['page_idx']}, page_number={chunk['page_number']}")
        print()
        print(chunk["text"][:1200])
        print()


if __name__ == "__main__":
    main()
