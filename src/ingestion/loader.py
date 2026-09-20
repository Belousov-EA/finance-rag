from collections.abc import Iterator
from pathlib import Path

import pymupdf

from src.types import Page


def iter_pdf_pages(path: Path) -> Iterator[Page]:
    doc = pymupdf.open(path)

    for page_idx, page in enumerate(doc):
        text = page.get_text("text").strip()

        if not text:
            continue

        yield {
            "document_id": path.stem,
            "page_idx": page_idx,
            "page_number": page_idx + 1,
            "text": text,
        }


def iter_corpus_pages(pdf_dir: Path) -> Iterator[Page]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{idx}/{len(pdf_paths)}] {pdf_path.name}")

        yield from iter_pdf_pages(pdf_path)
