from pathlib import Path

import pymupdf


def load_pdf(path: Path) -> list[dict]:
    doc = pymupdf.open(path)

    pages = []

    for page_idx, page in enumerate(doc):
        text = page.get_text("text").strip()

        if not text:
            continue

        pages.append(
            {
                "document_id": path.stem,
                "page_idx": page_idx,
                "page_number": page_idx + 1,
                "text": text,
            }
        )

    return pages


def iter_pdf_pages(path: Path):
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


def iter_corpus_pages(pdf_dir: Path):
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{idx}/{len(pdf_paths)}] {pdf_path.name}")

        yield from iter_pdf_pages(pdf_path)
