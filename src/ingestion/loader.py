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
                "page": page_idx + 1,
                "text": text,
            }
        )

    return pages
