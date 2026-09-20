from collections.abc import Sequence

from src.types import SearchResult


def chunks_to_unique_pages(
    results: Sequence[SearchResult],
) -> list[tuple[str, int]]:
    pages = []
    seen = set()

    for result in results:
        page = (
            result["document_id"],
            result["page_idx"],
        )

        if page in seen:
            continue

        seen.add(page)
        pages.append(page)

    return pages
