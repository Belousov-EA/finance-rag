def page_id(document_id: str, page_idx: int) -> str:
    return f"{document_id}::page::{page_idx}"


def extract_relevant_pages(
    sample: dict,
) -> set[tuple[str, int]]:
    return {
        (
            evidence["doc_name"],
            evidence["evidence_page_num"],
        )
        for evidence in sample["evidence"]
    }


def build_qrels(
    samples: list[dict],
) -> dict[str, dict[str, int]]:
    qrels = {}

    for sample in samples:
        query_id = str(sample["financebench_id"])

        qrels[query_id] = {
            page_id(document_id, page_idx): 1
            for document_id, page_idx in extract_relevant_pages(sample)
        }

    return qrels
