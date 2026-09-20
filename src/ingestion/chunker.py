from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def iter_chunks(
    pages,
    chunk_size: int = 400,
    overlap: int = 60,
):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap

    for page in pages:
        encoded = tokenizer(
            page["text"],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        offsets = encoded["offset_mapping"]

        for chunk_idx, start in enumerate(range(0, len(offsets), step)):
            end = min(start + chunk_size, len(offsets))

            if start >= end:
                break

            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            chunk_text = page["text"][char_start:char_end].strip()

            if not chunk_text:
                continue

            yield {
                "chunk_id": (
                    f"{page['document_id']}"
                    f"::page::{page['page_idx']}"
                    f"::chunk::{chunk_idx}"
                ),
                "document_id": page["document_id"],
                "page_idx": page["page_idx"],
                "page_number": page["page_number"],
                "chunk_idx": chunk_idx,
                "text": chunk_text,
            }

            if end == len(offsets):
                break
