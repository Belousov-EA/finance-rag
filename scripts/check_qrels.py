from pathlib import Path

import pymupdf

from src.utils.jsonl_utils import read_jsonl

TRAIN_PATH = Path("data/splits/train.jsonl")
PDF_DIR = Path("data/raw/financebench/pdfs")


def main() -> None:
    samples = read_jsonl(TRAIN_PATH)

    pdfs = {path.stem: path for path in PDF_DIR.glob("*.pdf")}

    missing_pdfs = []
    invalid_pages = []
    doc_mismatches = []

    evidence_count = 0
    relevant_pages = set()

    for sample in samples:
        for evidence in sample["evidence"]:
            evidence_count += 1

            doc_name = evidence["doc_name"]
            page_idx = evidence["evidence_page_num"]

            relevant_pages.add((doc_name, page_idx))

            # Gold evidence points to another document?
            if doc_name != sample["doc_name"]:
                doc_mismatches.append(
                    (
                        sample["financebench_id"],
                        sample["doc_name"],
                        doc_name,
                    )
                )

            # PDF exists?
            if doc_name not in pdfs:
                missing_pdfs.append((sample["financebench_id"], doc_name))
                continue

            # Page exists?
            with pymupdf.open(pdfs[doc_name]) as doc:
                if not 0 <= page_idx < len(doc):
                    invalid_pages.append(
                        (
                            sample["financebench_id"],
                            doc_name,
                            page_idx,
                            len(doc),
                        )
                    )

    print(f"Questions: {len(samples)}")
    print(f"Evidence entries: {evidence_count}")
    print(f"Unique relevant pages: {len(relevant_pages)}")

    print(f"\nMissing PDFs: {len(missing_pdfs)}")
    print(f"Invalid pages: {len(invalid_pages)}")
    print(f"Document mismatches: {len(doc_mismatches)}")

    if missing_pdfs:
        print("\nMissing PDFs:")
        for item in missing_pdfs[:10]:
            print(item)

    if invalid_pages:
        print("\nInvalid pages:")
        for item in invalid_pages[:10]:
            print(item)

    if doc_mismatches:
        print("\nDocument mismatches:")
        for item in doc_mismatches[:10]:
            print(item)


if __name__ == "__main__":
    main()
