# scripts/check_parsing.py

from pathlib import Path

from src.ingestion.loader import load_pdf

pdf_path = next(Path("data/raw/financebench/pdfs").glob("*.pdf"))

pages = load_pdf(pdf_path)

print(f"Document: {pdf_path.stem}")
print(f"Pages with text: {len(pages)}")

for page in pages[:3]:
    print("=" * 80)
    print(f"PAGE {page['page']}")
    print(page["text"][:1500])
