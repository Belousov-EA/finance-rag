from pathlib import Path

from src.evaluation.evaluator import (
    evaluate_retriever,
    print_results,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.jsonl_utils import read_jsonl

DATA_PATH = Path("data/splits/train.jsonl")


def main() -> None:
    samples = read_jsonl(DATA_PATH)
    retriever = HybridRetriever()

    results = evaluate_retriever(
        retriever.search,
        samples,
    )

    print_results(results)


if __name__ == "__main__":
    main()
