from pathlib import Path

from src.evaluation.evaluator import (
    evaluate_retriever,
    print_results,
)
from src.retrieval.reranked_retriever import RerankedRetriever
from src.utils.jsonl_utils import read_jsonl

DATA_PATH = Path("data/splits/train.jsonl")


def main() -> None:
    samples = read_jsonl(DATA_PATH)
    retriever = RerankedRetriever()

    results = evaluate_retriever(
        retriever.search,
        samples,
    )

    print_results(results)


if __name__ == "__main__":
    main()
