from pathlib import Path

from src.evaluation.evaluator import (
    evaluate_retriever,
    print_results,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.jsonl_utils import read_jsonl

DATA_PATH = Path("data/splits/train.jsonl")

CONFIGS = [
    {
        "name": "rrf_k=20",
        "rrf_k": 20,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
    },
    {
        "name": "rrf_k=60",
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
    },
    {
        "name": "rrf_k=100",
        "rrf_k": 100,
        "bm25_weight": 1.0,
        "dense_weight": 1.0,
    },
    {
        "name": "bm25_weight=0.25",
        "rrf_k": 60,
        "bm25_weight": 0.25,
        "dense_weight": 1.0,
    },
    {
        "name": "bm25_weight=0.50",
        "rrf_k": 60,
        "bm25_weight": 0.50,
        "dense_weight": 1.0,
    },
    {
        "name": "bm25_weight=0.75",
        "rrf_k": 60,
        "bm25_weight": 0.75,
        "dense_weight": 1.0,
    },
]


def main() -> None:
    samples = read_jsonl(DATA_PATH)

    for config in CONFIGS:
        print("\n" + "=" * 80)
        print(config["name"])
        print("=" * 80)

        retriever = HybridRetriever(
            rrf_k=config["rrf_k"],
            bm25_weight=config["bm25_weight"],
            dense_weight=config["dense_weight"],
        )

        results = evaluate_retriever(
            retriever.search,
            samples,
        )

        print_results(results)


if __name__ == "__main__":
    main()
