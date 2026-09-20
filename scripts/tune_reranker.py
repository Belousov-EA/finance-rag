from pathlib import Path

from src.evaluation.evaluator import (
    evaluate_retriever,
    print_results,
)
from src.retrieval.reranked_retriever import RerankedRetriever
from src.utils.jsonl_utils import read_jsonl

DATA_PATH = Path("data/splits/train.jsonl")

RRF_K = 60
BM25_WEIGHT = 0.25
DENSE_WEIGHT = 1.0
SOURCE_LIMIT = 50

CANDIDATE_LIMITS = [
    10,
    20,
    30,
    50,
]


def main() -> None:
    samples = read_jsonl(DATA_PATH)

    for candidate_limit in CANDIDATE_LIMITS:
        print("\n" + "=" * 80)
        print(f"candidate_limit={candidate_limit}")
        print("=" * 80)

        retriever = RerankedRetriever(
            candidate_limit=candidate_limit,
            source_limit=SOURCE_LIMIT,
            rrf_k=RRF_K,
            bm25_weight=BM25_WEIGHT,
            dense_weight=DENSE_WEIGHT,
        )

        results = evaluate_retriever(
            retriever.search,
            samples,
        )

        print_results(results)


if __name__ == "__main__":
    main()
