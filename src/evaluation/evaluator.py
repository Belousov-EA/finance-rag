from collections.abc import Callable, Sequence
from time import perf_counter

import numpy as np
from ranx import Qrels, Run, evaluate

from src.evaluation.qrels import build_qrels, page_id
from src.evaluation.ranking import chunks_to_unique_pages
from src.types import SearchResult

METRICS = [
    "hit_rate@1",
    "hit_rate@5",
    "hit_rate@10",
    "recall@10",
    "mrr@10",
]

SearchFn = Callable[
    [str, int],
    Sequence[SearchResult],
]


def evaluate_retriever(
    search: SearchFn,
    samples: list[dict],
    retrieval_limit: int = 50,
    page_limit: int = 10,
) -> dict[str, float]:
    run_dict = {}
    latencies = []

    for idx, sample in enumerate(samples, start=1):
        query_id = str(sample["financebench_id"])
        query = sample["question"]

        start = perf_counter()

        chunk_results = search(
            query,
            retrieval_limit,
        )

        latencies.append(perf_counter() - start)

        retrieved_pages = chunks_to_unique_pages(chunk_results)[:page_limit]

        run_dict[query_id] = {
            page_id(document_id, page_idx): 1.0 / rank
            for rank, (document_id, page_idx) in enumerate(
                retrieved_pages,
                start=1,
            )
        }

        print(f"[{idx}/{len(samples)}] {query_id}")

    results = evaluate(
        Qrels(build_qrels(samples)),
        Run(run_dict),
        METRICS,
    )

    return {
        **results,
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
    }


def print_results(results: dict[str, float]) -> None:
    print("\nranx:")

    for metric in METRICS:
        print(f"{metric:15} {results[metric]:.4f}")

    print("\nLatency:")
    print(f"p50: {results['p50_ms']:.1f} ms")
    print(f"p95: {results['p95_ms']:.1f} ms")
