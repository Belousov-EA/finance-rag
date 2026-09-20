from pathlib import Path
from time import perf_counter

import numpy as np
from ranx import Qrels, Run, evaluate

from src.evaluation.qrels import (
    extract_relevant_pages,
    page_id,
)
from src.evaluation.ranking import chunks_to_unique_pages
from src.retrieval.reranked_retriever import RerankedRetriever
from src.utils.jsonl_utils import read_jsonl

TRAIN_PATH = "data/splits/train.jsonl"

RETRIEVAL_LIMIT = 50
PAGE_LIMIT = 10


def main() -> None:
    samples = read_jsonl(Path(TRAIN_PATH))

    retriever = RerankedRetriever()

    latencies = []

    qrels_dict = {}
    run_dict = {}

    for idx, sample in enumerate(samples, start=1):
        query_id = str(sample["financebench_id"])
        query = sample["question"]

        relevant = extract_relevant_pages(sample)

        start = perf_counter()

        chunk_results = retriever.search(
            query=query,
            limit=RETRIEVAL_LIMIT,
        )

        latencies.append(perf_counter() - start)

        retrieved_pages = chunks_to_unique_pages(chunk_results)[:PAGE_LIMIT]

        # ranx qrels
        qrels_dict[query_id] = {
            page_id(document_id, page_idx): 1 for document_id, page_idx in relevant
        }

        # ranx run
        run_dict[query_id] = {
            page_id(document_id, page_idx): 1.0 / rank
            for rank, (document_id, page_idx) in enumerate(retrieved_pages, start=1)
        }

        print(f"[{idx}/{len(samples)}] {query_id}")

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    ranx_results = evaluate(
        qrels,
        run,
        [
            "hit_rate@1",
            "hit_rate@5",
            "hit_rate@10",
            "recall@10",
            "mrr@10",
        ],
    )

    print("\nranx:")
    for metric, value in ranx_results.items():
        print(f"{metric:15} {value:.4f}")

    print("\nLatency:")
    print(f"p50: {np.percentile(latencies, 50) * 1000:.1f} ms")
    print(f"p95: {np.percentile(latencies, 95) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
