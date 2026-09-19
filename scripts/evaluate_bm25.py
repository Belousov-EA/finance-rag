from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from ranx import Qrels, Run, evaluate

from src.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.qrels import (
    extract_relevant_pages,
    page_id,
)
from src.evaluation.ranking import chunks_to_unique_pages
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.jsonl_utils import read_jsonl

TRAIN_PATH = "data/splits/train.jsonl"

# Берём больше chunks, потому что после дедупликации
# 10 chunks могут превратиться, например, в 6 страниц.
RETRIEVAL_LIMIT = 50
PAGE_LIMIT = 10


def main() -> None:
    samples = read_jsonl(Path(TRAIN_PATH))

    retriever = BM25Retriever()

    manual_hit_1 = []
    manual_hit_5 = []
    manual_hit_10 = []
    manual_recall_10 = []
    manual_rr_10 = []

    qrels_dict = {}
    run_dict = {}

    latencies = []

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

        retrieved_pages = chunks_to_unique_pages(chunk_results)

        retrieved_pages = retrieved_pages[:PAGE_LIMIT]

        # -------------------------
        # Наши функции
        # -------------------------

        manual_hit_1.append(hit_at_k(retrieved_pages, relevant, 1))

        manual_hit_5.append(hit_at_k(retrieved_pages, relevant, 5))

        manual_hit_10.append(hit_at_k(retrieved_pages, relevant, 10))

        manual_recall_10.append(recall_at_k(retrieved_pages, relevant, 10))

        manual_rr_10.append(
            reciprocal_rank(
                retrieved_pages[:10],
                relevant,
            )
        )

        # -------------------------
        # ranx qrels
        # -------------------------

        qrels_dict[query_id] = {
            page_id(document_id, page_idx): 1 for document_id, page_idx in relevant
        }

        # ranx требует score.
        # Нам сейчас важен уже готовый порядок страниц,
        # поэтому score просто монотонно убывает.
        run_dict[query_id] = {
            page_id(document_id, page_idx): 1.0 / rank
            for rank, (document_id, page_idx) in enumerate(retrieved_pages, start=1)
        }

        print(f"[{idx}/{len(samples)}] {query_id}")

    manual_results = {
        "hit_rate@1": mean(manual_hit_1),
        "hit_rate@5": mean(manual_hit_5),
        "hit_rate@10": mean(manual_hit_10),
        "recall@10": mean(manual_recall_10),
        "mrr@10": mean(manual_rr_10),
    }

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

    print("\nManual:")
    for metric, value in manual_results.items():
        print(f"{metric:15} {value:.4f}")

    print("\nranx:")
    for metric, value in ranx_results.items():
        print(f"{metric:15} {value:.4f}")

    print("\nDifference:")
    for metric in manual_results:
        diff = abs(manual_results[metric] - ranx_results[metric])

        print(f"{metric:15} {diff:.10f}")

    print("\nLatency:")
    print(f"p50: {np.percentile(latencies, 50) * 1000:.1f} ms")
    print(f"p95: {np.percentile(latencies, 95) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
