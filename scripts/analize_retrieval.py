import csv
from pathlib import Path

from src.evaluation.qrels import extract_relevant_pages
from src.evaluation.ranking import chunks_to_unique_pages
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.utils.jsonl_utils import read_jsonl

DATA_PATH = Path("data/splits/train.jsonl")
OUTPUT_PATH = Path("results/retrieval_analysis.csv")

PAGE_LIMIT = 10
SOURCE_LIMIT = 50
CANDIDATE_LIMIT = 30

RRF_K = 60
BM25_WEIGHT = 0.25
DENSE_WEIGHT = 1.0


def relevant_rank(
    pages: list[tuple[str, int]],
    relevant: set[tuple[str, int]],
) -> int | None:
    for rank, page in enumerate(pages, start=1):
        if page in relevant:
            return rank

    return None


def is_hit(rank: int | None, limit: int = PAGE_LIMIT) -> bool:
    return rank is not None and rank <= limit


def main() -> None:
    samples = read_jsonl(DATA_PATH)

    dense = DenseRetriever()

    hybrid = HybridRetriever(
        rrf_k=RRF_K,
        bm25_weight=BM25_WEIGHT,
        dense_weight=DENSE_WEIGHT,
        source_limit=SOURCE_LIMIT,
    )

    reranker = Reranker()

    rows = []

    counters = {
        "dense_hit_hybrid_miss": 0,
        "dense_miss_hybrid_hit": 0,
        "hybrid_miss_reranker_hit": 0,
        "hybrid_hit_reranker_miss": 0,
        "candidate_miss": 0,
        "candidate_hit_reranker_miss": 0,
    }

    examples = {name: [] for name in counters}

    for idx, sample in enumerate(samples, start=1):
        query_id = str(sample["financebench_id"])
        query = sample["question"]
        relevant = extract_relevant_pages(sample)

        # Dense baseline
        dense_results = dense.search(
            query=query,
            limit=SOURCE_LIMIT,
        )

        dense_pages = chunks_to_unique_pages(dense_results)
        dense_rank = relevant_rank(dense_pages, relevant)

        # Tuned hybrid.
        # We keep 50 results here so we can inspect both top-10
        # quality and the top-30 reranker candidate pool.
        hybrid_results = hybrid.search(
            query=query,
            limit=SOURCE_LIMIT,
        )

        hybrid_pages = chunks_to_unique_pages(hybrid_results)
        hybrid_rank = relevant_rank(hybrid_pages, relevant)

        # Candidate pool passed to reranker
        candidates = hybrid_results[:CANDIDATE_LIMIT]

        candidate_pages = chunks_to_unique_pages(candidates)
        candidate_rank = relevant_rank(candidate_pages, relevant)
        candidate_has_relevant = candidate_rank is not None

        # Reranking exactly the candidate pool used by the tuned pipeline
        reranked_results = reranker.rerank(
            query=query,
            results=candidates,
        )

        reranked_pages = chunks_to_unique_pages(reranked_results)
        reranked_rank = relevant_rank(reranked_pages, relevant)

        dense_hit = is_hit(dense_rank)
        hybrid_hit = is_hit(hybrid_rank)
        reranker_hit = is_hit(reranked_rank)

        flags = {
            "dense_hit_hybrid_miss": dense_hit and not hybrid_hit,
            "dense_miss_hybrid_hit": not dense_hit and hybrid_hit,
            "hybrid_miss_reranker_hit": not hybrid_hit and reranker_hit,
            "hybrid_hit_reranker_miss": hybrid_hit and not reranker_hit,
            "candidate_miss": not candidate_has_relevant,
            "candidate_hit_reranker_miss": (
                candidate_has_relevant and not reranker_hit
            ),
        }

        for name, value in flags.items():
            if value:
                counters[name] += 1
                examples[name].append(query_id)

        rows.append(
            {
                "query_id": query_id,
                "question": query,
                "dense_rank": dense_rank,
                "hybrid_rank": hybrid_rank,
                "candidate_rank": candidate_rank,
                "reranked_rank": reranked_rank,
                "candidate_has_relevant": candidate_has_relevant,
                **flags,
            }
        )

        print(f"[{idx}/{len(samples)}] {query_id}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nFailure analysis:")
    for name, count in counters.items():
        print(f"{name:30} {count:>3}")

    print("\nExamples:")
    for name, query_ids in examples.items():
        print(f"\n{name}:")
        print(", ".join(query_ids[:10]) or "—")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
