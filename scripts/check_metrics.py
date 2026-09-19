from statistics import mean

from ranx import Qrels, Run, evaluate

from src.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Наши данные
relevant = {
    "q1": {
        ("doc_1", 41),
        ("doc_1", 42),
    },
    "q2": {
        ("doc_2", 7),
    },
}

retrieved = {
    "q1": [
        ("doc_1", 10),
        ("doc_1", 41),
        ("doc_1", 20),
    ],
    "q2": [
        ("doc_2", 7),
        ("doc_2", 1),
    ],
}


# Наши функции
manual_results = {
    "hit_rate@1": mean(hit_at_k(retrieved[qid], relevant[qid], 1) for qid in relevant),
    "hit_rate@5": mean(hit_at_k(retrieved[qid], relevant[qid], 5) for qid in relevant),
    "recall@10": mean(
        recall_at_k(retrieved[qid], relevant[qid], 10) for qid in relevant
    ),
    "mrr@10": mean(
        reciprocal_rank(
            retrieved[qid][:10],
            relevant[qid],
        )
        for qid in relevant
    ),
}


# ranx


def page_id(document_id: str, page_idx: int) -> str:
    return f"{document_id}::page::{page_idx}"


qrels_dict = {
    qid: {page_id(document_id, page_idx): 1 for document_id, page_idx in pages}
    for qid, pages in relevant.items()
}


# ranx ранжирует по score, поэтому задаём убывающие score
run_dict = {}

for qid, pages in retrieved.items():
    run_dict[qid] = {
        page_id(document_id, page_idx): 1.0 / rank
        for rank, (document_id, page_idx) in enumerate(
            pages,
            start=1,
        )
    }


qrels = Qrels(qrels_dict)
run = Run(run_dict)

ranx_results = evaluate(
    qrels,
    run,
    [
        "hit_rate@1",
        "hit_rate@5",
        "recall@10",
        "mrr@10",
    ],
)


# Сравнение
print("Manual:")
for metric, value in manual_results.items():
    print(f"  {metric}: {value:.4f}")

print("\nranx:")
for metric, value in ranx_results.items():
    print(f"  {metric}: {value:.4f}")

print("\nDifference:")
for metric in manual_results:
    diff = abs(manual_results[metric] - ranx_results[metric])

    print(f"  {metric}: {diff:.10f}")

    assert diff < 1e-9, f"{metric} does not match"

print("\nAll metrics match.")
