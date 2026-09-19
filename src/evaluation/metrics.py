def hit_at_k(
    retrieved: list[tuple[str, int]],
    relevant: set[tuple[str, int]],
    k: int,
) -> float:
    return float(any(item in relevant for item in retrieved[:k]))


def recall_at_k(
    retrieved: list[tuple[str, int]],
    relevant: set[tuple[str, int]],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def reciprocal_rank(
    retrieved: list[tuple[str, int]],
    relevant: set[tuple[str, int]],
) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0
