from src.retrieval.hybrid_retriever import HybridRetriever

QUERY = "What was Walmart's operating income?"


def main() -> None:
    retriever = HybridRetriever()

    results = retriever.search(
        query=QUERY,
        limit=10,
    )

    print(f"Query: {QUERY}\n")

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(
            f"Rank: {rank} | "
            f"RRF={result['rrf_score']:.6f} | "
            f"{result['document_id']} | "
            f"page={result['page_idx']}"
        )
        print()
        print(result["text"][:800])
        print()


if __name__ == "__main__":
    main()
