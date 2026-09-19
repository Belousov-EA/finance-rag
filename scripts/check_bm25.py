from src.retrieval.bm25_retriever import BM25Retriever

QUERY = "What was Walmart's operating income?"


def main() -> None:
    retriever = BM25Retriever()

    results = retriever.search(
        QUERY,
        limit=10,
    )

    print(f"Query: {QUERY}\n")

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(
            f"Rank: {rank} | "
            f"score={result['score']:.4f} | "
            f"{result['document_id']} | "
            f"page={result['page_idx']}"
        )
        print()
        print(result["text"][:800])
        print()


if __name__ == "__main__":
    main()
