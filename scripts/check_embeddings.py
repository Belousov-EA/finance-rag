from src.retrieval.embedder import Embedder


def main() -> None:
    embedder = Embedder()

    vectors = embedder.encode(
        [
            "Walmart reported operating income.",
            "What was Walmart's operating income?",
        ]
    )

    print(f"Vectors: {len(vectors)}")
    print(f"Dimension: {len(vectors[0])}")
    print(vectors[0][:5])


if __name__ == "__main__":
    main()
