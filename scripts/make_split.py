import json
import random
from pathlib import Path

SEED = 42
TRAIN_SIZE = 50

SOURCE = Path("data/raw/financebench/data/financebench_document_information.jsonl")
OUTPUT_DIR = Path("data/splits")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(data: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.writelines(json.dumps(row, ensure_ascii=False))


def main() -> None:
    data = read_jsonl(SOURCE)

    print(f"Readed {len(data)} lines")

    r = random.Random(SEED)
    r.shuffle(data)

    train = data[:TRAIN_SIZE]
    test = data[TRAIN_SIZE:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(train, OUTPUT_DIR / "train.jsonl")
    write_jsonl(test, OUTPUT_DIR / "test.jsonl")

    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")


if __name__ == "__main__":
    main()
