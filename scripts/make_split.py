import random
from pathlib import Path

from src.utils.jsonl_utils import read_jsonl, write_jsonl

SEED = 42
TRAIN_SIZE = 50

SOURCE = Path("data/raw/financebench/data/financebench_open_source.jsonl")
OUTPUT_DIR = Path("data/splits")


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
