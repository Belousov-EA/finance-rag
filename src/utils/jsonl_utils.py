import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(data: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n")
