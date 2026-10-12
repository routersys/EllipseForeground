import json
from collections.abc import Iterable
from pathlib import Path


def result_files(directory: Path, name: str) -> list[Path]:
    return sorted(directory.glob(f"{name}.jsonl")) + sorted(directory.glob(f"{name}.*.jsonl"))


def read_records(directory: Path, name: str) -> list[dict]:
    records: dict[tuple[int, int], dict] = {}
    for path in result_files(directory, name):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                records[(record["scene"], record["trial"])] = record
    return [records[key] for key in sorted(records)]


def finished_scenes(directory: Path, name: str) -> set[int]:
    return {record["scene"] for record in read_records(directory, name)}


def append_records(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
