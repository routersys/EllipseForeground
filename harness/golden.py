import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL
import scipy

GOLDEN_PATH = Path(__file__).with_name("golden.json")


class HarnessError(Exception):
    pass


@dataclass(frozen=True)
class GoldenCase:
    name: str
    hash: str


@dataclass(frozen=True)
class Golden:
    environment: str
    input: str
    cases: tuple[GoldenCase, ...]

    def save(self, path: Path = GOLDEN_PATH) -> None:
        document = {
            "environment": self.environment,
            "input": self.input,
            "cases": [{"name": case.name, "hash": case.hash} for case in self.cases],
        }
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: Path = GOLDEN_PATH) -> "Golden":
        if not path.exists():
            raise HarnessError(f"基準値がありません。先に --golden で書いてください。{path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            cases = tuple(GoldenCase(item["name"], item["hash"]) for item in document["cases"])
            golden = cls(document["environment"], document["input"], cases)
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise HarnessError(f"基準値を読めません。{path} {error}") from error
        if len({case.name for case in golden.cases}) != len(golden.cases):
            raise HarnessError(f"基準値のケース名が重複しています。{path}")
        return golden


def current_environment() -> str:
    return (
        f"python {platform.python_version()}; numpy {np.__version__}; scipy {scipy.__version__}; "
        f"pillow {PIL.__version__}; {platform.system()} {platform.machine()}"
    )


def digest(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(f"{contiguous.dtype.str}{contiguous.shape}".encode())
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest().upper()
