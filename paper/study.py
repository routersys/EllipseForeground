import json
from dataclasses import dataclass
from pathlib import Path

from ellipse_foreground.evaluation.statistics import Interval
from paper.numbers import interval_text
from paper.results import TREE, Measure, Results


@dataclass(frozen=True)
class Timing:
    tree_ms: float
    graph_cut_ms: float
    pixel_count: str
    environment: str
    instances: int
    problems: int
    boykov_kolmogorov_ms: float | None = None
    boykov_kolmogorov_library: str | None = None

    @property
    def ratio(self) -> float:
        return self.graph_cut_ms / self.tree_ms

    @property
    def boykov_kolmogorov_ratio(self) -> float | None:
        return None if self.boykov_kolmogorov_ms is None else self.boykov_kolmogorov_ms / self.tree_ms

    @staticmethod
    def read(path: Path) -> "Timing | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Timing(
            data["tree_ms"],
            data["graph_cut_ms"],
            data["pixel_count"],
            data["environment"],
            data["instances"],
            data["problems"],
            data.get("boykov_kolmogorov_ms"),
            data.get("boykov_kolmogorov_library"),
        )


class Study:
    def __init__(self, results: Results, timing: Timing | None) -> None:
        self.results = results
        self.measure = Measure(results)
        self.timing = timing

    def mean(self, key: str, method: str, quantity: str = "iou", family: str = TREE) -> float:
        return self.measure.mean(key, method, quantity, family)

    def interval(self, key: str, first: str, second: str, quantity: str = "iou", **families: str) -> Interval:
        return self.measure.interval(key, first, second, quantity, **families)

    def difference(self, key: str, first: str, second: str, quantity: str = "iou", **families: str) -> float:
        return self.measure.difference(key, first, second, quantity, **families)

    def difference_text(
        self, key: str, first: str, second: str, quantity: str = "iou", digits: int = 3, **families: str
    ) -> str:
        return interval_text(self.interval(key, first, second, quantity, **families), digits)
