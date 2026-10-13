from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ellipse_foreground.evaluation.results import read_records
from ellipse_foreground.evaluation.sprites import SCENES_PER_SPRITE
from ellipse_foreground.evaluation.statistics import Interval, bootstrap_interval, scene_means
from paper.numbers import separates_from_zero

TREE = "tree"
GRAPH = "graph"
OPEN_END = 10**6


@dataclass(frozen=True)
class Subset:
    key: str
    group: str
    name: str
    short: str
    scenes: range = range(OPEN_END)

    def includes(self, scene: int) -> bool:
        return scene in self.scenes


SUBSETS = {
    subset.key: subset
    for subset in (
        Subset("sprite1", "sprite1", "立ち絵1・全12条件", "立ち絵1"),
        Subset("dose0", "sprite1", "立ち絵1・星雲・転写0%", "立ち絵1の転写0%", range(0, 1)),
        Subset("dose25", "sprite1", "立ち絵1・星雲・転写25%", "立ち絵1の転写25%", range(1, 2)),
        Subset("dose50", "sprite1", "立ち絵1・星雲・転写50%", "立ち絵1の転写50%", range(2, 3)),
        Subset("dose75", "sprite1", "立ち絵1・星雲・転写75%", "立ち絵1の転写75%", range(3, 4)),
        Subset("dose100", "sprite1", "立ち絵1・星雲・転写100%", "立ち絵1の転写100%", range(4, 5)),
        Subset("others", "sprite1", "立ち絵1・他の7背景・100%", "立ち絵1の他の7背景", range(5, OPEN_END)),
        Subset("sprite2", "sprite2", "立ち絵2・8背景・100%", "立ち絵2"),
        Subset("collection0", "collection0", "17体・転写0%", "17体の転写0%"),
        Subset("collection100", "collection100", "17体・転写100%", "17体の転写100%"),
        Subset("public0", "public0", "公開素材・転写0%", "公開素材の転写0%"),
        Subset("public100", "public100", "公開素材・転写100%", "公開素材の転写100%"),
        Subset("photo", "photo", "写真（PPM-100）", "写真"),
    )
}
SETS = ("sprite1", "sprite2", "collection0", "collection100", "public0", "public100")
EARLIER_SETS = ("sprite1", "sprite2", "collection0", "collection100")
PUBLIC_SETS = ("public0", "public100")
PHOTO_SET = "photo"
DESIGN_SETS = ("sprite1", "collection0", "collection100")
DOSES = ("dose0", "dose25", "dose50", "dose75", "dose100")
METHODS = ("ellipse", "prior", "tex", "app", "mix", "adaptive_c", "adaptive", "oracle")
CUE_METHODS = ("prior", "tex", "app", "mix", "adaptive_c", "adaptive")
CUES = ("tex", "app", "mix", "adaptive_c", "adaptive")
MODEL_NAMES = ("S", "T", "L")
WORK_WIDTHS = (128, 256, 384)


class Results:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._records: dict[tuple[str, str], list[dict]] = {}
        self._intervals: dict[tuple, Interval] = {}

    def _load(self, kind: str, name: str) -> list[dict]:
        key = (kind, name)
        if key not in self._records:
            self._records[key] = read_records(self.directory / kind, name)
        return self._records[key]

    def _select(self, kind: str, name: str, subset: Subset) -> list[dict]:
        return [record for record in self._load(kind, name) if subset.includes(record["scene"])]

    def trials(self, key: str) -> list[dict]:
        subset = SUBSETS[key]
        return self._select("comparison", subset.group, subset)

    def ablations(self, key: str) -> list[dict]:
        subset = SUBSETS[key]
        return self._select("ablation", subset.group, subset)

    def model_trials(self, key: str, model: str) -> list[dict]:
        subset = SUBSETS[key]
        return self._select("models", f"{subset.group}-{model}", subset)

    def resolution_trials(self, key: str, width: int) -> list[dict]:
        subset = SUBSETS[key]
        return self._select("resolution", f"{subset.group}-{width}", subset)

    def scenes(self, key: str) -> int:
        return len({record["scene"] for record in self.trials(key)})

    def occupancy(self, key: str) -> float:
        return scene_mean(self.trials(key), lambda record: record["occupancy"])

    def spill(self, key: str) -> float:
        return scene_mean(self.trials(key), lambda record: record["spill"])


def scene_mean(records: Sequence[dict], value: Callable[[dict], float]) -> float:
    return float(scene_means(records, value).mean())


def sprite_means(records: Sequence[dict], value: Callable[[dict], float]) -> np.ndarray:
    by_scene: dict[int, list[float]] = {}
    for record in records:
        by_scene.setdefault(record["scene"], []).append(value(record))
    by_sprite: dict[int, list[float]] = {}
    for scene in sorted(by_scene):
        by_sprite.setdefault(scene // SCENES_PER_SPRITE, []).append(float(np.mean(by_scene[scene])))
    return np.array([np.mean(by_sprite[sprite]) for sprite in sorted(by_sprite)])


class Measure:
    def __init__(self, results: Results) -> None:
        self.results = results

    def mean(self, key: str, method: str, measure: str = "iou", family: str = TREE) -> float:
        return scene_mean(self.results.trials(key), lambda record: record[family][method][measure])

    def interval(
        self,
        key: str,
        first: str,
        second: str,
        measure: str = "iou",
        first_family: str = TREE,
        second_family: str = TREE,
    ) -> Interval:
        cache_key = (key, first, second, measure, first_family, second_family)
        cache = self.results._intervals
        if cache_key not in cache:
            records = self.results.trials(key)
            difference = scene_means(records, lambda record: record[first_family][first][measure]) - scene_means(
                records, lambda record: record[second_family][second][measure]
            )
            cache[cache_key] = bootstrap_interval(difference)
        return cache[cache_key]

    def sprite_interval(
        self,
        key: str,
        first: str,
        second: str,
        measure: str = "iou",
        first_family: str = TREE,
        second_family: str = TREE,
    ) -> Interval:
        records = self.results.trials(key)
        difference = sprite_means(records, lambda record: record[first_family][first][measure]) - sprite_means(
            records, lambda record: record[second_family][second][measure]
        )
        return bootstrap_interval(difference)

    def difference(self, key: str, first: str, second: str, measure: str = "iou", **families: str) -> float:
        return self.interval(key, first, second, measure, **families).mean

    def significant(self, key: str, first: str, second: str, measure: str = "iou", **families: str) -> bool:
        return separates_from_zero(self.interval(key, first, second, measure, **families))

    def ablation_mean(self, key: str, value: Callable[[dict], float]) -> float:
        return scene_mean(self.results.ablations(key), value)

    def model_mean(self, key: str, model: str, method: str, measure: str = "iou") -> float:
        return scene_mean(self.results.model_trials(key, model), lambda record: record[TREE][method][measure])

    def resolution_mean(self, key: str, width: int, measure: str = "iou") -> float:
        return scene_mean(self.results.resolution_trials(key, width), lambda record: record[measure])

    def best_mean(self, key: str, methods: Sequence[str] = CUE_METHODS) -> float:
        return max(self.mean(key, method) for method in methods)

    def regret(self, key: str, method: str, methods: Sequence[str] = CUE_METHODS) -> float:
        return self.best_mean(key, methods) - self.mean(key, method)
