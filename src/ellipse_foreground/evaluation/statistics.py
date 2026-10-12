from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SAMPLES = 2000
LOWER_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def scene_means(rows: Iterable[dict], value: Callable[[dict], float]) -> np.ndarray:
    by_scene: dict[int, list[float]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(value(row))
    return np.array([np.mean(by_scene[scene]) for scene in sorted(by_scene)])


def bootstrap_interval(per_scene: np.ndarray, samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> Interval:
    rng = np.random.default_rng(seed)
    means = [rng.choice(per_scene, len(per_scene)).mean() for _ in range(samples)]
    low, high = np.percentile(means, [LOWER_PERCENTILE, UPPER_PERCENTILE])
    return Interval(float(per_scene.mean()), float(low), float(high))


def paired_interval(
    rows: Iterable[dict], first: Callable[[dict], float], second: Callable[[dict], float], seed: int = 0
) -> Interval:
    materialized = list(rows)
    return bootstrap_interval(scene_means(materialized, first) - scene_means(materialized, second), seed=seed)
