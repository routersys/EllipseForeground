from collections.abc import Callable

from ellipse_foreground.evaluation.statistics import Interval, bootstrap_interval, scene_means
from paper.sections.baselines import classify
from paper.sections.refinement import REFINE_SETS, RefineSet


def value(name: str, measure: str = "iou") -> Callable[[dict], float]:
    return lambda row: row[name][measure]


def paired(rows: list[dict], first: Callable[[dict], float], second: Callable[[dict], float]) -> Interval:
    return bootstrap_interval(scene_means(rows, lambda row: first(row) - second(row)))


def classified_by(
    rows: dict[str, list[dict]], first: Callable[[dict], float], second: Callable[[dict], float]
) -> dict[str, list[RefineSet]]:
    groups: dict[str, list[RefineSet]] = {"higher": [], "lower": [], "unclear": []}
    for selection in REFINE_SETS:
        groups[classify(paired(rows[selection.name], first, second))].append(selection)
    return groups
