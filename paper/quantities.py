from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ellipse_foreground.evaluation.statistics import scene_means
from paper.numbers import above_zero, below_zero
from paper.results import CUE_METHODS, EARLIER_SETS, SETS, Measure, Results

CUE_KEYS = ("texture", "color", "patch")
SHORT_CUE = {"texture": "tex", "color": "color", "patch": "patch"}
PATCH_OVERLAP_BOUND = 0.1
BLIND_OVERLAP = 0.999
BLIND_AUC_MARGIN = 1e-9


def mean_overlap(results: Results, key: str, cue: str) -> float:
    return float(scene_means(results.trials(key), lambda record: record["overlaps"][cue]).mean())


def mean_auc(results: Results, key: str, cue: str) -> float:
    return float(scene_means(results.trials(key), lambda record: record["auc"][cue]).mean())


def mean_color_weight(results: Results, key: str) -> float:
    return float(scene_means(results.trials(key), lambda record: record["weights"]["adaptive_color"][1]).mean())


@dataclass(frozen=True)
class RankCorrelation:
    pooled: float
    per_set: dict[str, float]


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(spearmanr(first, second).statistic)


def overlap_auc_pairs(results: Results, key: str, cue: str) -> tuple[np.ndarray, np.ndarray]:
    records = results.trials(key)
    return (
        np.array([record["overlaps"][cue] for record in records]),
        np.array([record["auc"][cue] for record in records]),
    )


def rank_correlations(results: Results, keys: tuple[str, ...] = EARLIER_SETS) -> dict[str, RankCorrelation]:
    correlations = {}
    for cue in CUE_KEYS:
        pairs = {key: overlap_auc_pairs(results, key, cue) for key in keys}
        pooled = rank_correlation(
            np.concatenate([pairs[key][0] for key in keys]), np.concatenate([pairs[key][1] for key in keys])
        )
        correlations[cue] = RankCorrelation(pooled, {key: rank_correlation(*pairs[key]) for key in keys})
    return correlations


def trial_count(results: Results, keys: tuple[str, ...] = EARLIER_SETS) -> int:
    return sum(len(results.trials(key)) for key in keys)


def share_below(results: Results, cue: str, bound: float, keys: tuple[str, ...] = EARLIER_SETS) -> float:
    values = np.concatenate([overlap_auc_pairs(results, key, cue)[0] for key in keys])
    return float((values < bound).mean())


def ordering_disagreement(results: Results, keys: tuple[str, ...] = EARLIER_SETS) -> float:
    records = [record for key in keys for record in results.trials(key)]
    differing = sum(
        (record["overlaps"]["color"] < record["overlaps"]["texture"])
        != (record["auc"]["color"] > record["auc"]["texture"])
        for record in records
    )
    return differing / len(records)


@dataclass(frozen=True)
class Headline:
    ellipse_low: float
    ellipse_high: float
    adaptive_low: float
    adaptive_high: float
    mix_regret: float
    held_out_regret: float
    texture_regret: float
    appearance_regret: float
    adaptive_effect: float
    adaptive_patch_effect: float


def headline(measure: Measure) -> Headline:
    def regret_of(method: str, keys: tuple[str, ...]) -> float:
        return max(measure.regret(key, method, ("tex", "app", "mix", "adaptive_c", "adaptive")) for key in keys)

    return Headline(
        ellipse_low=min(measure.mean(key, "ellipse") for key in SETS),
        ellipse_high=max(measure.mean(key, "ellipse") for key in SETS),
        adaptive_low=min(measure.mean(key, "adaptive_c") for key in SETS),
        adaptive_high=max(measure.mean(key, "adaptive_c") for key in SETS),
        mix_regret=regret_of("mix", SETS),
        held_out_regret=regret_of("mix", ("sprite2", "public0", "public100")),
        texture_regret=regret_of("tex", SETS),
        appearance_regret=regret_of("app", SETS),
        adaptive_effect=max(abs(measure.difference(key, "adaptive_c", "mix")) for key in SETS),
        adaptive_patch_effect=max(abs(measure.difference(key, "adaptive", "mix")) for key in SETS),
    )


def best_methods(measure: Measure) -> dict[str, float]:
    return {key: max(measure.mean(key, method) for method in CUE_METHODS) for key in SETS}


@dataclass(frozen=True)
class GraphCutComparison:
    difference: dict[str, float]
    error_difference: dict[str, float]
    higher: list[str]
    lower: list[str]
    smaller_error_everywhere: bool


def graph_cut_comparison(measure: Measure) -> GraphCutComparison:
    def interval(key: str, quantity: str):
        return measure.interval(key, "adaptive_c", "adaptive_c", quantity, first_family="graph")

    return GraphCutComparison(
        difference={key: interval(key, "iou").mean for key in SETS},
        error_difference={key: interval(key, "mae").mean for key in SETS},
        higher=[key for key in SETS if above_zero(interval(key, "iou"))],
        lower=[key for key in SETS if below_zero(interval(key, "iou"))],
        smaller_error_everywhere=all(below_zero(interval(key, "mae"), 2) for key in SETS),
    )


@dataclass(frozen=True)
class Degenerate:
    scenes: int
    trials: int
    total_trials: int
    cue_iou: float
    with_mix: float
    without_mix: float


def degenerate_trials(results: Results, key: str) -> Degenerate | None:
    records = results.trials(key)

    def blind(record: dict) -> bool:
        return all(value > BLIND_OVERLAP for value in record["overlaps"].values()) and all(
            abs(value - 0.5) < BLIND_AUC_MARGIN for value in record["auc"].values()
        )

    blinded = [record for record in records if blind(record)]
    if not blinded:
        return None
    kept = [record for record in records if record["scene"] not in {item["scene"] for item in blinded}]
    cues = ("tex", "mix", "adaptive_c")
    return Degenerate(
        scenes=len({record["scene"] for record in blinded}),
        trials=len(blinded),
        total_trials=len(records),
        cue_iou=float(np.mean([record["tree"][method]["iou"] for record in blinded for method in cues])),
        with_mix=float(scene_means(records, lambda record: record["tree"]["mix"]["iou"]).mean()),
        without_mix=float(scene_means(kept, lambda record: record["tree"]["mix"]["iou"]).mean()),
    )
