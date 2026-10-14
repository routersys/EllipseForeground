import itertools
from pathlib import Path

import numpy as np

from ellipse_foreground.evaluation.results import append_records
from paper.results import SETS, SUBSETS

METHODS = ("prior", "tex", "app", "mix", "adaptive_c", "adaptive")
STAGES = ("bilinear", "jbu", "jbu_lut", "jbu_lutband", "jbu_guided", "final")
FORMS = {"mean_weakness": range(-6, 5), "weakness_sum": range(-6, 5), "merge_height": range(-7, 4)}
SHRINKS = ("1.0", "0.7", "0.5", "0.35", "0.25", "0.15", "0.0")
SCENE_COUNT = {"sprite1": 12, "sprite2": 8}
DEFAULT_SCENES = 6
TRIALS = {"sprite1": 2, "sprite2": 2}
DEFAULT_TRIALS = 2


def score(value: float, rng: np.random.Generator) -> dict:
    iou = float(np.clip(value + rng.normal(0, 0.01), 0.05, 0.99))
    return {"raw_iou": iou, "iou": iou, "mae": float(1.0 + (1.0 - iou))}


def comparison_record(group: str, scene: int, trial: int, rng: np.random.Generator) -> dict:
    base = 0.8 + 0.02 * rng.standard_normal()
    tree = {method: score(base + 0.01 * index, rng) for index, method in enumerate(METHODS)}
    tree["ellipse"] = score(base - 0.25, rng)
    tree["oracle"] = score(base + 0.08, rng)
    graph = {method: score(base + 0.03 + 0.01 * index, rng) for index, method in enumerate(METHODS)}
    overlap = float(rng.uniform(0.01, 0.9))
    return {
        "scene": scene,
        "name": f"{group}-{scene}",
        "trial": trial,
        "overlaps": {"color": overlap, "texture": float(rng.uniform(0.01, 0.5)), "patch": float(rng.uniform(0, 0.05))},
        "auc": {
            "color": float(1 - 0.4 * overlap),
            "texture": float(rng.uniform(0.7, 0.95)),
            "patch": float(rng.uniform(0.6, 0.95)),
        },
        "weights": {
            "adaptive_color": [1.0, float(rng.uniform(0.0, 0.5)), 0.3],
            "adaptive_color_patch": [1.0, float(rng.uniform(0.0, 0.5)), float(rng.uniform(0.0, 0.6))],
        },
        "occupancy": float(rng.uniform(0.4, 0.7)),
        "spill": float(rng.uniform(0.0, 0.2)),
        "tree": tree,
        "graph": graph,
    }


def boundary_sweep(rng: np.random.Generator) -> dict:
    sweep = {"none": {"iou": 0.8, "oracle_iou": 0.9}}
    for form, exponents in FORMS.items():
        peak = -1 if form != "merge_height" else -3
        sweep[form] = {
            str(e): {"iou": 0.9 - 0.004 * (e - peak) ** 2, "oracle_iou": 0.95 - 0.003 * (e - peak) ** 2}
            for e in exponents
        }
    return sweep


def ablation_record(group: str, scene: int, trial: int, rng: np.random.Generator) -> dict:
    return {
        "scene": scene,
        "name": f"{group}-{scene}",
        "trial": trial,
        "boundary": boundary_sweep(rng),
        "prior": {shrink: float(0.9 - abs(float(shrink) - 0.35) * 0.05) for shrink in SHRINKS},
        "texture_resolution": {
            source: {"texture_only": float(0.85 - drop), "mix": float(0.88 - drop)}
            for source, drop in (("full", 0.0), ("x640", 0.02), ("x320", 0.05))
        },
        "graph_cut": {str(e): float(0.9 - 0.004 * (e - 4) ** 2) for e in range(9)},
        "finish": {
            stage: {"iou": float(0.9 + 0.002 * index), "mae": float(1.0 - 0.05 * index)}
            for index, stage in enumerate(STAGES)
        },
    }


def model_record(group: str, scene: int, trial: int, model: str, rng: np.random.Generator) -> dict:
    base = 0.85 - {"S": 0.0, "T": -0.03, "L": 0.05}[model]
    tree = {
        "ellipse": score(base - 0.3, rng),
        "prior": score(base - 0.1, rng),
        "tex": score(base, rng),
        "oracle": score(base + 0.08, rng),
    }
    return {"scene": scene, "name": f"{group}-{scene}", "trial": trial, "model": model, "tree": tree}


def resolution_record(group: str, scene: int, trial: int, width: int, rng: np.random.Generator) -> dict:
    value = 0.9 - 0.02 * (256 / width - 1) ** 2
    return {"scene": scene, "name": f"{group}-{scene}", "trial": trial, "width": width, **score(value, rng)}


def refinement_record(group: str, scene: int, trial: int, rng: np.random.Generator) -> dict:
    base = 0.8 + 0.02 * rng.standard_normal()
    return {
        "scene": scene,
        "name": f"{group}-{scene}",
        "trial": trial,
        "tree": score(base, rng),
        "band": {str(radius): {**score(base + 0.004 * radius, rng), "share": 0.04 * radius} for radius in (1, 2, 3, 4)},
        "smoothness": {f"{value:g}": score(base + 0.01, rng) for value in (16.0, 32.0, 64.0, 128.0, 256.0, 512.0)},
        "agreement": {"tree": score(base + 0.002, rng), "band": score(base + 0.01, rng)},
    }


def write_results(directory: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    photos = [comparison_record("photo", scene, trial, rng) for scene in range(DEFAULT_SCENES) for trial in range(2)]
    append_records(directory / "comparison" / "photo.jsonl", photos)
    append_records(
        directory / "refinement" / "photo.jsonl",
        [refinement_record("photo", scene, trial, rng) for scene in range(DEFAULT_SCENES) for trial in range(2)],
    )
    for group in ("public0", "public100", "photo"):
        scenes = DEFAULT_SCENES
        records = [
            {
                "scene": scene,
                "name": f"{group}-{scene}",
                "trial": trial,
                **{
                    name: score(0.7 + shift, rng)
                    for name, shift in (("grabcut", 0.0), ("sam_single", 0.05), ("sam_best", 0.04))
                },
            }
            for scene in range(scenes)
            for trial in range(DEFAULT_TRIALS)
        ]
        append_records(directory / "baselines" / f"{group}.jsonl", records)
        coverage = [
            {"scene": record["scene"], "name": record["name"], "trial": record["trial"], "box_coverage": share}
            for record, share in zip(records, itertools.cycle((0.85, 0.93, 1.0)), strict=False)
        ]
        append_records(directory / "coverage" / f"{group}.jsonl", coverage)
    for key in SETS:
        group = SUBSETS[key].group
        scenes = SCENE_COUNT.get(group, DEFAULT_SCENES)
        trials = TRIALS.get(group, DEFAULT_TRIALS)
        pairs = [(scene, trial) for scene in range(scenes) for trial in range(trials)]
        append_records(directory / "comparison" / f"{group}.jsonl", [comparison_record(group, *p, rng) for p in pairs])
        append_records(directory / "ablation" / f"{group}.jsonl", [ablation_record(group, *p, rng) for p in pairs])
        append_records(directory / "refinement" / f"{group}.jsonl", [refinement_record(group, *p, rng) for p in pairs])
        for model in ("S", "T", "L"):
            append_records(
                directory / "models" / f"{group}-{model}.jsonl", [model_record(group, *p, model, rng) for p in pairs]
            )
        for width in (128, 256, 384):
            append_records(
                directory / "resolution" / f"{group}-{width}.jsonl",
                [resolution_record(group, *p, width, rng) for p in pairs],
            )
