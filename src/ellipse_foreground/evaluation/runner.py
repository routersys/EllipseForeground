from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ellipse_foreground.evaluation.experiments.ablation import ablation_record
from ellipse_foreground.evaluation.experiments.comparison import comparison_record
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.experiments.models import model_records
from ellipse_foreground.evaluation.experiments.resolution import WORK_WIDTHS, resolution_records
from ellipse_foreground.evaluation.own import own_scenes
from ellipse_foreground.evaluation.results import append_records, finished_scenes
from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.evaluation.sprites import public_scenes
from ellipse_foreground.evaluation.trials import ERROR_MODELS
from ellipse_foreground.settings import DEFAULT_SETTINGS, Settings

EXPERIMENTS = ("trials", "models", "resolution")
SECONDARY_MODELS = ("S", "T", "L")


@dataclass(frozen=True)
class GroupSpec:
    name: str
    scenes: Callable[[Path, range | None], Iterator[Scene]]
    trials: int
    resolution_stride: int | None


def own_group(name: str) -> GroupSpec:
    trials = 8 if name == "sprite1" else 10
    return GroupSpec(name, lambda data, selection: own_scenes(data, name), trials, 1)


def public_group(strength: float) -> GroupSpec:
    name = f"public{round(strength * 100)}"
    return GroupSpec(name, lambda data, selection: public_scenes(data, strength, selection=selection), 2, None)


GROUPS = {
    spec.name: spec for spec in (own_group("sprite1"), own_group("sprite2"), public_group(0.0), public_group(1.0))
}


def output_path(results: Path, experiment: str, name: str, selection: range) -> Path:
    suffix = "" if selection.start == 0 and selection.stop >= 10**6 else f".{selection.start}-{selection.stop}"
    return results / experiment / f"{name}{suffix}.jsonl"


def run_scene_experiments(
    scene: Scene,
    spec: GroupSpec,
    experiments: Sequence[str],
    results: Path,
    selection: range,
    settings: Settings = DEFAULT_SETTINGS,
) -> None:
    context = SceneContext(scene, settings)
    if "trials" in experiments:
        trials = list(context.trials(spec.trials))
        append_records(
            output_path(results, "comparison", spec.name, selection),
            (comparison_record(context, trial) for trial in trials),
        )
        append_records(
            output_path(results, "ablation", spec.name, selection),
            (ablation_record(context, trial) for trial in trials),
        )
    if "models" in experiments:
        for model in SECONDARY_MODELS:
            append_records(
                output_path(results, "models", f"{spec.name}-{model}", selection),
                model_records(context, ERROR_MODELS[model], spec.trials),
            )
    if "resolution" in experiments and spec.resolution_stride and scene.index % spec.resolution_stride == 0:
        for width in WORK_WIDTHS:
            append_records(
                output_path(results, "resolution", f"{spec.name}-{width}", selection),
                resolution_records(scene, width, spec.trials),
            )


def run_group(
    spec: GroupSpec,
    data: Path,
    results: Path,
    experiments: Sequence[str],
    selection: range,
    settings: Settings = DEFAULT_SETTINGS,
) -> int:
    finished = finished_scenes(results / "comparison", spec.name)
    processed = 0
    for scene in spec.scenes(data, selection):
        if scene.index in finished and "trials" in experiments:
            continue
        run_scene_experiments(scene, spec, experiments, results, selection, settings)
        processed += 1
        print(f"{spec.name} scene {scene.index} {scene.name}", flush=True)
    return processed
