from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ellipse_foreground.evaluation.experiments.ablation import ablation_record
from ellipse_foreground.evaluation.experiments.adaptation import adaptation_record
from ellipse_foreground.evaluation.experiments.comparison import comparison_record
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.experiments.holes import holes_record
from ellipse_foreground.evaluation.experiments.models import model_records
from ellipse_foreground.evaluation.experiments.refinement import refinement_record
from ellipse_foreground.evaluation.experiments.resolution import WORK_WIDTHS, resolution_records
from ellipse_foreground.evaluation.own import own_scenes
from ellipse_foreground.evaluation.photos import photo_scenes
from ellipse_foreground.evaluation.results import append_records, finished_scenes
from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.evaluation.sprites import collection_scenes, public_scenes
from ellipse_foreground.evaluation.trials import ERROR_MODELS
from ellipse_foreground.settings import DEFAULT_SETTINGS, Settings

EXPERIMENTS = ("trials", "comparison", "refinement", "adaptation", "holes", "models", "resolution")
DEFAULT_EXPERIMENTS = ("trials", "models", "resolution")
SECONDARY_MODELS = ("S", "T", "L")
COLLECTION_SEED = 78
COLLECTION_TRIALS = 2
COLLECTION_RESOLUTION_STRIDE = 3
PHOTO_TRIALS = 2


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


def collection_group(
    strength: float,
    sprites: Sequence[Path],
    backgrounds: Sequence[Path],
    seed: int = COLLECTION_SEED,
    grounded: bool = False,
) -> GroupSpec:
    name = f"collection{round(strength * 100)}"
    identified = [(f"sprite{index:02d}", path) for index, path in enumerate(sprites)]

    def scenes(data: Path, selection: range | None) -> Iterator[Scene]:
        return collection_scenes(identified, backgrounds, name, strength, seed, grounded, selection)

    return GroupSpec(name, scenes, COLLECTION_TRIALS, COLLECTION_RESOLUTION_STRIDE)


def photo_group(directory: Path) -> GroupSpec:
    return GroupSpec("photo", lambda data, selection: photo_scenes(directory, selection=selection), PHOTO_TRIALS, None)


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
) -> bool:
    worked = False

    @cache
    def context() -> SceneContext:
        return SceneContext(scene, settings)

    @cache
    def trials() -> list[Trial]:
        return list(context().trials(spec.trials))

    def store(experiment: str, name: str, produce: Callable[[], list[dict]]) -> None:
        nonlocal worked
        if scene.index not in finished_scenes(results / experiment, name):
            append_records(output_path(results, experiment, name, selection), produce())
            worked = True

    if "trials" in experiments or "comparison" in experiments:
        store("comparison", spec.name, lambda: [comparison_record(context(), trial) for trial in trials()])
    if "trials" in experiments or "refinement" in experiments:
        store("refinement", spec.name, lambda: [refinement_record(context(), trial) for trial in trials()])
    if "adaptation" in experiments:
        store("adaptation", spec.name, lambda: [adaptation_record(context(), trial) for trial in trials()])
    if "holes" in experiments:
        store("holes", spec.name, lambda: [holes_record(context(), trial) for trial in trials()])
    if "trials" in experiments:
        store("ablation", spec.name, lambda: [ablation_record(context(), trial) for trial in trials()])
    if "models" in experiments:
        for model in SECONDARY_MODELS:
            store(
                "models",
                f"{spec.name}-{model}",
                lambda model=model: model_records(context(), ERROR_MODELS[model], spec.trials),
            )
    if "resolution" in experiments and spec.resolution_stride and scene.index % spec.resolution_stride == 0:
        for width in WORK_WIDTHS:
            store(
                "resolution",
                f"{spec.name}-{width}",
                lambda width=width: resolution_records(scene, width, spec.trials),
            )
    return worked


def run_group(
    spec: GroupSpec,
    data: Path,
    results: Path,
    experiments: Sequence[str],
    selection: range,
    settings: Settings = DEFAULT_SETTINGS,
) -> int:
    processed = 0
    for scene in spec.scenes(data, selection):
        if run_scene_experiments(scene, spec, experiments, results, selection, settings):
            processed += 1
            print(f"{spec.name} scene {scene.index} {scene.name}", flush=True)
    return processed
