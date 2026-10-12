from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ellipse_foreground import BoundaryForm, Mix, Settings
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.own import own_scenes
from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.evaluation.sprites import public_scenes
from ellipse_foreground.evaluation.trials import ERROR_MODELS, draw_trials
from ellipse_foreground.settings import DEFAULT_SETTINGS, WorkSettings
from ellipse_foreground.weights import weights_for
from harness.golden import digest

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
FAST = Settings(work=WorkSettings(width=128))
PUBLIC_INDICES = (3, 40)
MAIN_SCENE = "sprite1-dose100"
TRIAL_COUNT = 4


@dataclass(frozen=True)
class CaseOutput:
    arrays: tuple[np.ndarray, ...]
    alpha: np.ndarray | None = None


@dataclass(frozen=True)
class Case:
    name: str
    run: Callable[["CaseContext"], CaseOutput]


@dataclass(frozen=True)
class CaseResult:
    name: str
    hash: str
    alpha: np.ndarray | None


def load_scenes(data_root: Path) -> dict[str, Scene]:
    own = {scene.name: scene for scene in own_scenes(data_root, "sprite1")}
    scenes = {name: own[name] for name in ("sprite1-dose0", MAIN_SCENE)}
    scenes["sprite2-bg4"] = next(own_scenes(data_root, "sprite2"))
    selection = range(min(PUBLIC_INDICES), max(PUBLIC_INDICES) + 1)
    for strength in (0.0, 1.0):
        for scene in public_scenes(data_root, strength, selection=selection):
            if scene.index in PUBLIC_INDICES:
                scenes[f"{scene.group}-{scene.index}"] = scene
    return scenes


class CaseContext:
    def __init__(self, data_root: Path = DATA_ROOT) -> None:
        self.scenes = load_scenes(data_root)
        self._contexts: dict[tuple[str, Settings], SceneContext] = {}
        self._trials: dict[tuple[str, Settings, int], Trial] = {}

    @property
    def identity(self) -> str:
        combined = "".join(digest(self.scenes[name].texture_source) for name in sorted(self.scenes))
        return f"paper-scenes {len(self.scenes)} {digest(np.frombuffer(combined.encode(), dtype=np.uint8))[:16]}"

    def scene_context(self, name: str, settings: Settings = DEFAULT_SETTINGS) -> SceneContext:
        key = (name, settings)
        if key not in self._contexts:
            self._contexts[key] = self.fresh_context(name, settings)
        return self._contexts[key]

    def fresh_context(self, name: str, settings: Settings = DEFAULT_SETTINGS) -> SceneContext:
        return SceneContext(self.scenes[name], settings)

    def trial(self, name: str, settings: Settings = DEFAULT_SETTINGS, index: int = 0) -> Trial:
        key = (name, settings, index)
        if key not in self._trials:
            ellipse = draw_trials(self.scenes[name], index + 1)[index]
            self._trials[key] = self.scene_context(name, settings).trial(index, ellipse)
        return self._trials[key]


def evaluate(case: Case, context: CaseContext) -> CaseResult:
    output = case.run(context)
    return CaseResult(case.name, digest(*output.arrays), output.alpha)


def stage_case(name: str, select: Callable[[CaseContext, Trial], tuple[np.ndarray, ...]]) -> Case:
    return Case(name, lambda context: CaseOutput(select(context, context.trial(MAIN_SCENE))))


def tree_arrays(context: CaseContext, _: Trial) -> tuple[np.ndarray, ...]:
    tree = context.scene_context(MAIN_SCENE).analysis.tree
    return tree.left, tree.right, tree.parent, tree.height, tree.area


def penalty_case(name: str, form: BoundaryForm) -> Case:
    def select(context: CaseContext, _: Trial) -> tuple[np.ndarray, ...]:
        return (context.scene_context(MAIN_SCENE).analysis.boundary.unit_penalty(form),)

    return stage_case(name, select)


def cue_case(name: str, field: str) -> Case:
    return stage_case(name, lambda _, trial: (getattr(trial.cues, field),))


def overlaps_case(name: str, scene: str) -> Case:
    def run(context: CaseContext) -> CaseOutput:
        overlaps = context.trial(scene).overlaps
        return CaseOutput((np.array([overlaps.color, overlaps.texture, overlaps.patch]),))

    return Case(name, run)


def ellipses_case(name: str) -> Case:
    def run(context: CaseContext) -> CaseOutput:
        scene = context.scenes[MAIN_SCENE]
        rows = [
            [ellipse.center_x, ellipse.center_y, ellipse.radius_x, ellipse.radius_y]
            for model in ERROR_MODELS.values()
            for ellipse in draw_trials(scene, TRIAL_COUNT, model)
        ]
        return CaseOutput((np.array(rows),))

    return Case(name, run)


def mix_alpha(context: CaseContext, scene: str, mix: Mix, settings: Settings = DEFAULT_SETTINGS) -> np.ndarray:
    trial = context.trial(scene, settings)
    analysis = context.scene_context(scene, settings).analysis
    scores = trial.cues.scores(weights_for(mix, trial.overlaps, settings.weights))
    return analysis.finish(analysis.cut(scores))


def mix_case(name: str, mix: Mix, scene: str = MAIN_SCENE, settings: Settings = DEFAULT_SETTINGS) -> Case:
    def run(context: CaseContext) -> CaseOutput:
        alpha = mix_alpha(context, scene, mix, settings)
        return CaseOutput((alpha,), alpha)

    return Case(name, run)


def graph_cut_case(name: str) -> Case:
    def run(context: CaseContext) -> CaseOutput:
        trial = context.trial(MAIN_SCENE)
        analysis = context.scene_context(MAIN_SCENE).analysis
        scores = trial.cues.scores(weights_for(Mix.FIXED_MIX, trial.overlaps, DEFAULT_SETTINGS.weights))
        alpha = analysis.finish(analysis.graph_cut(scores))
        return CaseOutput((alpha,), alpha)

    return Case(name, run)


def score_case(name: str) -> Case:
    def run(context: CaseContext) -> CaseOutput:
        alpha = mix_alpha(context, MAIN_SCENE, Mix.ADAPTIVE_COLOR)
        score = context.scene_context(MAIN_SCENE).scorer.score(alpha)
        return CaseOutput((np.array([score.iou, score.mae]),))

    return Case(name, run)


def model_case(name: str, model: str) -> Case:
    def run(context: CaseContext) -> CaseOutput:
        scene_context = context.scene_context(MAIN_SCENE, FAST)
        ellipse = draw_trials(context.scenes[MAIN_SCENE], 1, ERROR_MODELS[model])[0]
        trial = scene_context.trial(0, ellipse)
        scores = trial.cues.scores(weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, FAST.weights))
        alpha = scene_context.analysis.finish(scene_context.analysis.cut(scores))
        return CaseOutput((alpha,), alpha)

    return Case(name, run)


def form_settings(form: BoundaryForm, strength: float) -> Settings:
    return replace(FAST, boundary=replace(FAST.boundary, form=form, strength=strength))


def all_cases() -> tuple[Case, ...]:
    prior, patch = FAST.prior, FAST.patch
    return (
        stage_case("work-image", lambda context, _: (context.scene_context(MAIN_SCENE).analysis.work,)),
        stage_case("tree", tree_arrays),
        penalty_case("penalty-mean-weakness", BoundaryForm.MEAN_WEAKNESS),
        penalty_case("penalty-merge-height", BoundaryForm.MERGE_HEIGHT),
        penalty_case("penalty-weakness-sum", BoundaryForm.WEAKNESS_SUM),
        cue_case("cue-prior", "prior"),
        cue_case("cue-color", "color"),
        cue_case("cue-texture", "texture"),
        cue_case("cue-patch", "patch"),
        overlaps_case("overlaps", MAIN_SCENE),
        ellipses_case("trial-ellipses"),
        mix_case("prior-only", Mix.PRIOR_ONLY),
        mix_case("texture-only", Mix.TEXTURE_ONLY),
        mix_case("appearance-only", Mix.APPEARANCE_ONLY),
        mix_case("fixed-mix", Mix.FIXED_MIX),
        mix_case("adaptive-color", Mix.ADAPTIVE_COLOR),
        mix_case("adaptive-color-patch", Mix.ADAPTIVE_COLOR_PATCH),
        graph_cut_case("graph-cut"),
        score_case("score-adaptive-color"),
        mix_case("sprite1-dose0-adaptive-color", Mix.ADAPTIVE_COLOR, "sprite1-dose0"),
        mix_case("sprite2-adaptive-color-patch", Mix.ADAPTIVE_COLOR_PATCH, "sprite2-bg4"),
        mix_case("public0-3-adaptive-color", Mix.ADAPTIVE_COLOR, "public0-3"),
        mix_case("public100-3-adaptive-color-patch", Mix.ADAPTIVE_COLOR_PATCH, "public100-3"),
        mix_case("public0-40-fixed-mix", Mix.FIXED_MIX, "public0-40"),
        mix_case("public100-40-adaptive-color", Mix.ADAPTIVE_COLOR, "public100-40"),
        mix_case("fast-adaptive-color", Mix.ADAPTIVE_COLOR, settings=FAST),
        mix_case("form-merge-height", Mix.ADAPTIVE_COLOR, settings=form_settings(BoundaryForm.MERGE_HEIGHT, 8.0)),
        mix_case("form-weakness-sum", Mix.ADAPTIVE_COLOR, settings=form_settings(BoundaryForm.WEAKNESS_SUM, 8.0)),
        mix_case("strength-0.25", Mix.ADAPTIVE_COLOR, settings=form_settings(BoundaryForm.MEAN_WEAKNESS, 0.25)),
        mix_case("strength-2", Mix.ADAPTIVE_COLOR, settings=form_settings(BoundaryForm.MEAN_WEAKNESS, 2.0)),
        mix_case(
            "inside-shrink-1", Mix.ADAPTIVE_COLOR, settings=replace(FAST, prior=replace(prior, inside_shrink=1.0))
        ),
        mix_case(
            "patch-core-limit-200", Mix.ADAPTIVE_COLOR, settings=replace(FAST, patch=replace(patch, core_limit=200))
        ),
        model_case("model-S", "S"),
        model_case("model-T", "T"),
        model_case("model-L", "L"),
    )
