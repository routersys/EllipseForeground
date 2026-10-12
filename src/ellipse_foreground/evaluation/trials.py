from dataclasses import dataclass

import numpy as np

from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.geometry import Ellipse

TRIALS_PER_SCENE = {"sprite1": 8, "sprite2": 10, "public0": 2, "public100": 2}
TRIAL_SEED_OFFSET = 1000


@dataclass(frozen=True)
class ErrorModel:
    name: str
    scale_low: float
    scale_high: float
    shift: float
    log_uniform: bool = False


ERROR_MODELS = {
    model.name: model
    for model in (
        ErrorModel("M", 0.8, 1.6, 0.12),
        ErrorModel("T", 0.9, 1.15, 0.05),
        ErrorModel("L", 0.7, 2.0, 0.20),
        ErrorModel("S", 0.74, 1.35, 0.12, log_uniform=True),
    )
}
MAIN_MODEL = ERROR_MODELS["M"]


def draw_ellipse(rng: np.random.Generator, prototype: Ellipse, model: ErrorModel) -> Ellipse:
    if model.log_uniform:
        scales = np.exp(rng.uniform(np.log(model.scale_low), np.log(model.scale_high), size=2))
    else:
        scales = rng.uniform(model.scale_low, model.scale_high, size=2)
    shifts = rng.uniform(-model.shift, model.shift, size=2)
    return Ellipse(
        prototype.center_x + shifts[0] * prototype.radius_x * 2,
        prototype.center_y + shifts[1] * prototype.radius_y * 2,
        prototype.radius_x * scales[0],
        prototype.radius_y * scales[1],
    )


def draw_trials(scene: Scene, count: int, model: ErrorModel = MAIN_MODEL) -> list[Ellipse]:
    rng = np.random.default_rng(TRIAL_SEED_OFFSET + scene.index)
    return [draw_ellipse(rng, scene.prototype, model) for _ in range(count)]


def is_informative(inside: np.ndarray, truth: np.ndarray, minimum: int = 10) -> bool:
    return int(truth[inside].sum()) >= minimum and int((~truth[inside]).sum()) >= minimum
