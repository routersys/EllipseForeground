from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.rendering import blurred_composite

RENDER_RADIUS = 12.0
RENDER_EPSILON = 1e-4


@dataclass(frozen=True)
class Score:
    iou: float
    mae: float


def mask_iou(mask: np.ndarray, truth: np.ndarray) -> float:
    union = (mask | truth).sum()
    return float((mask & truth).sum() / union) if union else 0.0


class Scorer:
    def __init__(self, scene: Scene, radius: float = RENDER_RADIUS, epsilon: float = RENDER_EPSILON) -> None:
        self.scene = scene
        self.radius = radius
        self.epsilon = epsilon

    @cached_property
    def truth_mask(self) -> np.ndarray:
        return self.scene.alpha > 0.5

    @cached_property
    def truth_render(self) -> np.ndarray:
        return blurred_composite(self.scene.image, self.scene.alpha, self.radius, self.epsilon)

    def foreground_iou(self, alpha: np.ndarray) -> float:
        return mask_iou(alpha > 0.5, self.truth_mask)

    def render_error(self, alpha: np.ndarray) -> float:
        rendered = blurred_composite(self.scene.image, alpha, self.radius, self.epsilon)
        return float(np.abs(rendered - self.truth_render).mean(axis=2).mean() * 255)

    def score(self, alpha: np.ndarray) -> Score:
        return Score(self.foreground_iou(alpha), self.render_error(alpha))
