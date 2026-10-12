from dataclasses import dataclass

import numpy as np

from ellipse_foreground.geometry import Ellipse
from ellipse_foreground.imaging import area_resize, halve
from ellipse_foreground.settings import WorkSettings
from ellipse_foreground.work import work_shape


@dataclass(frozen=True)
class Scene:
    name: str
    group: str
    index: int
    image: np.ndarray
    alpha: np.ndarray
    texture_source: np.ndarray

    @property
    def prototype(self) -> Ellipse:
        return Ellipse.inscribed(self.alpha > 0.5)

    def work_alpha(self, settings: WorkSettings) -> np.ndarray:
        height, width = work_shape(self.image.shape, settings)
        return area_resize(self.alpha, width, height)


def scene_from_composite(name: str, group: str, index: int, composite: np.ndarray, alpha: np.ndarray) -> Scene:
    image = halve(composite.astype(np.float32) / 255.0).astype(np.float32)
    return Scene(name, group, index, image, halve(alpha.astype(np.float32)).astype(np.float32), composite)
