from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ellipse:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float

    @classmethod
    def inscribed(cls, mask: np.ndarray) -> "Ellipse":
        rows, columns = np.nonzero(mask)
        left, right = columns.min(), columns.max() + 1
        top, bottom = rows.min(), rows.max() + 1
        return cls((left + right) / 2.0, (top + bottom) / 2.0, (right - left) / 2.0, (bottom - top) / 2.0)

    def scaled(self, factor: float) -> "Ellipse":
        return Ellipse(self.center_x, self.center_y, self.radius_x * factor, self.radius_y * factor)

    def resized(self, scale_x: float, scale_y: float) -> "Ellipse":
        return Ellipse(
            self.center_x * scale_x, self.center_y * scale_y, self.radius_x * scale_x, self.radius_y * scale_y
        )

    def normalized_radius(self, shape: tuple[int, int]) -> np.ndarray:
        rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
        return np.sqrt(((columns - self.center_x) / self.radius_x) ** 2 + ((rows - self.center_y) / self.radius_y) ** 2)

    def inside(self, shape: tuple[int, int], factor: float = 1.0) -> np.ndarray:
        rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
        distance = ((columns - self.center_x) / (self.radius_x * factor)) ** 2
        distance = distance + ((rows - self.center_y) / (self.radius_y * factor)) ** 2
        return distance <= 1.0
