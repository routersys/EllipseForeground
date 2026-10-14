from dataclasses import dataclass

import numpy as np

MINIMUM_RADIUS = 2.0


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

    @classmethod
    def from_moments(cls, mask: np.ndarray) -> "Ellipse":
        rows, columns = np.nonzero(mask)
        return cls(
            columns.mean() + 0.5,
            rows.mean() + 0.5,
            max(2.0 * columns.std(), MINIMUM_RADIUS),
            max(2.0 * rows.std(), MINIMUM_RADIUS),
        )

    def scaled(self, factor: float) -> "Ellipse":
        return Ellipse(self.center_x, self.center_y, self.radius_x * factor, self.radius_y * factor)

    def stretched(self, factor_x: float, factor_y: float) -> "Ellipse":
        return Ellipse(self.center_x, self.center_y, self.radius_x * factor_x, self.radius_y * factor_y)

    def shifted(self, shift_x: float, shift_y: float) -> "Ellipse":
        return Ellipse(
            self.center_x + shift_x * self.radius_x,
            self.center_y + shift_y * self.radius_y,
            self.radius_x,
            self.radius_y,
        )

    def blended(self, other: "Ellipse", weight: float) -> "Ellipse":
        return Ellipse(
            (1.0 - weight) * self.center_x + weight * other.center_x,
            (1.0 - weight) * self.center_y + weight * other.center_y,
            self.radius_x ** (1.0 - weight) * other.radius_x**weight,
            self.radius_y ** (1.0 - weight) * other.radius_y**weight,
        )

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
