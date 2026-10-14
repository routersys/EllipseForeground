import numpy as np

from ellipse_foreground.geometry import Ellipse
from ellipse_foreground.settings import StabilitySettings

_SHIFTS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))


def perturbed_ellipses(ellipse: Ellipse, settings: StabilitySettings) -> list[Ellipse]:
    low, high = min(settings.scales), max(settings.scales)
    return [
        *(ellipse.scaled(scale) for scale in settings.scales),
        *(ellipse.shifted(settings.shift * shift_x, settings.shift * shift_y) for shift_x, shift_y in _SHIFTS),
        ellipse.stretched(low, high),
        ellipse.stretched(high, low),
    ]


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else float(np.count_nonzero(first & second) / union)
