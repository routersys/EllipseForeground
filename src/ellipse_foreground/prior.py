import numpy as np

from ellipse_foreground.geometry import Ellipse
from ellipse_foreground.settings import PriorSettings


def elliptical_prior(shape: tuple[int, int], ellipse: Ellipse, softness: float) -> np.ndarray:
    return np.clip((1.0 - ellipse.normalized_radius(shape)) / softness, -1.0, 1.0)


def ellipse_prior(shape: tuple[int, int], ellipse: Ellipse, settings: PriorSettings) -> np.ndarray:
    priors = []
    for scale in settings.scales:
        prior = elliptical_prior(shape, ellipse.scaled(scale), settings.softness)
        priors.append(np.where(prior > 0, settings.inside_shrink * prior, prior))
    return np.mean(priors, axis=0)
