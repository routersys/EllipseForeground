from dataclasses import dataclass

import numpy as np

from ellipse_foreground.cues.texture import TextureFeatures

_MINIMUM_REFERENCE = 1e-6


@dataclass(frozen=True)
class Overlaps:
    color: float
    texture: float
    patch: float


def gaussian_bhattacharyya(first: np.ndarray, second: np.ndarray, ridge: float) -> float:
    first, second = first.astype(np.float64), second.astype(np.float64)
    dimension = first.shape[1]
    if len(first) < dimension + 2 or len(second) < dimension + 2:
        return 1.0
    first_mean, second_mean = first.mean(axis=0), second.mean(axis=0)
    identity = np.eye(dimension)
    first_covariance = np.cov(first.T).reshape(dimension, dimension)
    second_covariance = np.cov(second.T).reshape(dimension, dimension)
    first_covariance = first_covariance + ridge * max(np.trace(first_covariance) / dimension, 1e-6) * identity
    second_covariance = second_covariance + ridge * max(np.trace(second_covariance) / dimension, 1e-6) * identity
    first_covariance = first_covariance + 1e-9 * identity
    second_covariance = second_covariance + 1e-9 * identity
    pooled = 0.5 * (first_covariance + second_covariance)
    difference = first_mean - second_mean
    try:
        mean_term = 0.125 * difference @ np.linalg.solve(pooled, difference)
        _, pooled_logdet = np.linalg.slogdet(pooled)
        _, first_logdet = np.linalg.slogdet(first_covariance)
        _, second_logdet = np.linalg.slogdet(second_covariance)
    except np.linalg.LinAlgError:
        return 1.0
    covariance_term = 0.5 * (pooled_logdet - 0.5 * (first_logdet + second_logdet))
    return float(np.exp(-(mean_term + covariance_term)))


def measure_overlaps(
    work: np.ndarray,
    texture: TextureFeatures,
    vectors: np.ndarray,
    core: np.ndarray,
    ring: np.ndarray,
    ridge: float,
) -> Overlaps:
    texture_points = np.stack([texture.flatness, texture.magnitude], axis=2)
    return Overlaps(
        color=gaussian_bhattacharyya(work[core], work[ring], ridge),
        texture=gaussian_bhattacharyya(texture_points[core], texture_points[ring], ridge),
        patch=gaussian_bhattacharyya(vectors[core.ravel()], vectors[ring.ravel()], ridge),
    )


def relative_advantage(candidate: float, reference: float) -> float:
    return float(np.clip((reference - candidate) / max(reference, _MINIMUM_REFERENCE), 0.0, 1.0))
