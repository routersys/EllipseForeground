from dataclasses import dataclass

import numpy as np

from ellipse_foreground import Analysis
from ellipse_foreground.cues.color import color_codes
from ellipse_foreground.cues.evidence import bounded_log_ratio
from ellipse_foreground.cues.texture import texture_codes

OCCUPANCY_EDGES = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4, 1.7, 2.2, 3.0, np.inf])
OCCUPANCY_FLOOR = 0.02
OCCUPANCY_CEILING = 0.98
POPULATION_EXTENT = 1.15
POPULATION_FLOOR = 0.02
POPULATION_CEILING = 0.97
POPULATION_WIDTH = 0.3
RIDGE = 0.02
TEXTURE_FLOOR = 1e-4


@dataclass(frozen=True)
class Feature:
    codes: np.ndarray
    size: int
    bins: int
    offset: float
    floor: float
    clip: float


def features_of(analysis: Analysis) -> list[Feature]:
    settings = analysis.settings
    color_bins = settings.color.bins
    texture_bins = settings.texture.bins
    return [
        Feature(
            color_codes(analysis.work, color_bins).ravel(),
            color_bins**3,
            color_bins,
            0.0,
            settings.color.floor,
            settings.color.clip,
        ),
        Feature(
            texture_codes(analysis.texture, texture_bins),
            texture_bins**2,
            0,
            settings.texture.count_offset,
            TEXTURE_FLOOR,
            settings.texture.clip,
        ),
    ]


def annulus_index(radius: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(radius.ravel(), OCCUPANCY_EDGES) - 1, 0, len(OCCUPANCY_EDGES) - 2)


def annulus_areas(index: np.ndarray) -> np.ndarray:
    return np.bincount(index, minlength=len(OCCUPANCY_EDGES) - 1).astype(np.float64)


def occupancy_of(index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    areas = annulus_areas(index)
    counts = np.bincount(index, weights=mask.ravel().astype(np.float64), minlength=len(areas))
    return np.clip(counts / np.maximum(areas, 1.0), OCCUPANCY_FLOOR, OCCUPANCY_CEILING)


def population_occupancy() -> np.ndarray:
    widths = np.minimum(np.diff(OCCUPANCY_EDGES), POPULATION_WIDTH)
    centers = OCCUPANCY_EDGES[:-1] + widths / 2.0
    return np.clip(1.0 - centers / POPULATION_EXTENT, POPULATION_FLOOR, POPULATION_CEILING)


def annulus_table(codes: np.ndarray, index: np.ndarray, size: int) -> np.ndarray:
    count = len(OCCUPANCY_EDGES) - 1
    return np.bincount(index * size + codes, minlength=count * size).reshape(count, size).astype(np.float64)


def smoothed(histogram: np.ndarray, bins: int) -> np.ndarray:
    padded = np.pad(histogram.reshape(bins, bins, bins), 1)
    total = sum(padded[i : i + bins, j : j + bins, k : k + bins] for i in range(3) for j in range(3) for k in range(3))
    return total.ravel()


def normalized(histogram: np.ndarray) -> np.ndarray:
    return histogram / max(histogram.sum(), 1e-9)


def sample_histogram(feature: Feature, selection: np.ndarray) -> np.ndarray:
    counts = np.bincount(feature.codes[selection.ravel()], minlength=feature.size).astype(np.float64)
    return normalized(smoothed(counts, feature.bins) if feature.bins else counts + feature.offset)


def unmix(
    table: np.ndarray,
    areas: np.ndarray,
    occupancy: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    ridge: float = RIDGE,
) -> tuple[np.ndarray, np.ndarray]:
    present = areas > 0
    design = np.stack([occupancy, 1.0 - occupancy], axis=1)[present]
    weights = areas[present]
    density = table[present] / weights[:, None]
    total = weights.sum()
    normal = design.T @ (weights[:, None] * design) + ridge * total * np.eye(2)
    target = design.T @ (weights[:, None] * density) + ridge * total * np.stack([foreground, background])
    zero = np.zeros(table.shape[1])
    candidates = np.stack(
        [
            np.linalg.solve(normal, target),
            np.stack([np.maximum(target[0] / normal[0, 0], 0.0), zero]),
            np.stack([zero, np.maximum(target[1] / normal[1, 1], 0.0)]),
            np.stack([zero, zero]),
        ]
    )
    costs = np.einsum("kic,ij,kjc->kc", candidates, normal, candidates)
    costs -= 2.0 * np.einsum("kic,ic->kc", candidates, target)
    costs[0, ~np.all(candidates[0] >= 0.0, axis=0)] = np.inf
    best = np.argmin(costs, axis=0)
    chosen = candidates[best, :, np.arange(table.shape[1])].T
    return chosen[0], chosen[1]


def unmixed_evidence(
    feature: Feature,
    index: np.ndarray,
    occupancy: np.ndarray,
    core: np.ndarray,
    ring: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    foreground, background = unmix(
        annulus_table(feature.codes, index, feature.size),
        annulus_areas(index),
        occupancy,
        sample_histogram(feature, core),
        sample_histogram(feature, ring),
    )
    if feature.bins:
        foreground, background = smoothed(foreground, feature.bins), smoothed(background, feature.bins)
    foreground, background = normalized(foreground), normalized(background)
    return bounded_log_ratio(
        foreground[feature.codes] + feature.floor, background[feature.codes] + feature.floor, feature.clip
    ).reshape(shape)
