from dataclasses import dataclass

import numpy as np

from ellipse_foreground.cues.evidence import bounded_log_ratio
from ellipse_foreground.cues.sampling import choose_samples
from ellipse_foreground.settings import PatchSettings

_UNREACHABLE_SQUARED = 1e12
_PRUNING_MINIMUM = 1024
_SEED_WIDTH = 256
_BOUND_MARGIN = 1e-6


def patch_vectors(work: np.ndarray, radius: int) -> np.ndarray:
    height, width = work.shape[:2]
    padded = np.pad(work, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    windows = [
        padded[dy : dy + height, dx : dx + width] for dy in range(2 * radius + 1) for dx in range(2 * radius + 1)
    ]
    return np.concatenate(windows, axis=2).reshape(height * width, -1).astype(np.float64)


def _exclude_neighborhood(
    scores: np.ndarray, start: int, column_of_pixel: np.ndarray, shape: tuple[int, int], exclusion: int
) -> None:
    height, width = shape
    pixels = np.arange(start, start + len(scores))
    rows, columns = pixels // width, pixels % width
    local = np.arange(len(scores))
    for dy in range(-exclusion, exclusion + 1):
        for dx in range(-exclusion, exclusion + 1):
            neighbor_rows, neighbor_columns = rows + dy, columns + dx
            inside = (
                (neighbor_rows >= 0) & (neighbor_rows < height) & (neighbor_columns >= 0) & (neighbor_columns < width)
            )
            sample_columns = column_of_pixel[neighbor_rows[inside] * width + neighbor_columns[inside]]
            sampled = sample_columns >= 0
            scores[local[inside][sampled], sample_columns[sampled]] = np.inf


def exhaustive_distances(
    vectors: np.ndarray, sample_indices: np.ndarray, shape: tuple[int, int], settings: PatchSettings
) -> np.ndarray:
    samples = vectors[sample_indices]
    sample_norms = np.einsum("ij,ij->i", samples, samples)
    sample_matrix = np.ascontiguousarray(np.hstack([samples, sample_norms[:, None]]).T)
    column_of_pixel = np.full(len(vectors), -1, dtype=np.int64)
    column_of_pixel[sample_indices] = np.arange(len(sample_indices))
    buffer = np.empty((settings.block_rows, len(sample_indices)))
    distances = np.empty(len(vectors))
    for start in range(0, len(vectors), settings.block_rows):
        queries = vectors[start : start + settings.block_rows]
        scores = buffer[: len(queries)]
        np.matmul(np.hstack([-2.0 * queries, np.ones((len(queries), 1))]), sample_matrix, out=scores)
        _exclude_neighborhood(scores, start, column_of_pixel, shape, settings.exclusion)
        squared = scores.min(axis=1) + np.einsum("ij,ij->i", queries, queries)
        distances[start : start + len(queries)] = np.sqrt(np.clip(squared, 0.0, _UNREACHABLE_SQUARED))
    return distances


@dataclass(frozen=True)
class _SortedSamples:
    center: np.ndarray
    direction: np.ndarray
    projections: np.ndarray
    rows: np.ndarray
    column_of_pixel: np.ndarray


def _sort_samples(vectors: np.ndarray, sample_indices: np.ndarray) -> _SortedSamples:
    samples = vectors[sample_indices]
    center = samples.mean(axis=0)
    centered = samples - center
    direction = np.linalg.eigh(centered.T @ centered)[1][:, -1]
    projections = centered @ direction
    order = np.argsort(projections, kind="stable")
    ordered = samples[order]
    rows = np.ascontiguousarray(np.hstack([ordered, np.einsum("ij,ij->i", ordered, ordered)[:, None]]))
    column_of_pixel = np.full(len(vectors), -1, dtype=np.int64)
    column_of_pixel[sample_indices[order]] = np.arange(len(sample_indices))
    return _SortedSamples(center, direction, projections[order], rows, column_of_pixel)


def _exclusion_table(column_of_pixel: np.ndarray, shape: tuple[int, int], exclusion: int) -> np.ndarray:
    height, width = shape
    padded = np.pad(column_of_pixel.reshape(shape), exclusion, constant_values=-1)
    offsets = range(2 * exclusion + 1)
    return np.stack([padded[dy : dy + height, dx : dx + width].ravel() for dy in offsets for dx in offsets], axis=1)


def _window_minimum(
    left: np.ndarray, samples: _SortedSamples, low: int, high: int, excluded: np.ndarray, buffer: np.ndarray
) -> np.ndarray:
    scores = buffer[: len(left) * (high - low)].reshape(len(left), high - low)
    np.matmul(left, samples.rows[low:high].T, out=scores)
    rows, slots = np.nonzero((excluded >= low) & (excluded < high))
    scores[rows, excluded[rows, slots] - low] = np.inf
    return scores.min(axis=1)


def _bound_groups(ranked: np.ndarray, block: int) -> list[np.ndarray]:
    groups = []
    rest = ranked
    while len(rest) > 2 * block:
        groups.append(rest[: len(rest) // 2])
        rest = rest[len(rest) // 2 :]
    groups.append(rest)
    return groups


def pruned_distances(
    vectors: np.ndarray, sample_indices: np.ndarray, shape: tuple[int, int], settings: PatchSettings
) -> np.ndarray:
    samples = _sort_samples(vectors, sample_indices)
    count, block = len(sample_indices), settings.block_rows
    projections = (vectors - samples.center) @ samples.direction
    norms = np.einsum("ij,ij->i", vectors, vectors)
    buffer = np.empty(block * count)
    table = _exclusion_table(samples.column_of_pixel, shape, settings.exclusion)

    def distances_of(pixels: np.ndarray, low: int, high: int) -> np.ndarray:
        queries = vectors[pixels]
        left = np.hstack([-2.0 * queries, np.ones((len(queries), 1))])
        squared = _window_minimum(left, samples, low, high, table[pixels], buffer) + norms[pixels]
        return np.sqrt(np.clip(squared, 0.0, _UNREACHABLE_SQUARED))

    by_projection = np.argsort(projections, kind="stable")
    upper = np.empty(len(vectors))
    for start in range(0, len(vectors), block):
        pixels = by_projection[start : start + block]
        low = int(np.searchsorted(samples.projections, projections[pixels[0]], side="left"))
        high = int(np.searchsorted(samples.projections, projections[pixels[-1]], side="right"))
        width = min(count, max(high - low, _SEED_WIDTH))
        low = min(max(0, (low + high - width) // 2), count - width)
        upper[pixels] = distances_of(pixels, low, low + width)
    distances = np.empty(len(vectors))
    for group in _bound_groups(np.argsort(upper, kind="stable"), block):
        group = group[np.argsort(projections[group], kind="stable")]
        for start in range(0, len(group), block):
            pixels = group[start : start + block]
            radius = float(upper[pixels].max()) * (1.0 + _BOUND_MARGIN) + _BOUND_MARGIN
            low = int(np.searchsorted(samples.projections, projections[pixels[0]] - radius, side="left"))
            high = int(np.searchsorted(samples.projections, projections[pixels[-1]] + radius, side="right"))
            distances[pixels] = distances_of(pixels, low, high)
    return distances


def nearest_sample_distances(
    vectors: np.ndarray, sample_indices: np.ndarray, shape: tuple[int, int], settings: PatchSettings
) -> np.ndarray:
    if len(sample_indices) < _PRUNING_MINIMUM:
        return exhaustive_distances(vectors, sample_indices, shape, settings)
    return pruned_distances(vectors, sample_indices, shape, settings)


def choose_patch_samples(
    core: np.ndarray, ring: np.ndarray, priorities: np.ndarray, settings: PatchSettings
) -> tuple[np.ndarray, np.ndarray]:
    core_samples = choose_samples(core, priorities, settings.core_limit)
    ring_limit = max(settings.ring_multiple * len(core_samples), settings.ring_minimum)
    return core_samples, choose_samples(ring, priorities, ring_limit)


def patch_evidence(
    vectors: np.ndarray,
    core_samples: np.ndarray,
    ring_samples: np.ndarray,
    shape: tuple[int, int],
    settings: PatchSettings,
) -> np.ndarray:
    core_distance = nearest_sample_distances(vectors, core_samples, shape, settings)
    ring_distance = nearest_sample_distances(vectors, ring_samples, shape, settings)
    evidence = bounded_log_ratio(
        ring_distance + settings.floor, core_distance + settings.floor, settings.clip, settings.sharpness
    )
    return evidence.reshape(shape)
