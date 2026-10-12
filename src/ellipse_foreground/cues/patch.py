import numpy as np

from ellipse_foreground.cues.evidence import bounded_log_ratio
from ellipse_foreground.cues.sampling import choose_samples
from ellipse_foreground.settings import PatchSettings

_UNREACHABLE_SQUARED = 1e12


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


def nearest_sample_distances(
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
        np.dot(np.hstack([-2.0 * queries, np.ones((len(queries), 1))]), sample_matrix, out=scores)
        _exclude_neighborhood(scores, start, column_of_pixel, shape, settings.exclusion)
        squared = scores.min(axis=1) + np.einsum("ij,ij->i", queries, queries)
        distances[start : start + len(queries)] = np.sqrt(np.clip(squared, 0.0, _UNREACHABLE_SQUARED))
    return distances


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
