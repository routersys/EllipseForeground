import numpy as np

from ellipse_foreground.cues.evidence import bounded_log_ratio
from ellipse_foreground.settings import ColorSettings


def color_codes(work: np.ndarray, bins: int) -> np.ndarray:
    indices = np.clip((work * bins).astype(np.int64), 0, bins - 1)
    return (indices[..., 0] * bins + indices[..., 1]) * bins + indices[..., 2]


def smoothed_color_histogram(codes: np.ndarray, bins: int) -> np.ndarray:
    histogram = np.bincount(codes, minlength=bins**3).astype(np.float64).reshape(bins, bins, bins)
    padded = np.pad(histogram, 1)
    smoothed = sum(
        padded[i : i + bins, j : j + bins, k : k + bins] for i in range(3) for j in range(3) for k in range(3)
    )
    return (smoothed / max(smoothed.sum(), 1e-9)).ravel()


def color_evidence(work: np.ndarray, core: np.ndarray, ring: np.ndarray, settings: ColorSettings) -> np.ndarray:
    codes = color_codes(work, settings.bins)
    foreground = smoothed_color_histogram(codes[core], settings.bins)
    background = smoothed_color_histogram(codes[ring], settings.bins)
    return bounded_log_ratio(foreground[codes] + settings.floor, background[codes] + settings.floor, settings.clip)
