from dataclasses import dataclass

import numpy as np

from ellipse_foreground.cues.evidence import bounded_log_ratio, normalized_histogram
from ellipse_foreground.imaging import area_resize, box_mean
from ellipse_foreground.settings import TextureSettings

_BOUNDARY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TextureFeatures:
    flatness: np.ndarray
    magnitude: np.ndarray


def neighborhood_contrast(source: np.ndarray) -> np.ndarray:
    values = source.astype(np.int16)
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    height, width = values.shape[:2]
    contrast = np.zeros((height, width), dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            contrast = np.maximum(contrast, np.abs(neighbor - values).max(axis=2))
    return contrast


def texture_features(source: np.ndarray, shape: tuple[int, int], settings: TextureSettings) -> TextureFeatures:
    contrast = neighborhood_contrast(source)
    flat = (contrast <= settings.flat_threshold).astype(np.float32)
    magnitude = np.minimum(contrast, settings.magnitude_cap).astype(np.float32) / settings.magnitude_cap
    height, width = shape
    return TextureFeatures(
        box_mean(area_resize(flat, width, height), settings.smoothing_radius),
        box_mean(area_resize(magnitude, width, height), settings.smoothing_radius),
    )


def quantize(values: np.ndarray, bins: int) -> np.ndarray:
    return np.clip(np.floor(values * bins + _BOUNDARY_TOLERANCE).astype(np.int64), 0, bins - 1)


def texture_codes(features: TextureFeatures, bins: int) -> np.ndarray:
    return (quantize(features.flatness, bins) * bins + quantize(features.magnitude, bins)).ravel()


def texture_evidence(
    features: TextureFeatures, core: np.ndarray, ring: np.ndarray, settings: TextureSettings
) -> np.ndarray:
    codes = texture_codes(features, settings.bins)
    size = settings.bins * settings.bins
    foreground = normalized_histogram(codes[core.ravel()], size, settings.count_offset)
    background = normalized_histogram(codes[ring.ravel()], size, settings.count_offset)
    return bounded_log_ratio(foreground[codes], background[codes], settings.clip).reshape(features.flatness.shape)
