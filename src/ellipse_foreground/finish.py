import numpy as np
from PIL import Image

from ellipse_foreground.imaging import box_mean, dilate, erode
from ellipse_foreground.settings import FinishSettings


def bilinear_upsample(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.float32)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def joint_bilateral_upsample(small: np.ndarray, mask: np.ndarray, image: np.ndarray, range_sigma: float) -> np.ndarray:
    height, width = image.shape[:2]
    small_height, small_width = mask.shape
    rows = (np.arange(height) + 0.5) * small_height / height - 0.5
    columns = (np.arange(width) + 0.5) * small_width / width - 0.5
    first_row, first_column = np.floor(rows).astype(np.int64), np.floor(columns).astype(np.int64)
    row_fraction, column_fraction = (rows - first_row)[:, None], (columns - first_column)[None, :]
    image64, small64 = image.astype(np.float64), small.astype(np.float64)
    numerator = np.zeros((height, width))
    denominator = np.full((height, width), 1e-9)
    for dy in (0, 1):
        for dx in (0, 1):
            source_rows = np.clip(first_row + dy, 0, small_height - 1)[:, None]
            source_columns = np.clip(first_column + dx, 0, small_width - 1)[None, :]
            spatial = (row_fraction if dy else 1 - row_fraction) * (column_fraction if dx else 1 - column_fraction)
            distance = ((image64 - small64[source_rows, source_columns]) ** 2).sum(axis=2)
            weight = spatial * np.exp(-distance / (2 * range_sigma**2))
            numerator += weight * mask[source_rows, source_columns]
            denominator += weight
    return (numerator / denominator).astype(np.float32)


def histogram_refinement(
    small: np.ndarray, mask: np.ndarray, image: np.ndarray, upsampled: np.ndarray, settings: FinishSettings
) -> np.ndarray:
    inner = erode(mask > 0.5, 1)
    outer = ~dilate(mask > 0.5, 1)
    bins = settings.histogram_bins
    log_likelihood = np.zeros(image.shape[:2])
    for channel in range(3):
        foreground = np.histogram(small[..., channel][inner], bins=bins, range=(0, 1))[0] + 1.0
        background = np.histogram(small[..., channel][outer], bins=bins, range=(0, 1))[0] + 1.0
        table = np.log((foreground / foreground.sum()) / (background / background.sum()))
        log_likelihood += table[np.clip((image[..., channel] * bins).astype(np.int64), 0, bins - 1)]
    evidence = np.clip(log_likelihood, -settings.histogram_clip, settings.histogram_clip) / settings.histogram_clip
    logit = settings.logit_gain * (upsampled - 0.5) + settings.histogram_gain * evidence
    return (1 / (1 + np.exp(-logit))).astype(np.float32)


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    guide, source = guide.astype(np.float64), source.astype(np.float64)
    height, width = source.shape
    mean_guide = np.stack([box_mean(guide[..., channel], radius) for channel in range(3)], axis=2)
    mean_source = box_mean(source, radius)
    correlation = np.stack([box_mean(guide[..., channel] * source, radius) for channel in range(3)], axis=2)
    covariance = correlation - mean_guide * mean_source[..., None]
    variance = np.empty((height, width, 3, 3))
    for first in range(3):
        for second in range(first, 3):
            product = box_mean(guide[..., first] * guide[..., second], radius)
            variance[..., first, second] = variance[..., second, first] = (
                product - mean_guide[..., first] * mean_guide[..., second]
            )
    variance = variance + epsilon * np.eye(3)
    coefficients = np.linalg.solve(variance, covariance[..., None])[..., 0]
    offset = mean_source - (coefficients * mean_guide).sum(axis=2)
    mean_coefficients = np.stack([box_mean(coefficients[..., channel], radius) for channel in range(3)], axis=2)
    return (mean_coefficients * guide).sum(axis=2) + box_mean(offset, radius)


def finish(small: np.ndarray, image: np.ndarray, mask: np.ndarray, settings: FinishSettings) -> np.ndarray:
    upsampled = joint_bilateral_upsample(small, mask, image, settings.range_sigma)
    refined = histogram_refinement(small, mask, image, upsampled, settings)
    band = dilate((upsampled > settings.band_low) & (upsampled < settings.band_high), settings.band_dilation)
    filtered = guided_filter(image, np.where(band, refined, upsampled), settings.guided_radius, settings.guided_epsilon)
    return np.clip(filtered, 0, 1).astype(np.float32)
