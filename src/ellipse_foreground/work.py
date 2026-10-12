import numpy as np

from ellipse_foreground.imaging import area_resize
from ellipse_foreground.settings import WorkSettings


def work_shape(image_shape: tuple[int, ...], settings: WorkSettings) -> tuple[int, int]:
    height, width = image_shape[:2]
    return round(settings.width * height / width), settings.width


def normalize_range(small: np.ndarray, settings: WorkSettings) -> np.ndarray:
    work = np.empty_like(small)
    for channel in range(small.shape[2]):
        low, high = np.percentile(small[..., channel], [settings.low_percentile, settings.high_percentile])
        scaled = (small[..., channel] - low) / max(high - low, settings.minimum_range)
        work[..., channel] = np.clip(scaled, 0, settings.ceiling)
    return work


def reduce_image(image: np.ndarray, settings: WorkSettings) -> tuple[np.ndarray, np.ndarray]:
    height, width = work_shape(image.shape, settings)
    small = area_resize(image, width, height)
    return small, normalize_range(small, settings)
