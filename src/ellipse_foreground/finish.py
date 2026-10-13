import numpy as np
from PIL import Image
from scipy import ndimage

from ellipse_foreground.imaging import dilate, erode
from ellipse_foreground.settings import FinishSettings


def bilinear_upsample(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.float32)).resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


class BilateralUpsampler:
    def __init__(self, small: np.ndarray, image: np.ndarray, range_sigma: float) -> None:
        height, width = image.shape[:2]
        small_height, small_width = small.shape[:2]
        rows = (np.arange(height) + 0.5) * small_height / height - 0.5
        columns = (np.arange(width) + 0.5) * small_width / width - 0.5
        first_row, first_column = np.floor(rows).astype(np.int64), np.floor(columns).astype(np.int64)
        row_fraction, column_fraction = (rows - first_row)[:, None], (columns - first_column)[None, :]
        image_planes = [image[..., channel].astype(np.float64) for channel in range(3)]
        small_planes = [small[..., channel].astype(np.float64) for channel in range(3)]
        self.taps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.denominator = np.full((height, width), 1e-9)
        for dy in (0, 1):
            for dx in (0, 1):
                source_rows = np.clip(first_row + dy, 0, small_height - 1)
                source_columns = np.clip(first_column + dx, 0, small_width - 1)
                spatial = (row_fraction if dy else 1 - row_fraction) * (column_fraction if dx else 1 - column_fraction)
                distance = np.zeros((height, width))
                for image_plane, small_plane in zip(image_planes, small_planes, strict=True):
                    difference = image_plane - small_plane[source_rows][:, source_columns]
                    distance += difference * difference
                weight = spatial * np.exp(-distance / (2 * range_sigma**2))
                self.taps.append((source_rows, source_columns, weight))
                self.denominator += weight

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        numerator = np.zeros(self.denominator.shape)
        for source_rows, source_columns, weight in self.taps:
            numerator += weight * mask[source_rows][:, source_columns]
        return (numerator / self.denominator).astype(np.float32)


def joint_bilateral_upsample(small: np.ndarray, mask: np.ndarray, image: np.ndarray, range_sigma: float) -> np.ndarray:
    return BilateralUpsampler(small, image, range_sigma)(mask)


def histogram_indices(image: np.ndarray, bins: int) -> list[np.ndarray]:
    return [np.clip((image[..., channel] * bins).astype(np.int64), 0, bins - 1) for channel in range(3)]


def histogram_refinement(
    small: np.ndarray,
    mask: np.ndarray,
    image: np.ndarray,
    upsampled: np.ndarray,
    settings: FinishSettings,
    indices: list[np.ndarray] | None = None,
) -> np.ndarray:
    inner = erode(mask > 0.5, 1)
    outer = ~dilate(mask > 0.5, 1)
    bins = settings.histogram_bins
    if indices is None:
        indices = histogram_indices(image, bins)
    log_likelihood = np.zeros(image.shape[:2])
    for channel in range(3):
        foreground = np.histogram(small[..., channel][inner], bins=bins, range=(0, 1))[0] + 1.0
        background = np.histogram(small[..., channel][outer], bins=bins, range=(0, 1))[0] + 1.0
        table = np.log((foreground / foreground.sum()) / (background / background.sum()))
        log_likelihood += table[indices[channel]]
    evidence = np.clip(log_likelihood, -settings.histogram_clip, settings.histogram_clip) / settings.histogram_clip
    logit = settings.logit_gain * (upsampled - 0.5) + settings.histogram_gain * evidence
    return (1 / (1 + np.exp(-logit))).astype(np.float32)


def local_mean(array: np.ndarray, radius: int) -> np.ndarray:
    size = (1,) * (array.ndim - 2) + (2 * radius + 1,) * 2
    return ndimage.uniform_filter(array, size=size, mode="nearest")


def inverse_3x3(matrices: np.ndarray) -> np.ndarray:
    m = matrices
    cofactors = np.empty_like(m)
    cofactors[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    cofactors[0, 1] = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
    cofactors[0, 2] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    cofactors[1, 0] = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]
    cofactors[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
    cofactors[1, 2] = m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]
    cofactors[2, 0] = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
    cofactors[2, 1] = m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]
    cofactors[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    determinant = m[0, 0] * cofactors[0, 0] + m[0, 1] * cofactors[0, 1] + m[0, 2] * cofactors[0, 2]
    return np.ascontiguousarray(cofactors.swapaxes(0, 1)) / determinant


class GuidedFilter:
    def __init__(self, guide: np.ndarray, radius: int, epsilon: float) -> None:
        self.planes = np.ascontiguousarray(np.moveaxis(guide.astype(np.float64), 2, 0))
        self.radius = radius
        self.means = local_mean(self.planes, radius)
        variance = np.empty((3, 3, *self.planes.shape[1:]))
        for first in range(3):
            for second in range(first, 3):
                product = local_mean(self.planes[first] * self.planes[second], radius)
                variance[first, second] = variance[second, first] = product - self.means[first] * self.means[second]
            variance[first, first] += epsilon
        self.inverse = inverse_3x3(variance)

    def __call__(self, source: np.ndarray) -> np.ndarray:
        source = source.astype(np.float64)
        radius = self.radius
        mean_source = local_mean(source, radius)
        covariance = local_mean(self.planes * source, radius) - self.means * mean_source
        coefficients = np.einsum("ijhw,jhw->ihw", self.inverse, covariance)
        offset = mean_source - np.einsum("chw,chw->hw", coefficients, self.means)
        return np.einsum("chw,chw->hw", local_mean(coefficients, radius), self.planes) + local_mean(offset, radius)


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    return GuidedFilter(guide, radius, epsilon)(source)


class Finisher:
    def __init__(self, small: np.ndarray, image: np.ndarray, settings: FinishSettings) -> None:
        self.small = small
        self.image = image
        self.settings = settings
        self.upsampler = BilateralUpsampler(small, image, settings.range_sigma)
        self.indices = histogram_indices(image, settings.histogram_bins)
        self.filter = GuidedFilter(image, settings.guided_radius, settings.guided_epsilon)

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        settings = self.settings
        upsampled = self.upsampler(mask)
        refined = histogram_refinement(self.small, mask, self.image, upsampled, settings, self.indices)
        band = dilate((upsampled > settings.band_low) & (upsampled < settings.band_high), settings.band_dilation)
        filtered = self.filter(np.where(band, refined, upsampled))
        return np.clip(filtered, 0, 1).astype(np.float32)


def finish(small: np.ndarray, image: np.ndarray, mask: np.ndarray, settings: FinishSettings) -> np.ndarray:
    return Finisher(small, image, settings)(mask)
