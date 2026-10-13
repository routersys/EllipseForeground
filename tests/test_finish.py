import numpy as np

from ellipse_foreground.finish import (
    Finisher,
    bilinear_upsample,
    finish,
    guided_filter,
    histogram_refinement,
    joint_bilateral_upsample,
)
from ellipse_foreground.settings import FinishSettings
from tests.helpers import random_work


def split_image(height: int = 32, width: int = 48) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, : width // 2] = [0.8, 0.2, 0.2]
    image[:, width // 2 :] = [0.1, 0.2, 0.7]
    image += 0.01 * random_work(height, width, seed=1)
    mask = np.zeros((height // 2, width // 2), dtype=np.float32)
    mask[:, : width // 4] = 1.0
    return image, mask


def test_bilinear_upsample_has_the_requested_shape_and_range():
    upsampled = bilinear_upsample(np.array([[0.0, 1.0], [1.0, 0.0]]), (6, 8))
    assert upsampled.shape == (6, 8)
    assert upsampled.min() >= 0.0
    assert upsampled.max() <= 1.0


def test_joint_bilateral_upsample_of_a_constant_mask_is_constant():
    image = np.full((16, 24, 3), 0.3, dtype=np.float32)
    small = np.full((8, 12, 3), 0.3, dtype=np.float32)
    assert np.allclose(joint_bilateral_upsample(small, np.ones((8, 12)), image, 0.12), 1.0)
    assert np.allclose(joint_bilateral_upsample(small, np.zeros((8, 12)), image, 0.12), 0.0)


def test_joint_bilateral_upsample_follows_the_image_edge():
    image, mask = split_image()
    small = image[::2, ::2]
    upsampled = joint_bilateral_upsample(small, mask, image, 0.12)
    assert upsampled[:, :20].min() > 0.95
    assert upsampled[:, 28:].max() < 0.05


def test_histogram_refinement_stays_in_the_unit_interval():
    image, mask = split_image()
    small = image[::2, ::2]
    upsampled = joint_bilateral_upsample(small, mask, image, 0.12)
    refined = histogram_refinement(small, mask, image, upsampled, FinishSettings())
    assert refined.min() >= 0.0
    assert refined.max() <= 1.0
    assert refined[:, :20].mean() > 0.9


def test_guided_filter_of_a_constant_source_is_constant():
    guide = random_work(12, 14, seed=4)
    assert np.allclose(guided_filter(guide, np.full((12, 14), 0.37), 2, 1e-3), 0.37)


def test_guided_filter_reproduces_a_source_that_is_linear_in_the_guide():
    guide = random_work(12, 14, seed=5)
    source = 0.5 * guide[..., 0] + 0.2
    assert np.allclose(guided_filter(guide, source, 2, 1e-9), source, atol=1e-4)


def test_a_finisher_keeps_no_state_between_masks():
    image, mask = split_image()
    small = image[::2, ::2]
    other = np.roll(mask, 3, axis=1)
    finisher = Finisher(small, image, FinishSettings())
    first = finisher(mask)
    finisher(other)
    assert np.array_equal(finisher(mask), first)
    assert np.array_equal(finish(small, image, other, FinishSettings()), finisher(other))


def direct_guided_filter(guide, source, radius, epsilon):
    height, width = source.shape
    padded_guide = np.pad(guide.astype(np.float64), ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    padded_source = np.pad(source.astype(np.float64), radius, mode="edge")
    size = 2 * radius + 1
    coefficients = np.empty((height, width, 3))
    offsets = np.empty((height, width))
    for row in range(height):
        for column in range(width):
            window_guide = padded_guide[row : row + size, column : column + size].reshape(-1, 3)
            window_source = padded_source[row : row + size, column : column + size].reshape(-1)
            mean_guide, mean_source = window_guide.mean(axis=0), window_source.mean()
            covariance = (window_guide * window_source[:, None]).mean(axis=0) - mean_guide * mean_source
            variance = (window_guide.T @ window_guide) / len(window_guide) - np.outer(mean_guide, mean_guide)
            coefficient = np.linalg.solve(variance + epsilon * np.eye(3), covariance)
            coefficients[row, column] = coefficient
            offsets[row, column] = mean_source - coefficient @ mean_guide
    padded_coefficients = np.pad(coefficients, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    padded_offsets = np.pad(offsets, radius, mode="edge")
    result = np.empty((height, width))
    for row in range(height):
        for column in range(width):
            mean_coefficient = padded_coefficients[row : row + size, column : column + size].reshape(-1, 3).mean(axis=0)
            result[row, column] = (
                mean_coefficient @ guide[row, column] + padded_offsets[row : row + size, column : column + size].mean()
            )
    return result


def test_guided_filter_matches_the_window_by_window_definition():
    guide = random_work(11, 13, seed=6)
    source = np.random.default_rng(7).random((11, 13))
    assert np.allclose(guided_filter(guide, source, 2, 1e-3), direct_guided_filter(guide, source, 2, 1e-3), atol=1e-9)


def test_finish_returns_an_alpha_in_the_unit_interval_with_the_image_shape():
    image, mask = split_image()
    alpha = finish(image[::2, ::2], image, mask, FinishSettings())
    assert alpha.shape == image.shape[:2]
    assert alpha.dtype == np.float32
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert alpha[:, :20].mean() > 0.9
    assert alpha[:, 28:].mean() < 0.1
