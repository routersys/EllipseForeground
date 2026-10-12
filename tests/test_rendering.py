import numpy as np

from ellipse_foreground.rendering import blurred_composite, disk_blur, disk_kernel
from tests.helpers import random_work


def test_disk_kernel_is_normalized_and_symmetric():
    kernel = disk_kernel((40, 50), 5.0)
    assert np.isclose(kernel.sum(), 1.0)
    assert kernel[0, 0] == kernel[0, 1] == kernel[1, 0]
    assert kernel[0, 5] > 0.0
    assert kernel[0, 6] == 0.0


def test_disk_blur_keeps_a_constant_image():
    assert np.allclose(disk_blur(np.full((30, 40), 0.4), 4.0), 0.4)


def test_disk_blur_preserves_the_mean_of_an_interior_impulse():
    image = np.zeros((60, 60))
    image[30, 30] = 1.0
    assert np.isclose(disk_blur(image, 6.0).sum(), 1.0)


def test_composite_with_full_alpha_returns_the_image():
    image = random_work(20, 30, seed=1)
    assert np.allclose(blurred_composite(image, np.ones((20, 30)), 5.0, 1e-4), image)


def test_composite_with_zero_alpha_blurs_everything():
    image = random_work(20, 30, seed=2)
    result = blurred_composite(image, np.zeros((20, 30)), 3.0, 1e-4)
    assert np.allclose(result, disk_blur(image, 3.0) / disk_blur(np.ones((20, 30)), 3.0)[..., None])


def test_foreground_color_does_not_bleed_into_the_blurred_background():
    image = np.zeros((40, 40, 3))
    image[:, 20:] = [1.0, 0.0, 0.0]
    alpha = np.zeros((40, 40))
    alpha[:, 20:] = 1.0
    result = blurred_composite(image, alpha, 4.0, 1e-4)
    assert np.allclose(result[:, :20], 0.0, atol=1e-9)
    assert np.allclose(result[:, 20:], image[:, 20:])
