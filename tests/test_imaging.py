import numpy as np

from ellipse_foreground.imaging import area_resize, box_mean, dilate, erode, halve


def test_halve_averages_blocks():
    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(halve(array), [[2.5, 4.5], [10.5, 12.5]])


def test_halve_drops_odd_edge():
    assert halve(np.ones((5, 7))).shape == (2, 3)


def test_area_resize_by_two_equals_halve():
    array = np.random.default_rng(0).random((8, 12)).astype(np.float32)
    assert np.allclose(area_resize(array, 6, 4), halve(array), atol=1e-6)


def test_area_resize_keeps_channels():
    assert area_resize(np.ones((8, 12, 3), dtype=np.float32), 6, 4).shape == (4, 6, 3)


def naive_box_mean(array: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(array, radius, mode="edge")
    size = 2 * radius + 1
    result = np.empty(array.shape)
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            result[row, column] = padded[row : row + size, column : column + size].mean()
    return result


def test_box_mean_matches_window_mean_with_edge_padding():
    array = np.random.default_rng(1).random((7, 9))
    for radius in (1, 2, 3):
        assert np.allclose(box_mean(array, radius), naive_box_mean(array, radius))


def test_box_mean_of_constant_is_constant():
    assert np.allclose(box_mean(np.full((5, 6), 0.3), 2), 0.3)


def test_dilate_single_pixel_is_cross():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 1:4] = True
    expected[1:4, 2] = True
    assert np.array_equal(dilate(mask), expected)


def test_erode_inverts_dilate_of_block():
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 3:6] = True
    assert np.array_equal(erode(dilate(mask)), mask)


def test_erode_keeps_mask_touching_border():
    assert erode(np.ones((4, 4), dtype=bool)).all()
