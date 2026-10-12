import numpy as np

from ellipse_foreground.evaluation.color_transfer import from_lab, to_lab, transfer_colors


def random_colors(shape, seed):
    return np.random.default_rng(seed).uniform(0.05, 0.95, (*shape, 3)).astype(np.float32)


def test_lab_round_trip():
    colors = random_colors((20, 30), 0)
    assert np.allclose(from_lab(to_lab(colors)), colors, atol=1e-5)


def test_zero_strength_returns_the_foreground_unchanged():
    foreground = random_colors((10, 10), 1)
    result = transfer_colors(foreground, np.ones((10, 10)), random_colors((8, 8), 2), 0.0)
    assert result is foreground


def test_full_strength_matches_the_background_statistics_of_the_opaque_pixels():
    foreground = random_colors((40, 40), 3)
    alpha = np.ones((40, 40))
    background = random_colors((50, 50), 4)
    result = transfer_colors(foreground, alpha, background, 1.0)
    result_lab = to_lab(result.reshape(-1, 3))
    background_lab = to_lab(background.reshape(-1, 3))
    assert np.allclose(result_lab.mean(axis=0), background_lab.mean(axis=0), atol=0.03)
    assert np.allclose(result_lab.std(axis=0), background_lab.std(axis=0), atol=0.03)


def test_statistics_are_taken_from_opaque_pixels_only():
    foreground = random_colors((20, 20), 5)
    alpha = np.zeros((20, 20))
    alpha[:, :10] = 1.0
    background = random_colors((30, 30), 6)
    changed = foreground.copy()
    changed[:, 10:] = 0.5
    assert np.allclose(
        transfer_colors(foreground, alpha, background, 1.0)[:, :10],
        transfer_colors(changed, alpha, background, 1.0)[:, :10],
    )


def test_partial_strength_lies_between_the_endpoints_in_lab_space():
    foreground = random_colors((20, 20), 7)
    alpha = np.ones((20, 20))
    background = random_colors((30, 30), 8)
    half = to_lab(transfer_colors(foreground, alpha, background, 0.5))
    start, end = to_lab(foreground), to_lab(transfer_colors(foreground, alpha, background, 1.0))
    assert np.allclose(half, 0.5 * (start + end), atol=0.05)
