import numpy as np

from ellipse_foreground.cues.evidence import bounded_log_ratio, normalized_histogram


def test_bounded_log_ratio_is_clipped_and_normalized():
    ratio = bounded_log_ratio(np.array([1e6, 1.0, 1e-6]), np.array([1.0, 1.0, 1.0]), 2.0)
    assert np.allclose(ratio, [1.0, 0.0, -1.0])


def test_normalized_histogram_sums_to_one():
    histogram = normalized_histogram(np.array([0, 1, 1, 3]), 5, 0.5)
    assert np.isclose(histogram.sum(), 1.0)
    assert histogram[1] > histogram[0] > histogram[2]
