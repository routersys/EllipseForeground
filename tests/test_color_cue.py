import numpy as np

from ellipse_foreground.cues.color import color_codes, color_evidence, smoothed_color_histogram
from ellipse_foreground.settings import ColorSettings


def test_color_codes_cover_the_cube_and_clip_overexposed_values():
    work = np.array([[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5], [0.5, 0.25, 0.75]]])
    assert color_codes(work, 8).tolist() == [[0, 511, 4 * 64 + 2 * 8 + 6]]


def test_smoothed_histogram_is_normalized_and_spreads_to_neighbors():
    histogram = smoothed_color_histogram(np.array([0, 0, 0]), 8).reshape(8, 8, 8)
    assert np.isclose(histogram.sum(), 1.0)
    assert histogram[1, 1, 1] > 0.0
    assert histogram[2, 2, 2] == 0.0


def test_color_evidence_prefers_the_core_color():
    work = np.zeros((10, 10, 3), dtype=np.float32)
    work[:, :5] = [0.9, 0.1, 0.1]
    work[:, 5:] = [0.1, 0.1, 0.9]
    core = np.zeros((10, 10), dtype=bool)
    core[:, :3] = True
    ring = np.zeros((10, 10), dtype=bool)
    ring[:, 7:] = True
    evidence = color_evidence(work, core, ring, ColorSettings())
    assert evidence[:, :5].min() > 0.9
    assert evidence[:, 5:].max() < -0.9
