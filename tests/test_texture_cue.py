import numpy as np

from ellipse_foreground.cues.texture import neighborhood_contrast, quantize, texture_evidence, texture_features
from ellipse_foreground.settings import TextureSettings


def test_neighborhood_contrast_is_zero_on_flat_regions_and_positive_next_to_edges():
    source = np.zeros((6, 6, 3), dtype=np.uint8)
    source[:, 3:] = 100
    contrast = neighborhood_contrast(source)
    assert contrast[:, 0].max() == 0
    assert contrast[2, 2] == contrast[2, 3] == 100


def test_neighborhood_contrast_is_the_largest_channel_difference_to_the_eight_neighbors():
    source = np.random.default_rng(5).integers(0, 256, (13, 17, 3), dtype=np.uint8)
    values = source.astype(np.int16)
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    expected = np.zeros(values.shape[:2], dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor = padded[1 + dy : 1 + dy + 13, 1 + dx : 1 + dx + 17]
            expected = np.maximum(expected, np.abs(neighbor - values).max(axis=2))
    assert np.array_equal(neighborhood_contrast(source), expected)


def test_texture_features_of_a_flat_image_are_fully_flat():
    features = texture_features(np.full((20, 40, 3), 77, dtype=np.uint8), (10, 20), TextureSettings())
    assert np.allclose(features.flatness, 1.0)
    assert np.allclose(features.magnitude, 0.0)


def test_quantize_puts_exact_boundaries_into_the_upper_bin():
    values = np.array([0.0, 1.0 / 6.0, 1.0 / 6.0 - 1e-12, 1.0])
    assert quantize(values, 6).tolist() == [0, 1, 1, 5]


def test_texture_evidence_separates_flat_from_busy_regions():
    source = np.zeros((40, 80, 3), dtype=np.uint8)
    source[:, 40:] = np.random.default_rng(0).integers(0, 255, (40, 40, 3))
    settings = TextureSettings()
    features = texture_features(source, (20, 40), settings)
    core = np.zeros((20, 40), dtype=bool)
    core[:, :15] = True
    ring = np.zeros((20, 40), dtype=bool)
    ring[:, 25:] = True
    evidence = texture_evidence(features, core, ring, settings)
    assert evidence[:, :12].mean() > 0.5
    assert evidence[:, 28:].mean() < -0.5
