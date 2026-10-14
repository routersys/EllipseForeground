import numpy as np
import pytest

from ellipse_foreground.evaluation.experiments.unmixing import (
    OCCUPANCY_EDGES,
    annulus_areas,
    annulus_index,
    annulus_table,
    occupancy_of,
    population_occupancy,
    smoothed,
    unmix,
)

COUNT = len(OCCUPANCY_EDGES) - 1


def test_each_radius_falls_into_one_annulus():
    radius = np.array([0.0, 0.14, 0.15, 1.1, 2.5, 50.0])
    index = annulus_index(radius)
    assert index.tolist() == [0, 0, 1, 7, 11, 12]
    assert annulus_areas(index).sum() == len(radius)


def test_occupancy_is_the_clipped_share_of_the_mask_in_each_annulus():
    radius = np.concatenate([np.full(10, 0.05), np.full(10, 0.2), np.full(10, 5.0)])
    mask = np.concatenate([np.ones(10), np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), np.zeros(10)]).astype(bool)
    occupancy = occupancy_of(annulus_index(radius), mask)
    assert occupancy[0] == pytest.approx(0.98)
    assert occupancy[1] == pytest.approx(0.4)
    assert occupancy[12] == pytest.approx(0.02)


def test_the_population_profile_falls_with_the_radius():
    profile = population_occupancy()
    assert len(profile) == COUNT
    assert np.all(np.diff(profile) <= 0.0)
    assert profile[0] > 0.8
    assert profile[-1] == pytest.approx(0.02)


def test_the_annulus_table_counts_codes_per_annulus():
    index = np.array([0, 0, 1, 1, 1])
    codes = np.array([2, 2, 0, 1, 1])
    table = annulus_table(codes, index, 3)
    assert table.shape == (COUNT, 3)
    assert table[0].tolist() == [0.0, 0.0, 2.0]
    assert table[1].tolist() == [1.0, 2.0, 0.0]


def test_smoothing_sums_each_color_with_its_neighbors():
    histogram = np.zeros(27)
    histogram[13] = 1.0
    assert smoothed(histogram, 3).tolist() == [1.0] * 27


def mixture(foreground, background, occupancy, pixels):
    areas = np.full(COUNT, float(pixels))
    table = areas[:, None] * (occupancy[:, None] * foreground + (1.0 - occupancy[:, None]) * background)
    return table, areas


def test_the_two_distributions_are_recovered_from_a_mixture_with_known_occupancy():
    rng = np.random.default_rng(0)
    foreground = rng.dirichlet(np.ones(12) * 0.5)
    background = rng.dirichlet(np.ones(12) * 0.5)
    occupancy = np.clip(np.linspace(0.95, 0.03, COUNT), 0.02, 0.98)
    table, areas = mixture(foreground, background, occupancy, 10000)
    found_foreground, found_background = unmix(table, areas, occupancy, np.full(12, 1 / 12), np.full(12, 1 / 12), 1e-9)
    assert np.allclose(found_foreground, foreground, atol=1e-3)
    assert np.allclose(found_background, background, atol=1e-3)


def test_negative_solutions_are_clamped_to_zero():
    foreground = np.array([0.7, 0.3, 0.0, 0.0])
    background = np.array([0.0, 0.0, 0.5, 0.5])
    occupancy = np.clip(np.linspace(0.95, 0.03, COUNT), 0.02, 0.98)
    table, areas = mixture(foreground, background, occupancy, 5000)
    found_foreground, found_background = unmix(table, areas, occupancy, np.full(4, 0.25), np.full(4, 0.25), 1e-9)
    assert found_foreground.min() >= 0.0
    assert found_background.min() >= 0.0
    assert found_foreground[2:].max() < 1e-3
    assert found_background[:2].max() < 1e-3


def test_a_strong_ridge_keeps_the_base_distributions():
    rng = np.random.default_rng(1)
    foreground = rng.dirichlet(np.ones(6))
    background = rng.dirichlet(np.ones(6))
    occupancy = np.clip(np.linspace(0.95, 0.03, COUNT), 0.02, 0.98)
    table, areas = mixture(foreground, background, occupancy, 1000)
    base_foreground, base_background = np.full(6, 1 / 6), np.full(6, 1 / 6)
    found_foreground, found_background = unmix(table, areas, occupancy, base_foreground, base_background, 1e6)
    assert np.allclose(found_foreground, base_foreground, atol=1e-3)
    assert np.allclose(found_background, base_background, atol=1e-3)
