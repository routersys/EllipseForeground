from itertools import product

import numpy as np

from ellipse_foreground.band import band_cut, boundary_band
from ellipse_foreground.graph_cut import pairwise_weights
from ellipse_foreground.grid import grid_edges
from ellipse_foreground.settings import BandSettings, GraphCutSettings
from tests.helpers import random_work

GRAPH = GraphCutSettings()


def energy(labels: np.ndarray, unary: np.ndarray, first, second, pairwise) -> float:
    flat = labels.ravel()
    data = np.where(flat, np.maximum(-unary, 0.0), np.maximum(unary, 0.0)).sum()
    return float(data + pairwise[flat[first] != flat[second]].sum())


def test_the_band_surrounds_the_boundary_of_the_mask():
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 3:6] = True
    band = boundary_band(mask, 1)
    assert band[3, 3] and band[2, 3] and band[5, 5] and band[6, 5]
    assert not band[4, 4]
    assert not band[1, 1]


def test_a_mask_without_a_boundary_has_an_empty_band():
    assert not boundary_band(np.zeros((6, 6), dtype=bool), 2).any()
    assert not boundary_band(np.ones((6, 6), dtype=bool), 2).any()


def test_a_radius_of_zero_leaves_the_mask_unchanged():
    work = random_work(6, 6, seed=1)
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True
    scores = np.full((6, 6), -50.0)
    assert np.array_equal(band_cut(work, scores, mask, BandSettings(radius=0), GRAPH), mask)


def test_pixels_outside_the_band_keep_their_labels():
    work = random_work(10, 10, seed=2)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    scores = np.random.default_rng(2).standard_normal((10, 10)) * 80.0
    result = band_cut(work, scores, mask, BandSettings(radius=1), GRAPH)
    fixed = ~boundary_band(mask, 1)
    assert np.array_equal(result[fixed], mask[fixed])


def test_strong_scores_move_the_boundary_inside_the_band():
    work = np.zeros((12, 12, 3), dtype=np.float32)
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True
    scores = np.full((12, 12), -200.0)
    scores[3:9, 3:9] = 200.0
    result = band_cut(work, scores, mask, BandSettings(radius=1), GRAPH)
    band = boundary_band(mask, 1)
    assert np.array_equal(result[band], (scores > 0)[band])
    assert np.array_equal(result[~band], mask[~band])
    assert result.sum() > mask.sum()


def test_labels_reach_the_minimum_energy_with_the_outside_fixed():
    for seed in range(6):
        work = random_work(3, 4, seed=seed)
        rng = np.random.default_rng(seed)
        scores = 4.0 * rng.standard_normal((3, 4))
        mask = rng.random((3, 4)) > 0.5
        free = boundary_band(mask, 1)
        first, second = grid_edges(mask.shape)
        pairwise = pairwise_weights(work, first, second, GRAPH.sigmas)
        band = BandSettings(radius=1)
        unary = scores.ravel() / band.smoothness
        found = energy(band_cut(work, scores, mask, band, GRAPH), unary, first, second, pairwise)
        best = np.inf
        for bits in product((False, True), repeat=int(free.sum())):
            candidate = mask.copy()
            candidate[free] = bits
            best = min(best, energy(candidate, unary, first, second, pairwise))
        assert found <= best + 1e-3
