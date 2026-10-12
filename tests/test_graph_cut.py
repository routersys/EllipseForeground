from itertools import product

import numpy as np

from ellipse_foreground.graph_cut import graph_cut_labels, pairwise_weights
from ellipse_foreground.grid import grid_edges
from ellipse_foreground.settings import GraphCutSettings
from tests.helpers import random_work


def energy(labels: np.ndarray, unary: np.ndarray, first, second, pairwise) -> float:
    flat = labels.ravel()
    data = np.where(flat, np.maximum(-unary, 0.0), np.maximum(unary, 0.0)).sum()
    return float(data + pairwise[flat[first] != flat[second]].sum())


def test_labels_reach_the_minimum_energy():
    settings = GraphCutSettings()
    for seed in range(6):
        work = random_work(2, 3, seed=seed)
        scores = 4.0 * np.random.default_rng(seed).standard_normal((2, 3))
        first, second = grid_edges(work.shape[:2])
        pairwise = pairwise_weights(work, first, second, settings.sigmas)
        unary = scores.ravel() / settings.smoothness
        found = energy(graph_cut_labels(work, scores, settings), unary, first, second, pairwise)
        best = min(
            energy(np.array(bits, dtype=bool), unary, first, second, pairwise)
            for bits in product((False, True), repeat=6)
        )
        assert found <= best + 1e-3


def test_strong_positive_scores_are_labeled_foreground():
    work = random_work(4, 4, seed=3)
    scores = np.full((4, 4), 100.0)
    assert graph_cut_labels(work, scores, GraphCutSettings()).all()


def test_strong_negative_scores_are_labeled_background():
    work = random_work(4, 4, seed=4)
    scores = np.full((4, 4), -100.0)
    assert not graph_cut_labels(work, scores, GraphCutSettings()).any()


def test_a_uniform_image_smooths_isolated_pixels():
    work = np.zeros((5, 5, 3), dtype=np.float32)
    scores = np.full((5, 5), -40.0)
    scores[2, 2] = 40.0
    settings = GraphCutSettings(smoothness=100.0)
    assert not graph_cut_labels(work, scores, settings).any()
