import math

import numpy as np

from ellipse_foreground.evaluation.auc import rank_auc


def test_perfect_separation_gives_one():
    assert rank_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])) == 1.0


def test_reversed_separation_gives_zero():
    assert rank_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([False, False, True, True])) == 0.0


def test_ties_count_as_half():
    assert rank_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([False, True, False, True])) == 0.5


def test_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, 200).astype(float)
    labels = rng.random(200) < 0.4
    positives, negatives = scores[labels], scores[~labels]
    expected = np.mean([(p > n) + 0.5 * (p == n) for p in positives for n in negatives])
    assert math.isclose(rank_auc(scores, labels), expected)


def test_a_single_class_is_undefined():
    assert math.isnan(rank_auc(np.array([0.1, 0.2]), np.array([True, True])))
    assert math.isnan(rank_auc(np.array([0.1, 0.2]), np.array([False, False])))
