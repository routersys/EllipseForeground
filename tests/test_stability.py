import numpy as np
import pytest

from ellipse_foreground import Analysis, Ellipse, Settings
from ellipse_foreground.settings import StabilitySettings, WorkSettings
from ellipse_foreground.stability import jaccard, perturbed_ellipses
from tests.helpers import noisy_scene

WORK = WorkSettings(width=64)
ELLIPSE = Ellipse(66.0, 41.0, 34.0, 40.0)


@pytest.fixture(scope="module")
def analysis():
    image, source, _ = noisy_scene(72, 128, seed=2)
    return Analysis(image, source, Settings(work=WORK))


def test_jaccard_is_the_ratio_of_intersection_to_union():
    first = np.array([[True, True], [False, False]])
    second = np.array([[True, False], [True, False]])
    assert jaccard(first, second) == pytest.approx(1 / 3)
    assert jaccard(first, first) == 1.0
    assert jaccard(first, ~first) == 0.0


def test_two_empty_masks_are_identical():
    empty = np.zeros((3, 3), dtype=bool)
    assert jaccard(empty, empty) == 1.0


def test_the_perturbations_scale_shift_and_stretch_the_ellipse():
    ellipse = Ellipse(50.0, 30.0, 20.0, 10.0)
    settings = StabilitySettings(scales=(0.8, 1.25), shift=0.1)
    perturbed = perturbed_ellipses(ellipse, settings)
    assert len(perturbed) == 8
    assert len(set(perturbed)) == 8
    assert ellipse.scaled(0.8) in perturbed
    assert ellipse.scaled(1.25) in perturbed
    assert Ellipse(52.0, 30.0, 20.0, 10.0) in perturbed
    assert Ellipse(50.0, 29.0, 20.0, 10.0) in perturbed
    assert ellipse.stretched(0.8, 1.25) in perturbed
    assert ellipse.stretched(1.25, 0.8) in perturbed


def test_solve_returns_the_scores_and_the_cut_of_those_scores(analysis):
    scores, mask = analysis.solve(ELLIPSE)
    assert scores.shape == analysis.shape
    assert np.array_equal(mask, analysis.cut(scores))


def test_extraction_is_the_finished_cut_of_the_solved_scores(analysis):
    _, mask = analysis.solve(ELLIPSE)
    assert np.array_equal(analysis.extract(ELLIPSE), analysis.finish(mask))


def test_a_clear_body_gives_a_stability_close_to_one(analysis):
    scores, mask = analysis.solve(ELLIPSE)
    assert 0.9 < analysis.stability(ELLIPSE, scores, mask) <= 1.0


def test_stability_is_lower_when_the_prior_decides_the_result(analysis):
    scores, _ = analysis.solve(ELLIPSE)
    prior = analysis.prior_map(ELLIPSE)
    mask = analysis.cut(prior)
    assert analysis.stability(ELLIPSE, prior, mask) < analysis.stability(ELLIPSE, scores, analysis.cut(scores))
