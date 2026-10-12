import numpy as np
import pytest

from ellipse_foreground.reliability import Overlaps, gaussian_bhattacharyya, relative_advantage
from ellipse_foreground.settings import WeightSettings
from ellipse_foreground.weights import CueWeights, Mix, weights_for

RIDGE = WeightSettings().ridge


def analytic_overlap(first_mean, first_deviation, second_mean, second_deviation):
    variances = first_deviation**2 + second_deviation**2
    scale = np.sqrt(2 * first_deviation * second_deviation / variances)
    return scale * np.exp(-((first_mean - second_mean) ** 2) / (4 * variances))


def test_identical_distributions_overlap_completely():
    samples = np.random.default_rng(0).standard_normal((20000, 3))
    assert gaussian_bhattacharyya(samples, samples.copy(), RIDGE) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    ("first_mean", "first_deviation", "second_mean", "second_deviation"),
    [(0, 1, 2, 1.5), (0, 1, 0, 3), (1, 0.5, -1, 0.8)],
)
def test_overlap_matches_the_closed_form_for_one_dimension(first_mean, first_deviation, second_mean, second_deviation):
    rng = np.random.default_rng(1)
    first = rng.normal(first_mean, first_deviation, (400000, 1))
    second = rng.normal(second_mean, second_deviation, (400000, 1))
    expected = analytic_overlap(first_mean, first_deviation, second_mean, second_deviation)
    assert gaussian_bhattacharyya(first, second, RIDGE) == pytest.approx(expected, abs=3e-3)


def test_distant_distributions_barely_overlap():
    rng = np.random.default_rng(2)
    first = rng.normal(0.0, 1.0, (5000, 2))
    second = rng.normal(20.0, 1.0, (5000, 2))
    assert gaussian_bhattacharyya(first, second, RIDGE) < 1e-6


def test_too_few_samples_give_no_information():
    rng = np.random.default_rng(3)
    assert gaussian_bhattacharyya(rng.random((3, 3)), rng.random((100, 3)), RIDGE) == 1.0


def test_relative_advantage_is_clipped_to_the_unit_interval():
    assert relative_advantage(0.5, 0.4) == 0.0
    assert relative_advantage(0.0, 0.4) == 1.0
    assert relative_advantage(0.1, 0.4) == pytest.approx(0.75)
    assert relative_advantage(0.3, 0.0) == 0.0


def test_fixed_and_single_cue_mixes_do_not_depend_on_overlaps():
    settings = WeightSettings()
    overlaps = Overlaps(0.1, 0.4, 0.2)
    assert weights_for(Mix.PRIOR_ONLY, overlaps, settings) == CueWeights(0.0, 0.0, 0.0)
    assert weights_for(Mix.TEXTURE_ONLY, overlaps, settings) == CueWeights(1.0, 0.0, 0.0)
    assert weights_for(Mix.APPEARANCE_ONLY, overlaps, settings) == CueWeights(0.0, 0.5, 0.6)
    assert weights_for(Mix.FIXED_MIX, overlaps, settings) == CueWeights(1.0, 0.25, 0.3)


def test_adaptive_color_weight_follows_the_overlap_ratio():
    settings = WeightSettings()
    weights = weights_for(Mix.ADAPTIVE_COLOR, Overlaps(0.1, 0.4, 0.2), settings)
    assert weights == CueWeights(1.0, pytest.approx(0.5 * 0.75), 0.3)


def test_adaptive_color_weight_vanishes_when_color_overlaps_more_than_texture():
    weights = weights_for(Mix.ADAPTIVE_COLOR, Overlaps(0.6, 0.4, 0.2), WeightSettings())
    assert weights.color == 0.0


def test_adaptive_patch_weight_follows_the_patch_overlap():
    weights = weights_for(Mix.ADAPTIVE_COLOR_PATCH, Overlaps(0.1, 0.4, 0.2), WeightSettings())
    assert weights.patch == pytest.approx(0.6 * 0.5)
