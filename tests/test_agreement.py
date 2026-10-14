import numpy as np

from ellipse_foreground import CueMaps, CueWeights
from ellipse_foreground.agreement import agreement_weights, correlation, cue_agreements
from ellipse_foreground.settings import AgreementSettings

WEIGHTS = CueWeights(1.0, 0.5, 0.3)


def maps(texture: np.ndarray, color: np.ndarray, patch: np.ndarray) -> CueMaps:
    return CueMaps(np.zeros_like(texture), color, texture, patch)


def test_correlation_of_identical_negated_and_constant_arrays():
    values = np.random.default_rng(0).standard_normal((6, 7))
    assert abs(correlation(values, values) - 1.0) < 1e-9
    assert abs(correlation(values, -values) + 1.0) < 1e-9
    assert correlation(values, np.ones_like(values)) == 0.0


def test_cues_that_agree_with_each_other_keep_their_weights():
    values = np.random.default_rng(1).standard_normal((8, 8))
    adjusted = agreement_weights(WEIGHTS, maps(values, values, values), AgreementSettings())
    assert adjusted == WEIGHTS


def test_the_cue_that_disagrees_loses_weight_to_the_others():
    rng = np.random.default_rng(2)
    shared = rng.standard_normal((30, 30))
    noise = rng.standard_normal((30, 30))
    cues = maps(shared, shared + 0.1 * rng.standard_normal((30, 30)), noise)
    agreements = cue_agreements(cues)
    assert agreements[2] < min(agreements[0], agreements[1])
    adjusted = agreement_weights(WEIGHTS, cues, AgreementSettings())
    assert adjusted.patch < WEIGHTS.patch
    assert adjusted.texture > WEIGHTS.texture
    assert adjusted.color > WEIGHTS.color


def test_factors_stay_within_the_floor_and_the_ceiling():
    rng = np.random.default_rng(3)
    shared = rng.standard_normal((30, 30))
    cues = maps(shared, shared, rng.standard_normal((30, 30)))
    settings = AgreementSettings(gain=50.0, floor=0.25, ceiling=1.5)
    adjusted = agreement_weights(WEIGHTS, cues, settings)
    assert abs(adjusted.patch - 0.25 * WEIGHTS.patch) < 1e-12
    assert abs(adjusted.texture - 1.5 * WEIGHTS.texture) < 1e-12
    assert abs(adjusted.color - 1.5 * WEIGHTS.color) < 1e-12
