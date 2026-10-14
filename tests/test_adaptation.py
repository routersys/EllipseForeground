import numpy as np
import pytest

from ellipse_foreground import Analysis, Ellipse, Mix, Settings
from ellipse_foreground.settings import RefitSettings, StabilitySettings, WorkSettings
from tests.helpers import noisy_scene

WORK = WorkSettings(width=64)
DISPLACED = Ellipse(80.0, 50.0, 55.0, 60.0)
ALWAYS = Settings(work=WORK, stability=StabilitySettings(threshold=2.0))
NEVER = Settings(work=WORK, stability=StabilitySettings(threshold=0.0))


def work_truth(body: np.ndarray) -> np.ndarray:
    return body.reshape(body.shape[0] // 2, 2, body.shape[1] // 2, 2).mean(axis=(1, 3)) > 0.5


@pytest.fixture(scope="module")
def scene():
    image, source, body = noisy_scene(72, 128, seed=2)
    return image, source, work_truth(body)


def analysis_of(scene, settings):
    image, source, _ = scene
    return Analysis(image, source, settings)


def test_a_threshold_of_zero_never_refits(scene):
    analysis = analysis_of(scene, NEVER)
    scores, mask = analysis.solve(DISPLACED)
    adapted_scores, adapted_mask = analysis.adapt(DISPLACED, Mix.ADAPTIVE_COLOR, scores, mask)
    assert adapted_scores is scores
    assert adapted_mask is mask


def test_a_threshold_above_one_always_refits(scene):
    analysis = analysis_of(scene, ALWAYS)
    scores, mask = analysis.solve(DISPLACED)
    adapted = analysis.adapt(DISPLACED, Mix.ADAPTIVE_COLOR, scores, mask)
    refit = analysis.refit(DISPLACED, Mix.ADAPTIVE_COLOR, scores, mask)
    assert np.array_equal(adapted[1], refit[1])
    assert np.array_equal(adapted[0], refit[0])


def test_refit_improves_a_displaced_ellipse(scene):
    analysis = analysis_of(scene, ALWAYS)
    truth = scene[2]
    scores, mask = analysis.solve(DISPLACED)
    _, refit = analysis.refit(DISPLACED, Mix.ADAPTIVE_COLOR, scores, mask)

    def iou(found: np.ndarray) -> float:
        return float((found & truth).sum() / (found | truth).sum())

    assert iou(refit) > iou(mask)
    assert iou(refit) > 0.98


def test_zero_iterations_leave_the_result_unchanged(scene):
    analysis = analysis_of(scene, Settings(work=WORK, refit=RefitSettings(iterations=0)))
    scores, mask = analysis.solve(DISPLACED)
    refit_scores, refit_mask = analysis.refit(DISPLACED, Mix.ADAPTIVE_COLOR, scores, mask)
    assert refit_scores is scores
    assert refit_mask is mask


def test_refit_stops_when_the_mask_is_smaller_than_the_minimum_area(scene):
    analysis = analysis_of(scene, Settings(work=WORK, refit=RefitSettings(minimum_area=10**6)))
    scores, mask = analysis.solve(DISPLACED)
    refit_scores, refit_mask = analysis.refit(DISPLACED, Mix.ADAPTIVE_COLOR, scores, mask)
    assert refit_scores is scores
    assert refit_mask is mask


def test_adaptive_extraction_without_a_refit_equals_plain_extraction(scene):
    analysis = analysis_of(scene, NEVER)
    assert np.array_equal(analysis.extract(DISPLACED, adaptive=True), analysis.extract(DISPLACED))


def test_adaptive_extraction_returns_an_alpha_of_the_image_shape(scene):
    analysis = analysis_of(scene, ALWAYS)
    alpha = analysis.extract(DISPLACED, refine=True, adaptive=True)
    assert alpha.shape == scene[0].shape[:2]
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0
