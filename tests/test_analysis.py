import numpy as np
import pytest

from ellipse_foreground import Analysis, BoundaryForm, CueWeights, Ellipse, Mix, Settings
from ellipse_foreground.band import boundary_band
from ellipse_foreground.settings import WorkSettings
from tests.helpers import noisy_scene

SETTINGS = Settings(work=WorkSettings(width=64))
ELLIPSE = Ellipse(66.0, 41.0, 34.0, 40.0)
FIXED_WEIGHTS = CueWeights(1.0, 0.25, 0.3)


def foreground_iou(alpha: np.ndarray, body: np.ndarray) -> float:
    found = alpha > 0.5
    return float((found & body).sum() / (found | body).sum())


@pytest.fixture(scope="module")
def scene():
    image, source, body = noisy_scene(72, 128, seed=2)
    return Analysis(image, source, SETTINGS), body


@pytest.fixture(scope="module")
def fixed_scores(scene):
    analysis, _ = scene
    return analysis.cue_maps(ELLIPSE, analysis.regions(ELLIPSE)).scores(FIXED_WEIGHTS)


def test_work_geometry_follows_the_settings(scene):
    analysis, _ = scene
    assert analysis.shape == (36, 64)
    assert analysis.scale_x == 0.5
    assert analysis.scale_y == 0.5
    assert analysis.tree.leaf_count == 36 * 64


def test_regions_are_nested(scene):
    analysis, _ = scene
    regions = analysis.regions(ELLIPSE)
    assert not (regions.core & regions.ring).any()
    assert not (regions.core & ~regions.inside).any()
    assert not (regions.ring & regions.inside).any()


@pytest.mark.parametrize("mix", [Mix.TEXTURE_ONLY, Mix.FIXED_MIX, Mix.ADAPTIVE_COLOR, Mix.ADAPTIVE_COLOR_PATCH])
def test_extraction_recovers_a_flat_body(scene, mix):
    analysis, body = scene
    alpha = analysis.extract(ELLIPSE, mix)
    assert alpha.shape == body.shape
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0
    assert foreground_iou(alpha, body) > 0.9


def test_extraction_is_deterministic(scene):
    analysis, _ = scene
    assert np.array_equal(analysis.extract(ELLIPSE), analysis.extract(ELLIPSE))


def test_a_new_analysis_gives_the_same_result(scene):
    analysis, _ = scene
    image, source, _ = noisy_scene(72, 128, seed=2)
    assert np.array_equal(Analysis(image, source, SETTINGS).extract(ELLIPSE), analysis.extract(ELLIPSE))


@pytest.mark.parametrize("form", list(BoundaryForm))
def test_every_boundary_form_gives_a_mask_of_the_work_shape(scene, fixed_scores, form):
    analysis, _ = scene
    mask = analysis.cut(fixed_scores, form, 0.5)
    assert mask.shape == analysis.shape
    assert mask.dtype == bool


def test_without_a_boundary_penalty_the_cut_selects_the_positive_pixels(scene, fixed_scores):
    analysis, _ = scene
    assert np.array_equal(analysis.cut(fixed_scores, strength=0.0), fixed_scores > 0)


def test_graph_cut_recovers_a_flat_body(scene, fixed_scores):
    analysis, body = scene
    assert foreground_iou(analysis.finish(analysis.graph_cut(fixed_scores)), body) > 0.9


def test_band_cut_keeps_the_pixels_far_from_the_cut_boundary(scene, fixed_scores):
    analysis, _ = scene
    mask = analysis.cut(fixed_scores)
    refined = analysis.band_cut(fixed_scores, mask)
    assert refined.shape == mask.shape
    assert refined.dtype == bool
    far = ~boundary_band(mask, analysis.settings.band.radius)
    assert np.array_equal(refined[far], mask[far])
    assert np.array_equal(analysis.band_cut(fixed_scores), refined)


def test_refined_extraction_recovers_a_flat_body(scene):
    analysis, body = scene
    alpha = analysis.extract(ELLIPSE, refine=True)
    assert alpha.shape == body.shape
    assert foreground_iou(alpha, body) > 0.9
    assert np.array_equal(alpha, analysis.extract(ELLIPSE, refine=True))


def test_individual_cue_methods_match_the_combined_cue_maps(scene):
    analysis, _ = scene
    regions = analysis.regions(ELLIPSE)
    maps = analysis.cue_maps(ELLIPSE, regions)
    assert np.array_equal(maps.prior, analysis.prior_map(ELLIPSE))
    assert np.array_equal(maps.color, analysis.color_map(regions))
    assert np.array_equal(maps.texture, analysis.texture_map(regions))
    assert np.array_equal(maps.patch, analysis.patch_map(regions))


def test_reusing_node_scores_gives_the_same_cut(scene, fixed_scores):
    analysis, _ = scene
    node_scores = analysis.node_scores(fixed_scores)
    for form in BoundaryForm:
        assert np.array_equal(analysis.cut_node_scores(node_scores, form, 3.0), analysis.cut(fixed_scores, form, 3.0))
