import numpy as np
import pytest

from ellipse_foreground import CueWeights, Mix, Settings
from ellipse_foreground.evaluation.experiments.ablation import (
    BOUNDARY_EXPONENTS,
    FINISH_STAGES,
    INSIDE_SHRINKS,
    ablation_record,
    finish_variants,
    halve_source,
    reduced_sources,
)
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite
from ellipse_foreground.settings import WorkSettings
from ellipse_foreground.weights import weights_for
from tests.helpers import noisy_scene

SETTINGS = Settings(work=WorkSettings(width=64))


def make_scene(index: int = 0) -> Scene:
    _, source, body = noisy_scene(144, 256, seed=3)
    return scene_from_composite(f"scene{index}", "group", index, source, body.astype(np.float32))


@pytest.fixture(scope="module")
def context() -> SceneContext:
    return SceneContext(make_scene(), SETTINGS)


@pytest.fixture(scope="module")
def trial(context):
    return next(context.trials(2))


def test_boundary_sweep_covers_every_form_and_a_zero_penalty(context, trial):
    boundary = ablation_record(context, trial)["boundary"]
    assert set(boundary) == {"none", *(form.value for form in BOUNDARY_EXPONENTS)}
    for form, exponents in BOUNDARY_EXPONENTS.items():
        assert set(boundary[form.value]) == {str(exponent) for exponent in exponents}


def test_the_zero_penalty_cut_selects_the_positive_scores(context, trial):
    weights = weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, SETTINGS.weights)
    expected = mask_iou(trial.cues.scores(weights) > 0, context.work_truth)
    assert ablation_record(context, trial)["boundary"]["none"]["iou"] == pytest.approx(expected)


def test_ablation_sweeps_have_the_documented_keys(context, trial):
    record = ablation_record(context, trial)
    assert set(record["prior"]) == {str(shrink) for shrink in INSIDE_SHRINKS}
    assert set(record["texture_resolution"]) == {"full", "x640", "x320"}
    assert set(record["graph_cut"]) == {str(exponent) for exponent in range(9)}
    assert set(record["finish"]) == set(FINISH_STAGES)


def test_the_final_finish_variant_is_the_standard_finish(context, trial):
    analysis = context.analysis
    mask = analysis.cut(trial.cues.scores(CueWeights(1.0, 0.25, 0.3)))
    assert np.array_equal(finish_variants(analysis, mask)["final"], analysis.finish(mask))


def test_reduced_sources_halve_twice():
    source = np.random.default_rng(0).integers(0, 255, (80, 120, 3), dtype=np.uint8)
    sources = reduced_sources(source)
    assert sources["full"].shape == (80, 120, 3)
    assert sources["x640"].shape == (40, 60, 3)
    assert sources["x320"].shape == (20, 30, 3)
    assert halve_source(source).dtype == np.uint8
