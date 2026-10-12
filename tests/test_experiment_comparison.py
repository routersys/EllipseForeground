import json

import numpy as np
import pytest

from ellipse_foreground import Mix, Settings
from ellipse_foreground.evaluation.experiments.comparison import comparison_record
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite
from ellipse_foreground.settings import WorkSettings
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


def test_context_yields_only_informative_trials(context):
    for item in context.trials(4):
        assert item.regions.inside.any()
        assert context.work_truth[item.regions.inside].any()
        assert (~context.work_truth[item.regions.inside]).any()


def test_comparison_record_has_every_method_and_valid_scores(context, trial):
    record = comparison_record(context, trial)
    assert {"ellipse", "oracle", *(mix.value for mix in Mix)} == set(record["tree"])
    assert {mix.value for mix in Mix} == set(record["graph"])
    for entry in [*record["tree"].values(), *record["graph"].values()]:
        assert 0.0 <= entry["iou"] <= 1.0
        assert entry["mae"] >= 0.0
    assert all(
        0.0 <= entry["raw_iou"] <= 1.0 for key, entry in record["tree"].items() if key not in ("ellipse", "oracle")
    )
    assert len(record["weights"]["adaptive_color"]) == 3
    assert 0.0 < record["occupancy"] <= 1.0
    assert 0.0 <= record["spill"] <= 1.0
    assert set(record["overlaps"]) == set(record["auc"]) == {"color", "texture", "patch"}


def test_comparison_record_is_deterministic_and_serializable(context, trial):
    first, second = comparison_record(context, trial), comparison_record(context, trial)
    assert first == second
    assert json.loads(json.dumps(first)) == first


def test_the_oracle_beats_the_ellipse(context, trial):
    record = comparison_record(context, trial)
    assert record["tree"]["oracle"]["iou"] >= record["tree"]["ellipse"]["iou"]
