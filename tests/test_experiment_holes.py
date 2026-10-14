import json

import numpy as np
import pytest

from ellipse_foreground import Settings
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.experiments.holes import holes_record
from ellipse_foreground.evaluation.results import read_records
from ellipse_foreground.evaluation.runner import GroupSpec, run_group
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


def test_the_record_scores_the_tree_the_fill_and_their_bands(context, trial):
    record = holes_record(context, trial)
    for name in ("tree", "fill", "band", "fill_band"):
        assert 0.0 <= record[name]["iou"] <= 1.0
        assert 0.0 <= record[name]["raw_iou"] <= 1.0
        assert record[name]["mae"] >= 0.0
    assert record["hole_share"] >= 0.0


def test_a_mask_without_holes_is_unchanged_by_the_fill(context, trial):
    record = holes_record(context, trial)
    if record["hole_share"] == 0.0:
        assert record["fill"] == record["tree"]


def test_the_record_is_deterministic_and_serializable(context, trial):
    first, second = holes_record(context, trial), holes_record(context, trial)
    assert first == second
    assert json.loads(json.dumps(first)) == first


def test_the_runner_writes_holes_records(tmp_path):
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene(0)]), 1, None)
    run_group(spec, tmp_path, tmp_path / "out", ["holes"], range(0, 10**6), SETTINGS)
    assert read_records(tmp_path / "out" / "holes", "tiny")
