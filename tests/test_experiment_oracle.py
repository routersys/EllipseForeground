import json

import numpy as np
import pytest

from ellipse_foreground import Settings
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.experiments.oracle import oracle_record
from ellipse_foreground.evaluation.results import read_records
from ellipse_foreground.evaluation.runner import GroupSpec, run_group
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite
from ellipse_foreground.settings import WorkSettings
from tests.helpers import noisy_scene

SETTINGS = Settings(work=WorkSettings(width=64))
VARIANTS = (
    "tree",
    "ellipse",
    "prior",
    "samples",
    "both",
    "samples_ct",
    "ring_far",
    "ring_near",
    "core_same",
    "core_all",
    "unmix_truth",
    "unmix_tree",
    "unmix_population",
)


def make_scene(index: int = 0) -> Scene:
    _, source, body = noisy_scene(144, 256, seed=3)
    return scene_from_composite(f"scene{index}", "group", index, source, body.astype(np.float32))


@pytest.fixture(scope="module")
def context() -> SceneContext:
    return SceneContext(make_scene(), SETTINGS)


@pytest.fixture(scope="module")
def record(context):
    return oracle_record(context, next(context.trials(2)))


def test_the_record_scores_every_variant(record):
    for name in VARIANTS:
        assert 0.0 <= record[name]["iou"] <= 1.0
        assert 0.0 <= record[name]["raw_iou"] <= 1.0
        assert record[name]["mae"] >= 0.0


def test_the_purity_is_a_share_of_the_samples(record):
    assert 0.0 <= record["purity"]["core"] <= 1.0
    assert 0.0 <= record["purity"]["ring"] <= 1.0


def test_the_true_samples_do_not_hurt_a_clear_scene(record):
    assert record["samples"]["raw_iou"] >= record["tree"]["raw_iou"] - 0.02


def test_the_record_is_deterministic_and_serializable(context, record):
    again = oracle_record(context, next(context.trials(2)))
    assert again == record
    assert json.loads(json.dumps(record)) == record


def test_the_runner_writes_oracle_records(tmp_path):
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene(0)]), 1, None)
    run_group(spec, tmp_path, tmp_path / "out", ["oracle"], range(0, 10**6), SETTINGS)
    assert read_records(tmp_path / "out" / "oracle", "tiny")
    assert not (tmp_path / "out" / "comparison").exists()
