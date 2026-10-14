import json

import numpy as np
import pytest

from ellipse_foreground import Settings
from ellipse_foreground.evaluation.experiments.adaptation import adaptation_record
from ellipse_foreground.evaluation.experiments.context import SceneContext
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


def test_the_record_scores_the_tree_the_refit_and_their_bands(context, trial):
    record = adaptation_record(context, trial)
    for name in ("tree", "refit", "band", "refit_band"):
        assert 0.0 <= record[name]["iou"] <= 1.0
        assert 0.0 <= record[name]["raw_iou"] <= 1.0
        assert record[name]["mae"] >= 0.0


def test_the_stability_is_a_ratio(context, trial):
    assert 0.0 <= adaptation_record(context, trial)["stability"] <= 1.0


def test_the_tree_score_matches_the_plain_extraction(context, trial):
    record = adaptation_record(context, trial)
    mask = context.analysis.solve(trial.ellipse)[1]
    assert record["tree"]["iou"] == context.finished_score(mask)["iou"]


def test_the_record_is_deterministic_and_serializable(context, trial):
    first, second = adaptation_record(context, trial), adaptation_record(context, trial)
    assert first == second
    assert json.loads(json.dumps(first)) == first


def test_the_runner_writes_adaptation_records(tmp_path):
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene(0)]), 2, None)
    run_group(spec, tmp_path, tmp_path / "out", ["adaptation"], range(0, 10**6), SETTINGS)
    records = read_records(tmp_path / "out" / "adaptation", "tiny")
    assert records
    assert not (tmp_path / "out" / "ablation").exists()
