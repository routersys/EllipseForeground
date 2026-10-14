import json

import numpy as np
import pytest

from ellipse_foreground import Settings
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.experiments.refinement import BAND_RADII, refinement_record
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


def test_the_record_has_a_score_for_every_radius(context, trial):
    record = refinement_record(context, trial)
    assert set(record["band"]) == {str(radius) for radius in BAND_RADII}
    for entry in [record["tree"], *record["band"].values()]:
        assert 0.0 <= entry["iou"] <= 1.0
        assert 0.0 <= entry["raw_iou"] <= 1.0
        assert entry["mae"] >= 0.0


def test_a_wider_band_covers_a_larger_share_of_the_image(context, trial):
    shares = [refinement_record(context, trial)["band"][str(radius)]["share"] for radius in BAND_RADII]
    assert shares == sorted(shares)
    assert 0.0 < shares[0] < shares[-1] < 1.0


def test_the_record_is_deterministic_and_serializable(context, trial):
    first, second = refinement_record(context, trial), refinement_record(context, trial)
    assert first == second
    assert json.loads(json.dumps(first)) == first


def test_the_runner_writes_refinement_records(tmp_path):
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene(0)]), 2, None)
    run_group(spec, tmp_path, tmp_path / "out", ["refinement"], range(0, 10**6), SETTINGS)
    records = read_records(tmp_path / "out" / "refinement", "tiny")
    assert records
    assert not (tmp_path / "out" / "ablation").exists()
