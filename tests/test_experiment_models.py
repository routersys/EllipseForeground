import numpy as np
import pytest

from ellipse_foreground import Settings
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.experiments.models import model_records
from ellipse_foreground.evaluation.experiments.resolution import resolution_records
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite
from ellipse_foreground.evaluation.trials import ERROR_MODELS
from ellipse_foreground.settings import WorkSettings
from tests.helpers import noisy_scene

SETTINGS = Settings(work=WorkSettings(width=64))


def make_scene(index: int = 0) -> Scene:
    _, source, body = noisy_scene(144, 256, seed=3)
    return scene_from_composite(f"scene{index}", "group", index, source, body.astype(np.float32))


@pytest.fixture(scope="module")
def context() -> SceneContext:
    return SceneContext(make_scene(), SETTINGS)


def test_model_records_cover_the_four_methods(context):
    records = model_records(context, ERROR_MODELS["S"], 3)
    assert records
    for record in records:
        assert record["model"] == "S"
        assert set(record["tree"]) == {"ellipse", "prior", "tex", "oracle"}


def test_resolution_records_report_the_requested_width():
    records = resolution_records(make_scene(), 48, 3)
    assert records
    assert {record["width"] for record in records} == {48}
    assert all(0.0 <= record["iou"] <= 1.0 and 0.0 <= record["raw_iou"] <= 1.0 for record in records)
