import numpy as np
import pytest

from ellipse_foreground.evaluation.baselines import (
    BASELINE_DIRECTORY,
    baseline_records,
    ellipse_box,
    grabcut_alpha,
    main,
    parse,
    run_baselines,
    sam_alphas,
)
from ellipse_foreground.evaluation.results import read_records
from ellipse_foreground.evaluation.runner import GroupSpec
from ellipse_foreground.evaluation.scene import scene_from_composite
from ellipse_foreground.geometry import Ellipse
from tests.helpers import noisy_scene


class FakePredictor:
    def __init__(self):
        self.image = None
        self.boxes = []

    def set_image(self, image):
        self.image = image

    def predict(self, *, box, multimask_output):
        self.boxes.append((tuple(box), multimask_output))
        height, width = self.image.shape[:2]
        masks = np.zeros((3 if multimask_output else 1, height, width), dtype=bool)
        masks[:, height // 4 : height * 3 // 4, width // 4 : width * 3 // 4] = True
        qualities = np.array([0.1, 0.9, 0.5][: len(masks)])
        return masks, qualities, np.zeros(1)


def make_scene():
    _, source, body = noisy_scene(144, 256, seed=3)
    return scene_from_composite("scene0", "tiny", 0, source, body.astype(np.float32))


def test_the_box_is_the_circumscribed_rectangle_clipped_to_the_image():
    assert ellipse_box(Ellipse(50.0, 40.0, 20.0, 10.0), (100, 200)) == (30.0, 30.0, 70.0, 50.0)
    assert ellipse_box(Ellipse(10.0, 95.0, 30.0, 20.0), (100, 200)) == (0.0, 75.0, 40.0, 100.0)


def test_sam_boxes_are_scaled_to_the_full_resolution_image_and_masks_are_halved():
    predictor = FakePredictor()
    composite = np.zeros((144, 256, 3), dtype=np.uint8)
    outputs = sam_alphas(predictor, composite, [(10.0, 20.0, 30.0, 40.0)], 2.0)
    assert predictor.image is composite
    assert predictor.boxes == [((20.0, 40.0, 60.0, 80.0), False), ((20.0, 40.0, 60.0, 80.0), True)]
    assert outputs[0]["sam_single"].shape == (72, 128)
    assert outputs[0]["sam_best"].shape == (72, 128)


def test_records_hold_one_score_per_method_for_each_informative_trial():
    pytest.importorskip("cv2")
    records = baseline_records(make_scene(), 3, FakePredictor())
    assert records
    for record in records:
        assert set(record) == {"scene", "name", "trial", "grabcut_converged", "grabcut", "sam_single", "sam_best"}
        for name in ("grabcut", "sam_single", "sam_best"):
            assert 0.0 <= record[name]["iou"] <= 1.0
            assert record[name]["mae"] >= 0.0


def test_without_a_predictor_only_grabcut_is_recorded():
    pytest.importorskip("cv2")
    records = baseline_records(make_scene(), 2, None)
    assert all("sam_single" not in record for record in records)
    assert all("grabcut" in record for record in records)


def test_run_baselines_writes_records_and_resumes(tmp_path):
    pytest.importorskip("cv2")
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene()]), 2, None)
    assert run_baselines(spec, tmp_path, tmp_path / "out", range(0, 10**6), None) == 1
    assert {record["scene"] for record in read_records(tmp_path / "out" / BASELINE_DIRECTORY, "tiny")} == {0}
    assert run_baselines(spec, tmp_path, tmp_path / "out", range(0, 10**6), None) == 0


def test_the_command_line_needs_the_photo_directory(tmp_path):
    assert main(["photo", "--results", str(tmp_path)]) == 2
    options = parse(["public0", "--checkpoint", "weights.pth"])
    assert options.model_type == "vit_b"


def test_grabcut_falls_back_to_the_rectangle_when_it_cannot_start():
    pytest.importorskip("cv2")
    image = np.random.default_rng(0).random((40, 60, 3)).astype(np.float32)
    alpha, converged = grabcut_alpha(image, (0.0, 0.0, 60.0, 40.0), 1)
    assert not converged
    assert alpha.min() == 1.0
