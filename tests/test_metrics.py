import numpy as np
import pytest

from ellipse_foreground.evaluation.metrics import Scorer, mask_iou
from ellipse_foreground.evaluation.scene import Scene


def scene() -> Scene:
    rng = np.random.default_rng(0)
    image = rng.random((60, 80, 3)).astype(np.float32)
    alpha = np.zeros((60, 80), dtype=np.float32)
    alpha[15:45, 20:60] = 1.0
    return Scene("s", "g", 0, image, alpha, np.zeros((120, 160, 3), dtype=np.uint8))


def test_mask_iou_basic_cases():
    first = np.zeros((4, 4), dtype=bool)
    first[:2] = True
    second = np.zeros((4, 4), dtype=bool)
    second[1:3] = True
    assert mask_iou(first, first) == 1.0
    assert mask_iou(first, second) == pytest.approx(1 / 3)
    assert mask_iou(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool)) == 0.0


def test_a_perfect_alpha_has_full_iou_and_no_render_error():
    subject = scene()
    score = Scorer(subject).score(subject.alpha)
    assert score.iou == 1.0
    assert score.mae == pytest.approx(0.0, abs=1e-9)


def test_a_wrong_alpha_scores_worse():
    subject = scene()
    shifted = np.roll(subject.alpha, 6, axis=1)
    score = Scorer(subject).score(shifted)
    assert score.iou < 1.0
    assert score.mae > 0.0


def test_render_error_is_larger_for_a_larger_mistake():
    subject = scene()
    scorer = Scorer(subject)
    assert scorer.render_error(np.roll(subject.alpha, 12, axis=1)) > scorer.render_error(
        np.roll(subject.alpha, 3, axis=1)
    )


def test_render_error_is_measured_on_a_zero_to_255_scale():
    subject = scene()
    assert Scorer(subject).render_error(np.zeros_like(subject.alpha)) > 1.0


def test_truth_render_is_cached():
    scorer = Scorer(scene())
    assert scorer.truth_render is scorer.truth_render
