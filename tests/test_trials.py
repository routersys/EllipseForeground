import numpy as np
import pytest

from ellipse_foreground.evaluation.scene import scene_from_composite
from ellipse_foreground.evaluation.trials import (
    ERROR_MODELS,
    MAIN_MODEL,
    draw_ellipse,
    draw_trials,
    is_informative,
)
from ellipse_foreground.geometry import Ellipse

PROTOTYPE = Ellipse(100.0, 80.0, 30.0, 50.0)


def scene_with_index(index: int):
    alpha = np.zeros((160, 200), dtype=np.float32)
    alpha[30:130, 70:130] = 1.0
    return scene_from_composite("scene", "group", index, np.zeros((160, 200, 3), dtype=np.uint8), alpha)


def test_error_models_follow_the_paper_definitions():
    assert (MAIN_MODEL.scale_low, MAIN_MODEL.scale_high, MAIN_MODEL.shift) == (0.8, 1.6, 0.12)
    assert (ERROR_MODELS["T"].scale_low, ERROR_MODELS["T"].scale_high, ERROR_MODELS["T"].shift) == (0.9, 1.15, 0.05)
    assert (ERROR_MODELS["L"].scale_low, ERROR_MODELS["L"].scale_high, ERROR_MODELS["L"].shift) == (0.7, 2.0, 0.20)
    assert ERROR_MODELS["S"].log_uniform
    assert not ERROR_MODELS["M"].log_uniform


@pytest.mark.parametrize("name", ["M", "T", "L", "S"])
def test_drawn_ellipses_stay_inside_the_model_ranges(name):
    model = ERROR_MODELS[name]
    rng = np.random.default_rng(0)
    for _ in range(300):
        ellipse = draw_ellipse(rng, PROTOTYPE, model)
        assert model.scale_low <= ellipse.radius_x / PROTOTYPE.radius_x <= model.scale_high
        assert model.scale_low <= ellipse.radius_y / PROTOTYPE.radius_y <= model.scale_high
        assert abs(ellipse.center_x - PROTOTYPE.center_x) <= 2 * model.shift * PROTOTYPE.radius_x + 1e-9
        assert abs(ellipse.center_y - PROTOTYPE.center_y) <= 2 * model.shift * PROTOTYPE.radius_y + 1e-9


def test_log_uniform_scales_are_symmetric_in_the_logarithm():
    model = ERROR_MODELS["S"]
    rng = np.random.default_rng(1)
    logs = np.log([draw_ellipse(rng, PROTOTYPE, model).radius_x / PROTOTYPE.radius_x for _ in range(4000)])
    assert abs(logs.mean() - 0.5 * (np.log(model.scale_low) + np.log(model.scale_high))) < 0.01


def test_trials_depend_only_on_the_scene_index():
    first = draw_trials(scene_with_index(7), 4)
    second = draw_trials(scene_with_index(7), 4)
    other = draw_trials(scene_with_index(8), 4)
    assert first == second
    assert first != other


def test_trials_are_a_prefix_of_longer_draws():
    assert draw_trials(scene_with_index(3), 2) == draw_trials(scene_with_index(3), 5)[:2]


def test_trials_start_from_the_scene_prototype():
    scene = scene_with_index(0)
    ellipse = draw_trials(scene, 1, ERROR_MODELS["T"])[0]
    assert abs(ellipse.center_x - scene.prototype.center_x) <= 0.1 * scene.prototype.radius_x + 1e-9


def test_informative_trials_need_foreground_and_background_inside_the_ellipse():
    truth = np.zeros((40, 40), dtype=bool)
    truth[10:30, 10:30] = True
    assert is_informative(Ellipse(20, 20, 15, 15).inside((40, 40)), truth)
    assert not is_informative(Ellipse(20, 20, 4, 4).inside((40, 40)), truth)
    assert not is_informative(Ellipse(100, 100, 4, 4).inside((40, 40)), truth)
