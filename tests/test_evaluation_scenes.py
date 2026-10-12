import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ellipse_foreground.evaluation.own import compose, conditions
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite
from ellipse_foreground.evaluation.sprites import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    Placement,
    draw_placement,
    place_sprite,
    public_scenes,
    trimmed_sprite,
)
from ellipse_foreground.settings import WorkSettings


def write_data_root(root: Path) -> Path:
    (root / "sprites").mkdir(parents=True)
    (root / "backgrounds").mkdir()
    rng = np.random.default_rng(0)
    sprites = []
    for number in range(2):
        pixels = np.zeros((90, 50, 4), dtype=np.uint8)
        pixels[10:80, 8:42, :3] = rng.integers(0, 255, 3)
        pixels[10:80, 8:42, 3] = 255
        Image.fromarray(pixels).save(root / "sprites" / f"s{number}.png")
        sprites.append({"id": f"s{number}", "file": f"s{number}.png"})
    backgrounds = []
    for number in range(3):
        Image.fromarray(rng.integers(0, 255, (108, 192, 3), dtype=np.uint8)).save(
            root / "backgrounds" / f"bg{number}.jpg"
        )
        backgrounds.append({"id": number, "file": f"bg{number}.jpg"})
    (root / "sprites" / "manifest.json").write_text(json.dumps(sprites), encoding="utf-8")
    (root / "backgrounds" / "manifest.json").write_text(json.dumps(backgrounds), encoding="utf-8")
    return root


def test_draw_placement_is_reproducible_and_within_range():
    first = [draw_placement(np.random.default_rng(5), 16) for _ in range(3)]
    second = [draw_placement(np.random.default_rng(5), 16) for _ in range(3)]
    assert first == second
    for placement in first:
        assert 0 <= placement.background < 16
        assert 0.55 <= placement.height_fraction <= 0.92
        assert 0.32 <= placement.center_fraction <= 0.68
        assert 0.0 <= placement.bottom_margin <= 0.08


def test_trimmed_sprite_removes_transparent_borders(tmp_path):
    pixels = np.zeros((30, 40, 4), dtype=np.uint8)
    pixels[5:20, 10:25] = 255
    Image.fromarray(pixels).save(tmp_path / "sprite.png")
    assert trimmed_sprite(tmp_path / "sprite.png").size == (15, 15)


def test_place_sprite_stands_on_the_bottom_edge_when_the_margin_is_zero():
    sprite = Image.fromarray(np.full((40, 20, 4), 255, dtype=np.uint8))
    background = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.float32)
    color, alpha = place_sprite(sprite, Placement(0, 0.5, 0.5, 0.0), background, 0.0)
    rows, columns = np.nonzero(alpha > 0.5)
    assert rows.max() == CANVAS_HEIGHT - 1
    assert rows.max() - rows.min() + 1 == round(0.5 * CANVAS_HEIGHT)
    assert abs((columns.min() + columns.max()) / 2 - CANVAS_WIDTH / 2) <= 1
    assert color[alpha > 0.5].min() == 1.0


def test_public_scenes_are_deterministic_and_complete(tmp_path):
    root = write_data_root(tmp_path)
    first = list(public_scenes(root, 0.0, per_sprite=2))
    second = list(public_scenes(root, 0.0, per_sprite=2))
    assert len(first) == 4
    assert [scene.index for scene in first] == [0, 1, 2, 3]
    for a, b in zip(first, second, strict=True):
        assert a.name == b.name
        assert np.array_equal(a.texture_source, b.texture_source)
    assert first[0].texture_source.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert first[0].image.shape == (CANVAS_HEIGHT // 2, CANVAS_WIDTH // 2, 3)
    assert first[0].group == "public0"


def test_a_selection_builds_only_the_chosen_scenes_with_unchanged_content(tmp_path):
    root = write_data_root(tmp_path)
    everything = list(public_scenes(root, 0.0, per_sprite=2))
    chosen = list(public_scenes(root, 0.0, per_sprite=2, selection=range(2, 4)))
    assert [scene.index for scene in chosen] == [2, 3]
    for scene in chosen:
        assert scene.name == everything[scene.index].name
        assert np.array_equal(scene.texture_source, everything[scene.index].texture_source)


def test_public_scenes_keep_placement_across_transfer_strengths(tmp_path):
    root = write_data_root(tmp_path)
    plain = list(public_scenes(root, 0.0, per_sprite=2))
    transferred = list(public_scenes(root, 1.0, per_sprite=2))
    for a, b in zip(plain, transferred, strict=True):
        assert a.name == b.name
        assert np.array_equal(a.alpha, b.alpha)
        assert not np.array_equal(a.texture_source, b.texture_source)


def test_public_alpha_is_quantized_to_eight_bits(tmp_path):
    scene = next(public_scenes(write_data_root(tmp_path), 0.0, per_sprite=1))
    doubled = scene.alpha.repeat(2, axis=0).repeat(2, axis=1)
    assert np.allclose(doubled * 255.0 * 4, np.rint(doubled * 255.0 * 4), atol=1e-3)
    assert scene.alpha.dtype == np.float32


def test_scene_from_composite_halves_the_resolution():
    composite = np.random.default_rng(1).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    scene = scene_from_composite("a", "g", 3, composite, np.ones((40, 60), dtype=np.float32))
    assert scene.image.shape == (20, 30, 3)
    assert scene.alpha.shape == (20, 30)
    assert scene.texture_source is composite
    assert scene.index == 3


def test_prototype_is_the_ellipse_inscribed_in_the_foreground_box():
    alpha = np.zeros((40, 60), dtype=np.float32)
    alpha[10:30, 20:50] = 1.0
    scene = scene_from_composite("a", "g", 0, np.zeros((40, 60, 3), dtype=np.uint8), alpha)
    assert scene.prototype.radius_x == 7.5
    assert scene.prototype.radius_y == 5.0


def test_work_alpha_follows_the_work_shape():
    scene = scene_from_composite(
        "a", "g", 0, np.zeros((72, 128, 3), dtype=np.uint8), np.ones((72, 128), dtype=np.float32)
    )
    assert scene.work_alpha(WorkSettings(width=32)).shape == (18, 32)


def test_condition_lists_match_the_evaluation_design():
    sprite1 = conditions("sprite1")
    assert [name for name, _, _ in sprite1[:5]] == ["dose0", "dose25", "dose50", "dose75", "dose100"]
    assert len(sprite1) == 12
    assert {strength for _, _, strength in sprite1[5:]} == {1.0}
    assert len(conditions("sprite2")) == 8


def test_compose_without_transfer_blends_by_alpha():
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[:, :2] = 1.0
    foreground = np.full((4, 4, 3), 0.8, dtype=np.float32)
    background = np.full((4, 4, 3), 0.2, dtype=np.float32)
    composite = compose(alpha, foreground, background, 0.0)
    assert composite.dtype == np.uint8
    assert composite[0, 0].tolist() == [204, 204, 204]
    assert composite[0, 3].tolist() == [51, 51, 51]


@pytest.mark.parametrize("group", ["sprite1", "sprite2"])
def test_scene_dataclass_is_immutable(group):
    scene = Scene("n", group, 0, np.zeros((2, 2, 3)), np.zeros((2, 2)), np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(AttributeError):
        scene.name = "other"
