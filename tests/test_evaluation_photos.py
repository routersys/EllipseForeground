import numpy as np
import pytest
from PIL import Image

from ellipse_foreground import Settings
from ellipse_foreground.evaluation.__main__ import main, parse, select_group
from ellipse_foreground.evaluation.photos import PHOTO_WIDTH, load_photo, photo_pairs, photo_scenes
from ellipse_foreground.evaluation.runner import GroupSpec, photo_group, run_group
from ellipse_foreground.evaluation.scene import scene_from_composite
from ellipse_foreground.settings import WorkSettings
from tests.helpers import noisy_scene

SIZES = {"b": (2000, 1500), "a": (301, 203), "c": (1500, 2500)}


def write_photos(root):
    (root / "image").mkdir(parents=True)
    (root / "matte").mkdir()
    rng = np.random.default_rng(1)
    for stem, (width, height) in SIZES.items():
        pixels = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "image" / f"{stem}.jpg")
        matte = np.zeros((height, width), dtype=np.uint8)
        matte[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4] = 255
        Image.fromarray(matte).save(root / "matte" / f"{stem}.png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "image" / "lonely.jpg")
    (root / "image" / "notes.txt").write_text("画像ではないファイル", encoding="utf-8")


def test_pairs_keep_sorted_stems_that_have_both_files(tmp_path):
    write_photos(tmp_path)
    assert [(image.stem, matte.stem) for image, matte in photo_pairs(tmp_path)] == [("a", "a"), ("b", "b"), ("c", "c")]


def test_large_photos_are_reduced_to_the_evaluation_width(tmp_path):
    write_photos(tmp_path)
    composite, alpha = load_photo(tmp_path / "image" / "b.jpg", tmp_path / "matte" / "b.png")
    assert composite.shape == (960, PHOTO_WIDTH, 3)
    assert composite.dtype == np.uint8
    assert alpha.shape == composite.shape[:2]
    assert alpha.dtype == np.float32
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_small_photos_are_not_enlarged_and_keep_even_sizes(tmp_path):
    write_photos(tmp_path)
    composite, alpha = load_photo(tmp_path / "image" / "a.jpg", tmp_path / "matte" / "a.png")
    assert composite.shape[1] == 300
    assert composite.shape[0] % 2 == 0
    assert alpha.shape == composite.shape[:2]


def test_scenes_are_named_without_revealing_files(tmp_path):
    write_photos(tmp_path)
    scenes = list(photo_scenes(tmp_path))
    assert [scene.name for scene in scenes] == ["photo000", "photo001", "photo002"]
    assert [scene.index for scene in scenes] == [0, 1, 2]
    assert all(scene.group == "photo" for scene in scenes)
    assert scenes[1].image.shape == (480, PHOTO_WIDTH // 2, 3)
    assert scenes[1].texture_source.shape == (960, PHOTO_WIDTH, 3)
    assert scenes[1].alpha.shape == scenes[1].image.shape[:2]


def test_selection_skips_other_photos(tmp_path):
    write_photos(tmp_path)
    assert [scene.index for scene in photo_scenes(tmp_path, selection=range(1, 2))] == [1]


def test_photo_group_uses_two_trials_and_no_resolution_study(tmp_path):
    write_photos(tmp_path)
    spec = photo_group(tmp_path)
    assert spec.name == "photo"
    assert spec.trials == 2
    assert spec.resolution_stride is None
    assert [scene.name for scene in spec.scenes(tmp_path, range(0, 2))] == ["photo000", "photo001"]


def test_command_line_requires_the_photo_directory(tmp_path):
    assert main(["photo", "--results", str(tmp_path)]) == 2
    options = parse(["photo", "--photos", str(tmp_path)])
    spec = select_group(options)
    assert spec is not None
    assert spec.name == "photo"


def test_the_default_experiments_leave_out_the_comparison_only_choice():
    assert parse(["sprite1"]).experiments == "trials,models,resolution"


@pytest.mark.parametrize("experiments, ablation", [(["comparison"], False), (["trials"], True)])
def test_comparison_runs_without_the_ablation_unless_trials_are_requested(tmp_path, experiments, ablation):
    _, source, body = noisy_scene(144, 256, seed=3)
    scene = scene_from_composite("scene0", "tiny", 0, source, body.astype(np.float32))
    spec = GroupSpec("tiny", lambda data, selection: iter([scene]), 2, None)
    settings = Settings(work=WorkSettings(width=64))
    assert run_group(spec, tmp_path, tmp_path / "out", experiments, range(0, 10**6), settings) == 1
    assert (tmp_path / "out" / "comparison" / "tiny.jsonl").exists()
    assert (tmp_path / "out" / "ablation" / "tiny.jsonl").exists() == ablation
