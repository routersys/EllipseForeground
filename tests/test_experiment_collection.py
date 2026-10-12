import numpy as np
from PIL import Image

from ellipse_foreground.evaluation.__main__ import main, parse, select_group
from ellipse_foreground.evaluation.listing import listed_paths
from ellipse_foreground.evaluation.runner import COLLECTION_SEED, collection_group


def write_images(root, sprites=2, backgrounds=3):
    (root / "sprites").mkdir(parents=True)
    (root / "backgrounds").mkdir()
    rng = np.random.default_rng(0)
    for number in range(sprites):
        pixels = np.zeros((90, 50, 4), dtype=np.uint8)
        pixels[10:80, 8:42, :3] = rng.integers(0, 255, 3)
        pixels[10:80, 8:42, 3] = 255
        Image.fromarray(pixels).save(root / "sprites" / f"character_{number}.png")
    for number in range(backgrounds):
        Image.fromarray(rng.integers(0, 255, (108, 192, 3), dtype=np.uint8)).save(
            root / "backgrounds" / f"b{number}.jpg"
        )
    (root / "backgrounds" / "notes.txt").write_text("画像ではないファイル", encoding="utf-8")


def test_listing_a_directory_keeps_sorted_images_only(tmp_path):
    write_images(tmp_path)
    names = [path.name for path in listed_paths(tmp_path / "backgrounds")]
    assert names == ["b0.jpg", "b1.jpg", "b2.jpg"]


def test_listing_a_text_file_keeps_the_given_order(tmp_path):
    (tmp_path / "paths.txt").write_text("b.png\n\n a.png \n", encoding="utf-8")
    assert [path.name for path in listed_paths(tmp_path / "paths.txt")] == ["b.png", "a.png"]


def test_collection_group_names_scenes_without_revealing_files(tmp_path):
    write_images(tmp_path)
    spec = collection_group(
        1.0, listed_paths(tmp_path / "sprites"), listed_paths(tmp_path / "backgrounds"), grounded=False
    )
    assert spec.name == "collection100"
    assert spec.trials == 2
    assert spec.resolution_stride == 3
    scenes = list(spec.scenes(tmp_path, range(0, 7)))
    assert [scene.index for scene in scenes] == list(range(7))
    assert scenes[0].name.startswith("sprite00_bg")
    assert scenes[6].name.startswith("sprite01_bg")
    assert all(scene.group == "collection100" for scene in scenes)
    assert "character" not in "".join(scene.name for scene in scenes)


def test_selection_skips_other_scenes_without_changing_placements(tmp_path):
    write_images(tmp_path)
    spec = collection_group(0.0, listed_paths(tmp_path / "sprites"), listed_paths(tmp_path / "backgrounds"))
    everything = {scene.index: scene.name for scene in spec.scenes(tmp_path, range(0, 12))}
    chosen = {scene.index: scene.name for scene in spec.scenes(tmp_path, range(3, 5))}
    assert chosen == {index: everything[index] for index in (3, 4)}


def test_command_line_requires_materials_for_collections(tmp_path):
    assert main(["collection0", "--results", str(tmp_path)]) == 2
    options = parse(["collection100", "--sprites", "a", "--backgrounds", "b"])
    assert options.seed == COLLECTION_SEED
    assert not options.grounded


def test_command_line_builds_a_group_from_given_materials(tmp_path):
    write_images(tmp_path)
    options = parse(
        ["collection0", "--sprites", str(tmp_path / "sprites"), "--backgrounds", str(tmp_path / "backgrounds")]
    )
    spec = select_group(options)
    assert spec is not None
    assert spec.name == "collection0"
