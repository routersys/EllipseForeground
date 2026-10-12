from collections.abc import Iterator
from pathlib import Path

from ellipse_foreground.evaluation.own import own_scenes
from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.evaluation.sprites import public_scenes
from paper.results import SUBSETS


def find_scene(data_root: Path, key: str, index: int) -> Scene:
    group = SUBSETS[key].group
    if group.startswith("public"):
        strength = 1.0 if group.endswith("100") else 0.0
        scenes: Iterator[Scene] = public_scenes(data_root, strength, selection=range(index, index + 1))
    else:
        scenes = own_scenes(data_root, group)
    return next(scene for scene in scenes if scene.index == index)
