from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ellipse_foreground.evaluation.backgrounds import background_paths, load_banner
from ellipse_foreground.evaluation.color_transfer import transfer_colors
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 640
TRANSFER_DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
SPRITE1_DOSE_BACKGROUND = 4
SPRITE1_OTHER_BACKGROUNDS = (0, 2, 5, 7, 9, 12, 14)
SPRITE2_BACKGROUNDS = (4, 1, 3, 6, 8, 10, 11, 13)


def conditions(group: str) -> list[tuple[str, int, float]]:
    if group == "sprite1":
        doses = [(f"dose{round(strength * 100)}", SPRITE1_DOSE_BACKGROUND, strength) for strength in TRANSFER_DOSES]
        return doses + [(f"bg{index}", index, 1.0) for index in SPRITE1_OTHER_BACKGROUNDS]
    return [(f"bg{index}", index, 1.0) for index in SPRITE2_BACKGROUNDS]


def own_sources(data_root: Path, group: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(data_root / "own" / f"{group}.npz") as stored:
        return stored["alpha"], stored["foreground"]


def own_reconstruction(data_root: Path, group: str) -> tuple[float, float]:
    with np.load(data_root / "own" / f"{group}.npz") as stored:
        return float(stored["reconstruction_mean"]), float(stored["reconstruction_max"])


def compose(alpha: np.ndarray, foreground: np.ndarray, background: np.ndarray, strength: float) -> np.ndarray:
    transferred = transfer_colors(foreground, alpha, background, strength)
    composite = alpha[..., None] * transferred + (1.0 - alpha[..., None]) * background
    return np.clip(np.rint(composite * 255.0), 0, 255).astype(np.uint8)


def own_scenes(data_root: Path, group: str) -> Iterator[Scene]:
    alpha, foreground = own_sources(data_root, group)
    backgrounds = background_paths(data_root)
    for index, (name, background_index, strength) in enumerate(conditions(group)):
        background = load_banner(backgrounds[background_index], CANVAS_WIDTH, CANVAS_HEIGHT)
        composite = compose(alpha, foreground, background, strength)
        yield scene_from_composite(f"{group}-{name}", group, index, composite, alpha)
