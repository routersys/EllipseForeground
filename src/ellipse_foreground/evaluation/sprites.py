import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ellipse_foreground.evaluation.backgrounds import background_paths, load_covering
from ellipse_foreground.evaluation.color_transfer import transfer_colors
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720
PUBLIC_SEED = 79
SCENES_PER_SPRITE = 6
_OPAQUE_THRESHOLD = 128


@dataclass(frozen=True)
class Placement:
    background: int
    height_fraction: float
    center_fraction: float
    bottom_margin: float


def draw_placement(rng: np.random.Generator, background_count: int) -> Placement:
    background = int(rng.integers(0, background_count))
    height_fraction = float(rng.uniform(0.55, 0.92))
    center_fraction = float(rng.uniform(0.32, 0.68))
    bottom_margin = float(rng.uniform(0.0, 0.08))
    return Placement(background, height_fraction, center_fraction, bottom_margin)


def trimmed_sprite(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    rows, columns = np.nonzero(np.asarray(image)[..., 3] > _OPAQUE_THRESHOLD)
    return image.crop((columns.min(), rows.min(), columns.max() + 1, rows.max() + 1))


def place_sprite(
    sprite: Image.Image, placement: Placement, background: np.ndarray, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    height = round(placement.height_fraction * CANVAS_HEIGHT)
    width = max(2, round(sprite.width * height / sprite.height))
    resized = np.asarray(sprite.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    left = max(0, min(CANVAS_WIDTH - width, round(placement.center_fraction * CANVAS_WIDTH - width / 2)))
    top = max(0, round(CANVAS_HEIGHT - height - placement.bottom_margin * CANVAS_HEIGHT))
    bottom, right = min(CANVAS_HEIGHT, top + height), min(CANVAS_WIDTH, left + width)
    alpha = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.float32)
    alpha[top:bottom, left:right] = resized[: bottom - top, : right - left, 3]
    color = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.float32)
    transferred = transfer_colors(resized[..., :3], resized[..., 3], background, strength)
    color[top:bottom, left:right] = transferred[: bottom - top, : right - left]
    return color, alpha


def sprite_files(data_root: Path) -> list[tuple[str, Path]]:
    manifest = json.loads((data_root / "sprites" / "manifest.json").read_text(encoding="utf-8"))
    return [(entry["id"], data_root / "sprites" / entry["file"]) for entry in manifest]


def quantize_unit(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)


def public_scenes(
    data_root: Path, strength: float, seed: int = PUBLIC_SEED, per_sprite: int = SCENES_PER_SPRITE
) -> Iterator[Scene]:
    backgrounds = background_paths(data_root)
    rng = np.random.default_rng(seed)
    group = f"public{round(strength * 100)}"
    index = 0
    for identifier, path in sprite_files(data_root):
        sprite = trimmed_sprite(path)
        for variant in range(per_sprite):
            placement = replace(draw_placement(rng, len(backgrounds)), bottom_margin=0.0)
            background = load_covering(backgrounds[placement.background], CANVAS_WIDTH, CANVAS_HEIGHT)
            color, alpha = place_sprite(sprite, placement, background, strength)
            composite = quantize_unit(alpha[..., None] * color + (1 - alpha[..., None]) * background)
            name = f"{identifier}_bg{placement.background}_{variant}"
            yield scene_from_composite(name, group, index, composite, quantize_unit(alpha).astype(np.float32) / 255.0)
            index += 1
