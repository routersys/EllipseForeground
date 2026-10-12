import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ellipse_foreground.evaluation.own import CANVAS_HEIGHT, CANVAS_WIDTH
from ellipse_foreground.evaluation.recovery import fill_foreground, read_rgb, recover_alpha

REPOSITORY = Path(__file__).resolve().parents[3]
SPRITE2_ALIGNMENT = (0.804, 125.5, 57.0)
EXPORT_GROUPS = ("sprite1", "sprite2")


@dataclass(frozen=True)
class Sources:
    alpha: np.ndarray
    foreground: np.ndarray
    reconstruction_mean: float
    reconstruction_max: float


def sprite1_color(folder: Path, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    original = read_rgb(folder / "original_colors_composite.png") / 255.0
    return np.clip(original - (1.0 - alpha)[..., None] * background / 255.0, 0.0, 1.0)


def sprite2_color(folder: Path) -> np.ndarray:
    scale, shift_x, shift_y = SPRITE2_ALIGNMENT
    original = Image.open(folder / "original_colors_foreground.png").convert("RGB")
    aligned = original.transform(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        Image.Transform.AFFINE,
        (1.0 / scale, 0, -shift_x / scale, 0, 1.0 / scale, -shift_y / scale),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(aligned, dtype=np.float64) / 255.0


def recover_sources(group: str, folder: Path) -> Sources:
    composite = read_rgb(folder / "composite.png")
    foreground_only = read_rgb(folder / "foreground_only.png")
    background = read_rgb(folder / "background_only.png")
    alpha = recover_alpha(composite, foreground_only, background)
    color = sprite1_color(folder, alpha, background) if group == "sprite1" else sprite2_color(folder)
    reconstructed = foreground_only + (1.0 - alpha)[..., None] * background
    error = np.abs(reconstructed - composite)
    return Sources(alpha.astype(np.float32), fill_foreground(color, alpha), float(error.mean()), float(error.max()))


def write_sources(group: str, folder: Path, data_root: Path) -> Path:
    sources = recover_sources(group, folder)
    foreground = np.where(sources.alpha[..., None] > 0, sources.foreground, 0.0).astype(np.float32)
    destination = data_root / "own" / f"{group}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        alpha=sources.alpha,
        foreground=foreground,
        reconstruction_mean=sources.reconstruction_mean,
        reconstruction_max=sources.reconstruction_max,
    )
    return destination


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m ellipse_foreground.evaluation.export")
    parser.add_argument("group", choices=EXPORT_GROUPS)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--data", type=Path, default=REPOSITORY / "data")
    options = parser.parse_args(arguments)
    print(write_sources(options.group, options.folder, options.data))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
