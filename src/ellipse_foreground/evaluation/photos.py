from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ellipse_foreground.evaluation.listing import IMAGE_SUFFIXES
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite

PHOTO_WIDTH = 1280


def photo_pairs(directory: Path) -> list[tuple[Path, Path]]:
    def listed(name: str) -> dict[str, Path]:
        return {path.stem: path for path in (directory / name).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}

    images, mattes = listed("image"), listed("matte")
    return [(images[stem], mattes[stem]) for stem in sorted(images) if stem in mattes]


def load_photo(image_path: Path, matte_path: Path, width: int = PHOTO_WIDTH) -> tuple[np.ndarray, np.ndarray]:
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path).convert("RGB")
    matte = Image.open(matte_path).convert("L")
    target_width = min(width, image.width) // 2 * 2
    target_height = max(2, round(image.height * target_width / image.width / 2)) * 2
    size = (target_width, target_height)
    composite = np.asarray(image.resize(size, Image.Resampling.BOX), dtype=np.uint8)
    alpha = np.asarray(matte.resize(size, Image.Resampling.BOX), dtype=np.float32) / 255.0
    return composite, alpha


def photo_scenes(directory: Path, group: str = "photo", selection: range | None = None) -> Iterator[Scene]:
    for index, (image_path, matte_path) in enumerate(photo_pairs(directory)):
        if selection is not None and index not in selection:
            continue
        composite, alpha = load_photo(image_path, matte_path)
        yield scene_from_composite(f"photo{index:03d}", group, index, composite, alpha)
