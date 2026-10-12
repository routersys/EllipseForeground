import json
from pathlib import Path

import numpy as np
from PIL import Image


def background_paths(data_root: Path) -> list[Path]:
    manifest = json.loads((data_root / "backgrounds" / "manifest.json").read_text(encoding="utf-8"))
    return [data_root / "backgrounds" / entry["file"] for entry in sorted(manifest, key=lambda entry: entry["id"])]


def load_covering(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    scale = max(width / image.width, height / image.height)
    image = image.resize((int(image.width * scale + 0.5), int(image.height * scale + 0.5)), Image.Resampling.LANCZOS)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.float32) / 255.0


def load_banner(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    crop_height = round(image.width * height / width)
    top = (image.height - crop_height) // 2
    cropped = image.crop((0, top, image.width, top + crop_height)).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(cropped, dtype=np.float32) / 255.0
