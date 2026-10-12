from pathlib import Path

import numpy as np
from PIL import Image


def alpha_bytes(alpha: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)


def save_alpha(path: Path, alpha: np.ndarray) -> None:
    Image.fromarray(alpha_bytes(alpha)).save(path)


def load_alpha(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))
