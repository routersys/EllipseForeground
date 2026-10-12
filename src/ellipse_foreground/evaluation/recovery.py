from pathlib import Path

import numpy as np
from PIL import Image

_CARDINAL_AND_DIAGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
_COLUMN_MARGIN = 6
_MINIMUM_ALPHA = 0.01
_FILL_ITERATIONS = 40
_FILL_THRESHOLD = 0.3


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64)


def character_columns(alpha: np.ndarray) -> np.ndarray:
    counts = (alpha > 0.5).sum(axis=0)
    peak = int(np.argmax(counts))
    low = peak
    while low > 0 and counts[low] > 0:
        low -= 1
    high = peak
    while high < len(counts) - 1 and counts[high] > 0:
        high += 1
    keep = np.zeros(alpha.shape, dtype=bool)
    keep[:, max(low - _COLUMN_MARGIN, 0) : min(high + _COLUMN_MARGIN + 1, alpha.shape[1])] = True
    return keep


def recover_alpha(composite: np.ndarray, foreground_only: np.ndarray, background_only: np.ndarray) -> np.ndarray:
    numerator = (background_only * (composite - foreground_only)).sum(axis=2)
    denominator = (background_only * background_only).sum(axis=2) + 1e-6
    alpha = np.clip(1.0 - numerator / denominator, 0.0, 1.0)
    alpha = np.where(character_columns(alpha), alpha, 0.0)
    alpha[alpha < _MINIMUM_ALPHA] = 0.0
    return alpha


def fill_foreground(color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    divided = np.clip(color / np.maximum(alpha, 1e-3)[..., None], 0.0, 1.0).astype(np.float32)
    filled = alpha >= 0.5
    current = np.where(filled[..., None], divided, 0.0).astype(np.float32)
    for _ in range(_FILL_ITERATIONS):
        if filled.all():
            break
        accumulated = np.zeros_like(current)
        counts = np.zeros(filled.shape, np.float32)
        for dy, dx in _CARDINAL_AND_DIAGONAL:
            shifted_filled = np.roll(filled, (dy, dx), (0, 1))
            shifted_color = np.roll(current, (dy, dx), (0, 1))
            accumulated += np.where(shifted_filled[..., None], shifted_color, 0.0)
            counts += shifted_filled
        added = (~filled) & (counts > 0)
        current[added] = accumulated[added] / counts[added][:, None]
        filled |= added
    return np.where((alpha >= _FILL_THRESHOLD)[..., None], divided, current).astype(np.float32)
