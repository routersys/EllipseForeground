from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageDifference:
    differing: int
    max_delta: float
    left: int
    top: int
    right: int
    bottom: int

    @property
    def is_empty(self) -> bool:
        return self.differing == 0

    def __str__(self) -> str:
        if self.is_empty:
            return "一致"
        return (
            f"differing={self.differing} maxDelta={self.max_delta:.6g} "
            f"box=({self.left},{self.top})-({self.right},{self.bottom})"
        )


def compare_arrays(first: np.ndarray, second: np.ndarray) -> ImageDifference:
    if first.shape != second.shape:
        raise ValueError("配列の寸法が一致しません。")
    delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
    changed = delta > 0
    if changed.ndim == 3:
        changed = changed.any(axis=2)
        delta = delta.max(axis=2)
    if not changed.any():
        return ImageDifference(0, 0.0, 0, 0, 0, 0)
    rows, columns = np.nonzero(changed)
    return ImageDifference(
        int(changed.sum()), float(delta.max()), int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())
    )
