import numpy as np
from PIL import Image


def area_resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.ndim == 3:
        return np.stack([area_resize(array[..., channel], width, height) for channel in range(array.shape[2])], axis=2)
    resized = Image.fromarray(array.astype(np.float32)).resize((width, height), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float32)


def halve(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[0] // 2 * 2, array.shape[1] // 2 * 2
    cropped = array[:height, :width]
    return 0.25 * (cropped[0::2, 0::2] + cropped[1::2, 0::2] + cropped[0::2, 1::2] + cropped[1::2, 1::2])


def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = mask.shape
    result = np.zeros_like(mask)
    result[max(dy, 0) : height + min(dy, 0), max(dx, 0) : width + min(dx, 0)] = mask[
        max(-dy, 0) : height + min(-dy, 0), max(-dx, 0) : width + min(-dx, 0)
    ]
    return result


def dilate(mask: np.ndarray, times: int = 1) -> np.ndarray:
    for _ in range(times):
        mask = mask | shifted(mask, 1, 0) | shifted(mask, -1, 0) | shifted(mask, 0, 1) | shifted(mask, 0, -1)
    return mask


def erode(mask: np.ndarray, times: int = 1) -> np.ndarray:
    return ~dilate(~mask, times)


def box_mean(array: np.ndarray, radius: int) -> np.ndarray:
    height, width = array.shape[:2]
    padding = ((radius + 1, radius), (radius + 1, radius)) + ((0, 0),) * (array.ndim - 2)
    sums = np.pad(array.astype(np.float64), padding, mode="edge").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = (
        sums[size : size + height, size : size + width]
        - sums[:height, size : size + width]
        - sums[size : size + height, :width]
        + sums[:height, :width]
    )
    return total / (size * size)
