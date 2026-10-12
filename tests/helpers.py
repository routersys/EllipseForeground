import numpy as np


def random_work(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((height, width, 3)).astype(np.float32)


def leaves_of(left: np.ndarray, right: np.ndarray, leaf_count: int, node: int) -> frozenset[int]:
    if node < leaf_count:
        return frozenset([node])
    return leaves_of(left, right, leaf_count, left[node]) | leaves_of(left, right, leaf_count, right[node])


def noisy_scene(height: int = 144, width: int = 256, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:height, 0:width]
    image = np.full((height, width, 3), 0.5, dtype=np.float32)
    image += 0.15 * np.sin(columns / 7.0 + rows / 11.0)[..., None]
    image += 0.03 * rng.standard_normal((height, width, 3)).astype(np.float32)
    body = ((columns - width / 2) / 40.0) ** 2 + ((rows - height * 0.55) / 50.0) ** 2 <= 1
    image[body] = np.array([0.9, 0.3, 0.2], dtype=np.float32)
    image = np.clip(image, 0, 1)
    source = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    return image, source, body
