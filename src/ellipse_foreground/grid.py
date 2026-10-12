import numpy as np


def grid_edges(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    index = np.arange(height * width).reshape(height, width)
    first = np.concatenate([index[:, :-1].ravel(), index[:-1].ravel()])
    second = np.concatenate([index[:, 1:].ravel(), index[1:].ravel()])
    return first, second


def edge_distances(work: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    flat = work.reshape(-1, work.shape[2])
    return np.sqrt(((flat[first] - flat[second]) ** 2).sum(axis=1))
