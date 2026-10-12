import numpy as np


def bounded_log_ratio(numerator: np.ndarray, denominator: np.ndarray, clip: float, gain: float = 1.0) -> np.ndarray:
    return np.clip(gain * np.log(numerator / denominator), -clip, clip) / clip


def normalized_histogram(codes: np.ndarray, size: int, offset: float = 0.0) -> np.ndarray:
    counts = np.bincount(codes, minlength=size).astype(np.float64) + offset
    return counts / max(counts.sum(), 1e-9)
