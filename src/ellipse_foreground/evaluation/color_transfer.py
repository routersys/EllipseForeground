import numpy as np

_RGB_TO_LMS = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])
_LMS_TO_LAB = np.diag([1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(2)]) @ np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
_LAB_TO_LMS = np.linalg.inv(_LMS_TO_LAB)
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)
_MINIMUM_LMS = 1e-4


def to_lab(rgb: np.ndarray) -> np.ndarray:
    lms = np.maximum(rgb @ _RGB_TO_LMS.T, _MINIMUM_LMS)
    return np.log10(lms) @ _LMS_TO_LAB.T


def from_lab(lab: np.ndarray) -> np.ndarray:
    lms = 10 ** (lab @ _LAB_TO_LMS.T)
    return np.clip(lms @ _LMS_TO_RGB.T, 0, 1)


def transfer_colors(foreground: np.ndarray, alpha: np.ndarray, background: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return foreground
    foreground_lab = to_lab(foreground)
    background_lab = to_lab(background.reshape(-1, 3))
    opaque = alpha > 0.5
    foreground_mean = foreground_lab[opaque].mean(axis=0)
    foreground_deviation = foreground_lab[opaque].std(axis=0) + 1e-6
    background_mean = background_lab.mean(axis=0)
    background_deviation = background_lab.std(axis=0) + 1e-6
    target = (foreground_lab - foreground_mean) / foreground_deviation * background_deviation + background_mean
    return from_lab((1 - strength) * foreground_lab + strength * target)
