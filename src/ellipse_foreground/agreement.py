import numpy as np

from ellipse_foreground.analysis import CueMaps
from ellipse_foreground.settings import AgreementSettings
from ellipse_foreground.weights import CueWeights

_MINIMUM_SPREAD = 1e-12


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first, second = first.ravel() - first.mean(), second.ravel() - second.mean()
    spread = np.sqrt((first**2).sum() * (second**2).sum())
    return float((first * second).sum() / spread) if spread > _MINIMUM_SPREAD else 0.0


def cue_agreements(cues: CueMaps) -> tuple[float, float, float]:
    texture, color, patch = cues.texture, cues.color, cues.patch
    return (
        correlation(texture, 0.5 * (color + patch)),
        correlation(color, 0.5 * (texture + patch)),
        correlation(patch, 0.5 * (texture + color)),
    )


def agreement_weights(weights: CueWeights, cues: CueMaps, settings: AgreementSettings) -> CueWeights:
    agreements = cue_agreements(cues)
    mean = sum(agreements) / len(agreements)
    factors = [
        float(np.clip(1.0 + settings.gain * (value - mean), settings.floor, settings.ceiling)) for value in agreements
    ]
    return CueWeights(weights.texture * factors[0], weights.color * factors[1], weights.patch * factors[2])
