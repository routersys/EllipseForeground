from dataclasses import dataclass
from enum import Enum

from ellipse_foreground.reliability import Overlaps, relative_advantage
from ellipse_foreground.settings import WeightSettings


class Mix(Enum):
    PRIOR_ONLY = "prior"
    TEXTURE_ONLY = "tex"
    APPEARANCE_ONLY = "app"
    FIXED_MIX = "mix"
    ADAPTIVE_COLOR = "adaptive_c"
    ADAPTIVE_COLOR_PATCH = "adaptive"


@dataclass(frozen=True)
class CueWeights:
    texture: float
    color: float
    patch: float


def weights_for(mix: Mix, overlaps: Overlaps, settings: WeightSettings) -> CueWeights:
    if mix is Mix.PRIOR_ONLY:
        return CueWeights(0.0, 0.0, 0.0)
    if mix is Mix.TEXTURE_ONLY:
        return CueWeights(settings.texture, 0.0, 0.0)
    if mix is Mix.APPEARANCE_ONLY:
        return CueWeights(0.0, settings.adaptive_color, settings.adaptive_patch)
    if mix is Mix.FIXED_MIX:
        return CueWeights(settings.texture, settings.fixed_color, settings.fixed_patch)
    color = settings.adaptive_color * relative_advantage(overlaps.color, overlaps.texture)
    if mix is Mix.ADAPTIVE_COLOR:
        return CueWeights(settings.texture, color, settings.fixed_patch)
    patch = settings.adaptive_patch * relative_advantage(overlaps.patch, overlaps.texture)
    return CueWeights(settings.texture, color, patch)
