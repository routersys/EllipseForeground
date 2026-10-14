from dataclasses import dataclass
from enum import Enum


class BoundaryForm(Enum):
    MEAN_WEAKNESS = "mean_weakness"
    WEAKNESS_SUM = "weakness_sum"
    MERGE_HEIGHT = "merge_height"


@dataclass(frozen=True)
class WorkSettings:
    width: int = 256
    low_percentile: float = 2.0
    high_percentile: float = 98.0
    minimum_range: float = 0.04
    ceiling: float = 1.5


@dataclass(frozen=True)
class PriorSettings:
    softness: float = 0.35
    inside_shrink: float = 0.35
    scales: tuple[float, ...] = (0.85, 1.0, 1.15)


@dataclass(frozen=True)
class SampleSettings:
    core_radius: float = 0.5
    ring_radius: float = 1.3
    seed: int = 0


@dataclass(frozen=True)
class ColorSettings:
    bins: int = 8
    floor: float = 1e-5
    clip: float = 2.0


@dataclass(frozen=True)
class TextureSettings:
    flat_threshold: int = 2
    magnitude_cap: int = 32
    smoothing_radius: int = 1
    bins: int = 6
    count_offset: float = 0.5
    clip: float = 2.0


@dataclass(frozen=True)
class PatchSettings:
    radius: int = 1
    core_limit: int = 3000
    ring_multiple: int = 8
    ring_minimum: int = 20
    exclusion: int = 2
    sharpness: float = 4.0
    floor: float = 1e-3
    clip: float = 2.0
    block_rows: int = 512


@dataclass(frozen=True)
class BoundarySettings:
    form: BoundaryForm = BoundaryForm.MEAN_WEAKNESS
    strength: float = 0.5
    sigmas: tuple[float, ...] = (0.05, 0.1)
    mean_lambda: float = 14.0 / 3.0
    shape_factor: float = 4.0
    reference_width: int = 128


@dataclass(frozen=True)
class WeightSettings:
    texture: float = 1.0
    fixed_color: float = 0.25
    fixed_patch: float = 0.3
    adaptive_color: float = 0.5
    adaptive_patch: float = 0.6
    ridge: float = 1e-4


@dataclass(frozen=True)
class FinishSettings:
    range_sigma: float = 0.12
    histogram_bins: int = 32
    histogram_clip: float = 6.0
    logit_gain: float = 6.0
    histogram_gain: float = 3.0
    band_low: float = 0.02
    band_high: float = 0.98
    band_dilation: int = 2
    guided_radius: int = 2
    guided_epsilon: float = 1e-3


@dataclass(frozen=True)
class GraphCutSettings:
    smoothness: float = 32.0
    sigmas: tuple[float, ...] = (0.05, 0.1)
    capacity_scale: int = 1 << 16


@dataclass(frozen=True)
class BandSettings:
    radius: int = 2
    smoothness: float = 128.0


@dataclass(frozen=True)
class StabilitySettings:
    scales: tuple[float, ...] = (0.8, 1.25)
    shift: float = 0.12
    threshold: float = 0.95


@dataclass(frozen=True)
class AgreementSettings:
    gain: float = 3.0
    floor: float = 0.2
    ceiling: float = 2.0


@dataclass(frozen=True)
class Settings:
    work: WorkSettings = WorkSettings()
    prior: PriorSettings = PriorSettings()
    samples: SampleSettings = SampleSettings()
    color: ColorSettings = ColorSettings()
    texture: TextureSettings = TextureSettings()
    patch: PatchSettings = PatchSettings()
    boundary: BoundarySettings = BoundarySettings()
    weights: WeightSettings = WeightSettings()
    finish: FinishSettings = FinishSettings()
    graph_cut: GraphCutSettings = GraphCutSettings()
    band: BandSettings = BandSettings()
    agreement: AgreementSettings = AgreementSettings()
    stability: StabilitySettings = StabilitySettings()


DEFAULT_SETTINGS = Settings()
