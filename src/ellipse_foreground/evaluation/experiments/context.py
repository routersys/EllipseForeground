from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ellipse_foreground import Analysis, CueMaps, Ellipse, Regions, Settings
from ellipse_foreground.evaluation.metrics import Scorer
from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.evaluation.trials import MAIN_MODEL, ErrorModel, draw_trials, is_informative
from ellipse_foreground.reliability import Overlaps
from ellipse_foreground.settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class Trial:
    index: int
    ellipse: Ellipse
    regions: Regions
    cues: CueMaps
    overlaps: Overlaps


class SceneContext:
    def __init__(self, scene: Scene, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.scene = scene
        self.settings = settings
        self.analysis = Analysis(scene.image, scene.texture_source, settings)
        self.scorer = Scorer(scene)
        self.work_alpha = scene.work_alpha(settings.work)
        self.work_truth = self.work_alpha > 0.5

    def informative(self, ellipse: Ellipse) -> bool:
        return is_informative(self.analysis.regions(ellipse).inside, self.work_truth)

    def trial(self, index: int, ellipse: Ellipse) -> Trial:
        regions = self.analysis.regions(ellipse)
        return Trial(
            index,
            ellipse,
            regions,
            self.analysis.cue_maps(ellipse, regions),
            self.analysis.overlaps(regions),
        )

    def trials(self, count: int, model: ErrorModel = MAIN_MODEL) -> Iterator[Trial]:
        for index, ellipse in enumerate(draw_trials(self.scene, count, model)):
            if self.informative(ellipse):
                yield self.trial(index, ellipse)

    def finished_score(self, mask: np.ndarray) -> dict[str, float]:
        score = self.scorer.score(self.analysis.finish(mask))
        return {"iou": score.iou, "mae": score.mae}
