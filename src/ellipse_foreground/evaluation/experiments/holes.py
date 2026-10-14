import numpy as np
from scipy import ndimage

from ellipse_foreground import Mix
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.weights import weights_for


def holes_record(context: SceneContext, trial: Trial) -> dict:
    analysis = context.analysis
    truth = context.work_truth
    weights = weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, context.settings.weights)
    scores = trial.cues.scores(weights)
    mask = analysis.cut(scores)
    filled = ndimage.binary_fill_holes(mask)

    def result(found: np.ndarray) -> dict:
        return {"raw_iou": mask_iou(found, truth), **context.finished_score(found)}

    return {
        "scene": context.scene.index,
        "name": context.scene.name,
        "trial": trial.index,
        "hole_share": float((filled & ~mask).sum() / max(int(mask.sum()), 1)),
        "tree": result(mask),
        "fill": result(filled),
        "band": result(analysis.band_cut(scores, mask)),
        "fill_band": result(analysis.band_cut(scores, filled)),
    }
