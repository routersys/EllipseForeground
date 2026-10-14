import numpy as np

from ellipse_foreground import Mix
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.weights import weights_for


def adaptation_record(context: SceneContext, trial: Trial) -> dict:
    analysis = context.analysis
    truth = context.work_truth
    weights = weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, context.settings.weights)
    scores = trial.cues.scores(weights)
    mask = analysis.cut(scores)
    refit_scores, refit_mask = analysis.refit(trial.ellipse, Mix.ADAPTIVE_COLOR, scores, mask)

    def result(found: np.ndarray) -> dict:
        return {"raw_iou": mask_iou(found, truth), **context.finished_score(found)}

    return {
        "scene": context.scene.index,
        "name": context.scene.name,
        "trial": trial.index,
        "stability": analysis.stability(trial.ellipse, scores, mask),
        "tree": result(mask),
        "refit": result(refit_mask),
        "band": result(analysis.band_cut(scores, mask)),
        "refit_band": result(analysis.band_cut(refit_scores, refit_mask)),
    }
