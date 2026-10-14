from dataclasses import replace

from ellipse_foreground import Mix
from ellipse_foreground.band import band_cut, boundary_band
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.weights import weights_for

BAND_RADII = (1, 2, 3, 4)


def refinement_record(context: SceneContext, trial: Trial) -> dict:
    analysis = context.analysis
    settings = context.settings
    truth = context.work_truth
    weights = weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, settings.weights)
    scores = trial.cues.scores(weights)
    mask = analysis.cut(scores)
    band = {}
    for radius in BAND_RADII:
        refined = band_cut(analysis.work, scores, mask, replace(settings.band, radius=radius), settings.graph_cut)
        band[str(radius)] = {
            "raw_iou": mask_iou(refined, truth),
            "share": float(boundary_band(mask, radius).mean()),
            **context.finished_score(refined),
        }
    return {
        "scene": context.scene.index,
        "name": context.scene.name,
        "trial": trial.index,
        "tree": {"raw_iou": mask_iou(mask, truth), **context.finished_score(mask)},
        "band": band,
    }
