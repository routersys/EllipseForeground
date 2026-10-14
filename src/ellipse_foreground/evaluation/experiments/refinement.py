from dataclasses import replace

from ellipse_foreground import Mix
from ellipse_foreground.agreement import agreement_weights
from ellipse_foreground.band import band_cut, boundary_band
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.weights import weights_for

BAND_RADII = (1, 2, 3, 4)
BAND_SMOOTHNESS = (16.0, 32.0, 64.0, 128.0, 256.0, 512.0)


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
    smoothness = {}
    for value in BAND_SMOOTHNESS:
        refined = band_cut(analysis.work, scores, mask, replace(settings.band, smoothness=value), settings.graph_cut)
        smoothness[f"{value:g}"] = {"raw_iou": mask_iou(refined, truth), **context.finished_score(refined)}
    agreed = trial.cues.scores(agreement_weights(weights, trial.cues, settings.agreement))
    agreed_mask = analysis.cut(agreed)
    agreed_band = band_cut(analysis.work, agreed, agreed_mask, settings.band, settings.graph_cut)
    return {
        "scene": context.scene.index,
        "name": context.scene.name,
        "trial": trial.index,
        "tree": {"raw_iou": mask_iou(mask, truth), **context.finished_score(mask)},
        "band": band,
        "smoothness": smoothness,
        "agreement": {
            "tree": {"raw_iou": mask_iou(agreed_mask, truth), **context.finished_score(agreed_mask)},
            "band": {"raw_iou": mask_iou(agreed_band, truth), **context.finished_score(agreed_band)},
        },
    }
