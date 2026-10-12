import numpy as np

from ellipse_foreground import Mix
from ellipse_foreground.evaluation.auc import rank_auc
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.weights import CueWeights, weights_for


def cue_auc(cue: np.ndarray, trial: Trial, truth: np.ndarray) -> float:
    return rank_auc(cue[trial.regions.inside], truth[trial.regions.inside])


def weights_record(weights: CueWeights) -> list[float]:
    return [weights.texture, weights.color, weights.patch]


def comparison_record(context: SceneContext, trial: Trial) -> dict:
    analysis = context.analysis
    truth = context.work_truth
    weights = {mix: weights_for(mix, trial.overlaps, context.settings.weights) for mix in Mix}
    scores = {mix: trial.cues.scores(weights[mix]) for mix in Mix}
    tree = {"ellipse": context.finished_score(trial.regions.inside)}
    graph = {}
    for mix in Mix:
        mask = analysis.cut(scores[mix])
        tree[mix.value] = {"raw_iou": mask_iou(mask, truth), **context.finished_score(mask)}
        labels = analysis.graph_cut(scores[mix])
        graph[mix.value] = {"raw_iou": mask_iou(labels, truth), **context.finished_score(labels)}
    tree["oracle"] = context.finished_score(analysis.cut(2.0 * context.work_alpha - 1.0))
    inside = trial.regions.inside
    return {
        "scene": context.scene.index,
        "name": context.scene.name,
        "trial": trial.index,
        "overlaps": {
            "color": trial.overlaps.color,
            "texture": trial.overlaps.texture,
            "patch": trial.overlaps.patch,
        },
        "auc": {
            "color": cue_auc(trial.cues.color, trial, truth),
            "texture": cue_auc(trial.cues.texture, trial, truth),
            "patch": cue_auc(trial.cues.patch, trial, truth),
        },
        "weights": {
            "adaptive_color": weights_record(weights[Mix.ADAPTIVE_COLOR]),
            "adaptive_color_patch": weights_record(weights[Mix.ADAPTIVE_COLOR_PATCH]),
        },
        "occupancy": float(truth[inside].mean()),
        "spill": float(1.0 - truth[inside].sum() / truth.sum()),
        "tree": tree,
        "graph": graph,
    }
