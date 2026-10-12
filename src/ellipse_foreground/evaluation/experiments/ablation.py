from dataclasses import replace

import numpy as np

from ellipse_foreground import Analysis, BoundaryForm, Mix
from ellipse_foreground.cues.texture import texture_evidence, texture_features
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.finish import (
    bilinear_upsample,
    guided_filter,
    histogram_refinement,
    joint_bilateral_upsample,
)
from ellipse_foreground.graph_cut import graph_cut_labels
from ellipse_foreground.imaging import dilate, halve
from ellipse_foreground.prior import ellipse_prior
from ellipse_foreground.weights import CueWeights, weights_for

BOUNDARY_EXPONENTS = {
    BoundaryForm.MEAN_WEAKNESS: range(-6, 5),
    BoundaryForm.WEAKNESS_SUM: range(-6, 5),
    BoundaryForm.MERGE_HEIGHT: range(-7, 4),
}
INSIDE_SHRINKS = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.0)
GRAPH_CUT_EXPONENTS = range(9)
FINISH_STAGES = ("bilinear", "jbu", "jbu_lut", "jbu_lutband", "jbu_guided", "final")


def halve_source(source: np.ndarray) -> np.ndarray:
    halved = halve(source.astype(np.float32) / 255.0)
    return np.clip(np.rint(halved * 255.0), 0, 255).astype(np.uint8)


def reduced_sources(source: np.ndarray) -> dict[str, np.ndarray]:
    half = halve_source(source)
    return {"full": source, "x640": half, "x320": halve_source(half)}


def finish_variants(analysis: Analysis, mask: np.ndarray) -> dict[str, np.ndarray]:
    settings = analysis.settings.finish
    image, small = analysis.image, analysis.small
    mask32 = mask.astype(np.float32)
    upsampled = joint_bilateral_upsample(small, mask32, image, settings.range_sigma)
    refined = histogram_refinement(small, mask32, image, upsampled, settings)
    band = dilate((upsampled > settings.band_low) & (upsampled < settings.band_high), settings.band_dilation)
    banded = np.where(band, refined, upsampled).astype(np.float32)

    def filtered(source: np.ndarray) -> np.ndarray:
        result = guided_filter(image, source, settings.guided_radius, settings.guided_epsilon)
        return np.clip(result, 0, 1).astype(np.float32)

    return {
        "bilinear": bilinear_upsample(mask32, image.shape[:2]),
        "jbu": upsampled,
        "jbu_lut": refined,
        "jbu_lutband": banded,
        "jbu_guided": filtered(upsampled),
        "final": filtered(banded),
    }


def boundary_sweep(context: SceneContext, scores: np.ndarray) -> dict:
    analysis, truth = context.analysis, context.work_truth
    data = analysis.node_scores(scores)
    oracle_data = analysis.node_scores(2.0 * context.work_alpha - 1.0)

    def measure(strength: float, form: BoundaryForm = BoundaryForm.MEAN_WEAKNESS) -> dict[str, float]:
        return {
            "iou": mask_iou(analysis.cut_node_scores(data, form, strength), truth),
            "oracle_iou": mask_iou(analysis.cut_node_scores(oracle_data, form, strength), truth),
        }

    sweep = {"none": measure(0.0)}
    for form, exponents in BOUNDARY_EXPONENTS.items():
        sweep[form.value] = {str(exponent): measure(2.0**exponent, form) for exponent in exponents}
    return sweep


def prior_sweep(context: SceneContext, trial: Trial, weights: CueWeights) -> dict[str, float]:
    analysis, settings = context.analysis, context.settings
    ellipse = analysis.work_ellipse(trial.ellipse)
    result = {}
    for shrink in INSIDE_SHRINKS:
        prior = ellipse_prior(analysis.shape, ellipse, replace(settings.prior, inside_shrink=shrink))
        scores = replace(trial.cues, prior=prior).scores(weights)
        result[str(shrink)] = mask_iou(analysis.cut(scores), context.work_truth)
    return result


def texture_resolution_sweep(context: SceneContext, trial: Trial, weights: CueWeights) -> dict[str, dict[str, float]]:
    analysis, settings = context.analysis, context.settings
    only = CueWeights(settings.weights.texture, 0.0, 0.0)
    result = {}
    for name, source in reduced_sources(context.scene.texture_source).items():
        features = texture_features(source, analysis.shape, settings.texture)
        evidence = texture_evidence(features, trial.regions.core, trial.regions.ring, settings.texture)
        cues = replace(trial.cues, texture=evidence)
        result[name] = {
            "texture_only": mask_iou(analysis.cut(cues.scores(only)), context.work_truth),
            "mix": mask_iou(analysis.cut(cues.scores(weights)), context.work_truth),
        }
    return result


def graph_cut_sweep(context: SceneContext, scores: np.ndarray) -> dict[str, float]:
    analysis, settings = context.analysis, context.settings
    return {
        str(exponent): mask_iou(
            graph_cut_labels(analysis.work, scores, replace(settings.graph_cut, smoothness=2.0**exponent)),
            context.work_truth,
        )
        for exponent in GRAPH_CUT_EXPONENTS
    }


def finish_sweep(context: SceneContext, scores: np.ndarray) -> dict[str, dict[str, float]]:
    variants = finish_variants(context.analysis, context.analysis.cut(scores))
    result = {}
    for name in FINISH_STAGES:
        score = context.scorer.score(variants[name])
        result[name] = {"iou": score.iou, "mae": score.mae}
    return result


def ablation_record(context: SceneContext, trial: Trial) -> dict:
    weights = weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, context.settings.weights)
    scores = trial.cues.scores(weights)
    return {
        "scene": context.scene.index,
        "name": context.scene.name,
        "trial": trial.index,
        "boundary": boundary_sweep(context, scores),
        "prior": prior_sweep(context, trial, weights),
        "texture_resolution": texture_resolution_sweep(context, trial, weights),
        "graph_cut": graph_cut_sweep(context, scores),
        "finish": finish_sweep(context, scores),
    }
