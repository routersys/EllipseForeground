from dataclasses import replace

import numpy as np

from ellipse_foreground import Analysis, CueMaps, Mix, Regions
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.experiments.unmixing import (
    annulus_index,
    features_of,
    occupancy_of,
    population_occupancy,
    unmixed_evidence,
)
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.imaging import erode
from ellipse_foreground.weights import CueWeights, weights_for

ERODE_DEPTH = 3
MINIMUM_SAMPLES = 60


def usable(mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    return mask if np.count_nonzero(mask) >= MINIMUM_SAMPLES else fallback


def appearance_scores(
    analysis: Analysis, prior: np.ndarray, regions: Regions, weights: CueWeights | None
) -> np.ndarray:
    cues = CueMaps(prior, analysis.color_map(regions), analysis.texture_map(regions), analysis.patch_map(regions))
    if weights is None:
        weights = weights_for(Mix.ADAPTIVE_COLOR, analysis.overlaps(regions), analysis.settings.weights)
    return cues.scores(weights)


def oracle_masks(context: SceneContext, trial: Trial) -> dict[str, np.ndarray]:
    analysis = context.analysis
    settings = context.settings
    truth = context.work_truth
    weights = weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, settings.weights)
    prior = trial.cues.prior
    inside, core, ring = trial.regions.inside, trial.regions.core, trial.regions.ring
    radius = analysis.work_ellipse(trial.ellipse).normalized_radius(analysis.shape)
    prototype = context.scene.prototype
    true_prior = analysis.prior_map(prototype)
    pure_foreground = erode(truth, ERODE_DEPTH)
    pure_background = erode(~truth, ERODE_DEPTH)
    sample_regions = Regions(inside, usable(pure_foreground, core), usable(pure_background, ring))
    sample_cues = CueMaps(
        prior,
        analysis.color_map(sample_regions),
        analysis.texture_map(sample_regions),
        analysis.patch_map(sample_regions),
    )
    sample_weights = weights_for(Mix.ADAPTIVE_COLOR, analysis.overlaps(sample_regions), settings.weights)
    masks = {
        "tree": analysis.cut(trial.cues.scores(weights)),
        "ellipse": analysis.solve(prototype)[1],
        "prior": analysis.cut(replace(trial.cues, prior=true_prior).scores(weights)),
        "samples": analysis.cut(sample_cues.scores(sample_weights)),
        "both": analysis.cut(replace(sample_cues, prior=true_prior).scores(sample_weights)),
        "samples_ct": analysis.cut(replace(sample_cues, patch=trial.cues.patch).scores(weights)),
    }
    far = radius > settings.samples.ring_radius
    same = radius < settings.samples.core_radius
    decomposition = {
        "ring_far": Regions(inside, core, usable(pure_background & far, ring)),
        "ring_near": Regions(inside, core, usable(pure_background & ~far, ring)),
        "core_same": Regions(inside, usable(pure_foreground & same, core), ring),
        "core_all": Regions(inside, usable(pure_foreground, core), ring),
    }
    for name, regions in decomposition.items():
        masks[name] = analysis.cut(appearance_scores(analysis, prior, regions, None))
    features = features_of(analysis)
    index = annulus_index(radius)
    occupancies = {
        "unmix_truth": occupancy_of(index, truth),
        "unmix_tree": occupancy_of(index, masks["tree"]),
        "unmix_population": population_occupancy(),
    }
    for name, occupancy in occupancies.items():
        color, texture = (
            unmixed_evidence(feature, index, occupancy, core, ring, analysis.shape) for feature in features
        )
        masks[name] = analysis.cut(replace(trial.cues, color=color, texture=texture).scores(weights))
    return masks


def share(mask: np.ndarray, selection: np.ndarray) -> float | None:
    chosen = mask[selection]
    return float(chosen.mean()) if chosen.size else None


def oracle_record(context: SceneContext, trial: Trial) -> dict:
    truth = context.work_truth
    record = {
        "scene": context.scene.index,
        "name": context.scene.name,
        "trial": trial.index,
        "purity": {"core": share(truth, trial.regions.core), "ring": share(~truth, trial.regions.ring)},
    }
    for name, mask in oracle_masks(context, trial).items():
        record[name] = {"raw_iou": mask_iou(mask, truth), **context.finished_score(mask)}
    return record
