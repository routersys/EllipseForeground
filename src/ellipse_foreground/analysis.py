from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ellipse_foreground.band import band_cut
from ellipse_foreground.boundary import BoundaryTerms
from ellipse_foreground.cues.color import color_evidence
from ellipse_foreground.cues.patch import choose_patch_samples, patch_evidence, patch_vectors
from ellipse_foreground.cues.sampling import pixel_priorities
from ellipse_foreground.cues.texture import TextureFeatures, texture_evidence, texture_features
from ellipse_foreground.cut import optimal_cut
from ellipse_foreground.finish import Finisher
from ellipse_foreground.geometry import Ellipse
from ellipse_foreground.graph_cut import graph_cut_labels
from ellipse_foreground.prior import ellipse_prior
from ellipse_foreground.reliability import Overlaps, measure_overlaps
from ellipse_foreground.settings import DEFAULT_SETTINGS, BoundaryForm, Settings
from ellipse_foreground.tree import build_tree, leaf_sums
from ellipse_foreground.weights import CueWeights, Mix, weights_for
from ellipse_foreground.work import reduce_image


@dataclass(frozen=True)
class Regions:
    inside: np.ndarray
    core: np.ndarray
    ring: np.ndarray


@dataclass(frozen=True)
class CueMaps:
    prior: np.ndarray
    color: np.ndarray
    texture: np.ndarray
    patch: np.ndarray

    def scores(self, weights: CueWeights) -> np.ndarray:
        return self.prior + weights.texture * self.texture + weights.color * self.color + weights.patch * self.patch


class Analysis:
    def __init__(self, image: np.ndarray, texture_source: np.ndarray, settings: Settings = DEFAULT_SETTINGS) -> None:
        self.settings = settings
        self.image = image.astype(np.float32)
        self.small, self.work = reduce_image(self.image, settings.work)
        self.shape = self.work.shape[:2]
        self.scale_x = self.shape[1] / self.image.shape[1]
        self.scale_y = self.shape[0] / self.image.shape[0]
        self.tree = build_tree(self.work)
        self.boundary = BoundaryTerms(self.tree, self.work, settings.boundary)
        self.texture: TextureFeatures = texture_features(texture_source, self.shape, settings.texture)
        self.priorities = pixel_priorities(self.tree.leaf_count, settings.samples.seed)

    @cached_property
    def vectors(self) -> np.ndarray:
        return patch_vectors(self.work, self.settings.patch.radius)

    def work_ellipse(self, ellipse: Ellipse) -> Ellipse:
        return ellipse.resized(self.scale_x, self.scale_y)

    def regions(self, ellipse: Ellipse) -> Regions:
        work_ellipse = self.work_ellipse(ellipse)
        samples = self.settings.samples
        return Regions(
            work_ellipse.inside(self.shape),
            work_ellipse.inside(self.shape, samples.core_radius),
            ~work_ellipse.inside(self.shape, samples.ring_radius),
        )

    def prior_map(self, ellipse: Ellipse) -> np.ndarray:
        return ellipse_prior(self.shape, self.work_ellipse(ellipse), self.settings.prior)

    def color_map(self, regions: Regions) -> np.ndarray:
        return color_evidence(self.work, regions.core, regions.ring, self.settings.color)

    def texture_map(self, regions: Regions) -> np.ndarray:
        return texture_evidence(self.texture, regions.core, regions.ring, self.settings.texture)

    def patch_map(self, regions: Regions) -> np.ndarray:
        patch = self.settings.patch
        core_samples, ring_samples = choose_patch_samples(regions.core, regions.ring, self.priorities, patch)
        return patch_evidence(self.vectors, core_samples, ring_samples, self.shape, patch)

    def cue_maps(self, ellipse: Ellipse, regions: Regions) -> CueMaps:
        return CueMaps(
            prior=self.prior_map(ellipse),
            color=self.color_map(regions),
            texture=self.texture_map(regions),
            patch=self.patch_map(regions),
        )

    def overlaps(self, regions: Regions) -> Overlaps:
        return measure_overlaps(
            self.work, self.texture, self.vectors, regions.core, regions.ring, self.settings.weights.ridge
        )

    def node_scores(self, scores: np.ndarray) -> np.ndarray:
        return leaf_sums(self.tree, scores)

    def cut_node_scores(
        self, node_scores: np.ndarray, form: BoundaryForm | None = None, strength: float | None = None
    ) -> np.ndarray:
        penalized = node_scores - self.boundary.penalty(form, strength)
        return optimal_cut(self.tree, penalized).reshape(self.shape)

    def cut(self, scores: np.ndarray, form: BoundaryForm | None = None, strength: float | None = None) -> np.ndarray:
        return self.cut_node_scores(self.node_scores(scores), form, strength)

    def graph_cut(self, scores: np.ndarray) -> np.ndarray:
        return graph_cut_labels(self.work, scores, self.settings.graph_cut)

    def band_cut(self, scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        mask = self.cut(scores) if mask is None else mask
        return band_cut(self.work, scores, mask, self.settings.band, self.settings.graph_cut)

    @cached_property
    def finisher(self) -> Finisher:
        return Finisher(self.small, self.image, self.settings.finish)

    def finish(self, mask: np.ndarray) -> np.ndarray:
        return self.finisher(mask.astype(np.float32))

    def extract(self, ellipse: Ellipse, mix: Mix = Mix.ADAPTIVE_COLOR, refine: bool = False) -> np.ndarray:
        regions = self.regions(ellipse)
        weights = weights_for(mix, self.overlaps(regions), self.settings.weights)
        scores = self.cue_maps(ellipse, regions).scores(weights)
        return self.finish(self.band_cut(scores) if refine else self.cut(scores))
