from dataclasses import dataclass

import numpy as np

from ellipse_foreground.graph_cut import minimum_cut_labels, pairwise_weights
from ellipse_foreground.grid import grid_edges
from ellipse_foreground.imaging import dilate, erode
from ellipse_foreground.settings import BandSettings, GraphCutSettings


@dataclass(frozen=True)
class BandProblem:
    free: np.ndarray
    count: int
    first: np.ndarray
    second: np.ndarray
    pairwise: np.ndarray
    source_weights: np.ndarray
    sink_weights: np.ndarray


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    return dilate(mask, radius) & ~erode(mask, radius)


def band_problem(
    work: np.ndarray, scores: np.ndarray, mask: np.ndarray, band: BandSettings, graph: GraphCutSettings
) -> BandProblem | None:
    free = boundary_band(mask, band.radius).ravel()
    if not free.any():
        return None
    first, second = grid_edges(mask.shape)
    pairwise = pairwise_weights(work, first, second, graph.sigmas)
    fixed = mask.ravel()
    node_of_pixel = np.cumsum(free) - 1
    inside_first, inside_second = free[first], free[second]
    both = inside_first & inside_second
    unary = scores.ravel()[free] / graph.smoothness
    source_weights = np.maximum(unary, 0.0)
    sink_weights = np.maximum(-unary, 0.0)
    for outer_free, inner, outer in (
        (inside_first & ~inside_second, first, second),
        (inside_second & ~inside_first, second, first),
    ):
        nodes = node_of_pixel[inner[outer_free]]
        anchored = fixed[outer[outer_free]]
        np.add.at(source_weights, nodes[anchored], pairwise[outer_free][anchored])
        np.add.at(sink_weights, nodes[~anchored], pairwise[outer_free][~anchored])
    return BandProblem(
        free,
        int(free.sum()),
        node_of_pixel[first[both]],
        node_of_pixel[second[both]],
        pairwise[both],
        source_weights,
        sink_weights,
    )


def band_cut(
    work: np.ndarray, scores: np.ndarray, mask: np.ndarray, band: BandSettings, graph: GraphCutSettings
) -> np.ndarray:
    problem = band_problem(work, scores, mask, band, graph)
    if problem is None:
        return mask
    labels = minimum_cut_labels(
        problem.count,
        problem.first,
        problem.second,
        problem.pairwise,
        problem.source_weights,
        problem.sink_weights,
        graph.capacity_scale,
    )
    result = mask.ravel().copy()
    result[problem.free] = labels
    return result.reshape(mask.shape)
