import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from ellipse_foreground.grid import edge_distances, grid_edges
from ellipse_foreground.settings import GraphCutSettings

_CAPACITY_LIMIT = 1 << 30


def pairwise_weights(work: np.ndarray, first: np.ndarray, second: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    distances = edge_distances(work, first, second).astype(np.float64)
    return np.mean([np.exp(-(distances**2) / (2.0 * sigma * sigma)) for sigma in sigmas], axis=0)


def minimum_cut_labels(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
    pairwise: np.ndarray,
    source_weights: np.ndarray,
    sink_weights: np.ndarray,
    capacity_scale: int,
) -> np.ndarray:
    total = float(source_weights.sum() + sink_weights.sum())
    scale = min(capacity_scale, _CAPACITY_LIMIT // max(int(np.ceil(total)), 1))
    source, sink = count, count + 1
    foreground, background = np.flatnonzero(source_weights > 0), np.flatnonzero(sink_weights > 0)
    rows = np.concatenate([first, second, np.full(len(foreground), source), background])
    columns = np.concatenate([second, first, foreground, np.full(len(background), sink)])
    weights = np.concatenate([pairwise, pairwise, source_weights[foreground], sink_weights[background]])
    capacities = np.rint(weights * scale).astype(np.int32)
    present = capacities > 0
    graph = csr_array((capacities[present], (rows[present], columns[present])), shape=(count + 2, count + 2))
    residual = (graph - maximum_flow(graph, source, sink).flow).tocsr()
    residual.data = np.where(residual.data > 0, residual.data, 0)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    labels = np.zeros(count + 2, dtype=bool)
    labels[reachable] = True
    return labels[:count]


def graph_cut_labels(work: np.ndarray, scores: np.ndarray, settings: GraphCutSettings) -> np.ndarray:
    first, second = grid_edges(work.shape[:2])
    pairwise = pairwise_weights(work, first, second, settings.sigmas)
    unary = scores.ravel() / settings.smoothness
    labels = minimum_cut_labels(
        work.shape[0] * work.shape[1],
        first,
        second,
        pairwise,
        np.maximum(unary, 0.0),
        np.maximum(-unary, 0.0),
        settings.capacity_scale,
    )
    return labels.reshape(work.shape[:2])
