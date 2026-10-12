import numpy as np

from ellipse_foreground.grid import edge_distances, grid_edges
from ellipse_foreground.settings import BoundaryForm, BoundarySettings
from ellipse_foreground.tree import Tree, accumulate


def lowest_common_ancestors(parent: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    node_count = len(parent)
    depth = np.zeros(node_count, dtype=np.int64)
    for node in range(node_count - 2, -1, -1):
        depth[node] = depth[parent[node]] + 1
    levels = max(1, int(depth.max()).bit_length())
    ancestors = [parent]
    for _ in range(1, levels):
        ancestors.append(ancestors[-1][ancestors[-1]])
    first, second = first.copy(), second.copy()
    deeper_second = depth[first] < depth[second]
    first[deeper_second], second[deeper_second] = second[deeper_second], first[deeper_second].copy()
    difference = depth[first] - depth[second]
    for level in range(levels):
        lifting = ((difference >> level) & 1).astype(bool)
        first[lifting] = ancestors[level][first[lifting]]
    for level in range(levels - 1, -1, -1):
        above_first, above_second = ancestors[level][first], ancestors[level][second]
        diverging = above_first != above_second
        first[diverging] = above_first[diverging]
        second[diverging] = above_second[diverging]
    return np.where(first == second, first, parent[first])


def boundary_sums(
    tree: Tree, first: np.ndarray, second: np.ndarray, ancestors: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    own = np.zeros(tree.node_count)
    np.add.at(own, first, weights)
    np.add.at(own, second, weights)
    np.add.at(own, ancestors, -2.0 * weights)
    totals = accumulate(tree, own)
    totals[tree.root] = 0.0
    return totals


class BoundaryTerms:
    def __init__(self, tree: Tree, work: np.ndarray, settings: BoundarySettings) -> None:
        self.settings = settings
        first, second = grid_edges(work.shape[:2])
        distances = edge_distances(work, first, second).astype(np.float64)
        ancestors = lowest_common_ancestors(tree.parent, first, second)
        counts = np.maximum(boundary_sums(tree, first, second, ancestors, np.ones(len(distances))), 1.0)
        sums = [
            boundary_sums(tree, first, second, ancestors, np.exp(-(distances**2) / (2.0 * sigma * sigma)))
            for sigma in settings.sigmas
        ]
        self.weakness_sum = np.mean(sums, axis=0)
        self.mean_weakness = np.mean([total / counts for total in sums], axis=0)
        parent_height = np.full(tree.node_count, np.inf)
        parent_height[: tree.root] = tree.height[tree.parent[: tree.root]]
        self.merge_weakness = np.mean(
            [np.exp(-(parent_height**2) / (2.0 * sigma * sigma)) for sigma in settings.sigmas], axis=0
        )
        self.shape = settings.shape_factor * np.sqrt(tree.area)
        self.resolution = work.shape[1] / settings.reference_width

    def unit_penalty(self, form: BoundaryForm) -> np.ndarray:
        lam = self.settings.mean_lambda
        if form is BoundaryForm.MEAN_WEAKNESS:
            return self.resolution * lam * self.shape * self.mean_weakness
        if form is BoundaryForm.MERGE_HEIGHT:
            return self.resolution * lam * self.shape * self.merge_weakness
        return lam * self.weakness_sum

    def penalty(self, form: BoundaryForm | None = None, strength: float | None = None) -> np.ndarray:
        form = self.settings.form if form is None else form
        strength = self.settings.strength if strength is None else strength
        return strength * self.unit_penalty(form)
