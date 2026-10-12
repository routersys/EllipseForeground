import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from ellipse_foreground.grid import edge_distances, grid_edges
from ellipse_foreground.tree import build_tree, leaf_sums
from tests.helpers import leaves_of, random_work


def tree_partition(tree, threshold: float) -> set[frozenset[int]]:
    maximal = [
        node
        for node in range(tree.node_count)
        if tree.height[node] <= threshold and (node == tree.root or tree.height[tree.parent[node]] > threshold)
    ]
    return {leaves_of(tree.left, tree.right, tree.leaf_count, node) for node in maximal}


def graph_partition(work: np.ndarray, threshold: float) -> set[frozenset[int]]:
    first, second = grid_edges(work.shape[:2])
    distances = edge_distances(work, first, second)
    kept = distances <= threshold
    count = work.shape[0] * work.shape[1]
    graph = coo_array((np.ones(kept.sum()), (first[kept], second[kept])), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    return {frozenset(np.flatnonzero(labels == label).tolist()) for label in np.unique(labels)}


def test_node_counts_and_root_area():
    tree = build_tree(random_work(4, 5))
    assert tree.leaf_count == 20
    assert tree.node_count == 39
    assert tree.area[tree.root] == 20


def test_children_point_back_to_their_parent():
    tree = build_tree(random_work(4, 5, seed=3))
    for node in range(tree.leaf_count, tree.node_count):
        assert tree.parent[tree.left[node]] == node
        assert tree.parent[tree.right[node]] == node
        assert tree.area[node] == tree.area[tree.left[node]] + tree.area[tree.right[node]]


def test_heights_do_not_decrease_toward_the_root():
    tree = build_tree(random_work(5, 6, seed=4))
    for node in range(tree.node_count - 1):
        assert tree.height[tree.parent[node]] >= tree.height[node]


def test_cutting_at_a_height_gives_the_connected_components():
    work = random_work(4, 5, seed=5)
    tree = build_tree(work)
    first, second = grid_edges(work.shape[:2])
    for threshold in np.unique(edge_distances(work, first, second)):
        assert tree_partition(tree, threshold) == graph_partition(work, threshold)


def test_leaf_sums_accumulate_subtrees():
    tree = build_tree(random_work(3, 4, seed=6))
    values = np.random.default_rng(0).random(12)
    sums = leaf_sums(tree, values)
    for node in range(tree.node_count):
        leaves = leaves_of(tree.left, tree.right, tree.leaf_count, node)
        assert np.isclose(sums[node], sum(values[leaf] for leaf in leaves))
