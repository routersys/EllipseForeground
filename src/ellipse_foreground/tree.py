from dataclasses import dataclass

import numpy as np

from ellipse_foreground.grid import edge_distances, grid_edges


@dataclass(frozen=True)
class Tree:
    leaf_count: int
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray
    height: np.ndarray
    area: np.ndarray

    @property
    def node_count(self) -> int:
        return len(self.parent)

    @property
    def root(self) -> int:
        return self.node_count - 1


class DisjointSets:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def merge(self, larger_root: int, smaller_root: int) -> None:
        self.parent[smaller_root] = larger_root
        self.size[larger_root] += self.size[smaller_root]


def build_tree(work: np.ndarray) -> Tree:
    leaf_count = work.shape[0] * work.shape[1]
    first, second = grid_edges(work.shape[:2])
    distances = edge_distances(work, first, second)
    order = np.argsort(distances, kind="stable")
    node_count = 2 * leaf_count - 1
    left = np.full(node_count, -1, dtype=np.int64)
    right = np.full(node_count, -1, dtype=np.int64)
    parent = np.arange(node_count)
    height = np.zeros(node_count)
    area = np.ones(node_count)
    sets = DisjointSets(leaf_count)
    node_of_root = list(range(leaf_count))
    next_node = leaf_count
    for a, b, distance in zip(first[order].tolist(), second[order].tolist(), distances[order].tolist(), strict=True):
        root_a, root_b = sets.find(a), sets.find(b)
        if root_a == root_b:
            continue
        if sets.size[root_a] < sets.size[root_b]:
            root_a, root_b = root_b, root_a
        node_a, node_b = node_of_root[root_a], node_of_root[root_b]
        left[next_node], right[next_node] = node_a, node_b
        parent[node_a] = parent[node_b] = next_node
        height[next_node] = distance
        area[next_node] = area[node_a] + area[node_b]
        sets.merge(root_a, root_b)
        node_of_root[root_a] = next_node
        next_node += 1
    return Tree(leaf_count, left, right, parent, height, area)


def accumulate(tree: Tree, own: np.ndarray) -> np.ndarray:
    total = own.astype(np.float64)
    for node in range(tree.leaf_count, tree.node_count):
        total[node] += total[tree.left[node]] + total[tree.right[node]]
    return total


def leaf_sums(tree: Tree, leaf_values: np.ndarray) -> np.ndarray:
    own = np.zeros(tree.node_count)
    own[: tree.leaf_count] = leaf_values.ravel()
    return accumulate(tree, own)
