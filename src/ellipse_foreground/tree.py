from dataclasses import dataclass
from functools import cached_property

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

    @cached_property
    def left_nodes(self) -> list[int]:
        return self.left.tolist()

    @cached_property
    def right_nodes(self) -> list[int]:
        return self.right.tolist()

    @cached_property
    def parent_nodes(self) -> list[int]:
        return self.parent.tolist()


def build_tree(work: np.ndarray) -> Tree:
    leaf_count = work.shape[0] * work.shape[1]
    first, second = grid_edges(work.shape[:2])
    distances = edge_distances(work, first, second)
    order = np.argsort(distances, kind="stable")
    node_count = 2 * leaf_count - 1
    left = [-1] * node_count
    right = [-1] * node_count
    parent = list(range(node_count))
    height = [0.0] * node_count
    area = [1.0] * node_count
    set_parent = list(range(leaf_count))
    set_size = [1] * leaf_count
    node_of_root = list(range(leaf_count))
    next_node = leaf_count
    for a, b, distance in zip(first[order].tolist(), second[order].tolist(), distances[order].tolist(), strict=True):
        root_a, root_b = a, b
        while set_parent[root_a] != root_a:
            root_a = set_parent[root_a]
        while set_parent[a] != root_a:
            set_parent[a], a = root_a, set_parent[a]
        while set_parent[root_b] != root_b:
            root_b = set_parent[root_b]
        while set_parent[b] != root_b:
            set_parent[b], b = root_b, set_parent[b]
        if root_a == root_b:
            continue
        if set_size[root_a] < set_size[root_b]:
            root_a, root_b = root_b, root_a
        node_a, node_b = node_of_root[root_a], node_of_root[root_b]
        left[next_node], right[next_node] = node_a, node_b
        parent[node_a] = parent[node_b] = next_node
        height[next_node] = distance
        area[next_node] = area[node_a] + area[node_b]
        set_parent[root_b] = root_a
        set_size[root_a] += set_size[root_b]
        node_of_root[root_a] = next_node
        next_node += 1
    return Tree(
        leaf_count,
        np.array(left, dtype=np.int64),
        np.array(right, dtype=np.int64),
        np.array(parent, dtype=np.int64),
        np.array(height),
        np.array(area),
    )


def accumulate(tree: Tree, own: np.ndarray) -> np.ndarray:
    total = own.astype(np.float64).tolist()
    left, right = tree.left_nodes, tree.right_nodes
    for node in range(tree.leaf_count, tree.node_count):
        total[node] += total[left[node]] + total[right[node]]
    return np.array(total)


def leaf_sums(tree: Tree, leaf_values: np.ndarray) -> np.ndarray:
    own = np.zeros(tree.node_count)
    own[: tree.leaf_count] = leaf_values.ravel()
    return accumulate(tree, own)
