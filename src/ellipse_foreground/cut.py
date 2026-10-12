import numpy as np

from ellipse_foreground.tree import Tree


def optimal_cut(tree: Tree, scores: np.ndarray) -> np.ndarray:
    leaf_count, node_count = tree.leaf_count, tree.node_count
    left, right, parent = tree.left_nodes, tree.right_nodes, tree.parent_nodes
    values = scores.tolist()
    leaf_best = np.zeros(node_count)
    leaf_best[:leaf_count] = np.maximum(0.0, scores[:leaf_count])
    best = leaf_best.tolist()
    for node in range(leaf_count, node_count):
        best[node] = max(values[node], best[left[node]] + best[right[node]])
    covered = [False] * node_count
    pending = [tree.root]
    while pending:
        node = pending.pop()
        if node < leaf_count:
            covered[node] = values[node] > 0
        elif values[node] > 0 and values[node] >= best[left[node]] + best[right[node]]:
            covered[node] = True
        else:
            pending.extend((left[node], right[node]))
    for node in range(node_count - 2, -1, -1):
        if covered[parent[node]]:
            covered[node] = True
    return np.array(covered[:leaf_count], dtype=bool)
