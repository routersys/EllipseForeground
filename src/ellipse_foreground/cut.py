import numpy as np

from ellipse_foreground.tree import Tree


def optimal_cut(tree: Tree, scores: np.ndarray) -> np.ndarray:
    leaf_count = tree.leaf_count
    best = np.zeros(tree.node_count)
    best[:leaf_count] = np.maximum(0.0, scores[:leaf_count])
    for node in range(leaf_count, tree.node_count):
        best[node] = max(scores[node], best[tree.left[node]] + best[tree.right[node]])
    selected = np.zeros(tree.node_count, dtype=bool)
    pending = [tree.root]
    while pending:
        node = pending.pop()
        if node < leaf_count:
            selected[node] = scores[node] > 0
        elif scores[node] > 0 and scores[node] >= best[tree.left[node]] + best[tree.right[node]]:
            selected[node] = True
        else:
            pending.extend((tree.left[node], tree.right[node]))
    covered = selected.copy()
    for node in range(tree.node_count - 2, -1, -1):
        if covered[tree.parent[node]]:
            covered[node] = True
    return covered[:leaf_count]
