from itertools import combinations

import numpy as np

from ellipse_foreground.cut import optimal_cut
from ellipse_foreground.tree import build_tree
from tests.helpers import leaves_of, random_work


def ancestors_of(tree, node: int) -> set[int]:
    chain = {node}
    while node != tree.root:
        node = int(tree.parent[node])
        chain.add(node)
    return chain


def best_antichain_cover(tree, scores: np.ndarray) -> np.ndarray:
    nodes = range(tree.node_count)
    ancestors = {node: ancestors_of(tree, node) for node in nodes}
    best_value, best_cover = 0.0, np.zeros(tree.leaf_count, dtype=bool)
    for size in range(1, tree.node_count + 1):
        for chosen in combinations(nodes, size):
            if any(a != b and a in ancestors[b] for a in chosen for b in chosen):
                continue
            value = sum(scores[node] for node in chosen)
            if value > best_value:
                cover = np.zeros(tree.leaf_count, dtype=bool)
                for node in chosen:
                    cover[list(leaves_of(tree.left, tree.right, tree.leaf_count, node))] = True
                best_value, best_cover = value, cover
    return best_cover


def test_cut_matches_exhaustive_search_over_all_cuts():
    tree = build_tree(random_work(2, 3, seed=12))
    for seed in range(8):
        scores = np.random.default_rng(seed).standard_normal(tree.node_count)
        assert np.array_equal(optimal_cut(tree, scores), best_antichain_cover(tree, scores))


def test_all_negative_scores_select_nothing():
    tree = build_tree(random_work(3, 3, seed=13))
    assert not optimal_cut(tree, -np.ones(tree.node_count)).any()


def test_a_positive_root_with_negative_leaves_selects_everything():
    tree = build_tree(random_work(3, 3, seed=14))
    scores = -np.ones(tree.node_count)
    scores[tree.root] = 100.0
    assert optimal_cut(tree, scores).all()


def test_positive_leaves_without_internal_bonus_are_selected_individually():
    tree = build_tree(random_work(3, 3, seed=15))
    scores = np.full(tree.node_count, -100.0)
    scores[[0, 4, 8]] = 1.0
    expected = np.zeros(9, dtype=bool)
    expected[[0, 4, 8]] = True
    assert np.array_equal(optimal_cut(tree, scores), expected)
