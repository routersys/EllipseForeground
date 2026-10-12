import numpy as np

from ellipse_foreground.boundary import BoundaryTerms, boundary_sums, lowest_common_ancestors
from ellipse_foreground.grid import edge_distances, grid_edges
from ellipse_foreground.settings import BoundaryForm, BoundarySettings
from ellipse_foreground.tree import build_tree
from tests.helpers import leaves_of, random_work


def ancestors_of(tree, node: int) -> list[int]:
    chain = [node]
    while chain[-1] != tree.root:
        chain.append(int(tree.parent[chain[-1]]))
    return chain


def naive_lowest_common_ancestor(tree, first: int, second: int) -> int:
    second_chain = set(ancestors_of(tree, second))
    return next(node for node in ancestors_of(tree, first) if node in second_chain)


def test_lowest_common_ancestors_match_walking_up():
    work = random_work(5, 6, seed=2)
    tree = build_tree(work)
    first, second = grid_edges(work.shape[:2])
    found = lowest_common_ancestors(tree.parent, first, second)
    expected = [naive_lowest_common_ancestor(tree, a, b) for a, b in zip(first.tolist(), second.tolist(), strict=True)]
    assert found.tolist() == expected


def test_lowest_common_ancestor_of_a_pair_with_itself_is_itself():
    tree = build_tree(random_work(3, 3, seed=1))
    nodes = np.arange(tree.node_count)
    assert np.array_equal(lowest_common_ancestors(tree.parent, nodes, nodes), nodes)


def naive_boundary_sum(tree, first, second, weights, node: int) -> float:
    inside = leaves_of(tree.left, tree.right, tree.leaf_count, node)
    return sum(
        weight
        for a, b, weight in zip(first.tolist(), second.tolist(), weights, strict=True)
        if (a in inside) != (b in inside)
    )


def test_boundary_sums_equal_weights_of_edges_leaving_each_region():
    work = random_work(4, 5, seed=7)
    tree = build_tree(work)
    first, second = grid_edges(work.shape[:2])
    weights = np.random.default_rng(3).random(len(first))
    ancestors = lowest_common_ancestors(tree.parent, first, second)
    sums = boundary_sums(tree, first, second, ancestors, weights)
    for node in range(tree.root):
        assert np.isclose(sums[node], naive_boundary_sum(tree, first, second, weights.tolist(), node))
    assert sums[tree.root] == 0.0


def test_mean_weakness_is_a_weighted_average_between_zero_and_one():
    work = random_work(5, 6, seed=8)
    terms = BoundaryTerms(build_tree(work), work, BoundarySettings())
    assert terms.mean_weakness.min() >= 0.0
    assert terms.mean_weakness.max() <= 1.0 + 1e-12


def test_merge_weakness_is_zero_at_the_root_and_uses_the_parent_height():
    work = random_work(4, 4, seed=9)
    tree = build_tree(work)
    terms = BoundaryTerms(tree, work, BoundarySettings())
    sigmas = BoundarySettings().sigmas
    expected = np.mean([np.exp(-(tree.height[tree.parent[0]] ** 2) / (2 * sigma * sigma)) for sigma in sigmas])
    assert np.isclose(terms.merge_weakness[0], expected)
    assert terms.merge_weakness[tree.root] == 0.0


def test_penalty_scales_linearly_with_strength():
    work = random_work(4, 4, seed=10)
    terms = BoundaryTerms(build_tree(work), work, BoundarySettings())
    for form in BoundaryForm:
        assert np.allclose(terms.penalty(form, 3.0), 3.0 * terms.penalty(form, 1.0))


def test_forms_use_the_documented_resolution_factor():
    work = random_work(4, 8, seed=11)
    tree = build_tree(work)
    settings = BoundarySettings()
    terms = BoundaryTerms(tree, work, settings)
    resolution = work.shape[1] / settings.reference_width
    shape = settings.shape_factor * np.sqrt(tree.area)
    assert np.allclose(
        terms.unit_penalty(BoundaryForm.MEAN_WEAKNESS), resolution * settings.mean_lambda * shape * terms.mean_weakness
    )
    assert np.allclose(terms.unit_penalty(BoundaryForm.WEAKNESS_SUM), settings.mean_lambda * terms.weakness_sum)


def test_weakness_sum_counts_boundary_edges_for_a_zero_distance_image():
    work = np.zeros((3, 4, 3), dtype=np.float32)
    tree = build_tree(work)
    terms = BoundaryTerms(tree, work, BoundarySettings())
    first, second = grid_edges(work.shape[:2])
    assert np.allclose(edge_distances(work, first, second), 0.0)
    for node in range(tree.root):
        inside = leaves_of(tree.left, tree.right, tree.leaf_count, node)
        leaving = sum((a in inside) != (b in inside) for a, b in zip(first.tolist(), second.tolist(), strict=True))
        assert np.isclose(terms.weakness_sum[node], leaving)
        assert np.isclose(terms.mean_weakness[node], 1.0)
