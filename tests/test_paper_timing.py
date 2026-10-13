import shutil
import subprocess

import numpy as np
import pytest

from ellipse_foreground import Analysis
from ellipse_foreground.settings import Settings, WorkSettings
from paper.timing import (
    build_instance,
    build_solver,
    pixel_range,
    read_graph_cut_input,
    run_boykov_kolmogorov,
    run_solver,
)
from tests.helpers import noisy_scene

PROBLEMS = 3


def dotnet_supports_the_solvers() -> bool:
    if shutil.which("dotnet") is None:
        return False
    sdks = subprocess.run(["dotnet", "--list-sdks"], capture_output=True, text=True).stdout
    return any(line.startswith("10.") for line in sdks.splitlines())


@pytest.fixture(scope="module")
def instance_inputs():
    image, source, _ = noisy_scene(72, 128, seed=4)
    analysis = Analysis(image, source, Settings(work=WorkSettings(width=48)))
    rng = np.random.default_rng(5)
    unaries = [rng.normal(0.0, 1.0, analysis.shape).astype(np.float32) for _ in range(PROBLEMS)]
    return analysis, build_instance(analysis, unaries)


def test_tree_input_has_the_documented_layout(instance_inputs):
    analysis, instance = instance_inputs
    nodes, leaves = analysis.tree.node_count, analysis.tree.leaf_count
    header = np.frombuffer(instance.tree_input[:12], dtype=np.int32)
    assert header.tolist() == [leaves, nodes, PROBLEMS]
    assert len(instance.tree_input) == 12 + 4 * (2 * nodes + nodes + PROBLEMS * leaves)


def test_graph_cut_input_has_the_documented_layout(instance_inputs):
    analysis, instance = instance_inputs
    height, width = analysis.shape
    edges = height * (width - 1) + (height - 1) * width
    header = np.frombuffer(instance.graph_cut_input[:12], dtype=np.int32)
    assert header.tolist() == [width, height, PROBLEMS]
    assert len(instance.graph_cut_input) == 12 + 4 * (edges + PROBLEMS * height * width)
    assert instance.pixel_count == height * width


def test_reference_labels_have_one_row_per_problem(instance_inputs):
    analysis, instance = instance_inputs
    assert instance.tree_labels.shape == (PROBLEMS, analysis.shape[0] * analysis.shape[1])
    assert instance.graph_cut_labels.shape == instance.tree_labels.shape


def test_graph_cut_input_is_read_back(instance_inputs):
    analysis, instance = instance_inputs
    height, width = analysis.shape
    read_width, read_height, weights, unaries = read_graph_cut_input(instance.graph_cut_input)
    assert (read_width, read_height) == (width, height)
    assert weights.shape == (height * (width - 1) + (height - 1) * width,)
    assert unaries.shape == (PROBLEMS, height * width)


def test_pixel_range_collapses_equal_values():
    assert pixel_range([32768, 36864]) == "3.3万〜3.7万"
    assert pixel_range([32768, 32768]) == "3.3万"


@pytest.mark.skipif(not dotnet_supports_the_solvers(), reason=".NET 10のSDKがありません")
def test_compiled_solvers_agree_with_the_reference_implementation(instance_inputs):
    _, instance = instance_inputs
    _, tree_agreement = run_solver(build_solver("TreeDp"), instance.tree_input, instance.tree_labels, 1)
    _, graph_agreement = run_solver(build_solver("GraphCut"), instance.graph_cut_input, instance.graph_cut_labels, 1)
    assert tree_agreement == 1.0
    assert graph_agreement > 0.99


def test_boykov_kolmogorov_agrees_with_the_reference_implementation(instance_inputs):
    library = pytest.importorskip("maxflow")
    _, instance = instance_inputs
    milliseconds, agreement = run_boykov_kolmogorov(library, instance.graph_cut_input, instance.graph_cut_labels, 2)
    assert len(milliseconds) == 2
    assert agreement > 0.99
