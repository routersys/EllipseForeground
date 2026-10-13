import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ellipse_foreground import Analysis, Mix
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.graph_cut import graph_cut_labels, pairwise_weights
from ellipse_foreground.grid import grid_edges
from ellipse_foreground.settings import DEFAULT_SETTINGS
from ellipse_foreground.weights import weights_for
from paper.scenes import find_scene

SOLVERS = Path(__file__).resolve().parent / "solvers"
TIMED_SCENES = (("sprite1", 4), ("sprite1", 5), ("public100", 10), ("public100", 20), ("public100", 30))
PROBLEMS_PER_SCENE = 8
TREE_REPETITIONS = 5
GRAPH_CUT_REPETITIONS = 3
ELAPSED = re.compile(r"in ([0-9.]+) ms")
TEN_THOUSAND = 10000


@dataclass(frozen=True)
class Instance:
    pixel_count: int
    tree_input: bytes
    graph_cut_input: bytes
    tree_labels: np.ndarray
    graph_cut_labels: np.ndarray


def tree_input(analysis: Analysis, unaries: list[np.ndarray]) -> bytes:
    tree = analysis.tree
    left = np.where(tree.left < 0, 0, tree.left).astype(np.int32)
    right = np.where(tree.right < 0, 0, tree.right).astype(np.int32)
    penalty = analysis.boundary.penalty().astype(np.float32)
    header = np.array([tree.leaf_count, tree.node_count, len(unaries)], dtype=np.int32)
    body = [header, left, right, penalty, *(unary.ravel().astype(np.float32) for unary in unaries)]
    return b"".join(array.tobytes() for array in body)


def graph_cut_input(analysis: Analysis, unaries: list[np.ndarray]) -> bytes:
    height, width = analysis.shape
    settings = DEFAULT_SETTINGS.graph_cut
    first, second = grid_edges(analysis.shape)
    pairwise = pairwise_weights(analysis.work, first, second, settings.sigmas).astype(np.float32)
    header = np.array([width, height, len(unaries)], dtype=np.int32)
    scaled = [(unary.ravel() / settings.smoothness).astype(np.float32) for unary in unaries]
    return b"".join(array.tobytes() for array in [header, pairwise, *scaled])


def build_instance(analysis: Analysis, unaries: list[np.ndarray]) -> Instance:
    settings = DEFAULT_SETTINGS.graph_cut
    return Instance(
        analysis.shape[0] * analysis.shape[1],
        tree_input(analysis, unaries),
        graph_cut_input(analysis, unaries),
        np.stack([analysis.cut(unary).ravel() for unary in unaries]),
        np.stack([graph_cut_labels(analysis.work, unary, settings).ravel() for unary in unaries]),
    )


def scene_instance(data_root: Path, key: str, index: int) -> Instance:
    context = SceneContext(find_scene(data_root, key, index))
    unaries = []
    for trial in context.trials(PROBLEMS_PER_SCENE):
        weights = weights_for(Mix.FIXED_MIX, trial.overlaps, context.settings.weights)
        unaries.append(trial.cues.scores(weights))
    return build_instance(context.analysis, unaries)


def build_solver(name: str) -> Path:
    project = SOLVERS / name
    subprocess.run(["dotnet", "build", "-c", "Release", "-nologo", "-v", "quiet"], cwd=project, check=True)
    return next((project / "bin" / "Release").rglob(f"{name}.dll"))


def run_solver(
    library: Path, payload: bytes, expected_labels: np.ndarray, repetitions: int
) -> tuple[list[float], float]:
    problems = len(expected_labels)
    with tempfile.TemporaryDirectory() as directory:
        source, destination = Path(directory) / "input.bin", Path(directory) / "output.bin"
        source.write_bytes(payload)
        milliseconds = []
        for _ in range(repetitions):
            completed = subprocess.run(
                ["dotnet", str(library), str(source), str(destination)], check=True, capture_output=True, text=True
            )
            milliseconds.append(float(ELAPSED.search(completed.stdout).group(1)) / problems)
        produced = np.frombuffer(destination.read_bytes(), dtype=np.uint8).reshape(expected_labels.shape)
    return milliseconds, float((produced == expected_labels).mean())


def read_graph_cut_input(payload: bytes) -> tuple[int, int, np.ndarray, np.ndarray]:
    width, height, count = np.frombuffer(payload[:12], dtype=np.int32).tolist()
    body = np.frombuffer(payload, dtype=np.float32, offset=12)
    edges = height * (width - 1) + (height - 1) * width
    return (
        width,
        height,
        body[:edges].astype(np.float64),
        body[edges:].reshape(count, height * width).astype(np.float64),
    )


def maxflow_library():
    try:
        import maxflow
    except ImportError:
        return None
    return maxflow


def run_boykov_kolmogorov(
    library, payload: bytes, expected_labels: np.ndarray, repetitions: int
) -> tuple[list[float], float]:
    width, height, weights, unaries = read_graph_cut_input(payload)
    first, second = grid_edges((height, width))
    problems = len(unaries)
    produced = np.zeros(expected_labels.shape, dtype=np.uint8)
    milliseconds = []
    for repetition in range(repetitions + 1):
        elapsed = 0.0
        for problem, unary in enumerate(unaries):
            graph = library.Graph[float](width * height, len(weights))
            nodes = graph.add_grid_nodes((height, width))
            graph.add_edges(first, second, weights, weights)
            graph.add_grid_tedges(
                nodes, np.maximum(unary, 0.0).reshape(height, width), np.maximum(-unary, 0.0).reshape(height, width)
            )
            start = time.perf_counter()
            graph.maxflow()
            elapsed += time.perf_counter() - start
            produced[problem] = ~graph.get_grid_segments(nodes).ravel()
        if repetition > 0:
            milliseconds.append(elapsed * 1000.0 / problems)
    return milliseconds, float((produced == expected_labels).mean())


def processor_name() -> str:
    if platform.system() == "Windows":
        query = "(Get-CimInstance Win32_Processor | Select-Object -First 1 | ForEach-Object { $_.Name + '|' + $_.NumberOfCores })"
        completed = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True)
        name, _, cores = completed.stdout.strip().partition("|")
        if name:
            compact = re.sub(r"\s+", " ", name)
            return f"{compact}（{cores}コア）"
    return platform.processor() or platform.machine()


def runtime_version() -> str:
    completed = subprocess.run(["dotnet", "--version"], capture_output=True, text=True, check=True)
    return completed.stdout.strip().split(".")[0]


def pixel_range(counts: list[int]) -> str:
    low, high = min(counts) / TEN_THOUSAND, max(counts) / TEN_THOUSAND
    return f"{low:.1f}万" if f"{low:.1f}" == f"{high:.1f}" else f"{low:.1f}万〜{high:.1f}万"


def measure(data_root: Path, output: Path) -> dict:
    tree_library, graph_library = build_solver("TreeDp"), build_solver("GraphCut")
    tree_samples: list[float] = []
    graph_samples: list[float] = []
    flow_samples: list[float] = []
    flow_agreement: list[float] = []
    flow_library = maxflow_library()
    tree_agreement: list[float] = []
    graph_agreement: list[float] = []
    counts = []
    problems = 0
    for key, index in TIMED_SCENES:
        instance = scene_instance(data_root, key, index)
        counts.append(instance.pixel_count)
        problems += len(instance.tree_labels)
        samples, agreement = run_solver(tree_library, instance.tree_input, instance.tree_labels, TREE_REPETITIONS)
        tree_samples += samples
        tree_agreement.append(agreement)
        samples, agreement = run_solver(
            graph_library, instance.graph_cut_input, instance.graph_cut_labels, GRAPH_CUT_REPETITIONS
        )
        graph_samples += samples
        graph_agreement.append(agreement)
        if flow_library:
            samples, agreement = run_boykov_kolmogorov(
                flow_library, instance.graph_cut_input, instance.graph_cut_labels, GRAPH_CUT_REPETITIONS
            )
            flow_samples += samples
            flow_agreement.append(agreement)
    summary = {
        "tree_ms": statistics.median(tree_samples),
        "graph_cut_ms": statistics.median(graph_samples),
        "pixel_count": pixel_range(counts),
        "environment": f"{processor_name()}、.NET {runtime_version()}",
        "instances": len(TIMED_SCENES),
        "problems": problems,
        "tree_agreement": min(tree_agreement),
        "graph_cut_agreement": min(graph_agreement),
        "tree_samples": tree_samples,
        "graph_cut_samples": graph_samples,
    }
    if flow_library:
        summary |= {
            "boykov_kolmogorov_ms": statistics.median(flow_samples),
            "boykov_kolmogorov_agreement": min(flow_agreement),
            "boykov_kolmogorov_library": f"PyMaxflow {importlib.metadata.version('PyMaxflow')}",
            "boykov_kolmogorov_samples": flow_samples,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
