import statistics
import time
from collections.abc import Callable

from ellipse_foreground import Analysis
from ellipse_foreground.cues.color import color_evidence
from ellipse_foreground.cues.patch import choose_patch_samples, patch_evidence
from ellipse_foreground.cues.texture import texture_evidence
from ellipse_foreground.settings import DEFAULT_SETTINGS, Settings
from ellipse_foreground.weights import CueWeights
from harness.cases import MAIN_SCENE, CaseContext

ROUNDS = 5


def measure(action: Callable[[], object]) -> tuple[float, float]:
    durations = []
    for _ in range(ROUNDS):
        started = time.perf_counter()
        action()
        durations.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(durations), min(durations)


def stage_timings(context: CaseContext, scene_name: str, settings: Settings) -> list[tuple[str, tuple[float, float]]]:
    scene = context.scenes[scene_name]
    trial = context.trial(scene_name, settings)
    analysis = context.scene_context(scene_name, settings).analysis
    regions = trial.regions
    core_samples, ring_samples = choose_patch_samples(regions.core, regions.ring, analysis.priorities, settings.patch)
    scores = trial.cues.scores(CueWeights(1.0, 0.25, 0.3))
    mask = analysis.cut(scores)
    return [
        (
            "解析の構築(縮小、木、境界項、テクスチャ)",
            measure(lambda: Analysis(scene.image, scene.texture_source, settings)),
        ),
        ("色の手がかり", measure(lambda: color_evidence(analysis.work, regions.core, regions.ring, settings.color))),
        (
            "テクスチャの手がかり",
            measure(lambda: texture_evidence(analysis.texture, regions.core, regions.ring, settings.texture)),
        ),
        (
            "パッチの手がかり",
            measure(
                lambda: patch_evidence(analysis.vectors, core_samples, ring_samples, analysis.shape, settings.patch)
            ),
        ),
        ("信頼度", measure(lambda: analysis.overlaps(regions))),
        ("木の最適な切断", measure(lambda: analysis.cut(scores))),
        ("グラフカット", measure(lambda: analysis.graph_cut(scores))),
        ("精緻化", measure(lambda: analysis.finish(mask))),
        ("全体", measure(lambda: analysis.extract(trial.ellipse))),
    ]


def benchmark(context: CaseContext) -> int:
    height, width = context.scenes[MAIN_SCENE].image.shape[:2]
    print(f"{MAIN_SCENE} {width}x{height} 縮小画像の幅 {DEFAULT_SETTINGS.work.width}(中央値 ms / 最小 ms、{ROUNDS} 回)")
    for label, (median, fastest) in stage_timings(context, MAIN_SCENE, DEFAULT_SETTINGS):
        print(f"  {label}: {median:.1f} / {fastest:.1f}")
    return 0
