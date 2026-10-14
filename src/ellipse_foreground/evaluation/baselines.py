import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.results import append_records, finished_scenes
from ellipse_foreground.evaluation.runner import GROUPS, GroupSpec, photo_group
from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.evaluation.trials import draw_trials
from ellipse_foreground.geometry import Ellipse
from ellipse_foreground.imaging import halve

GRABCUT_ITERATIONS = 5
BASELINE_DIRECTORY = "baselines"
COVERAGE_DIRECTORY = "coverage"
REPOSITORY = Path(__file__).resolve().parents[3]
OPEN_END = 10**6


class BoxPredictor(Protocol):
    def set_image(self, image: np.ndarray) -> None: ...

    def predict(self, *, box: np.ndarray, multimask_output: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def ellipse_box(ellipse: Ellipse, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape
    return (
        max(0.0, ellipse.center_x - ellipse.radius_x),
        max(0.0, ellipse.center_y - ellipse.radius_y),
        min(float(width), ellipse.center_x + ellipse.radius_x),
        min(float(height), ellipse.center_y + ellipse.radius_y),
    )


def grabcut_alpha(
    image: np.ndarray, box: tuple[float, float, float, float], iterations: int
) -> tuple[np.ndarray, bool]:
    import cv2

    pixels = np.ascontiguousarray(np.rint(image * 255.0).astype(np.uint8)[..., ::-1])
    left, top = int(np.floor(box[0])), int(np.floor(box[1]))
    width, height = max(1, int(np.ceil(box[2])) - left), max(1, int(np.ceil(box[3])) - top)
    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    background, foreground = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(
            pixels, labels, (left, top, width, height), background, foreground, iterations, cv2.GC_INIT_WITH_RECT
        )
    except cv2.error:
        rectangle = np.zeros(image.shape[:2], dtype=np.float32)
        rectangle[top : top + height, left : left + width] = 1.0
        return rectangle, False
    return ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.float32), True


def sam_alphas(
    predictor: BoxPredictor, composite: np.ndarray, boxes: Sequence[tuple[float, float, float, float]], scale: float
) -> list[dict[str, np.ndarray]]:
    predictor.set_image(composite)
    outputs = []
    for box in boxes:
        scaled = np.array(box, dtype=np.float64) * scale
        single, _, _ = predictor.predict(box=scaled, multimask_output=False)
        candidates, qualities, _ = predictor.predict(box=scaled, multimask_output=True)
        outputs.append(
            {
                "sam_single": halve(single[0].astype(np.float32)),
                "sam_best": halve(candidates[int(np.argmax(qualities))].astype(np.float32)),
            }
        )
    return outputs


def informative_trials(scene: Scene, count: int) -> list[tuple[int, Ellipse]]:
    context = SceneContext(scene)
    return [(index, ellipse) for index, ellipse in enumerate(draw_trials(scene, count)) if context.informative(ellipse)]


def box_coverage(scene: Scene, ellipse: Ellipse) -> float:
    truth = scene.alpha > 0.5
    left, top, right, bottom = ellipse_box(ellipse, truth.shape)
    box = np.zeros_like(truth)
    box[int(np.floor(top)) : int(np.ceil(bottom)), int(np.floor(left)) : int(np.ceil(right))] = True
    return float((truth & box).sum() / truth.sum())


def coverage_records(scene: Scene, count: int) -> list[dict]:
    return [
        {"scene": scene.index, "name": scene.name, "trial": index, "box_coverage": box_coverage(scene, ellipse)}
        for index, ellipse in informative_trials(scene, count)
    ]


def baseline_records(scene: Scene, count: int, predictor: BoxPredictor | None) -> list[dict]:
    context = SceneContext(scene)
    trials = informative_trials(scene, count)
    shape = scene.image.shape[:2]
    boxes = [ellipse_box(ellipse, shape) for _, ellipse in trials]
    grabcut = [grabcut_alpha(scene.image, box, GRABCUT_ITERATIONS) for box in boxes]
    alphas = [{"grabcut": alpha} for alpha, _ in grabcut]
    if predictor is not None:
        scale = scene.texture_source.shape[1] / scene.image.shape[1]
        for found, extra in zip(alphas, sam_alphas(predictor, scene.texture_source, boxes, scale), strict=True):
            found.update(extra)
    records = []
    for (index, _), found, (_, converged) in zip(trials, alphas, grabcut, strict=True):
        record = {"scene": scene.index, "name": scene.name, "trial": index, "grabcut_converged": converged}
        for name, alpha in found.items():
            score = context.scorer.score(alpha)
            record[name] = {"iou": score.iou, "mae": score.mae}
        records.append(record)
    return records


def run_baselines(
    spec: GroupSpec,
    data: Path,
    results: Path,
    selection: range,
    predictor: BoxPredictor | None,
    coverage: bool = False,
) -> int:
    directory = results / (COVERAGE_DIRECTORY if coverage else BASELINE_DIRECTORY)
    processed = 0
    for scene in spec.scenes(data, selection):
        if scene.index in finished_scenes(directory, spec.name):
            continue
        suffix = "" if selection.start == 0 and selection.stop >= OPEN_END else f".{selection.start}-{selection.stop}"
        records = coverage_records(scene, spec.trials) if coverage else baseline_records(scene, spec.trials, predictor)
        append_records(directory / f"{spec.name}{suffix}.jsonl", records)
        processed += 1
        print(f"{spec.name} scene {scene.index} {scene.name}", flush=True)
    return processed


def load_predictor(checkpoint: Path, model_type: str = "vit_b") -> BoxPredictor:
    from segment_anything import SamPredictor, sam_model_registry

    return SamPredictor(sam_model_registry[model_type](checkpoint=str(checkpoint)))


def parse(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ellipse_foreground.evaluation.baselines")
    parser.add_argument("group", choices=sorted([*GROUPS, "photo"]))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=OPEN_END)
    parser.add_argument("--data", type=Path, default=REPOSITORY / "data")
    parser.add_argument("--results", type=Path, default=REPOSITORY / "results")
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--coverage", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    options = parse(arguments)
    if options.group == "photo":
        if options.photos is None:
            print("photo には --photos が必要です。", file=sys.stderr)
            return 2
        spec = photo_group(options.photos)
    else:
        spec = GROUPS[options.group]
    predictor = None if options.checkpoint is None else load_predictor(options.checkpoint, options.model_type)
    run_baselines(spec, options.data, options.results, range(options.start, options.end), predictor, options.coverage)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
