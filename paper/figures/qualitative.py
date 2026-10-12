from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ellipse_foreground import Mix
from ellipse_foreground.evaluation.experiments.context import SceneContext, Trial
from ellipse_foreground.evaluation.trials import draw_trials
from ellipse_foreground.rendering import blurred_composite
from ellipse_foreground.weights import weights_for
from paper.figures.style import ellipse_outline, hide_frame, save
from paper.results import Results
from paper.scenes import find_scene

RENDER_RADIUS = 12.0
RENDER_EPSILON = 1e-4
CROP_PADDING = 0.1
COLUMN_METHODS = ("ellipse", "prior", "tex", "mix", "adaptive_c", "graph")
COLUMN_TITLES = [
    "入力と楕円",
    "楕円をそのまま",
    "事前分布のみ",
    "テクスチャのみ",
    "固定配合",
    "適応（色）",
    "グラフカット",
    "正解のα",
]
MIXES = {"prior": Mix.PRIOR_ONLY, "tex": Mix.TEXTURE_ONLY, "mix": Mix.FIXED_MIX, "adaptive_c": Mix.ADAPTIVE_COLOR}
LOWER_EIGHTH = 8


@dataclass(frozen=True)
class Pick:
    title: str
    key: str
    scene: int
    trial: int


def representative(results: Results, key: str, kind: str, scene: int | None = None) -> tuple[int, int]:
    records = [record for record in results.trials(key) if scene is None or record["scene"] == scene]
    values = np.array([record["tree"]["adaptive_c"]["iou"] for record in records])
    index = (
        int(np.argmin(np.abs(values - np.median(values))))
        if kind == "median"
        else int(np.argsort(values)[len(values) // LOWER_EIGHTH])
    )
    return records[index]["scene"], records[index]["trial"]


def qualitative_picks(results: Results) -> list[Pick]:
    picks = []
    for title, key, scene, kind in (
        ("立ち絵1・星雲・転写100%", "dose100", None, "median"),
        ("立ち絵1・夜景・転写100%", "others", 5, "median"),
        ("立ち絵2・8背景から", "sprite2", None, "median"),
        ("公開素材・転写0%", "public0", None, "median"),
        ("公開素材・転写100%", "public100", None, "median"),
        ("公開素材・転写100%（IoUの下位1/8）", "public100", None, "low"),
    ):
        picked_scene, picked_trial = representative(results, key, kind, scene)
        picks.append(Pick(title, key, picked_scene, picked_trial))
    return picks


def finished_alphas(context: SceneContext, trial: Trial) -> dict[str, np.ndarray]:
    analysis = context.analysis
    settings = context.settings
    alphas = {"ellipse": analysis.finish(trial.regions.inside)}
    scores = {}
    for name, mix in MIXES.items():
        scores[name] = trial.cues.scores(weights_for(mix, trial.overlaps, settings.weights))
        alphas[name] = analysis.finish(analysis.cut(scores[name]))
    alphas["graph"] = analysis.finish(analysis.graph_cut(scores["adaptive_c"]))
    return alphas


def error_picture(alpha: np.ndarray, truth: np.ndarray) -> np.ndarray:
    mask = alpha > 0.5
    picture = np.zeros((*truth.shape, 3))
    picture[mask & truth] = [1, 1, 1]
    picture[mask & ~truth] = [0.95, 0.2, 0.2]
    picture[~mask & truth] = [0.25, 0.5, 1.0]
    return picture


def crop_box(truth: np.ndarray, ellipse, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    rows, columns = np.nonzero(truth)
    left = min(columns.min(), ellipse.center_x - ellipse.radius_x)
    right = max(columns.max() + 1, ellipse.center_x + ellipse.radius_x)
    top = min(rows.min(), ellipse.center_y - ellipse.radius_y)
    bottom = max(rows.max() + 1, ellipse.center_y + ellipse.radius_y)
    pad_x, pad_y = CROP_PADDING * (right - left), CROP_PADDING * (bottom - top)
    return (
        int(max(0, left - pad_x)),
        int(min(width, right + pad_x)),
        int(max(0, top - pad_y)),
        int(min(height, bottom + pad_y)),
    )


def prepare(data_root: Path, results: Results, picks: list[Pick]) -> list[tuple[Pick, SceneContext, Trial, dict]]:
    prepared = []
    for pick in picks:
        scene = find_scene(data_root, pick.key, pick.scene)
        context = SceneContext(scene)
        ellipse = draw_trials(scene, pick.trial + 1)[pick.trial]
        trial = context.trial(pick.trial, ellipse)
        prepared.append((pick, context, trial, finished_alphas(context, trial)))
    return prepared


def comparison_figure(directory: Path, data_root: Path, results: Results) -> list[Pick]:
    picks = qualitative_picks(results)
    prepared = prepare(data_root, results, picks)
    boxes = []
    for _, context, trial, _ in prepared:
        truth = context.scene.alpha > 0.5
        boxes.append(crop_box(truth, trial.ellipse, truth.shape))
    ratios = [(box[3] - box[2]) / (box[1] - box[0]) for box in boxes]
    column_width = 0.9
    figure = plt.figure(figsize=(7.7, sum(column_width * ratio for ratio in ratios) + 0.7))
    grid = figure.add_gridspec(
        len(prepared), 8, height_ratios=ratios, hspace=0.12, wspace=0.03, left=0.0, right=1.0, top=0.955, bottom=0.0
    )
    for row, ((pick, context, trial, alphas), box) in enumerate(zip(prepared, boxes, strict=True)):
        scene = context.scene
        truth = scene.alpha > 0.5
        left, right, top, bottom = box
        for column in range(8):
            axis = figure.add_subplot(grid[row, column])
            hide_frame(axis)
            if column == 0:
                axis.imshow(np.clip(scene.image[top:bottom, left:right], 0, 1))
                xs, ys = ellipse_outline(
                    (trial.ellipse.center_x - left, trial.ellipse.center_y - top),
                    (trial.ellipse.radius_x, trial.ellipse.radius_y),
                )
                axis.plot(xs, ys, color="#facc15", lw=1.0)
                axis.set_xlim(0, right - left)
                axis.set_ylim(bottom - top, 0)
                axis.set_ylabel(pick.title.replace("・", "\n", 1), fontsize=6.3, rotation=90, labelpad=3)
            elif column == 7:
                axis.imshow(truth[top:bottom, left:right], cmap="gray", vmin=0, vmax=1)
            else:
                alpha = alphas[COLUMN_METHODS[column - 1]]
                axis.imshow(error_picture(alpha, truth)[top:bottom, left:right])
                mask = alpha > 0.5
                iou = (mask & truth).sum() / (mask | truth).sum()
                axis.text(
                    0.03,
                    0.97,
                    f"{iou:.2f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.8,
                    color="#111827",
                    bbox={"fc": "white", "ec": "none", "pad": 0.8, "alpha": 0.8},
                )
            if row == 0:
                axis.set_title(COLUMN_TITLES[column], fontsize=7.4, pad=2.5)
    save(figure, directory, "fig_qual", vector=False)
    return picks


RENDER_PICKS = (("立ち絵1・星雲・転写100%", "dose100", None), ("公開素材・転写0%", "public0", None))
RENDER_METHODS = ("ellipse", "prior", "tex", "adaptive_c")
RENDER_TITLES = ["基準描画（正解α）", "楕円をそのまま", "事前分布のみ", "テクスチャのみ", "適応（色）"]
RENDER_ROWS_FRACTION = 0.45
RENDER_MARGIN = 20
RENDER_AMPLIFICATION = 6


def render_figure(directory: Path, data_root: Path, results: Results) -> None:
    picks = [Pick(title, key, *representative(results, key, "median", scene)) for title, key, scene in RENDER_PICKS]
    prepared = prepare(data_root, results, picks)
    figure, axes = plt.subplots(4, 5, figsize=(7.7, 5.3))
    for index, (pick, context, _, alphas) in enumerate(prepared):
        scene = context.scene
        reference = blurred_composite(scene.image, scene.alpha, RENDER_RADIUS, RENDER_EPSILON)
        width = scene.alpha.shape[1]
        rows, columns = np.nonzero(scene.alpha > 0.5)
        left, right = max(0, columns.min() - RENDER_MARGIN), min(width, columns.max() + 1 + RENDER_MARGIN)
        top = max(0, rows.min() - RENDER_MARGIN)
        bottom = rows.min() + int(RENDER_ROWS_FRACTION * (rows.max() - rows.min()))
        for column in range(5):
            picture_axis, difference_axis = axes[2 * index, column], axes[2 * index + 1, column]
            if column == 0:
                output = reference
            else:
                output = blurred_composite(
                    scene.image, alphas[RENDER_METHODS[column - 1]], RENDER_RADIUS, RENDER_EPSILON
                )
            picture_axis.imshow(np.clip(output[top:bottom, left:right], 0, 1))
            difference = np.abs(output - reference).mean(axis=2)
            difference_axis.imshow(
                np.clip(difference[top:bottom, left:right] * RENDER_AMPLIFICATION, 0, 1), cmap="magma", vmin=0, vmax=1
            )
            hide_frame(picture_axis)
            hide_frame(difference_axis)
            if index == 0:
                picture_axis.set_title(RENDER_TITLES[column], fontsize=8)
            if column > 0:
                difference_axis.text(
                    0.03,
                    0.95,
                    f"MAE {difference.mean() * 255:.2f}",
                    transform=difference_axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                )
        axes[2 * index, 0].set_ylabel(pick.title.replace("・", "\n", 1), fontsize=7)
        axes[2 * index + 1, 0].set_ylabel("差（6倍）", fontsize=7)
    figure.subplots_adjust(wspace=0.03, hspace=0.04, left=0.07, right=0.995, top=0.95, bottom=0.0)
    save(figure, directory, "fig_render", vector=False)
