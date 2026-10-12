from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ellipse_foreground import Mix
from ellipse_foreground.cues.texture import neighborhood_contrast
from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.metrics import RENDER_EPSILON, RENDER_RADIUS
from ellipse_foreground.evaluation.own import own_scenes
from ellipse_foreground.evaluation.trials import draw_trials
from ellipse_foreground.rendering import blurred_composite
from ellipse_foreground.tree import Tree
from ellipse_foreground.weights import weights_for
from paper.figures.style import ellipse_outline, hide_frame, save

OVERVIEW_SCENE = 1
OVERVIEW_TRIAL = 7
OVERVIEW_TRIALS = 10
MOSAIC_REGIONS = 4000
CUE_SCENES = (("星雲\n転写100%", 4), ("夜景\n転写100%", 5), ("教室\n転写100%", 11))
CUE_CROP = (120, 280, 400, 720)


def region_labels(tree: Tree, height: float) -> np.ndarray:
    top = np.arange(tree.node_count)
    for node in range(tree.node_count - 2, -1, -1):
        parent = tree.parent[node]
        top[node] = top[parent] if tree.height[parent] <= height else node
    return top[: tree.leaf_count]


def tree_mosaic(tree: Tree, work: np.ndarray, target: int = MOSAIC_REGIONS) -> tuple[np.ndarray, int]:
    low, high = 0.0, float(tree.height.max())
    labels = region_labels(tree, high)
    for _ in range(30):
        middle = 0.5 * (low + high)
        labels = region_labels(tree, middle)
        if len(np.unique(labels)) > target:
            low = middle
        else:
            high = middle
    colors = work.reshape(-1, 3)
    unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    mosaic = np.zeros_like(colors)
    for channel in range(3):
        mosaic[:, channel] = (np.bincount(inverse, weights=colors[:, channel]) / counts)[inverse]
    return mosaic.reshape(work.shape), len(unique)


def scene_trial(data_root: Path, group: str, scene_index: int, trial_index: int, count: int):
    scene = next(scene for scene in own_scenes(data_root, group) if scene.index == scene_index)
    context = SceneContext(scene)
    ellipse = draw_trials(scene, count)[trial_index]
    return context, context.trial(trial_index, ellipse)


def overview_figure(directory: Path, data_root: Path) -> None:
    context, trial = scene_trial(data_root, "sprite2", OVERVIEW_SCENE, OVERVIEW_TRIAL, OVERVIEW_TRIALS)
    analysis = context.analysis
    weights = weights_for(Mix.ADAPTIVE_COLOR, trial.overlaps, context.settings.weights)
    scores = trial.cues.scores(weights)
    mask = analysis.cut(scores)
    alpha = analysis.finish(mask)
    rendered = blurred_composite(analysis.image, alpha, RENDER_RADIUS, RENDER_EPSILON)
    mosaic, _ = tree_mosaic(analysis.tree, analysis.work)
    ellipse = analysis.work_ellipse(trial.ellipse)
    height, width = analysis.shape
    panels = [
        (analysis.work, "(a) 入力画像と指定した楕円", "image"),
        (mosaic, "(b) 単連結階層木（高さで切った区画）", "image"),
        (scores, "(c) 画素ごとの手がかりの和 u", "heat"),
        (mask.astype(np.float32), "(d) 木の上の最適切断", "mask"),
        (alpha, "(e) 縁の仕上げ後のα", "mask"),
        (rendered, "(f) ぼかし描画", "image"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(7.4, 3.35))
    for axis, (picture, title, kind) in zip(axes.flat, panels, strict=True):
        if kind == "image":
            axis.imshow(np.clip(picture, 0, 1))
        elif kind == "heat":
            axis.imshow(np.clip(picture, -2, 2), cmap="coolwarm", vmin=-2, vmax=2)
        else:
            axis.imshow(picture, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        if title.startswith(("(a)", "(c)")):
            xs, ys = ellipse_outline((ellipse.center_x, ellipse.center_y), (ellipse.radius_x, ellipse.radius_y))
            axis.plot(xs, ys, color="#facc15", lw=1.4)
            axis.set_xlim(0, width)
            axis.set_ylim(height, 0)
    figure.subplots_adjust(wspace=0.04, hspace=0.2, left=0.005, right=0.995, top=0.93, bottom=0.0)
    save(figure, directory, "fig_overview", vector=False)


def texture_cues(directory: Path, data_root: Path) -> None:
    top, bottom, left, right = CUE_CROP
    figure, axes = plt.subplots(len(CUE_SCENES), 5, figsize=(9.2, 3.5))
    titles = ["原寸画像の部分", "平坦領域", "最大絶対差", "テクスチャの証拠", "正解のα"]
    for row, (label, scene_index) in enumerate(CUE_SCENES):
        context, trial = scene_trial(data_root, "sprite1", scene_index, 0, 8)
        source = context.scene.texture_source
        contrast = neighborhood_contrast(source)
        flat = (contrast <= context.settings.texture.flat_threshold).astype(np.float32)
        evidence = trial.cues.texture
        ellipse = context.analysis.work_ellipse(trial.ellipse)
        axes[row, 0].imshow(source[top:bottom, left:right], interpolation="nearest")
        axes[row, 1].imshow(flat[top:bottom, left:right], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        capped = np.minimum(contrast, context.settings.texture.magnitude_cap)
        axes[row, 2].imshow(
            capped[top:bottom, left:right],
            cmap="magma",
            vmin=0,
            vmax=context.settings.texture.magnitude_cap,
            interpolation="nearest",
        )
        axes[row, 3].imshow(evidence, cmap="coolwarm", vmin=-1, vmax=1)
        xs, ys = ellipse_outline((ellipse.center_x, ellipse.center_y), (ellipse.radius_x, ellipse.radius_y))
        axes[row, 3].plot(xs, ys, color="#111827", lw=1.0)
        axes[row, 3].set_xlim(0, evidence.shape[1])
        axes[row, 3].set_ylim(evidence.shape[0], 0)
        axes[row, 4].imshow(context.work_alpha, cmap="gray", vmin=0, vmax=1)
        if row == 0:
            for column, title in enumerate(titles):
                axes[row, column].set_title(title, fontsize=9.5)
        axes[row, 0].set_ylabel(label, fontsize=8.5, labelpad=4)
        for axis in axes[row]:
            hide_frame(axis)
    figure.subplots_adjust(wspace=0.05, hspace=0.06, left=0.07, right=0.995, top=0.95, bottom=0.0)
    save(figure, directory, "fig_cues", vector=False)
