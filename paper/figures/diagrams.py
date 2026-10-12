from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from ellipse_foreground.evaluation.backgrounds import background_paths, load_banner
from ellipse_foreground.evaluation.own import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    SPRITE1_DOSE_BACKGROUND,
    compose,
    own_sources,
)
from paper.figures.style import save

LEAVES = ("a", "b", "c", "d", "e", "f")
CHILDREN = {"n1": ("a", "b"), "n2": ("n1", "c"), "n3": ("d", "e"), "n4": ("n3", "f"), "r": ("n2", "n4")}
UNARY = {"a": 0.9, "b": 0.6, "c": -0.5, "d": -0.7, "e": 0.8, "f": -0.9}
PENALTY = {
    "a": 0.5,
    "b": 0.5,
    "c": 0.5,
    "d": 0.5,
    "e": 0.5,
    "f": 0.5,
    "n1": 0.4,
    "n2": 1.9,
    "n3": 1.6,
    "n4": 1.9,
    "r": 2.6,
}
POSITIONS = {
    "a": (0, 0),
    "b": (1, 0),
    "c": (2, 0),
    "d": (3, 0),
    "e": (4, 0),
    "f": (5, 0),
    "n1": (0.5, 1),
    "n2": (1.25, 2),
    "n3": (3.5, 1),
    "n4": (4.25, 2),
    "r": (2.75, 3.1),
}


def solve_example() -> tuple[dict[str, float], dict[str, float], list[str]]:
    totals = dict(UNARY)
    for node, (left, right) in CHILDREN.items():
        totals[node] = totals[left] + totals[right]
    scores = {node: totals[node] - PENALTY[node] for node in totals}
    best = {leaf: max(0.0, scores[leaf]) for leaf in LEAVES}
    keep = {}
    for node, (left, right) in CHILDREN.items():
        children = best[left] + best[right]
        best[node] = max(scores[node], children)
        keep[node] = scores[node] >= children and scores[node] > 0
    cut: list[str] = []

    def walk(node: str) -> None:
        if node in LEAVES:
            if scores[node] > 0:
                cut.append(node)
            return
        if keep[node]:
            cut.append(node)
            return
        for child in CHILDREN[node]:
            walk(child)

    walk("r")
    return scores, best, cut


def leaves_under(node: str) -> list[str]:
    if node in LEAVES:
        return [node]
    left, right = CHILDREN[node]
    return leaves_under(left) + leaves_under(right)


def dynamic_programming(directory: Path) -> None:
    scores, best, cut = solve_example()
    figure, axis = plt.subplots(figsize=(7.6, 3.6))
    axis.set_xlim(-0.8, 6.6)
    axis.set_ylim(-1.55, 3.75)
    axis.axis("off")
    for node, children in CHILDREN.items():
        for child in children:
            axis.plot(
                [POSITIONS[node][0], POSITIONS[child][0]],
                [POSITIONS[node][1], POSITIONS[child][1]],
                color="#6b7280",
                lw=1.0,
                zorder=1,
            )
    selected = set(cut)
    for node, (x, y) in POSITIONS.items():
        leaf = node in LEAVES
        width, height = (0.62, 0.52) if leaf else (1.12, 0.74)
        chosen = node in selected
        axis.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc="#dbeafe" if chosen else "white",
                ec="#1d4ed8" if chosen else "#111827",
                lw=1.8 if chosen else 0.9,
                zorder=3,
            )
        )
        if leaf:
            axis.text(x, y + 0.03, f"u={UNARY[node]:+.1f}", ha="center", va="center", fontsize=7.6, zorder=4)
            axis.text(
                x, y - 0.5, f"B={best[node]:.1f}", ha="center", va="center", fontsize=7.6, color="#1d4ed8", zorder=4
            )
        else:
            axis.text(x, y + 0.11, f"s={scores[node]:+.1f}", ha="center", va="center", fontsize=7.8, zorder=4)
            axis.text(
                x, y - 0.13, f"B={best[node]:.1f}", ha="center", va="center", fontsize=7.8, color="#1d4ed8", zorder=4
            )
    axis.text(-0.75, -1.12, "前景", fontsize=8.5, va="center")
    covered = {leaf for node in cut for leaf in leaves_under(node)}
    for leaf in LEAVES:
        x = POSITIONS[leaf][0]
        axis.add_patch(
            Rectangle((x - 0.3, -1.3), 0.6, 0.36, fc="#1d4ed8" if leaf in covered else "#e5e7eb", ec="#111827", lw=0.7)
        )
    axis.text(6.5, 3.62, r"$s(v)=\sum_{p\in v}u(p)-\pi(v)$", fontsize=9, ha="right", va="top")
    axis.text(6.5, 3.2, r"$B(v)=\max\{s(v),\,B(l)+B(r)\}$", fontsize=9, ha="right", va="top")
    save(figure, directory, "fig_dp")


def recovery(directory: Path, data_root: Path) -> None:
    alpha, color = own_sources(data_root, "sprite1")
    background = load_banner(background_paths(data_root)[SPRITE1_DOSE_BACKGROUND], CANVAS_WIDTH, CANVAS_HEIGHT)
    panels = [
        (alpha[..., None] * color, "(a) 背景を非表示にした出力 K"),
        (np.stack([alpha] * 3, axis=2), "(b) 復元したα"),
        (np.where(alpha[..., None] > 0.02, color, 0.0), "(c) 前景の色 F"),
        (compose(alpha, color, background, 0.0) / 255.0, "(d) 合成（転写0%）"),
        (compose(alpha, color, background, 1.0) / 255.0, "(e) 合成（転写100%）"),
    ]
    figure, axes = plt.subplots(1, 5, figsize=(10.2, 2.35))
    for axis, (image, title) in zip(axes, panels, strict=True):
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.subplots_adjust(wspace=0.04, left=0.005, right=0.995, top=0.9, bottom=0.0)
    save(figure, directory, "fig_recovery", vector=False)
