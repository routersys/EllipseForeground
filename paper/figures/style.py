from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "BIZ UDPGothic"
COLORS = {
    "ellipse": "#6b7280",
    "prior": "#0f766e",
    "tex": "#1d4ed8",
    "app": "#c2410c",
    "mix": "#7c3aed",
    "adaptive_c": "#111827",
    "adaptive": "#0e7490",
    "oracle": "#9ca3af",
}
NAMES = {
    "ellipse": "楕円をそのまま使用",
    "prior": "事前分布のみ",
    "tex": "テクスチャのみ",
    "app": "色とパッチのみ",
    "mix": "固定配合",
    "adaptive_c": "適応（色）",
    "adaptive": "適応（色とパッチ）",
    "oracle": "正解の事前分布",
}
PATCH_COLOR = "#a16207"


def configure() -> None:
    if any(entry.name == FONT for entry in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = FONT
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 9,
            "mathtext.fontset": "stix",
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.5,
        }
    )


def save(figure, directory: Path, name: str, vector: bool = True, dpi: int = 220) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
    figure.savefig(directory / f"{name}.png", dpi=dpi, **options)
    if vector:
        figure.savefig(directory / f"{name}.svg", **options)
    plt.close(figure)


def ellipse_outline(center: tuple[float, float], radii: tuple[float, float], points: int = 200):
    angle = np.linspace(0, 2 * np.pi, points)
    return center[0] + radii[0] * np.cos(angle), center[1] + radii[1] * np.sin(angle)


def hide_frame(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
