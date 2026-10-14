from pathlib import Path

import matplotlib.pyplot as plt

from paper.figures.style import save
from paper.sections.refinement import REFINE_SETS
from paper.sections.stability import all_adaptation_rows, mean_iou, threshold, unstable
from paper.study import Study

PALETTE = ("#1d4ed8", "#0f766e", "#c2410c", "#a16207", "#7c3aed", "#be185d", "#0e7490", "#4b5563")
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*")


def stability_plot(study: Study, directory: Path) -> None:
    rows = all_adaptation_rows(study)
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw={"width_ratios": [1.5, 1]})
    for index, selection in enumerate(REFINE_SETS):
        data = rows[selection.name]
        axes[0].scatter(
            [row["stability"] for row in data],
            [row["tree"]["iou"] for row in data],
            s=6,
            alpha=0.45,
            color=PALETTE[index],
            marker=MARKERS[index],
            linewidths=0,
            label=selection.short,
        )
        low = mean_iou([row for row in data if unstable(row)])
        high = mean_iou([row for row in data if not unstable(row)])
        if low is not None and high is not None:
            axes[1].plot([0, 1], [low, high], color=PALETTE[index], marker=MARKERS[index], lw=1.2, ms=4.5)
    axes[0].axvline(threshold(), color="#9ca3af", ls=":", lw=1)
    axes[0].set_xlabel("安定性")
    axes[0].set_ylabel("IoU（精緻化後）")
    axes[0].set_title("(a) 1点が1試行", fontsize=9)
    axes[0].set_xlim(0.55, 1.01)
    axes[0].set_ylim(0.0, 1.01)
    axes[0].grid(alpha=0.25, lw=0.6)
    axes[0].legend(loc="lower right", frameon=False, fontsize=6.6, markerscale=1.6, ncol=2)
    axes[1].set_xticks([0, 1], ["閾値未満", "閾値以上"])
    axes[1].set_xlim(-0.3, 1.3)
    axes[1].set_title("(b) 安定性の高低ごとの平均", fontsize=9)
    axes[1].grid(alpha=0.25, lw=0.6)
    figure.tight_layout(w_pad=1.0)
    save(figure, directory, "fig_stab")
