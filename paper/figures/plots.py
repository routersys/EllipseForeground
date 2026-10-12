from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ellipse_foreground.evaluation.statistics import bootstrap_interval, scene_means
from paper.figures.style import COLORS, NAMES, PATCH_COLOR, save
from paper.quantities import (
    mean_auc,
    mean_color_weight,
    mean_overlap,
    overlap_auc_pairs,
    rank_correlations,
)
from paper.results import DOSES, EARLIER_SETS, SETS
from paper.sections.ablation import FORMS, BoundaryCurves, graph_cut_values
from paper.study import Study

DOSE_PERCENT = [0, 25, 50, 75, 100]
SET_COLORS = {"sprite1": "#1d4ed8", "sprite2": "#0f766e", "collection0": "#c2410c", "collection100": "#a16207"}
SET_MARKERS = {"sprite1": "o", "sprite2": "s", "collection0": "^", "collection100": "v"}
SET_TITLES = {
    "sprite1": "立ち絵1",
    "sprite2": "立ち絵2",
    "collection0": "17体・転写0%",
    "collection100": "17体・転写100%",
}
BAR_TITLES = {
    "sprite1": "立ち絵1",
    "sprite2": "立ち絵2",
    "collection0": "17体\n転写0%",
    "collection100": "17体\n転写100%",
    "public0": "公開素材\n転写0%",
    "public100": "公開素材\n転写100%",
}
CUE_STYLES = (
    ("texture", "テクスチャ", COLORS["tex"], "D"),
    ("color", "色", COLORS["app"], "s"),
    ("patch", "パッチ", PATCH_COLOR, "v"),
)
LEGEND = {"loc": "upper center", "frameon": False, "fontsize": 7.4, "columnspacing": 1.0, "handletextpad": 0.4}


def dose_plot(study: Study, directory: Path) -> None:
    results = study.results
    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for cue, name, color, marker in CUE_STYLES:
        values = [mean_overlap(results, dose, cue) for dose in DOSES]
        axes[0].plot(DOSE_PERCENT, values, color=color, marker=marker, ms=4.2, lw=1.5, label=name)
    axes[0].set_ylabel("バタチャリヤ係数")
    axes[0].set_ylim(-0.02, 1.0)
    axes[0].set_title("(a) 内側と外側の分布の重なり", fontsize=9)
    axes[0].legend(bbox_to_anchor=(0.5, -0.30), ncol=3, **LEGEND)
    for cue, name, color, marker in CUE_STYLES:
        values = [mean_auc(results, dose, cue) for dose in DOSES]
        axes[1].plot(DOSE_PERCENT, values, color=color, marker=marker, ms=4.2, lw=1.5, label=name)
    axes[1].axhline(0.5, color="#9ca3af", ls=":", lw=1)
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_ylabel("楕円内のAUC")
    axes[1].set_title("(b) 手がかりの識別性能", fontsize=9)
    twin = axes[1].twinx()
    twin.plot(
        DOSE_PERCENT,
        [mean_color_weight(results, dose) for dose in DOSES],
        color="#111827",
        marker="o",
        ms=4,
        lw=1.4,
        ls="--",
        label="色の重み（右軸）",
    )
    twin.set_ylim(0, 0.65)
    twin.set_ylabel("色の重み")
    handles, labels = axes[1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + twin_handles, labels + twin_labels, bbox_to_anchor=(0.5, -0.30), ncol=2, **LEGEND)
    for method, style, marker in (
        ("ellipse", ":", "x"),
        ("tex", "-", "D"),
        ("mix", "-", "^"),
        ("adaptive_c", "-", "o"),
        ("app", "--", "s"),
    ):
        values = [study.mean(dose, method) for dose in DOSES]
        axes[2].plot(
            DOSE_PERCENT, values, color=COLORS[method], marker=marker, ms=4.2, lw=1.5, ls=style, label=NAMES[method]
        )
    axes[2].set_ylim(0.4, 1.0)
    axes[2].set_ylabel("前景のIoU")
    axes[2].set_title("(c) 前景のIoU", fontsize=9)
    axes[2].legend(bbox_to_anchor=(0.5, -0.30), ncol=2, **LEGEND)
    for axis in axes:
        axis.set_xlabel("色転写の強さ（%）")
        axis.set_xticks(DOSE_PERCENT)
        axis.grid(alpha=0.25, lw=0.6)
    figure.tight_layout(w_pad=1.1)
    save(figure, directory, "fig_dose")


def scatter_plot(study: Study, directory: Path) -> None:
    results = study.results
    correlations = rank_correlations(results)
    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), sharey=True)
    titles = {"texture": "(a) テクスチャ", "color": "(b) 色", "patch": "(c) パッチ"}
    for column, cue in enumerate(("texture", "color", "patch")):
        for key in EARLIER_SETS:
            overlaps, aucs = overlap_auc_pairs(results, key, cue)
            axes[column].scatter(
                overlaps,
                aucs,
                s=7,
                alpha=0.5,
                color=SET_COLORS[key],
                marker=SET_MARKERS[key],
                linewidths=0,
                label=SET_TITLES[key] if column == 0 else None,
            )
        rho = f"{correlations[cue].pooled:.2f}".replace("-", "−")
        axes[column].set_title(f"{titles[cue]}  順位相関 {rho}", fontsize=9)
        axes[column].set_xlabel("バタチャリヤ係数")
        axes[column].set_xlim(-0.02, 1.02)
        axes[column].set_ylim(0.4, 1.01)
        axes[column].axhline(0.5, color="#9ca3af", ls=":", lw=1)
        axes[column].grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("楕円内のAUC")
    axes[0].legend(loc="lower left", frameon=False, fontsize=7.4, markerscale=1.8)
    figure.tight_layout(w_pad=0.8)
    save(figure, directory, "fig_scatter")


def boundary_plot(study: Study, directory: Path) -> None:
    curves = BoundaryCurves(study)
    graph = graph_cut_values(study)
    figure, axes = plt.subplots(2, 4, figsize=(7.6, 4.0), sharey="col")
    styles = {
        FORMS[0][0].value: ("平均弱さ型（採用）", COLORS["adaptive_c"], "o", "-"),
        FORMS[1][0].value: ("総和型", COLORS["mix"], "^", "-."),
        FORMS[2][0].value: ("併合高さ型", COLORS["app"], "s", "--"),
    }
    for column, key in enumerate(EARLIER_SETS):
        for form, _ in FORMS:
            name, color, marker, style = styles[form.value]
            curve = curves.curves[(form.value, key)]
            exponents = sorted(curve)
            axes[0, column].plot(
                exponents,
                [curve[exponent] for exponent in exponents],
                color=color,
                marker=marker,
                ms=3.4,
                lw=1.3,
                ls=style,
                label=name if column == 0 else None,
            )
        axes[0, column].axhline(
            curves.none[key], color="#9ca3af", ls=":", lw=1, label="境界項なし" if column == 0 else None
        )
        axes[0, column].set_title(SET_TITLES[key], fontsize=8.6)
        exponents = sorted(graph[key])
        axes[1, column].plot(
            exponents,
            [graph[key][exponent] for exponent in exponents],
            color="#374151",
            marker="d",
            ms=3.4,
            lw=1.3,
            label="グラフカット" if column == 0 else None,
        )
        best_tree = max(curves.curves[(FORMS[0][0].value, key)].values())
        axes[1, column].axhline(
            best_tree,
            color=COLORS["adaptive_c"],
            ls="-",
            lw=1.1,
            label="木の切断（平均弱さ型の最良）" if column == 0 else None,
        )
        axes[0, column].set_xlabel(r"$\log_2$（境界項の倍率）", fontsize=8)
        axes[1, column].set_xlabel(r"$\log_2\,\lambda$（グラフカット）", fontsize=8)
        for row in (0, 1):
            axes[row, column].grid(alpha=0.25, lw=0.6)
            axes[row, column].tick_params(labelsize=7.5)
        axes[0, column].set_xticks([-6, -4, -2, 0, 2, 4])
        axes[1, column].set_xticks([0, 2, 4, 6, 8])
    axes[0, 0].set_ylabel("IoU（木の切断）")
    axes[1, 0].set_ylabel("IoU（グラフカット）")
    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    bottom_handles, bottom_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        top_handles + bottom_handles,
        top_labels + bottom_labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=7.2,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.tight_layout(h_pad=0.8, w_pad=0.5, rect=(0, 0, 1, 0.95))
    save(figure, directory, "fig_reg")


def bars_plot(study: Study, directory: Path) -> None:
    methods = ("ellipse", "prior", "tex", "app", "mix", "adaptive_c")
    figure, axes = plt.subplots(1, len(SETS), figsize=(7.6, 2.6), sharey=True)
    for column, key in enumerate(SETS):
        records = study.results.trials(key)
        for position, method in enumerate(methods):
            interval = bootstrap_interval(
                scene_means(records, lambda record, method=method: record["tree"][method]["iou"])
            )
            axes[column].bar(position, interval.mean, color=COLORS[method], width=0.72)
            axes[column].errorbar(
                position,
                interval.mean,
                yerr=[[interval.mean - interval.low], [interval.high - interval.mean]],
                color="#111827",
                lw=0.9,
                capsize=1.6,
            )
        axes[column].axhline(study.mean(key, "oracle"), color="#6b7280", ls=":", lw=1.0)
        axes[column].set_title(BAR_TITLES[key], fontsize=8.0)
        axes[column].set_xticks([])
        axes[column].set_ylim(0.4, 1.0)
        axes[column].grid(axis="y", alpha=0.25, lw=0.6)
    axes[0].set_ylabel("前景のIoU")
    handles = [Patch(fc=COLORS[method], label=NAMES[method]) for method in methods]
    handles.append(Line2D([0], [0], color="#6b7280", ls=":", label="正解の事前分布（上限）"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, -0.17))
    figure.tight_layout(w_pad=0.6)
    save(figure, directory, "fig_bars")
