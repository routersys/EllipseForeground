from collections.abc import Callable

import numpy as np

from ellipse_foreground.evaluation.auc import rank_auc
from ellipse_foreground.evaluation.statistics import (
    BOOTSTRAP_SAMPLES,
    LOWER_PERCENTILE,
    UPPER_PERCENTILE,
    Interval,
)
from ellipse_foreground.settings import DEFAULT_SETTINGS
from paper.markup import figure
from paper.numbers import f2, f3, interval_text, percent, plain, signed
from paper.quantities import rank_correlation
from paper.sections.baselines import classify
from paper.sections.pairs import classified_by, mean_difference, paired, value, with_values
from paper.sections.refinement import REFINE_SETS, RefineSet, names_of
from paper.study import Study
from paper.table import table_html

QUALITY_LEVEL = 0.8
CHANGE_LEVEL = 0.05
GROUP_STRIDE = 10**4
UNSEEN_SETS = (
    ("public0", 3, "公開素材の転写0%"),
    ("public100", 3, "公開素材の転写100%"),
    ("photo", 4, "写真"),
)
UNSEEN_NAME = "選択に使っていない場面"
POOLED_NAME = "全体"


def threshold() -> float:
    return DEFAULT_SETTINGS.stability.threshold


def adaptation_rows(study: Study, selection: RefineSet) -> list[dict]:
    return [record for record in study.results.adaptations(selection.key) if selection.accepts(record["scene"])]


def all_adaptation_rows(study: Study) -> dict[str, list[dict]]:
    return study.remembered(
        "adaptation_rows", lambda: {selection.name: adaptation_rows(study, selection) for selection in REFINE_SETS}
    )


def unseen_rows(study: Study) -> list[dict]:
    def collect() -> list[dict]:
        rows = []
        for index, (key, modulus, _) in enumerate(UNSEEN_SETS):
            for record in study.results.adaptations(key):
                if record["scene"] % modulus != 0:
                    rows.append({**record, "scene": record["scene"] + GROUP_STRIDE * (index + 1)})
        return rows

    return study.remembered("unseen_rows", collect)


def correlation_intervals(study: Study) -> dict[str, Interval]:
    def compute() -> dict[str, Interval]:
        rows = all_adaptation_rows(study)
        intervals = {selection.name: correlation_interval(rows[selection.name]) for selection in REFINE_SETS}
        intervals[POOLED_NAME] = correlation_interval(pooled_rows(rows))
        return intervals

    return study.remembered("correlation_intervals", compute)


def unstable(row: dict) -> bool:
    return row["stability"] < threshold()


def gated_value(adapted: str, plain_name: str, measure: str = "iou") -> Callable[[dict], float]:
    return lambda row: row[adapted][measure] if unstable(row) else row[plain_name][measure]


def unstable_difference(rows: list[dict], adapted: str, plain_name: str) -> Interval | None:
    chosen = [row for row in rows if unstable(row)]
    if not chosen:
        return None
    return paired(chosen, value(adapted), value(plain_name))


def correlation_of(rows: list[dict]) -> float:
    statistic = rank_correlation(
        np.array([row["stability"] for row in rows]), np.array([row["tree"]["iou"] for row in rows])
    )
    return 0.0 if np.isnan(statistic) else statistic


def correlation_interval(rows: list[dict], seed: int = 0) -> Interval:
    by_scene: dict[int, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        picks = rng.integers(0, len(scenes), len(scenes))
        values.append(correlation_of([row for pick in picks for row in by_scene[scenes[pick]]]))
    low, high = np.percentile(values, [LOWER_PERCENTILE, UPPER_PERCENTILE])
    return Interval(correlation_of(rows), float(low), float(high))


def quality_auc(rows: list[dict]) -> float:
    labels = np.array([row["tree"]["iou"] >= QUALITY_LEVEL for row in rows])
    return rank_auc(np.array([row["stability"] for row in rows]), labels)


def auc_text(rows: list[dict]) -> str:
    result = quality_auc(rows)
    return "−" if np.isnan(result) else f2(result)


def mean_iou(rows: list[dict]) -> float | None:
    return float(np.mean([row["tree"]["iou"] for row in rows])) if rows else None


def mean_text(rows: list[dict]) -> str:
    result = mean_iou(rows)
    return "−" if result is None else f3(result)


def pooled_rows(rows: dict[str, list[dict]]) -> list[dict]:
    return [
        {**row, "scene": row["scene"] + GROUP_STRIDE * index}
        for index, selection in enumerate(REFINE_SETS)
        for row in rows[selection.name]
    ]


def auc_note(rows: dict[str, list[dict]]) -> str:
    missing = [selection for selection in REFINE_SETS if np.isnan(quality_auc(rows[selection.name]))]
    if not missing:
        return ""
    return f"正例と負例の一方がない行（{names_of(missing)}）は、AUCを求められないので−とした。"


def stability_cells(name: str, data: list[dict], interval: Interval) -> list[str]:
    low = [row for row in data if unstable(row)]
    high = [row for row in data if not unstable(row)]
    return [
        name,
        str(len(data)),
        f3(float(np.mean([row["stability"] for row in data]))),
        interval_text(interval, 2),
        auc_text(data),
        percent(len(low) / len(data)),
        mean_text(low),
        mean_text(high),
    ]


def stability_table(rows: dict[str, list[dict]], intervals: dict[str, Interval]) -> str:
    body = [
        stability_cells(selection.short, rows[selection.name], intervals[selection.name]) for selection in REFINE_SETS
    ]
    body.append(stability_cells(POOLED_NAME, pooled_rows(rows), intervals[POOLED_NAME]))
    return table_html(
        "t:stab",
        f"楕円の摂動に対する切断の安定性と、精緻化後のIoUとの関係。順位相関はSpearmanの値で、場面を単位とするブートストラップ法の区間を付けた。AUCは、IoUが{plain(QUALITY_LEVEL)}以上の試行を、安定性が高いほど正例と見なして求めた。{auc_note(rows)}右の三列は、安定性が閾値{plain(threshold())}未満の試行の割合と、閾値未満、閾値以上の試行のIoUの平均である。",
        [
            "データセット",
            "試行数",
            "安定性の平均",
            "順位相関<br>［95%信頼区間］",
            "AUC",
            "閾値未満の割合",
            "閾値未満のIoU",
            "閾値以上のIoU",
        ],
        body,
        align=["l", "", "", "", "", "", "", ""],
        sub_rows=(len(REFINE_SETS),),
        widths=["17%", "8%", "10%", "19%", "8%", "12%", "13%", "13%"],
    )


def refit_cells(name: str, data: list[dict]) -> list[str]:
    low = unstable_difference(data, "refit", "tree")
    return [
        name,
        interval_text(paired(data, value("refit"), value("tree"))),
        interval_text(paired(data, gated_value("refit", "tree"), value("tree"))),
        "−" if low is None else interval_text(low),
        interval_text(paired(data, gated_value("refit_band", "band"), value("band"))),
    ]


def refit_table(rows: dict[str, list[dict]], unseen: list[dict]) -> str:
    body = [refit_cells(selection.short, rows[selection.name]) for selection in REFINE_SETS]
    if unseen:
        body.append(refit_cells(UNSEEN_NAME, unseen))
    return table_html(
        "t:refit",
        f"楕円の再推定の効果（精緻化後のIoUの差）。左から、すべての試行で再推定した場合の木の切断との差、安定性が閾値{plain(threshold())}未満のときだけ再推定した場合の木の切断との差、閾値未満の試行だけで見た差、閾値未満のときだけ再推定した場合の、帯の再判定との差である。場面を単位とするブートストラップ法の区間を付けた。開発用に使ったのは、立ち絵1、17体、写真の25枚である。最終行は、再推定の方法と閾値を選ぶ際に結果を見ていない場面で、公開素材の場面番号が3で割り切れないものと、写真の番号が4で割り切れないものを合わせた。",
        [
            "データセット",
            "常に再推定<br>［95%信頼区間］",
            "不安定なときだけ<br>［95%信頼区間］",
            "閾値未満の試行だけ<br>［95%信頼区間］",
            "帯と併用（不安定なときだけ）<br>［95%信頼区間］",
        ],
        body,
        align=["l", "", "", "", ""],
        sub_rows=(len(REFINE_SETS),) if unseen else (),
        widths=["20%", "20%", "20%", "20%", "20%"],
    )


def stability_text(rows: dict[str, list[dict]], intervals: dict[str, Interval]) -> str:
    per_set = {selection: intervals[selection.name] for selection in REFINE_SETS}
    pooled = pooled_rows(rows)
    overall = intervals[POOLED_NAME]
    positive = [selection for selection, interval in per_set.items() if interval.low > 0]
    weakest = min(per_set.values(), key=lambda interval: interval.mean)
    strongest = max(per_set.values(), key=lambda interval: interval.mean)
    if len(positive) == len(REFINE_SETS):
        sign_text = (
            f"順位相関は、表のすべての行で正で、信頼区間は0を含まない（{f2(weakest.mean)}から{f2(strongest.mean)}）。"
        )
    else:
        sign_text = f"順位相関が正で、信頼区間が0を含まないのは、{names_of(positive) or 'なし'}である（行ごとの値は{f2(weakest.mean)}から{f2(strongest.mean)}）。"
    low = [row for row in pooled if unstable(row)]
    high = [row for row in pooled if not unstable(row)]
    return (
        f'<p>安定性 \\(S\\) を、すべての試行で求めた。<ref k="t:stab"></ref>と<ref k="f:stab"></ref>に、\\(S\\) と、木の切断（精緻化後）のIoUの関係を示す。'
        f"{sign_text}全体では順位相関が{interval_text(overall, 2)}、IoUが{plain(QUALITY_LEVEL)}以上かどうかを分けるAUCが{f2(quality_auc(pooled))}である。"
        f"\\(S\\) が閾値{plain(threshold())}未満の試行は、全体の{percent(len(low) / len(pooled))}で、IoUの平均は{f3(mean_iou(low))}、"
        f"閾値以上の試行は{f3(mean_iou(high))}である。"
        "安定性は、正解を使わずに求められる値で、切断が事前分布の摂動に左右される画像ほど、IoUが低い。</p>"
    )


def refit_text(rows: dict[str, list[dict]], unseen: list[dict]) -> str:
    always = classified_by(rows, value("refit"), value("tree"))
    gated = classified_by(rows, gated_value("refit", "tree"), value("tree"))
    higher_values = with_values(mean_difference(rows, always["higher"], value("refit"), value("tree")))
    lower_values = with_values(mean_difference(rows, always["lower"], value("refit"), value("tree")))
    if always["higher"] and always["lower"]:
        always_text = f"すべての試行で再推定すると、信頼区間が0を含まない差で木の切断より高いのは{higher_values}で、低いのは{lower_values}である。"
    elif always["higher"]:
        always_text = (
            f"すべての試行で再推定すると、信頼区間が0を含まない差で木の切断より高いのは{higher_values}である。"
        )
    elif always["lower"]:
        always_text = f"すべての試行で再推定しても、信頼区間が0を含まない差で木の切断より高い行はない。低いのは{lower_values}である。"
    else:
        always_text = "すべての試行で再推定しても、信頼区間が0を含まない差で木の切断より高い行も低い行もない。"
    lower_text = (
        f"信頼区間が0を含まない差で低いのは{names_of(gated['lower'])}である。"
        if gated["lower"]
        else "信頼区間が0を含まない差で低い行はない。"
    )
    if gated["higher"]:
        higher_text = f"高いのは{with_values(mean_difference(rows, gated['higher'], gated_value('refit', 'tree'), value('tree')))}である。"
    else:
        higher_text = "高い行はない。"
    gated_text = (
        f"安定性が閾値未満のときだけ再推定すると、信頼区間が0を含まない差で木の切断より{higher_text}{lower_text}"
    )
    unstable_pairs = []
    for selection in REFINE_SETS:
        interval = unstable_difference(rows[selection.name], "refit", "tree")
        if interval is not None:
            unstable_pairs.append((selection, interval.mean))
    negative = [(selection, mean) for selection, mean in unstable_pairs if mean < 0]
    means = [mean for _, mean in unstable_pairs]
    if negative:
        sign_text = f"閾値未満の試行だけで見ても、差の平均が負になるのは{with_values(negative)}である。"
    elif unstable_pairs:
        sign_text = f"閾値未満の試行だけで見ると、再推定の差の平均は、{len(unstable_pairs)}行のすべてで正である（{signed(min(means))}から{signed(max(means))}）。"
    else:
        sign_text = ""
    unseen_text = ""
    if unseen:
        gated_interval = paired(unseen, gated_value("refit", "tree"), value("tree"))
        always_interval = paired(unseen, value("refit"), value("tree"))
        unseen_text = (
            f"{UNSEEN_NAME}を合わせると、安定性が閾値未満のときだけ再推定した場合の差は{interval_text(gated_interval)}、"
            f"すべての試行で再推定した場合の差は{interval_text(always_interval)}である。"
        )
    return (
        f'<p>安定性が低い画像で、楕円を切断から求め直す効果を<ref k="t:refit"></ref>に示す。{always_text}'
        f"{gated_text}{sign_text}{unseen_text}{spread_text(rows)}</p>"
    )


def spread_text(rows: dict[str, list[dict]]) -> str:
    chosen = [row for row in pooled_rows(rows) if unstable(row)]
    if not chosen:
        return ""
    changes = np.array([row["refit"]["iou"] - row["tree"]["iou"] for row in chosen])
    return (
        f"安定性が閾値未満の{len(chosen)}試行のうち、再推定でIoUが{plain(CHANGE_LEVEL)}以上高くなったのは{percent(float(np.mean(changes >= CHANGE_LEVEL)))}、"
        f"{plain(CHANGE_LEVEL)}以上低くなったのは{percent(float(np.mean(changes <= -CHANGE_LEVEL)))}で、"
        f"最も低くなった試行では{signed(float(changes.min()))}になった。"
    )


def refit_verdict(rows: dict[str, list[dict]], unseen: list[dict]) -> str:
    gated = classified_by(rows, gated_value("refit", "tree"), value("tree"))
    always = classified_by(rows, value("refit"), value("tree"))
    held_out = [selection for selection in REFINE_SETS if not selection.development]
    held_higher = [selection for selection in held_out if selection in gated["higher"]]
    development = [selection for selection in REFINE_SETS if selection.development]
    dev_higher = [selection for selection in development if selection in gated["higher"]]
    protective = [selection for selection in held_out if selection in always["lower"]]
    protect_text = (
        f"すべての試行で再推定すると{names_of(protective)}では低くなるため、安定性による限定は、この低下を防ぐ役割を持つ。"
        if protective
        else ""
    )
    unseen_class = classify(paired(unseen, gated_value("refit", "tree"), value("tree"))) if unseen else "unclear"
    if unseen and unseen_class == "higher" and len(held_higher) == len(held_out):
        return f"開発用に使っていない行のすべてと、{UNSEEN_NAME}で、信頼区間が0を含まない向上が得られた。{protect_text}"
    if unseen and unseen_class == "higher":
        return f"{UNSEEN_NAME}では、信頼区間が0を含まない向上が得られた。{protect_text}"
    if dev_higher and not held_higher:
        return (
            "向上が確かめられたのは開発用の行だけで、開発用に使っていない行では、信頼区間が0を含まない向上はなかった。"
            f"{protect_text}この段階は、既定では行わない。"
        )
    return (
        f"開発用に使っていない行で、信頼区間が0を含まない向上があるのは{names_of(held_higher) or 'なし'}に限る。"
        f"{UNSEEN_NAME}では、信頼区間が0を含まない向上は確かめられていない。{protect_text}この段階は、既定では行わない。"
    )


def stability_figure() -> str:
    return figure(
        "f:stab",
        "fig_stab.svg",
        "安定性とIoU",
        f"楕円の摂動に対する切断の安定性と、精緻化後のIoU。(a)は1点が1試行で、点線は閾値{plain(threshold())}である。(b)は、データセットごとの、安定性が閾値未満と閾値以上の試行のIoUの平均を線で結んだ。",
    )


def stability_section(study: Study) -> str:
    rows = all_adaptation_rows(study)
    unseen = unseen_rows(study)
    intervals = correlation_intervals(study)
    return (
        stability_text(rows, intervals)
        + "\n"
        + stability_table(rows, intervals)
        + "\n"
        + stability_figure()
        + "\n"
        + refit_text(rows, unseen)
        + f"<p>{verdict_of(study)}</p>\n"
        + refit_table(rows, unseen)
    )


def verdict_of(study: Study) -> str:
    return study.remembered("refit_verdict", lambda: refit_verdict(all_adaptation_rows(study), unseen_rows(study)))


def abstract_of(study: Study) -> str:
    return study.remembered("refit_abstract", lambda: refit_abstract(all_adaptation_rows(study), unseen_rows(study)))


def adaptation_summary(study: Study) -> dict[str, str]:
    rows = all_adaptation_rows(study)
    settings = DEFAULT_SETTINGS
    pooled = pooled_rows(rows)
    low = [row for row in pooled if unstable(row)]
    high = [row for row in pooled if not unstable(row)]
    overall = correlation_intervals(study)[POOLED_NAME]
    return {
        "stab_low": plain(min(settings.stability.scales)),
        "stab_high": plain(max(settings.stability.scales)),
        "stab_shift": plain(settings.stability.shift),
        "stab_threshold": plain(settings.stability.threshold),
        "refit_damping": plain(settings.refit.damping),
        "refit_iterations": plain(settings.refit.iterations),
        "stab_rho": f2(overall.mean),
        "stab_auc": f2(quality_auc(pooled)),
        "stab_share": percent(len(low) / len(pooled)),
        "stab_iou_low": f3(mean_iou(low)),
        "stab_iou_high": f3(mean_iou(high)),
        "stab_count": str(len(pooled)),
        "stab_quality": plain(QUALITY_LEVEL),
    }


def refit_abstract(rows: dict[str, list[dict]], unseen: list[dict]) -> str:
    gated = classified_by(rows, gated_value("refit", "tree"), value("tree"))
    development = [selection for selection in REFINE_SETS if selection.development]
    dev_higher = [selection for selection in development if selection in gated["higher"]]
    unseen_class = classify(paired(unseen, gated_value("refit", "tree"), value("tree"))) if unseen else "unclear"
    if unseen_class == "higher":
        return "安定性が低いときだけ楕円を切断から求め直すと、選択に使っていない場面でもIoUが高くなる。"
    if dev_higher:
        return "安定性が低いときだけ楕円を切断から求め直すと、IoUは開発用の行で高くなるが、選択に使っていない場面では信頼区間が0を含む。"
    return "安定性が低いときだけ楕円を切断から求め直しても、信頼区間が0を含まない向上は得られなかった。"


def abstract_adaptation(study: Study) -> str:
    summary = adaptation_summary(study)
    return (
        f"楕円を摂動させたときの切断の安定性は、正解を使わずに求められ、IoUとの順位相関は{summary['stab_rho']}である。"
        f"{abstract_of(study)}"
    )


def intro_stability(study: Study) -> str:
    summary = adaptation_summary(study)
    return (
        f"{summary['stab_count']}試行での順位相関は{summary['stab_rho']}で、"
        f"安定性が閾値未満の試行（{summary['stab_share']}）のIoUの平均は{summary['stab_iou_low']}、閾値以上は{summary['stab_iou_high']}である。"
        f"{abstract_of(study)}"
    )


def weakest_stable(rows: dict[str, list[dict]]) -> tuple[RefineSet, float] | None:
    means = {}
    for selection in REFINE_SETS:
        result = mean_iou([row for row in rows[selection.name] if not unstable(row)])
        if result is not None:
            means[selection] = result
    if not means:
        return None
    lowest = min(means, key=means.get)
    return lowest, means[lowest]


def discussion_stability(study: Study) -> str:
    weakest = weakest_stable(all_adaptation_rows(study))
    stable_text = (
        f"安定性が閾値以上の試行でも、IoUの平均が最も低い{weakest[0].short}では{f3(weakest[1])}にとどまる。"
        if weakest
        else ""
    )
    return (
        "<p><b>切断の安定性は、結果の信頼度を表すが、再推定の効果は限られる。</b>"
        f"{verdict_of(study)}"
        f"安定性は、手がかりを固定して楕円だけを摂動させるため、手がかりの誤りは反映しない。{stable_text}"
        "また、閾値は、開発用の行と、立ち絵2、公開素材の一部の結果を見て選んだ値で、データセットごとの最適値は確かめていない。</p>\n"
    )


def conclusion_stability(study: Study) -> str:
    summary = adaptation_summary(study)
    return f"楕円を摂動させたときの切断の安定性は、IoUと順位相関{summary['stab_rho']}で対応する。{abstract_of(study)}"
