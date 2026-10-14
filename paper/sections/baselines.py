import numpy as np
from scipy.stats import spearmanr

from ellipse_foreground.evaluation.statistics import (
    BOOTSTRAP_SAMPLES,
    LOWER_PERCENTILE,
    UPPER_PERCENTILE,
    Interval,
    bootstrap_interval,
    scene_means,
)
from paper.numbers import above_zero, below_zero, f2, f3, interval_text, signed
from paper.results import SUBSETS, scene_mean
from paper.sections.photos import relation
from paper.study import Study
from paper.table import table_html

BASELINE_SETS = ("public0", "public100", "photo")
BASELINE_NAMES = (
    ("grabcut", "GrabCut"),
    ("sam_single", "SAM ViT-B（単一マスク）"),
    ("sam_best", "SAM ViT-B（推定IoU最大）"),
)
SET_NAMES = {key: SUBSETS[key].short for key in BASELINE_SETS}


def baseline_table(study: Study) -> str:
    columns = []
    for key in BASELINE_SETS:
        rows = [
            (study.mean(key, "adaptive_c"), study.mean(key, "adaptive_c", "mae")),
            (study.mean(key, "adaptive_c", family="graph"), study.mean(key, "adaptive_c", "mae", family="graph")),
            *(
                (study.measure.baseline_mean(key, name), study.measure.baseline_mean(key, name, "mae"))
                for name, _ in BASELINE_NAMES
            ),
        ]
        columns.append(rows)
    labels = ["適応（色）、木の切断", "適応（色）、グラフカット", *(label for _, label in BASELINE_NAMES)]
    best = set()
    for number, rows in enumerate(columns):
        best.add((max(range(len(rows)), key=lambda index: rows[index][0]), 1 + 2 * number))
        best.add((min(range(len(rows)), key=lambda index: rows[index][1]), 2 + 2 * number))
    table_rows = [
        [label, *(cell for rows in columns for cell in (f3(rows[index][0]), f2(rows[index][1])))]
        for index, label in enumerate(labels)
    ]
    return table_html(
        "t:baseline",
        "GrabCutとSAMとの比較（IoUとMAE。提案手法は精緻化後。各列の最良を太字で示す）。GrabCutとSAMには、楕円の外接矩形を与えた。",
        ["手法", "IoU", "MAE", "IoU", "MAE", "IoU", "MAE"],
        table_rows,
        align=["l", "", "", "", "", "", ""],
        best=best,
        widths=["31%", *["11.5%"] * 6],
        group=[("", 1), *((SET_NAMES[key], 2) for key in BASELINE_SETS)],
    )


def classify(interval: Interval, digits: int = 3) -> str:
    if above_zero(interval, digits):
        return "higher"
    if below_zero(interval, digits):
        return "lower"
    return "unclear"


def summary_sentence(study: Study, name: str, label: str, measure: str, higher: str, lower: str, digits: int) -> str:
    groups: dict[str, list[str]] = {"higher": [], "lower": [], "unclear": []}
    for key in BASELINE_SETS:
        groups[classify(study.measure.baseline_interval(key, name, measure), digits)].append(SET_NAMES[key])
    parts = []
    if groups["higher"]:
        parts.append(f"{'、'.join(groups['higher'])}では{higher}")
    if groups["lower"]:
        parts.append(f"{'、'.join(groups['lower'])}では{lower}")
    if groups["unclear"]:
        parts.append(f"{'、'.join(groups['unclear'])}では信頼区間が0を含み、差がはっきりしない")
    return f"{label}は、{'、'.join(parts)}。"


def failed_grabcut_trials(study: Study) -> tuple[int, int]:
    records = [record for key in BASELINE_SETS for record in study.results.baselines(key)]
    return sum(1 for record in records if record.get("grabcut_converged") is False), len(records)


def baseline_results(study: Study) -> str:
    measure = study.measure
    trials = {key: len(study.results.baselines(key)) for key in BASELINE_SETS}
    failed, total = failed_grabcut_trials(study)
    sentences = []
    for name, label in BASELINE_NAMES:
        parts = [f"{SET_NAMES[key]}で{interval_text(measure.baseline_interval(key, name))}" for key in BASELINE_SETS]
        sentences.append(f"{label}は{'、'.join(parts)}である。")
    errors = []
    for name, label in BASELINE_NAMES:
        values = "、".join(
            f"{SET_NAMES[key]}で{interval_text(measure.baseline_interval(key, name, 'mae'), 2)}"
            for key in BASELINE_SETS
        )
        errors.append(f"{label}は{values}")
    summary = "".join(
        summary_sentence(study, name, label, "iou", "提案手法より高く", "提案手法より低く", 3)
        for name, label in BASELINE_NAMES
    )
    fallback = (
        f"GrabCutは、矩形が画像全体を覆うと背景の標本が取れず初期化できない。この場合は矩形そのものを結果にしており、{total}試行のうち{failed}試行が該当する。"
        if failed
        else ""
    )
    return (
        '<p>適応（色）のスコアによる木の切断を、GrabCut<cite k="rother2004"></cite>と、SAM<cite k="kirillov2023"></cite>のViT-B版と比べる。'
        "手法の設計に使っていない公開素材（転写0%と100%）と写真を対象にし、試行は他の比較と同じである。"
        f"試行数は、公開素材の転写0%が{trials['public0']}、転写100%が{trials['public100']}、写真が{trials['photo']}である。"
        "GrabCutとSAMには、提案手法に与えた楕円の外接矩形を入力する。楕円の丸みの情報は渡さない。"
        "GrabCutは、OpenCVの実装を矩形で初期化し、評価用の解像度の画像で反復5回を実行する。"
        f"{fallback}"
        "SAMは、Metaが配布する重みを使い、元の解像度の画像に矩形を与える。出力は、単一のマスクと、三つの候補のうち推定IoUが最大のものの二通りを記録し、どちらも二値マスクを評価の解像度に縮めて評価する。"
        '結果を<ref k="t:baseline"></ref>に示す。</p>\n'
        "<p>IoUの差は、各手法から適応（色）の木の切断を引いた値で、場面を単位とするブートストラップ法の区間とあわせて、"
        + "".join(sentences)
        + "</p>\n"
        "<p>" + summary + "</p>\n"
        "<p>ぼかし描画のMAEの差は、" + "。".join(errors) + "である。</p>\n"
        "<p>これらの差には、入力が矩形であることと、SAMが大規模な学習で得た知識を使うことの影響が含まれる。"
        "他のSAMの派生や、マッティングのモデル、LooseCutやDeep GrabCutとの比較は行っていない。</p>\n"
    )


def baseline_section(study: Study) -> str:
    return (
        '\n<h3 id="s:baselines">既存の手法との比較</h3>\n'
        + baseline_results(study)
        + "\n"
        + baseline_table(study)
        + '\n<h3 id="s:coverage">外接矩形が前景を覆う割合による違い</h3>\n'
        + coverage_section(study)
    )


def sets_where(study: Study, name: str, outcome: str) -> list[str]:
    return [SET_NAMES[key] for key in BASELINE_SETS if classify(study.measure.baseline_interval(key, name)) == outcome]


def abstract_baselines(study: Study) -> str:
    summary = coverage_summary(study)

    def clause(name: str) -> str:
        higher, lower = sets_where(study, name, "higher"), sets_where(study, name, "lower")
        if higher and lower:
            return f"{'、'.join(higher)}で提案手法より高く、{'、'.join(lower)}で低い"
        if higher:
            return f"{'、'.join(higher)}で提案手法より高い"
        if lower:
            return f"{'、'.join(lower)}で提案手法より低い"
        return "どのデータセットでも信頼区間が0を含む"

    return (
        "GrabCutとSAM（ViT-B）に楕円の外接矩形を与えた比較では、"
        f"IoUの差は、GrabCutでは{clause('grabcut')}。SAMの推定IoU最大の出力では{clause('sam_best')}。"
        "一方、外接矩形が前景を覆う割合と、提案手法から各手法を引いたIoUの差との順位相関は、"
        f"GrabCutに対して{summary['cov_rho_grabcut']}、SAMに対して{summary['cov_rho_sam']}で、覆う割合が大きいほど提案手法に有利な差は小さい。"
        f"覆う割合が90%未満の{summary['cov_n_low']}試行では、提案手法のIoUは、{summary['cov_grabcut_low']}、{summary['cov_sam_low']}。"
        "この90%は結果を見たあとに決めた閾値である。"
    )


def baseline_discussion(study: Study) -> str:
    shares = coverage_summary(study)
    summary = "".join(
        summary_sentence(study, name, label, "iou", "提案手法より高く", "提案手法より低く", 3)
        for name, label in BASELINE_NAMES
    )
    better = any(sets_where(study, name, "lower") for name in ("grabcut", "sam_best"))
    caution = "" if better else "提案手法が、GrabCutやSAMのViT-Bより精度が高いとは、本稿の実験からは言えない。"
    return (
        "<p><b>既存の手法との比較。</b>IoUの差の区間を提案手法に対して見ると、"
        f"{summary}{caution}"
        "ただし、外接矩形が前景を覆う割合と、提案手法から各手法を引いたIoUの差との順位相関は、"
        f"GrabCutに対して{shares['cov_rho_grabcut']}、SAMに対して{shares['cov_rho_sam']}で、覆う割合が大きいほど提案手法に有利な差は小さい。"
        f"覆う割合が90%未満の{shares['cov_n_low']}試行では、提案手法のIoUは、{shares['cov_grabcut_low']}、{shares['cov_sam_low']}。"
        "覆う割合が97%以上の試行では、GrabCutとSAMのほうが高い。信頼区間が0を含まずに提案手法が上回ったのは、外接矩形が前景の90%未満しか覆わない場合だけであり、この90%は結果を見たあとに決めた閾値である。"
        "提案手法は学習を使わず、楕円の丸みと内側の割合を事前分布に使う。GrabCutとSAMは矩形だけを使い、入力の情報が同じではない。"
        '詳しくは<ref k="s:baselines"></ref>と<ref k="s:coverage"></ref>に示した。</p>\n'
    )


STRATA = (
    (0.0, 0.9, "90%未満"),
    (0.9, 0.97, "90%以上97%未満"),
    (0.97, 1.01, "97%以上"),
)


def coverage_rows(study: Study) -> list[dict]:
    rows = []
    for key in BASELINE_SETS:
        trials = {(record["scene"], record["trial"]): record for record in study.results.trials(key)}
        coverage = {
            (record["scene"], record["trial"]): record["box_coverage"] for record in study.results.coverage(key)
        }
        for record in study.results.baselines(key):
            identifier = (record["scene"], record["trial"])
            if identifier not in trials or identifier not in coverage:
                continue
            tree = trials[identifier]["tree"]["adaptive_c"]["iou"]
            rows.append(
                {
                    "scene": f"{key}-{record['scene']}",
                    "coverage": coverage[identifier],
                    "tree": tree,
                    "graph": trials[identifier]["graph"]["adaptive_c"]["iou"],
                    "grabcut": record["grabcut"]["iou"],
                    "sam_best": record["sam_best"]["iou"],
                }
            )
    return rows


def stratum(rows: list[dict], low: float, high: float) -> list[dict]:
    return [row for row in rows if low <= row["coverage"] < high]


def mean_by_scene(rows: list[dict], name: str) -> float:
    return scene_mean(rows, lambda row: row[name])


def difference_interval(rows: list[dict], first: str, second: str) -> Interval:
    return bootstrap_interval(scene_means(rows, lambda row: row[first] - row[second]))


def coverage_table(study: Study) -> str:
    rows = coverage_rows(study)
    body = []
    for low, high, label in STRATA:
        part = stratum(rows, low, high)
        body.append(
            [
                label,
                str(len(part)),
                f3(mean_by_scene(part, "tree")),
                f3(mean_by_scene(part, "grabcut")),
                f3(mean_by_scene(part, "sam_best")),
                interval_text(difference_interval(part, "tree", "grabcut")),
                interval_text(difference_interval(part, "tree", "sam_best")),
            ]
        )
    return table_html(
        "t:coverage",
        "外接矩形が正解の前景を覆う割合ごとのIoU。公開素材の転写0%と100%、写真の試行をまとめた。SAMは推定IoU最大の出力である。右の二列は、木の切断（適応（色）、精緻化後）から各手法を引いた値で、場面を単位とするブートストラップ法の区間を示した。",
        ["覆う割合", "試行数", "木の切断", "GrabCut", "SAM", "木の切断−GrabCut", "木の切断−SAM"],
        body,
        align=["l", "", "", "", "", "", ""],
        widths=["22%", "8%", "10%", "10%", "10%", "20%", "20%"],
    )


def trend_value(rows: list[dict], other: str) -> float:
    return float(spearmanr([row["coverage"] for row in rows], [row["tree"] - row[other] for row in rows]).statistic)


def coverage_trend(rows: list[dict], other: str, seed: int = 0) -> Interval:
    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(row["scene"], []).append(row)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        picks = rng.integers(0, len(scenes), len(scenes))
        values.append(trend_value([row for pick in picks for row in by_scene[scenes[pick]]], other))
    low, high = np.percentile(values, [LOWER_PERCENTILE, UPPER_PERCENTILE])
    return Interval(trend_value(rows, other), float(low), float(high))


def trend_text(interval: Interval) -> str:
    return f"{signed(interval.mean)}（区間は{signed(interval.low)}から{signed(interval.high)}）"


def dataset_trends(rows: list[dict], other: str) -> list[tuple[str, float]]:
    trends = []
    for key in BASELINE_SETS:
        part = [row for row in rows if row["scene"].startswith(f"{key}-")]
        if len(part) > 2:
            trends.append((SET_NAMES[key], trend_value(part, other)))
    return trends


def trend_sentence(rows: list[dict]) -> str:
    grabcut = dataset_trends(rows, "grabcut")
    sam = dataset_trends(rows, "sam_best")
    values = [value for _, value in grabcut + sam]
    ending = "すべて負である。" if values and all(value < 0 for value in values) else "符号はデータセットにより異なる。"
    return (
        "閾値に依らない見方として、覆う割合と、この差との順位相関（Spearman）を全試行で求めた。"
        f"GrabCutに対して{trend_text(coverage_trend(rows, 'grabcut'))}、SAMに対して{trend_text(coverage_trend(rows, 'sam_best'))}で、"
        "覆う割合が大きいほど提案手法に有利な差は小さい。"
        f"データセットごとの順位相関は、{'、'.join(name for name, _ in grabcut)}の順に、"
        f"GrabCutでは{'、'.join(signed(value) for _, value in grabcut)}、"
        f"SAMでは{'、'.join(signed(value) for _, value in sam)}で、{ending}"
    )


def margin_text(interval: Interval, name: str, ending: str) -> str:
    if above_zero(interval):
        return f"{name}より{f3(interval.mean)}（区間は{f3(interval.low)}から{f3(interval.high)}）{ending}"
    return f"{name}との差は{interval_text(interval)}"


def coverage_summary(study: Study) -> dict[str, str]:
    rows = coverage_rows(study)
    low = stratum(rows, STRATA[0][0], STRATA[0][1])
    high = stratum(rows, STRATA[2][0], STRATA[2][1])
    return {
        "cov_n_low": str(len(low)),
        "cov_n_all": str(len(rows)),
        "cov_rho_grabcut": trend_text(coverage_trend(rows, "grabcut")),
        "cov_rho_sam": trend_text(coverage_trend(rows, "sam_best")),
        "cov_grabcut_low": margin_text(difference_interval(low, "tree", "grabcut"), "GrabCut", "高く"),
        "cov_sam_low": margin_text(difference_interval(low, "tree", "sam_best"), "SAM", "高い"),
        "cov_grabcut_high": interval_text(difference_interval(high, "tree", "grabcut")),
        "cov_sam_high": interval_text(difference_interval(high, "tree", "sam_best")),
    }


def level_range(rows: list[dict], name: str) -> tuple[float, float]:
    means = [mean_by_scene(stratum(rows, low, high), name) for low, high, _ in STRATA]
    return min(means), max(means)


def dataset_margins(rows: list[dict], other: str) -> list[tuple[str, int, float]]:
    low = stratum(rows, STRATA[0][0], STRATA[0][1])
    margins = []
    for key in BASELINE_SETS:
        part = [row for row in low if row["scene"].startswith(f"{key}-")]
        if part:
            margins.append((SET_NAMES[key], len(part), scene_mean(part, lambda row: row["tree"] - row[other])))
    return margins


def dataset_sentence(rows: list[dict]) -> str:
    grabcut = dataset_margins(rows, "grabcut")
    sam = dataset_margins(rows, "sam_best")
    if not grabcut:
        return ""
    counts = "、".join(f"{name}が{count}" for name, count, _ in grabcut)
    values = [value for _, _, value in grabcut + sam]
    ending = (
        "これらの値はすべて正である。" if all(value > 0 for value in values) else "符号はデータセットにより異なる。"
    )
    return (
        f"覆う割合が90%未満の試行をデータセットごとに分けると、試行数は{counts}と少なく、区間は求めていない。"
        f"木の切断からGrabCutを引いた差の平均は{'、'.join(signed(value) for _, _, value in grabcut)}、"
        f"SAMを引いた差の平均は{'、'.join(signed(value) for _, _, value in sam)}（いずれもデータセットの記載順）で、{ending}"
    )


def coverage_results(study: Study) -> str:
    rows = coverage_rows(study)
    sentences = []
    for low, high, label in STRATA:
        part = stratum(rows, low, high)
        parts = []
        for other, name in (("grabcut", "GrabCut"), ("sam_best", "SAM")):
            interval = difference_interval(part, "tree", other)
            parts.append(
                f"{name}に対して{interval_text(interval)}（{relation(interval, '提案手法が高い', '提案手法が低い')}）"
            )
        scenes = len({row["scene"] for row in part})
        sentences.append(f"覆う割合が{label}の{len(part)}試行（{scenes}場面）では、{'、'.join(parts)}。")
    tree_low, tree_high = level_range(rows, "tree")
    grabcut_low, grabcut_high = level_range(rows, "grabcut")
    sam_low, sam_high = level_range(rows, "sam_best")
    return (
        '<p><ref k="s:baselines"></ref>の比較を、楕円の外接矩形が前景を覆う割合で分けて見る。'
        "覆う割合を、正解の前景（0.5で二値化）の面積のうち、外接矩形に入る面積の割合と定義する。"
        "楕円の誤差モデルMでは、楕円の半径が前景の内接楕円の0.8倍まで小さくなり、中心も動くため、前景の一部が矩形の外に出る試行が含まれる。"
        f'公開素材の転写0%と100%、写真を合わせた{len(rows)}試行を、90%と97%を閾値として覆う割合で三つに分けた結果を<ref k="t:coverage"></ref>に示す。'
        "この二つの閾値は結果を見たあとに決めたため、層ごとの比較は探索的であり、別のデータでの確認が必要である。"
        "また、層ごとに六つの区間を示し、多重性の調整はしていない。</p>\n"
        '<p>IoUの差は、<ref k="s:baselines"></ref>とは符号を逆にして、適応（色）の木の切断から各手法を引いた値とし、正のときは提案手法が高い。'
        "場面を単位とするブートストラップ法の区間とあわせて示す。"
        f"{trend_sentence(rows)}</p>\n"
        f"<p>三つの層に分けると、{''.join(sentences)}{dataset_sentence(rows)}</p>\n"
        f"<p>木の切断のIoUの平均は、三つの層で{f3(tree_low)}から{f3(tree_high)}にとどまる。"
        f"これに対して、GrabCutは{f3(grabcut_low)}から{f3(grabcut_high)}、SAMは{f3(sam_low)}から{f3(sam_high)}で、層による差が大きい。"
        "ただし、層ごとにデータセットの構成が異なるため、この差のすべてが覆う割合によるとは限らない。"
        "覆う割合が低い試行では、GrabCutは矩形の外側の画素を背景に固定するため、矩形の外に出た前景を抽出できない。"
        "SAMも、矩形の外側を前景にしにくいと考えられる。"
        "提案手法は、手がかりを楕円の外側の画素にも与え、事前分布も楕円の境界で連続的に変わるため、楕円からはみ出した前景も抽出できる。"
        "この違いが、覆う割合が低い試行での差の理由と考えられるが、本稿では要因を分けて確かめていない。</p>\n"
    )


def coverage_section(study: Study) -> str:
    return coverage_results(study) + "\n" + coverage_table(study)
