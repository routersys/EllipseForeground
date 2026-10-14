from ellipse_foreground.evaluation.statistics import Interval, bootstrap_interval, scene_means
from paper.numbers import above_zero, below_zero, f2, f3, interval_text
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
        + '\n<h3 id="s:coverage">入力が前景を覆う割合による違い</h3>\n'
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
        f"一方、外接矩形が前景の90%未満しか覆わない{summary['cov_n_low']}試行では、提案手法のIoUの差は、GrabCutに対して{summary['cov_grabcut_low']}、SAMに対して{summary['cov_sam_low']}である。"
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
        "提案手法は学習を使わず、楕円の丸みと内側の割合を事前分布に使う。GrabCutとSAMは矩形だけを使い、入力の情報が同じではない。"
        f"一方、外接矩形が前景の90%未満しか覆わない{shares['cov_n_low']}試行では、提案手法のIoUの差が、GrabCutに対して{shares['cov_grabcut_low']}、SAMに対して{shares['cov_sam_low']}で、提案手法のほうが高い。"
        "覆う割合が97%以上の試行では、GrabCutとSAMのほうが高い。提案手法の強みは、入力が前景を覆わない粗い場合の頑健さである。"
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
        widths=["15%", "9%", "11%", "11%", "11%", "21.5%", "21.5%"],
    )


def coverage_summary(study: Study) -> dict[str, str]:
    rows = coverage_rows(study)
    low = stratum(rows, STRATA[0][0], STRATA[0][1])
    high = stratum(rows, STRATA[2][0], STRATA[2][1])
    return {
        "cov_n_low": str(len(low)),
        "cov_n_all": str(len(rows)),
        "cov_grabcut_low": interval_text(difference_interval(low, "tree", "grabcut")),
        "cov_sam_low": interval_text(difference_interval(low, "tree", "sam_best")),
        "cov_grabcut_high": interval_text(difference_interval(high, "tree", "grabcut")),
        "cov_sam_high": interval_text(difference_interval(high, "tree", "sam_best")),
    }


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
        sentences.append(f"覆う割合が{label}の{len(part)}試行では、提案手法のIoUの差は、{'、'.join(parts)}である。")
    return (
        "<p>入力の矩形が前景を覆う割合によって、結果は大きく変わる。"
        "覆う割合を、正解の前景（0.5で二値化）のうち、楕円の外接矩形に入る面積の割合と定義する。"
        "楕円の誤差モデルMでは、楕円の半径が前景の内接楕円の0.8倍まで小さくなるため、矩形が前景の一部を切る試行が含まれる。"
        f'公開素材の転写0%と100%、写真の{len(rows)}試行を、覆う割合で三つに分けた結果を<ref k="t:coverage"></ref>に示す。'
        "区切りの90%と97%は、結果を見たあとに決めたもので、探索的な分析である。</p>\n"
        f"<p>{''.join(sentences)}</p>\n"
        "<p>矩形が前景を十分に覆うときは、GrabCutとSAMのほうが高い。矩形が前景を覆わないときは、GrabCutは矩形の外側を背景に固定し、SAMも矩形を前景を囲むものとして扱うため、切られた部分を取り戻しにくい。"
        "提案手法は、手がかりを楕円の外側の画素にも与え、事前分布も楕円の境界で連続的に変わるため、楕円からはみ出した前景も取れる。"
        "この違いが、覆う割合が低い試行での差の理由と考えられるが、本稿では要因を分けて確かめていない。</p>\n"
    )


def coverage_section(study: Study) -> str:
    return coverage_results(study) + "\n" + coverage_table(study)
