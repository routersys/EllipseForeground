from ellipse_foreground.evaluation.statistics import Interval
from paper.numbers import above_zero, below_zero, f2, f3, interval_text
from paper.results import SUBSETS
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
    return '\n<h3 id="s:baselines">既存の手法との比較</h3>\n' + baseline_results(study) + "\n" + baseline_table(study)


def sets_where(study: Study, name: str, outcome: str) -> list[str]:
    return [SET_NAMES[key] for key in BASELINE_SETS if classify(study.measure.baseline_interval(key, name)) == outcome]


def abstract_baselines(study: Study) -> str:
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
    )


def baseline_discussion(study: Study) -> str:
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
        '詳しくは<ref k="s:baselines"></ref>に示した。</p>\n'
    )
