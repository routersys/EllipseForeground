from ellipse_foreground.evaluation.statistics import Interval
from paper.numbers import above_zero, below_zero, f2, f3, interval_text
from paper.quantities import mean_auc, mean_color_weight, mean_overlap
from paper.results import CUE_METHODS, METHODS, PHOTO_SET, SETS
from paper.sections.labels import METHOD_LABELS
from paper.study import Study
from paper.table import table_html

NEAR_LOWER_LIMIT = 0.02


def relation(interval: Interval, higher: str = "上回る", lower: str = "下回る") -> str:
    if above_zero(interval):
        return f"信頼区間が0を含まず、{higher}"
    if below_zero(interval):
        return f"信頼区間が0を含まず、{lower}"
    return "信頼区間が0を含み、はっきりしない"


def photo_table(study: Study) -> str:
    iou = [study.mean(PHOTO_SET, method) for method in METHODS]
    error = [study.mean(PHOTO_SET, method, "mae") for method in METHODS]
    candidates = range(len(CUE_METHODS) + 1)
    best = {
        (max(candidates, key=lambda index: iou[index]), 1),
        (min(candidates, key=lambda index: error[index]), 2),
    }
    rows = [[METHOD_LABELS[method], f3(iou[index]), f2(error[index])] for index, method in enumerate(METHODS)]
    return table_html(
        "t:photo",
        "写真での前景のIoUとMAE（精緻化後。各列の最良を太字で示す。正解スコアは上限として別に示し、太字の対象に含めない）。",
        ["構成", "IoU", "MAE"],
        rows,
        align=["l", "", ""],
        best=best,
        widths=["50%", "25%", "25%"],
    )


def position_in(value: float, others: list[float]) -> str:
    if value < min(others) - 0.005:
        return "その下限を下回る"
    if value <= min(others) + NEAR_LOWER_LIMIT:
        return "その下限に近い"
    return "その範囲の内側にある"


def photo_results(study: Study) -> str:
    key = PHOTO_SET
    results = study.results
    mean = study.mean
    interval = study.interval
    scenes = results.scenes(key)
    trials = len(results.trials(key))
    best = max(CUE_METHODS, key=lambda method: mean(key, method))
    regret = mean(key, best) - mean(key, "mix")
    auc = {cue: mean_auc(results, key, cue) for cue in ("texture", "color", "patch")}
    overlap = {cue: mean_overlap(results, key, cue) for cue in ("texture", "color", "patch")}
    others = [mean(other, "adaptive_c") for other in SETS]
    cue_order = ""
    if auc["texture"] < min(auc["color"], auc["patch"]):
        cue_order = (
            "写真では、テクスチャの手がかりの識別性能が三つのうち最も低く、立ち絵1とは順位が逆である。"
            "撮像のノイズや圧縮の影響で平坦な領域が少ないことが理由として考えられるが、本稿では確かめていない。"
        )
    graph = interval(key, "adaptive_c", "adaptive_c", first_family="graph")
    graph_error = interval(key, "adaptive_c", "adaptive_c", "mae", first_family="graph")
    app_tex = interval(key, "app", "tex")
    mix_tex = interval(key, "mix", "tex")
    mix_app = interval(key, "mix", "app")
    return (
        f'<p>写真の人物での評価として、Keらが公開したPPM-100<cite k="ke2022"></cite>の{scenes}枚を使う（<ref k="s:data"></ref>）。'
        "手法の設計にも定数の選択にも使っておらず、境界項の倍率、グラフカットの \\(\\lambda\\)、手がかりの重みを含め、すべての定数は他のデータセットで定めた値のままである。"
        f'色転写はかけない。場面数は{scenes}、試行数は{trials}である。結果を<ref k="t:photo"></ref>に示す。</p>\n'
        f"<p>楕円をそのまま使うとIoUは{f3(mean(key, 'ellipse'))}になる。事前分布のみでは{f3(mean(key, 'prior'))}、テクスチャのみでは{f3(mean(key, 'tex'))}、"
        f"色とパッチのみでは{f3(mean(key, 'app'))}、固定重みでは{f3(mean(key, 'mix'))}、適応（色）では{f3(mean(key, 'adaptive_c'))}、"
        f"適応（色とパッチ）では{f3(mean(key, 'adaptive'))}である。正解スコアを用いると、上限は{f3(mean(key, 'oracle'))}になる。"
        f"手がかりの構成のうちIoUが最大なのは{METHOD_LABELS[best]}（{f3(mean(key, best))}）で、固定重みとの差は{f3(regret)}である。"
        f"適応（色）のIoUは、六つのデータセットでは{f3(min(others))}〜{f3(max(others))}で、写真の{f3(mean(key, 'adaptive_c'))}は{position_in(mean(key, 'adaptive_c'), others)}。</p>\n"
        f"<p>色とパッチのみのテクスチャのみに対するIoUの差は{interval_text(app_tex)}で、{relation(app_tex)}。"
        f"固定重みのテクスチャのみに対する差は{interval_text(mix_tex)}で、{relation(mix_tex)}。"
        f"固定重みの色とパッチのみに対する差は{interval_text(mix_app)}で、{relation(mix_app)}。"
        f"適応（色）の固定重みに対する差は{study.difference_text(key, 'adaptive_c', 'mix')}、適応（色とパッチ）の固定重みに対する差は{study.difference_text(key, 'adaptive', 'mix')}である。</p>\n"
        f"<p>楕円内のAUCの平均は、テクスチャが{f2(auc['texture'])}、色が{f2(auc['color'])}、パッチが{f2(auc['patch'])}である。"
        f"バタチャリヤ係数の平均は、テクスチャが{f2(overlap['texture'])}、色が{f2(overlap['color'])}、パッチが{f2(overlap['patch'])}で、"
        f"適応（色）で定まる色の重みの平均は{f2(mean_color_weight(results, key))}である。{cue_order}</p>\n"
        f"<p>画素単位のグラフカットと木の切断（適応（色）のスコア）の差は、IoUで{interval_text(graph)}、MAEで{interval_text(graph_error, 2)}である。"
        f"差はグラフカットから木の切断を引いた値である。IoUでは{relation(graph, 'グラフカットが高い', '木の切断が高い')}。"
        f"MAEでは{relation(graph_error, 'グラフカットのほうが大きい', 'グラフカットのほうが小さい')}。</p>\n"
    )


def photo_section(study: Study) -> str:
    return '\n<h3 id="s:photo">写真の人物での評価</h3>\n' + photo_results(study) + "\n" + photo_table(study)
