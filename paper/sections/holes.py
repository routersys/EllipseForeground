import numpy as np

from paper.numbers import interval_stack, percent, signed
from paper.sections.pairs import classified_by, paired, value
from paper.sections.refinement import REFINE_SETS, RefineSet, names_of
from paper.study import Study
from paper.table import table_html


def hole_rows(study: Study, selection: RefineSet) -> list[dict]:
    return [record for record in study.results.holes(selection.key) if selection.accepts(record["scene"])]


def all_hole_rows(study: Study) -> dict[str, list[dict]]:
    return {selection.name: hole_rows(study, selection) for selection in REFINE_SETS}


def hole_share(rows: list[dict]) -> float:
    return float(np.mean([row["hole_share"] for row in rows]))


def holes_table(rows: dict[str, list[dict]]) -> str:
    body = []
    for selection in REFINE_SETS:
        data = rows[selection.name]
        body.append(
            [
                selection.short,
                percent(hole_share(data)),
                interval_stack(paired(data, value("fill"), value("tree"))),
                interval_stack(paired(data, value("band"), value("tree"))),
                interval_stack(paired(data, value("fill_band"), value("tree"))),
                interval_stack(paired(data, value("fill_band"), value("band"))),
            ]
        )
    return table_html(
        "t:holes",
        "木の切断のマスクの穴を埋める処理と、帯の再判定との比較（精緻化後のIoU）。穴は、前景に囲まれた背景の連結成分である。左から、マスクの面積に対する穴の面積の割合、穴埋めのあと、帯の再判定のあと、穴埋めのあとに帯の再判定を加えたあとの、木の切断との差と、穴埋めを加えた場合の帯の再判定との差である。場面を単位とするブートストラップ法の区間を付けた。",
        [
            "データセット",
            "穴の割合",
            "穴埋め<br>［95%信頼区間］",
            "帯<br>［95%信頼区間］",
            "穴埋めと帯<br>［95%信頼区間］",
            "穴埋めと帯−帯<br>［95%信頼区間］",
        ],
        body,
        align=["l", "", "", "", "", ""],
        widths=["20%", "10%", "17.5%", "17.5%", "17.5%", "17.5%"],
        dense=True,
    )


def holes_text(rows: dict[str, list[dict]]) -> str:
    fill = classified_by(rows, value("fill"), value("tree"))
    extra = classified_by(rows, value("fill_band"), value("band"))
    band_means = {s: paired(rows[s.name], value("band"), value("tree")).mean for s in REFINE_SETS}
    fill_means = {s: paired(rows[s.name], value("fill"), value("tree")).mean for s in REFINE_SETS}
    shares = {s: fill_means[s] / band_means[s] for s in REFINE_SETS if band_means[s] > 0.005}
    low = [s for s, ratio in shares.items() if ratio < 0.5]
    low_text = (
        f"割合が半分に満たないのは{names_of(low)}で、これらの行では、帯の再判定の向上の半分以上が、穴を埋める効果では説明できない。"
        if low
        else ""
    )
    share_text = (
        f"穴埋めだけで得られる向上は、帯の再判定の向上に対する割合で、{percent_range(list(shares.values()))}である（帯の向上が0.005を超える{len(shares)}行）。{low_text}"
        if shares
        else ""
    )
    if len(fill["higher"]) == len(REFINE_SETS):
        fill_text = "穴埋めは、表のすべての行で、信頼区間が0を含まない差で木の切断より高い。"
    elif fill["higher"]:
        fill_text = f"穴埋めが、信頼区間が0を含まない差で木の切断より高いのは{names_of(fill['higher'])}である。"
    else:
        fill_text = "穴埋めが、信頼区間が0を含まない差で木の切断より高い行はない。"
    if fill["lower"]:
        fill_text += f"低いのは{names_of(fill['lower'])}である。"
    if extra["higher"]:
        extra_text = f"穴埋めのあとに帯の再判定を加えると、帯の再判定だけより、信頼区間が0を含まない差で高いのは{names_of(extra['higher'])}である。"
    else:
        extra_text = "穴埋めのあとに帯の再判定を加えても、帯の再判定だけより、信頼区間が0を含まない差で高い行はない。"
    if extra["lower"]:
        extra_text += f"低いのは{names_of(extra['lower'])}である。"
    return (
        "<p>帯の再判定が直す誤りのうち、どの程度が、マスクの内部の穴によるかを調べた。"
        "木の切断のマスクで、前景に囲まれた背景の連結成分を前景に変えた場合の結果を、帯の再判定と比べる。"
        '結果を<ref k="t:holes"></ref>に示す。'
        f"{fill_text}{share_text}{extra_text}</p>"
    )


def percent_range(values: list[float]) -> str:
    low, high = round(100 * min(values)), round(100 * max(values))
    return f"{low}%" if low == high else f"{low}%から{high}%"


def hole_lead(shares: list[float]) -> str:
    if min(shares) >= 0.5:
        return "帯の再判定の効果の大半は、マスクの穴を埋める効果で説明できる。"
    if max(shares) < 0.5:
        return "帯の再判定の効果のうち、マスクの穴を埋める効果で説明できるのは一部である。"
    return "帯の再判定の効果のうち、マスクの穴を埋める効果で説明できる割合は、データセットにより大きく異なる。"


def holes_section(study: Study) -> str:
    rows = all_hole_rows(study)
    return holes_text(rows) + "\n" + holes_table(rows)


def holes_discussion(study: Study) -> str:
    rows = all_hole_rows(study)
    band_means = {s: paired(rows[s.name], value("band"), value("tree")).mean for s in REFINE_SETS}
    fill_means = {s: paired(rows[s.name], value("fill"), value("tree")).mean for s in REFINE_SETS}
    shares = [fill_means[s] / band_means[s] for s in REFINE_SETS if band_means[s] > 0.005]
    if not shares:
        return ""
    median = float(np.median(shares))
    lead = hole_lead(shares)
    extra = [paired(rows[s.name], value("fill_band"), value("band")).mean for s in REFINE_SETS]
    extra_text = (
        f'穴埋めのあとに帯の再判定を加えた場合と、帯の再判定だけの差は、<ref k="t:holes"></ref>の右端の列のとおり、平均で{signed(min(extra))}から{signed(max(extra))}と小さい。'
        if max(abs(difference) for difference in extra) < 0.005
        else ""
    )
    return (
        f"<p><b>{lead}</b>"
        f"穴埋めだけで得られる向上は、帯の再判定の向上に対する割合で、{percent_range(shares)}（中央値{round(100 * median)}%）である。"
        "穴埋めは最小カットを必要とせず、画素数に比例する時間で計算できる。"
        f"{extra_text}</p>\n"
    )
