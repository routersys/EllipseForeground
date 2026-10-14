from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ellipse_foreground.evaluation.experiments.refinement import BAND_RADII, BAND_SMOOTHNESS
from ellipse_foreground.evaluation.statistics import Interval, bootstrap_interval, scene_means
from ellipse_foreground.settings import DEFAULT_SETTINGS
from paper.numbers import f2, f3, interval_text, percent, plain, signed
from paper.results import PHOTO_SET, SETS, SUBSETS, scene_mean
from paper.sections.baselines import classify
from paper.study import Study
from paper.table import table_html

ADAPTIVE = "adaptive_c"
DEVELOPMENT_MODULUS = 4


@dataclass(frozen=True)
class RefineSet:
    key: str
    name: str
    short: str
    development: bool
    accepts: Callable[[int], bool]


def development_photo(scene: int) -> bool:
    return scene % DEVELOPMENT_MODULUS == 0


def other_photo(scene: int) -> bool:
    return scene % DEVELOPMENT_MODULUS != 0


def accept_all(scene: int) -> bool:
    return True


DEVELOPMENT_SETS = ("sprite1", "collection0", "collection100")
REFINE_SETS = (
    *(RefineSet(key, SUBSETS[key].name, SUBSETS[key].short, key in DEVELOPMENT_SETS, accept_all) for key in SETS),
    RefineSet(PHOTO_SET, "写真・開発用の25枚", "写真の開発用25枚", True, development_photo),
    RefineSet(PHOTO_SET, "写真・それ以外の75枚", "写真のそれ以外75枚", False, other_photo),
)
COLUMN_TITLES = {
    "sprite1": "立ち絵1",
    "sprite2": "立ち絵2",
    "collection0": "17体0%",
    "collection100": "17体100%",
    "public0": "公開0%",
    "public100": "公開100%",
}


def column_title(selection: RefineSet) -> str:
    if selection.key == PHOTO_SET:
        return "写真の開発用" if selection.development else "写真のそれ以外"
    return COLUMN_TITLES[selection.key]


def refinement_rows(study: Study, selection: RefineSet) -> list[dict]:
    comparison = {(record["scene"], record["trial"]): record for record in study.results.trials(selection.key)}
    rows = []
    for record in study.results.refinements(selection.key):
        paired = comparison.get((record["scene"], record["trial"]))
        if paired is not None and selection.accepts(record["scene"]):
            rows.append({**record, "graph": paired["graph"][ADAPTIVE]})
    return rows


def tree_value(measure: str) -> Callable[[dict], float]:
    return lambda row: row["tree"][measure]


def band_value(measure: str, radius: int | None = None) -> Callable[[dict], float]:
    chosen = str(DEFAULT_SETTINGS.band.radius if radius is None else radius)
    return lambda row: row["band"][chosen][measure]


def smoothness_value(setting: float, measure: str) -> Callable[[dict], float]:
    return lambda row: row["smoothness"][f"{setting:g}"][measure]


def graph_value(measure: str) -> Callable[[dict], float]:
    return lambda row: row["graph"][measure]


def agreement_value(name: str, measure: str) -> Callable[[dict], float]:
    return lambda row: row["agreement"][name][measure]


def paired(rows: list[dict], first: Callable[[dict], float], second: Callable[[dict], float]) -> Interval:
    return bootstrap_interval(scene_means(rows, lambda row: first(row) - second(row)))


def mean_of(rows: list[dict], value: Callable[[dict], float]) -> float:
    return scene_mean(rows, value)


def all_rows(study: Study) -> dict[str, list[dict]]:
    return {selection.name: refinement_rows(study, selection) for selection in REFINE_SETS}


def band_table(study: Study, rows: dict[str, list[dict]]) -> str:
    body = []
    for selection in REFINE_SETS:
        data = rows[selection.name]
        body.append(
            [
                selection.short,
                f3(mean_of(data, tree_value("iou"))),
                f3(mean_of(data, band_value("iou"))),
                interval_text(paired(data, band_value("iou"), tree_value("iou"))),
                f3(mean_of(data, graph_value("iou"))),
                interval_text(paired(data, band_value("iou"), graph_value("iou"))),
                interval_text(paired(data, band_value("mae"), tree_value("mae")), 2),
            ]
        )
    return table_html(
        "t:band",
        "木の切断に帯の再判定を加えた結果（適応（色）のスコア、精緻化後）。差は帯の再判定の値から木の切断またはグラフカットの値を引いた値で、場面を単位とするブートストラップ法の区間を付けた。開発用に使ったのは、立ち絵1、17体、写真の25枚である。",
        [
            "データセット",
            "木の切断<br>IoU",
            "帯の再判定<br>IoU",
            "IoUの差（帯−木）<br>［95%信頼区間］",
            "グラフカット<br>IoU",
            "IoUの差（帯−グラフカット）<br>［95%信頼区間］",
            "MAEの差（帯−木）<br>［95%信頼区間］",
        ],
        body,
        align=["l", *[""] * 6],
        widths=["16%", "9%", "10%", "20%", "10%", "20%", "15%"],
    )


def parameter_rows() -> list[tuple[str, Callable[[dict], float]]]:
    chosen_radius = DEFAULT_SETTINGS.band.radius
    chosen_lambda = DEFAULT_SETTINGS.band.smoothness
    entries = [(f"半径{radius}、強さ{plain(chosen_lambda)}", band_value("iou", radius)) for radius in BAND_RADII]
    entries += [
        (f"半径{chosen_radius}、強さ{plain(setting)}", smoothness_value(setting, "iou"))
        for setting in BAND_SMOOTHNESS
        if setting != chosen_lambda
    ]
    return entries


def parameter_table(study: Study, rows: dict[str, list[dict]]) -> str:
    chosen_radius = DEFAULT_SETTINGS.band.radius
    chosen_lambda = DEFAULT_SETTINGS.band.smoothness
    body = []
    for label, value in parameter_rows():
        cells = [label]
        for selection in REFINE_SETS:
            data = rows[selection.name]
            cells.append(
                signed(float(scene_means(data, lambda row, value=value: value(row) - row["tree"]["iou"]).mean()))
            )
        body.append(cells)
    header = ["設定", *[column_title(selection) for selection in REFINE_SETS]]
    return table_html(
        "t:bandparam",
        f"帯の半径と強さへの感度。帯の再判定のIoUから木の切断のIoUを引いた平均を示す（精緻化後）。半径{chosen_radius}、強さ{plain(chosen_lambda)}が選んだ設定である。",
        header,
        body,
        align=["l", *[""] * len(REFINE_SETS)],
    )


def agreement_table(study: Study, rows: dict[str, list[dict]]) -> str:
    body = []
    for selection in REFINE_SETS:
        data = rows[selection.name]
        body.append(
            [
                selection.short,
                interval_text(paired(data, agreement_value("tree", "iou"), tree_value("iou"))),
                interval_text(paired(data, agreement_value("band", "iou"), band_value("iou"))),
            ]
        )
    return table_html(
        "t:agree",
        "手がかりの一致度による重みの調整の効果（精緻化後のIoUの差）。左は木の切断どうし、右は帯の再判定どうしの差で、場面を単位とするブートストラップ法の区間を付けた。",
        ["データセット", "木の切断の差<br>［95%信頼区間］", "帯の再判定の差<br>［95%信頼区間］"],
        body,
        align=["l", "", ""],
        widths=["34%", "33%", "33%"],
    )


def names_of(selections: list[RefineSet]) -> str:
    return "、".join(selection.short for selection in selections)


def row_phrase(chosen: list[RefineSet]) -> str:
    if len(chosen) == len(REFINE_SETS):
        return "表のすべての行"
    if 2 * len(chosen) > len(REFINE_SETS):
        rest = [selection for selection in REFINE_SETS if selection not in chosen]
        return f"{names_of(rest)}を除く{len(chosen)}行"
    return names_of(chosen)


def classified(
    rows: dict[str, list[dict]], first: Callable[[dict], float], second: Callable[[dict], float]
) -> dict[str, list[RefineSet]]:
    groups: dict[str, list[RefineSet]] = {"higher": [], "lower": [], "unclear": []}
    for selection in REFINE_SETS:
        groups[classify(paired(rows[selection.name], first, second))].append(selection)
    return groups


def band_results(study: Study, rows: dict[str, list[dict]]) -> str:
    radius, setting = DEFAULT_SETTINGS.band.radius, DEFAULT_SETTINGS.band.smoothness
    versus_tree = classified(rows, band_value("iou"), tree_value("iou"))
    versus_graph = classified(rows, band_value("iou"), graph_value("iou"))
    error = classified(rows, tree_value("mae"), band_value("mae"))
    held_out = [selection for selection in REFINE_SETS if not selection.development]
    held_out_higher = [selection for selection in held_out if selection in versus_tree["higher"]]
    differences = [paired(rows[selection.name], band_value("iou"), tree_value("iou")).mean for selection in REFINE_SETS]
    tree_text = (
        f"帯の再判定のIoUは、{row_phrase(versus_tree['higher'])}で、信頼区間が0を含まない差で木の切断より高い"
        f"（差の平均は{signed(min(differences))}から{signed(max(differences))}）。"
    )
    if versus_tree["lower"]:
        tree_text += f"{names_of(versus_tree['lower'])}では、信頼区間が0を含まない差で低い。"
    if versus_tree["unclear"] and len(versus_tree["higher"]) < len(REFINE_SETS):
        tree_text += f"{names_of(versus_tree['unclear'])}では、信頼区間が0を含み、差がはっきりしない。"
    held_text = (
        f"開発用に使っていない{len(held_out)}行（{names_of(held_out)}）のすべてで、この差は信頼区間が0を含まない。"
        if len(held_out_higher) == len(held_out)
        else f"開発用に使っていない行のうち、信頼区間が0を含まない差で高いのは、{names_of(held_out_higher) or 'なし'}である。"
    )
    error_text = f"ぼかし描画のMAEは、{row_phrase(error['higher'])}で、信頼区間が0を含まない差で小さい。"
    graph_text = "全画素のグラフカットと比べると、"
    if versus_graph["higher"]:
        graph_text += f"信頼区間が0を含まない差で帯の再判定が高いのは{names_of(versus_graph['higher'])}である。"
    if versus_graph["lower"]:
        graph_text += f"信頼区間が0を含まない差でグラフカットが高いのは{names_of(versus_graph['lower'])}である。"
    rest = len(versus_graph["unclear"])
    if rest == len(REFINE_SETS):
        graph_text += "差は、すべての行で信頼区間が0を含む。"
    elif rest:
        graph_text += f"残りの{rest}行では、信頼区間が0を含み、差がはっきりしない。"
    return (
        f'<p>木の切断のあとに、<ref k="s:band"></ref>の帯の再判定を加えた結果を<ref k="t:band"></ref>に示す。'
        f"帯の半径は{radius}、強さは \\(\\lambda_b={plain(setting)}\\) とした。"
        "この二つは、開発用の画像（立ち絵1、17体、写真の25枚）で選んだ値である。"
        '比較に使うグラフカットは<ref k="s:abl_gc"></ref>と同じもので、強さは \\(\\lambda\\) である。</p>\n'
        f"<p>{tree_text}{held_text}{error_text}{graph_text}</p>"
    )


def sensitivity_text(study: Study, rows: dict[str, list[dict]]) -> str:
    values = [
        float(scene_means(rows[selection.name], lambda row, value=value: value(row) - row["tree"]["iou"]).mean())
        for _, value in parameter_rows()
        for selection in REFINE_SETS
    ]
    best_radius: dict[int, list[RefineSet]] = {}
    for selection in REFINE_SETS:
        means = {
            radius: float(
                np.mean(
                    scene_means(
                        rows[selection.name],
                        lambda row, radius=radius: band_value("iou", radius)(row) - row["tree"]["iou"],
                    )
                )
            )
            for radius in BAND_RADII
        }
        best_radius.setdefault(max(means, key=means.get), []).append(selection)
    positive = all(value > 0 for value in values)
    sign_text = (
        f"試した設定のすべてと、表のすべての行で、差の平均は正である（最小は{signed(min(values))}）。"
        if positive
        else f"差の平均が負になる設定と行がある（最小は{signed(min(values))}）。"
    )
    best_text = "、".join(
        f"半径{radius}が最大になるのは{names_of(selections)}" for radius, selections in sorted(best_radius.items())
    )
    return (
        f'<p><ref k="t:bandparam"></ref>に、帯の半径と強さを変えたときの結果を示す。{sign_text}'
        f"差の平均が最大になる半径は行によって異なり、{best_text}である。"
        "選んだ設定は、すべての行で最良とは限らない。</p>"
    )


def with_means(selections: list[RefineSet], rows: dict[str, list[dict]]) -> str:
    return "、".join(
        f"{selection.short}（{signed(paired(rows[selection.name], agreement_value('tree', 'iou'), tree_value('iou')).mean)}）"
        for selection in selections
    )


def agreement_text(study: Study, rows: dict[str, list[dict]]) -> str:
    tree_groups = classified(rows, agreement_value("tree", "iou"), tree_value("iou"))
    band_groups = classified(rows, agreement_value("band", "iou"), band_value("iou"))
    development = [selection for selection in REFINE_SETS if selection.development]
    held_out = [selection for selection in REFINE_SETS if not selection.development]
    gain_on_development = [selection for selection in development if selection in tree_groups["higher"]]
    gain_held_out = [selection for selection in held_out if selection in tree_groups["higher"]]
    loss_held_out = [selection for selection in held_out if selection in tree_groups["lower"]]
    development_text = (
        f"開発用の{len(development)}行のうち、信頼区間が0を含まない差で木の切断より高いのは{with_means(gain_on_development, rows)}である。"
        if gain_on_development
        else "開発用の行に、信頼区間が0を含まない差で木の切断より高い行はない。"
    )
    held_out_text = (
        f"開発用に使っていない{len(held_out)}行では、信頼区間が0を含まない差で高いのは{with_means(gain_held_out, rows)}である。"
        if gain_held_out
        else f"開発用に使っていない{len(held_out)}行には、信頼区間が0を含まない差で高い行がない。"
    )
    loss_text = f"信頼区間が0を含まない差で低いのは{with_means(loss_held_out, rows)}である。" if loss_held_out else ""
    band_text = (
        f"帯の再判定と併用した場合に、信頼区間が0を含まない差で併用のほうが高いのは{names_of(band_groups['higher'])}である。"
        if band_groups["higher"]
        else "帯の再判定と併用しても、信頼区間が0を含まない差で高い行はない。"
    )
    return (
        "<p>手がかりの信頼度を、三つの手がかりの出力の一致度で見積もる方法も試した。"
        "各手がかりのスコアの画素ごとの値と、ほかの二つの手がかりのスコアの平均との相関を、その手がかりの一致度とする。"
        f"一致度が三つの平均より \\(\\Delta\\) だけ高い手がかりの重みを、\\(1+{plain(DEFAULT_SETTINGS.agreement.gain)}\\Delta\\) 倍する"
        f"（倍率は{plain(DEFAULT_SETTINGS.agreement.floor)}から{plain(DEFAULT_SETTINGS.agreement.ceiling)}の間に収める）。"
        '<ref k="t:agree"></ref>に結果を示す。'
        f"{development_text}{held_out_text}{loss_text}{band_text}"
        f"{agreement_verdict(gain_on_development, gain_held_out, held_out)}</p>"
    )


def agreement_verdict(
    on_development: list[RefineSet], held_out_gain: list[RefineSet], held_out: list[RefineSet]
) -> str:
    if len(held_out_gain) == len(held_out):
        return "設計に使っていない行のすべてで向上したため、採用の候補になる。"
    if on_development and not held_out_gain:
        return "この方法は、開発用の画像では向上を示したが、設計に使っていない画像では信頼区間が0を含まない向上がなかったため、採用しない。"
    return "設計に使っていない画像での向上は一部にとどまり、差も小さいため、採用の根拠としては弱い。"


def timing_text(study: Study, rows: dict[str, list[dict]]) -> str:
    timing = study.timing
    if timing is None or timing.band_ms is None:
        return ""
    return (
        f"<p>最小カットの計算時間を、幅256の画像（\\(N\\) は{timing.pixel_count}）の{timing.instances}場面で測った。"
        f"同じ問題を、KolmogorovによるC++の実装を呼び出す{timing.boykov_kolmogorov_library}で解き、maxflowの呼び出しだけの時間を、"
        f"全画素のグラフと、半径{timing.band_radius}の帯のグラフで比べた。"
        f"帯の画素数は \\(N\\) の平均{100 * timing.band_share:.1f}%で、1回あたりの中央値は、全画素が{timing.band_full_ms:.2f}ミリ秒、"
        f"帯が{timing.band_ms:.2f}ミリ秒で、約{timing.band_speedup:.1f}分の1である。"
        "帯の再判定には、木の切断の計算に加えて、帯のグラフの構築と、辺の重みの計算が必要で、この時間は含めていない。</p>"
    )


def band_mean_share(study: Study, rows: dict[str, list[dict]]) -> float:
    radius = str(DEFAULT_SETTINGS.band.radius)
    values = [row["band"][radius]["share"] for data in rows.values() for row in data]
    return float(np.mean(values))


def agreement_future(study: Study) -> str:
    rows = all_rows(study)
    groups = classified(rows, agreement_value("tree", "iou"), tree_value("iou"))
    held_out = [selection for selection in REFINE_SETS if not selection.development]
    gained = [selection for selection in held_out if selection in groups["higher"]]
    if gained:
        small = all(
            paired(rows[s.name], agreement_value("tree", "iou"), tree_value("iou")).mean < 0.005 for s in gained
        )
        size = "で、向上は小さかった" if small else "である"
        return f'手がかり間の一致を使う方法は、<ref k="s:abl_band"></ref>で試したが、設計に使っていない画像で信頼区間が0を含まない向上があったのは{with_means(gained, rows)}だけ{size}。'
    return '手がかり間の一致を使う方法は、<ref k="s:abl_band"></ref>で試したが、設計に使っていない画像では信頼区間が0を含まない向上がなかった。'


def refinement_facts(study: Study) -> dict:
    rows = all_rows(study)
    versus_tree = classified(rows, band_value("iou"), tree_value("iou"))
    versus_graph = classified(rows, band_value("iou"), graph_value("iou"))
    differences = [paired(rows[s.name], band_value("iou"), tree_value("iou")).mean for s in REFINE_SETS]
    return {
        "tree": versus_tree,
        "graph": versus_graph,
        "low": min(differences),
        "high": max(differences),
        "everywhere": len(versus_tree["higher"]) == len(REFINE_SETS),
        "count": len(REFINE_SETS),
    }


def datasets_phrase() -> str:
    count = len({selection.key for selection in REFINE_SETS})
    numerals = {5: "五", 6: "六", 7: "七", 8: "八"}
    return f"{numerals.get(count, str(count))}つのデータセット（写真は開発用とそれ以外に分けた）"


def abstract_refinement(study: Study) -> str:
    facts = refinement_facts(study)
    radius = DEFAULT_SETTINGS.band.radius
    if facts["everywhere"]:
        tree_text = f"IoUは、{datasets_phrase()}のすべてで、信頼区間が0を含まない差で{signed(facts['low'])}から{signed(facts['high'])}高くなる"
    else:
        tree_text = f"IoUは、{names_of(facts['tree']['higher'])}で、信頼区間が0を含まない差で木の切断より高くなる"
    graph_text = (
        f"。全画素のグラフカットと比べると、{names_of(facts['graph']['higher'])}では帯の再判定のほうが高い"
        if facts["graph"]["higher"]
        else ""
    )
    timing = study.timing
    speed = (
        f"。帯のグラフの最小カットは、全画素のグラフの約{timing.band_speedup:.1f}分の1の時間で解ける"
        if timing is not None and timing.band_speedup is not None
        else ""
    )
    return f"木の切断で得たマスクの境界から{radius}画素以内の帯だけを、画素単位のグラフカットで再判定する段階を加えると、{tree_text}{graph_text}{speed}。"


def discussion_refinement(study: Study) -> str:
    facts = refinement_facts(study)
    radius = DEFAULT_SETTINGS.band.radius
    graph_lower = names_of(facts["graph"]["lower"])
    graph_higher = names_of(facts["graph"]["higher"])
    comparison = (
        f"全画素のグラフカットと比べると、信頼区間が0を含まない差で帯の再判定が高いのは{graph_higher}である。"
        f"グラフカットが高いのは{graph_lower}である。"
        if graph_higher and graph_lower
        else (
            f"全画素のグラフカットが、信頼区間が0を含まない差で帯の再判定より高いのは{graph_lower}である。"
            if graph_lower
            else f"帯の再判定が、信頼区間が0を含まない差で全画素のグラフカットより高いのは{graph_higher}である。"
        )
    )
    effect = (
        f"帯の再判定のIoUは、{datasets_phrase()}のすべてで木の切断より{signed(facts['low'])}から{signed(facts['high'])}高い。"
        if facts["everywhere"]
        else f"帯の再判定のIoUが木の切断より高いのは{names_of(facts['tree']['higher'])}である。"
    )
    residual = (
        "これは、全画素のグラフカットとの差が残る一因と考えられるが、本稿では要因を分けて確かめていない。"
        if graph_lower
        else ""
    )
    return (
        "<p><b>境界の帯の再判定は、木の切断の制約を一部補う。</b>"
        f"{effect}{comparison}"
        f"帯の外のラベルを木の切断のまま固定するため、半径{radius}を超えて木の切断が誤った場所は直らない。"
        f"{residual}"
        "帯の再判定を加えるかどうかは、最適化の速さを優先するか、精度を優先するかで選べる。</p>\n"
    )


def conclusion_refinement(study: Study) -> str:
    facts = refinement_facts(study)
    if facts["everywhere"]:
        return (
            f"木の切断のあとに、境界の帯だけを画素単位のグラフカットで再判定すると、{datasets_phrase()}のすべてで、"
            f"IoUが{signed(facts['low'])}から{signed(facts['high'])}高くなった。"
        )
    return f"木の切断のあとの帯の再判定で、IoUが高くなったのは{names_of(facts['tree']['higher'])}である。"


def refinement_section(study: Study) -> str:
    rows = all_rows(study)
    return (
        band_results(study, rows)
        + "\n"
        + band_table(study, rows)
        + "\n"
        + sensitivity_text(study, rows)
        + "\n"
        + parameter_table(study, rows)
        + "\n"
        + timing_text(study, rows)
        + "\n"
        + agreement_text(study, rows)
        + "\n"
        + agreement_table(study, rows)
    )


def intro_item(study: Study) -> str:
    facts = refinement_facts(study)
    held_out = [selection for selection in REFINE_SETS if not selection.development]
    if facts["everywhere"]:
        return (
            f"{datasets_phrase()}のすべてで、IoUが木の切断より{signed(facts['low'])}から{signed(facts['high'])}高くなる"
            f"（信頼区間は0を含まない）。設計に使っていない{names_of(held_out)}でも同じである。"
        )
    return f"IoUが木の切断より高いのは、{names_of(facts['tree']['higher'])}である。"


def refinement_summary(study: Study) -> dict[str, str]:
    rows = all_rows(study)
    differences = {
        selection.name: paired(rows[selection.name], band_value("iou"), tree_value("iou")) for selection in REFINE_SETS
    }
    held_out = [selection for selection in REFINE_SETS if not selection.development]
    return {
        "band_radius": plain(DEFAULT_SETTINGS.band.radius),
        "band_lambda": plain(DEFAULT_SETTINGS.band.smoothness),
        "band_share": percent(band_mean_share(study, rows)),
        "band_low": signed(min(interval.mean for interval in differences.values())),
        "band_high": signed(max(interval.mean for interval in differences.values())),
        "band_item": intro_item(study),
        "agree_gain": plain(DEFAULT_SETTINGS.agreement.gain),
        "agree_floor": plain(DEFAULT_SETTINGS.agreement.floor),
        "agree_ceiling": plain(DEFAULT_SETTINGS.agreement.ceiling),
        "band_rows": str(len(REFINE_SETS)),
        "band_held": str(len(held_out)),
        "band_mae_low": f2(min(paired(rows[s.name], band_value("mae"), tree_value("mae")).mean for s in REFINE_SETS)),
    }
