from dataclasses import dataclass

import numpy as np

from ellipse_foreground import BoundaryForm
from ellipse_foreground.settings import DEFAULT_SETTINGS
from paper.numbers import ceil3, f2, f3, f3_zero, plain, signed
from paper.quantities import graph_cut_comparison
from paper.results import (
    DESIGN_SETS,
    EARLIER_SETS,
    MODEL_NAMES,
    SETS,
    SUBSETS,
    WORK_WIDTHS,
)
from paper.study import Study
from paper.table import table_html

FORMS = (
    (BoundaryForm.MEAN_WEAKNESS, "平均類似度型"),
    (BoundaryForm.WEAKNESS_SUM, "総和型"),
    (BoundaryForm.MERGE_HEIGHT, "併合高さ型"),
)
FORM_NAMES = dict(FORMS)
SHORT_NAMES = {key: SUBSETS[key].short for key in SETS}
COLUMN_NAMES = {
    "sprite1": "立ち絵1",
    "sprite2": "立ち絵2",
    "collection0": "17体・転写0%",
    "collection100": "17体・転写100%",
    "public0": "公開素材・転写0%",
    "public100": "公開素材・転写100%",
}
STAGES = (
    ("bilinear", "二値マスクの双一次拡大"),
    ("jbu", "ジョイントバイラテラルアップサンプリング"),
    ("jbu_lut", "＋色の補正（全画素）"),
    ("jbu_lutband", "＋色の補正（帯の中だけ）"),
    ("jbu_guided", "＋ガイデッドフィルタ（補正なし）"),
    ("final", "＋色の補正（帯）＋ガイデッドフィルタ（本稿）"),
)
SOURCE_NAMES = (("full", "原寸（幅1280）"), ("x640", "1/2に縮小（幅640）"), ("x320", "1/4に縮小（幅320）"))
SETTLED_STEP = 0.005
MODEL_ORDER = ("M", *MODEL_NAMES)


def exponent_key(exponent: int) -> str:
    return str(exponent)


@dataclass(frozen=True)
class FormSummary:
    exponent: int
    design_mean: float
    lower: float
    upper: float
    oracle: float
    at_set: dict[str, float]


class BoundaryCurves:
    def __init__(self, study: Study) -> None:
        self.study = study
        self.curves: dict[tuple[str, str], dict[int, float]] = {}
        self.oracle: dict[tuple[str, str], dict[int, float]] = {}
        self.none: dict[str, float] = {}
        for key in EARLIER_SETS:
            self.none[key] = study.measure.ablation_mean(key, lambda record: record["boundary"]["none"]["iou"])
            for form, _ in FORMS:
                exponents = sorted(int(name) for name in study.results.ablations(key)[0]["boundary"][form.value])
                self.curves[(form.value, key)] = {
                    exponent: study.measure.ablation_mean(
                        key,
                        lambda record, form=form, exponent=exponent: record["boundary"][form.value][str(exponent)][
                            "iou"
                        ],
                    )
                    for exponent in exponents
                }
                self.oracle[(form.value, key)] = {
                    exponent: study.measure.ablation_mean(
                        key,
                        lambda record, form=form, exponent=exponent: record["boundary"][form.value][str(exponent)][
                            "oracle_iou"
                        ],
                    )
                    for exponent in exponents
                }

    def design_mean(self, form: BoundaryForm, exponent: int) -> float:
        return float(np.mean([self.curves[(form.value, key)][exponent] for key in DESIGN_SETS]))

    def best_exponent(self, form: BoundaryForm) -> int:
        exponents = sorted(self.curves[(form.value, DESIGN_SETS[0])])
        candidates = [e for e in exponents if e - 2 in exponents and e + 2 in exponents]
        return max(candidates, key=lambda exponent: self.design_mean(form, exponent))

    def summary(self, form: BoundaryForm) -> FormSummary:
        exponent = self.best_exponent(form)
        design = self.design_mean(form, exponent)
        oracle = float(np.mean([self.oracle[(form.value, key)][exponent] for key in DESIGN_SETS]))
        return FormSummary(
            exponent,
            design,
            self.design_mean(form, exponent - 2) - design,
            self.design_mean(form, exponent + 2) - design,
            oracle,
            {key: self.curves[(form.value, key)][exponent] for key in EARLIER_SETS},
        )

    def best_for_set(self, form: BoundaryForm, key: str) -> int:
        curve = self.curves[(form.value, key)]
        return max(curve, key=curve.get)


def largest_drop_form(study: Study) -> str:
    curves = BoundaryCurves(study)
    summaries = {form: curves.summary(form) for form, _ in FORMS}
    return FORM_NAMES[min(summaries, key=lambda form: summaries[form].upper)]


def default_exponent() -> int:
    return round(float(np.log2(DEFAULT_SETTINGS.boundary.strength)))


def forms_section(study: Study) -> str:
    curves = BoundaryCurves(study)
    summaries = {form: curves.summary(form) for form, _ in FORMS}
    mean_form, sum_form, height_form = (form for form, _ in FORMS)
    chosen = summaries[mean_form].exponent
    if chosen != default_exponent():
        raise ValueError(
            f"選んだ倍率が既定の設定と一致しません。{2.0**chosen:g}と{DEFAULT_SETTINGS.boundary.strength:g}"
        )
    rows = []
    for form, name in FORMS:
        summary = summaries[form]
        label = f"{name}（採用）" if form is mean_form else name
        rows.append(
            [
                label,
                plain(2.0**summary.exponent),
                *(f3(summary.at_set[key]) for key in EARLIER_SETS),
                f3(summary.design_mean),
                signed(summary.lower),
                signed(summary.upper),
                f3(summary.oracle),
            ]
        )
    header = [
        "境界項の形",
        "選んだ<br>倍率",
        *(COLUMN_NAMES[key] for key in EARLIER_SETS),
        "開発用の<br>画像の平均",
        "倍率を<br>1/4倍",
        "倍率を<br>4倍",
        "正解の<br>事前分布",
    ]
    table = table_html(
        "t:forms",
        "境界項の形ごとの切断のIoU（精緻化前の二値マスク）。倍率は開発用の画像（立ち絵1と17体）の平均を最大にするものを形ごとに選んだ。右の三列は、開発用の画像の平均について、選んだ倍率の1/4倍と4倍のときの変化、および正解スコアを用いたときのIoUである。",
        header,
        rows,
        align=["l", *[""] * 10],
        widths=["15%", "7%", "8.5%", "8.5%", "9%", "9%", "9.5%", "8.5%", "8.5%", "8.5%"],
    )
    design = {form: summaries[form].design_mean for form, _ in FORMS}
    lower = {form: summaries[form].lower for form, _ in FORMS}
    upper = {form: summaries[form].upper for form, _ in FORMS}
    oracle = {form: summaries[form].oracle for form, _ in FORMS}
    strength = {form: 2.0 ** summaries[form].exponent for form, _ in FORMS}
    drop_names = {form: FORM_NAMES[form] for form, _ in FORMS}
    largest_drop = max(upper, key=lambda form: -upper[form])
    best_form = max(oracle, key=oracle.get)
    per_best = {key: 2.0 ** curves.best_for_set(mean_form, key) for key in EARLIER_SETS}
    gap = {
        key: curves.curves[(mean_form.value, key)][chosen] - max(curves.curves[(mean_form.value, key)].values())
        for key in EARLIER_SETS
    }
    spread = max(per_best.values()) / min(per_best.values())
    low_spread = max(-lower[form] for form, _ in FORMS) - min(-lower[form] for form, _ in FORMS)
    text = f"""<p>境界項の三つの形（<ref k="s:boundary"></ref>）を、同じスコア（適応（色））のもとで比べる（<ref k="t:forms"></ref>、<ref k="f:reg"></ref>上段）。倍率は形ごとに、開発用の画像の平均IoUが最大になる値を選んだ。選んだ倍率での開発用の画像の平均IoUは、平均類似度型が{f3(design[mean_form])}、総和型が{f3(design[sum_form])}、併合高さ型が{f3(design[height_form])}で、差は{ceil3(max(design.values()) - min(design.values()))}以内である。境界項の強さを調整すれば、三つの形の精度は同程度になる。</p>
<p>差は倍率への感度に現れる。選んだ倍率を4倍にすると、IoUは併合高さ型で{f3(-upper[height_form])}、総和型で{f3(-upper[sum_form])}、平均類似度型で{f3(-upper[mean_form])}下がり、{drop_names[largest_drop]}の低下が最も大きい。1/4倍にすると、総和型で{f3(-lower[sum_form])}、平均類似度型で{f3(-lower[mean_form])}、併合高さ型で{f3(-lower[height_form])}下がり、形による差は{ceil3(low_spread)}以内である。併合高さ型は、境界辺のうち最も弱い一本の弱さを使う。この値は境界辺の弱さの平均以上なので、同じ倍率でも罰が強くなり、最適な倍率が小さい（併合高さ型が{plain(strength[height_form])}、平均類似度型が{plain(strength[mean_form])}、総和型が{plain(strength[sum_form])}）。正解スコアを用いたときのIoUは、総和型が{f3(oracle[sum_form])}、平均類似度型が{f3(oracle[mean_form])}、併合高さ型が{f3(oracle[height_form])}であり、{drop_names[best_form]}が正解の形を最もよく保つ。</p>
<p>最適な倍率はデータセットによって異なる。平均類似度型では、立ち絵1が{plain(per_best["sprite1"])}、立ち絵2が{plain(per_best["sprite2"])}、17体の転写0%が{plain(per_best["collection0"])}、転写100%が{plain(per_best["collection100"])}で最大になり、最大で{spread:.0f}倍の差がある。倍率を{plain(strength[mean_form])}の一つに決めると、各データセットの最良に対するIoUの低下は、立ち絵1で{f3_zero(-gap["sprite1"])}、立ち絵2で{f3_zero(-gap["sprite2"])}、17体の転写0%で{f3_zero(-gap["collection0"])}、転写100%で{f3_zero(-gap["collection100"])}である。境界項をなくすと、IoUは{f3(curves.none["sprite1"])}、{f3(curves.none["sprite2"])}、{f3(curves.none["collection0"])}、{f3(curves.none["collection100"])}に下がり、いずれのデータセットでも境界項は有効である。</p>"""
    return text + "\n" + table


def graph_cut_values(study: Study) -> dict[str, dict[int, float]]:
    values = {}
    for key in EARLIER_SETS:
        exponents = sorted(int(name) for name in study.results.ablations(key)[0]["graph_cut"])
        values[key] = {
            exponent: study.measure.ablation_mean(
                key, lambda record, exponent=exponent: record["graph_cut"][str(exponent)]
            )
            for exponent in exponents
        }
    return values


def chosen_smoothness(study: Study) -> float:
    values = graph_cut_values(study)
    exponents = sorted(values[DESIGN_SETS[0]])
    pooled = {exponent: float(np.mean([values[key][exponent] for key in DESIGN_SETS])) for exponent in exponents}
    return 2.0 ** max(pooled, key=pooled.get)


def graph_cut_section(study: Study) -> str:
    comparison = graph_cut_comparison(study.measure)
    rows = []
    for key in SETS:
        rows.append(
            [
                SUBSETS[key].name,
                f3(study.mean(key, "adaptive_c")),
                f3(study.mean(key, "adaptive_c", "iou", "graph")),
                study.difference_text(key, "adaptive_c", "adaptive_c", "iou", 3, first_family="graph"),
                f2(study.mean(key, "adaptive_c", "mae")),
                f2(study.mean(key, "adaptive_c", "mae", "graph")),
                study.difference_text(key, "adaptive_c", "adaptive_c", "mae", 2, first_family="graph"),
            ]
        )
    header = [
        "データセット",
        "木の切断<br>IoU",
        "グラフカット<br>IoU",
        "IoUの差<br>［95%信頼区間］",
        "木の切断<br>MAE",
        "グラフカット<br>MAE",
        "MAEの差<br>［95%信頼区間］",
    ]
    table = table_html(
        "t:tvg",
        "木の切断とグラフカットの比較（適応（色）のスコア、精緻化後）。差はグラフカットから木の切断を引いた値で、場面を単位とするブートストラップ法で求めた。",
        header,
        rows,
        align=["l", *[""] * 6],
        widths=["20%", "10%", "12%", "19%", "10%", "12%", "17%"],
    )
    values = graph_cut_values(study)
    best_exponent = {key: max(values[key], key=values[key].get) for key in EARLIER_SETS}
    smoothness = chosen_smoothness(study)
    names = {key: SUBSETS[key].short for key in SETS}
    higher = "、".join(names[key] for key in comparison.higher)
    lower = "、".join(names[key] for key in comparison.lower)
    higher_values = "、".join(signed(comparison.difference[key]) for key in comparison.higher)
    lower_values = "、".join(signed(comparison.difference[key]) for key in comparison.lower)
    if comparison.higher:
        higher_sentence = (
            f"グラフカットのIoUは、{higher}で、信頼区間が0を含まない差で木の切断より高い（差は順に{higher_values}）。"
        )
    else:
        higher_sentence = "信頼区間が0を含まない差でグラフカットのIoUが木の切断より高いデータセットはない。"
    if comparison.lower:
        lower_sentence = f"木の切断のほうが高いのは、{lower}である（差は{lower_values}）。"
    else:
        lower_sentence = "信頼区間が0を含まない差で木の切断のほうが高いデータセットはない。"
    error = comparison.error_difference
    error_note = (
        "いずれもグラフカットのほうが小さく、信頼区間は0を含まない"
        if comparison.smaller_error_everywhere
        else "符号と信頼区間は表のとおりである"
    )
    smallest, largest = min(best_exponent.values()), max(best_exponent.values())
    undecided = [key for key in SETS if key not in comparison.higher + comparison.lower]
    unclear = f"{'、'.join(names[key] for key in undecided)}では、差の信頼区間が0を含む。" if undecided else ""
    dependence = (
        "色転写100%のデータセットどうしでも、17体では木の切断が、公開素材ではグラフカットが高く、この向きは画像のデータセットに依存する。"
        if "collection100" in comparison.lower and "public100" in comparison.higher
        else ""
    )
    paragraphs = [
        "<p>同じスコア \\(u\\)、同じ辺の重み \\(\\omega_e\\)、同じ精緻化のもとで、画素単位のグラフカットと比べる。グラフカットは、"
        '前景の画素の集合を \\(F\\) として、\\(-\\sum_{p\\in F}u(p)+\\lambda\\sum_{(p,q):\\,p\\in F,\\,q\\notin F}\\omega_{pq}\\) を、画素の4近傍のグラフ上で最小カットにより最小にする<cite k="boykov2001"></cite>。'
        f'\\(\\lambda\\) は、開発用の画像の平均IoUが最大になる{plain(smoothness)}とした（<ref k="f:reg"></ref>下段）。</p>',
        f'<p><ref k="t:tvg"></ref>に結果を示す。{higher_sentence}{lower_sentence}{unclear}'
        f"MAEの差は、六つのデータセットの順に{'、'.join(signed(error[key]) for key in SETS)}で、{error_note}。"
        "木の切断が選べるのは、単連結の併合で作った領域だけであり、輪郭が弱い場所で背景の領域と細くつながっていると、その部分だけを分けられないことがある。"
        f"{dependence}"
        f"グラフカットの \\(\\lambda\\) も、データセットごとの最良の値が{plain(2.0**smallest)}から{plain(2.0**largest)}まで変わる。</p>",
    ]
    if study.timing:
        timing = study.timing
        paragraphs.append(
            f"<p>最適化1回の計算時間を、同じスコアのもとで測った。幅256の画像（\\(N\\) は{timing.pixel_count}）の{timing.instances}場面について、計{timing.problems}個のスコアを解いた。"
            f"木の動的計画法（C#、単一スレッド、スコアの合計から切断の復元まで）は1回あたり中央値{timing.tree_ms:.2f}ミリ秒、"
            f"同じ画素グラフに対する素朴な Dinic 法による最小カット（C#）は{timing.graph_cut_ms:.1f}ミリ秒で、約{timing.ratio:.0f}倍である。"
            f"どちらも、木とグラフの構築と、1回目の実行を除いた、2回目の実行の時間を問題数で割った値である。測定は{timing.environment}で行った。"
            "木の構築と境界項の計算は画像ごとに1回で済み、楕円を動かす操作では、スコアの再計算のあとに動的計画法だけを繰り返す。"
            '画像のグラフでは、BoykovとKolmogorovのアルゴリズム<cite k="boykov2004"></cite>のように、最小カットをより速く求める実装もあり、この比は実装によって変わる。</p>'
        )
    return "\n".join(paragraphs) + "\n" + table


def texture_resolution_values(study: Study) -> dict[str, dict[str, dict[str, float]]]:
    return {
        source: {
            variant: {
                key: study.measure.ablation_mean(
                    key, lambda record, source=source, variant=variant: record["texture_resolution"][source][variant]
                )
                for key in EARLIER_SETS
            }
            for variant in ("texture_only", "mix")
        }
        for source, _ in SOURCE_NAMES
    }


def work_width_values(study: Study) -> dict[int, dict[str, tuple[float, float]]]:
    return {
        width: {
            key: (
                study.measure.resolution_mean(key, width, "iou"),
                study.measure.resolution_mean(key, width, "mae"),
            )
            for key in EARLIER_SETS
        }
        for width in WORK_WIDTHS
    }


def resolution_section(study: Study) -> str:
    texture = texture_resolution_values(study)
    widths = work_width_values(study)
    texture_rows = [
        [
            name,
            *(f3(texture[source]["texture_only"][key]) for key in EARLIER_SETS),
            *(f3(texture[source]["mix"][key]) for key in EARLIER_SETS),
        ]
        for source, name in SOURCE_NAMES
    ]
    texture_table = table_html(
        "t:texres",
        "テクスチャの手がかりを計算する入力画像の解像度と、切断のIoU（精緻化前の二値マスク）。「固定の重み」は、テクスチャの重みを1.0、色とパッチの重みを適応（色）の値にした構成である。",
        [
            "テクスチャの入力",
            *(COLUMN_NAMES[key] for key in EARLIER_SETS),
            *(COLUMN_NAMES[key] for key in EARLIER_SETS),
        ],
        texture_rows,
        align=["l", *[""] * 8],
        widths=["24%", *["9.5%"] * 8],
        group=[("", 1), ("テクスチャのみ", 4), ("固定の重み", 4)],
    )
    width_rows = [
        [str(width), *(f"{f3(widths[width][key][0])} / {f2(widths[width][key][1])}" for key in EARLIER_SETS)]
        for width in WORK_WIDTHS
    ]
    width_table = table_html(
        "t:res",
        "縮小画像の幅 \\(L\\) と、前景のIoU / MAE（テクスチャのみ、精緻化後）。17体のデータセットは3場面ごとに1場面で評価した。",
        ["縮小画像の幅 \\(L\\)", *(COLUMN_NAMES[key] for key in EARLIER_SETS)],
        width_rows,
        align=["l", *[""] * 4],
    )
    tex_only = texture
    drop_half = {
        key: tex_only["full"]["texture_only"][key] - tex_only["x640"]["texture_only"][key] for key in EARLIER_SETS
    }
    drop_quarter = {
        key: tex_only["full"]["texture_only"][key] - tex_only["x320"]["texture_only"][key] for key in EARLIER_SETS
    }
    drop_half_mix = {key: tex_only["full"]["mix"][key] - tex_only["x640"]["mix"][key] for key in EARLIER_SETS}

    def iou(width: int, key: str) -> float:
        return widths[width][key][0]

    def step(first: int, second: int, key: str) -> str:
        return signed(iou(second, key) - iou(first, key))

    largest_step = max(abs(iou(384, key) - iou(256, key)) for key in EARLIER_SETS)
    settled = (
        f"変化の絶対値は{ceil3(largest_step)}以下で、幅を256より大きくしても、IoUはほとんど変わらない。"
        if largest_step <= SETTLED_STEP
        else "幅を256より大きくしたときの変化は、データセットにより異なる。"
    )

    text = f"""<p>テクスチャの手がかりを、原寸画像と、縮小した画像から計算したときの比較を<ref k="t:texres"></ref>に示す。入力を1/2（幅640）に縮小すると、テクスチャのみのIoUは、立ち絵1で{f3(drop_half["sprite1"])}、立ち絵2で{f3(drop_half["sprite2"])}、17体の転写0%と100%で{f3(drop_half["collection0"])}と{f3(drop_half["collection100"])}下がる。1/4（幅320）では、それぞれ{f3(drop_quarter["sprite1"])}、{f3(drop_quarter["sprite2"])}、{f3(drop_quarter["collection0"])}、{f3(drop_quarter["collection100"])}下がる。色とパッチを併用した場合も、1/2への縮小で{f3(drop_half_mix["sprite1"])}、{f3(drop_half_mix["sprite2"])}、{f3(drop_half_mix["collection0"])}、{f3(drop_half_mix["collection100"])}下がる。平坦な領域は縮小の平均で失われるため、テクスチャの手がかりは原寸の画素から計算する。</p>
<p>縮小画像の幅 \\(L\\)（<ref k="s:work"></ref>）を変えた結果を<ref k="t:res"></ref>に示す。\\(L=128\\) から256に上げると、テクスチャのみのIoUは、立ち絵1で{step(128, 256, "sprite1")}、17体の転写0%と100%で{step(128, 256, "collection0")}と{step(128, 256, "collection100")}変わり、立ち絵2では{step(128, 256, "sprite2")}変わる。256から384への変化は、立ち絵1で{step(256, 384, "sprite1")}、立ち絵2で{step(256, 384, "sprite2")}、17体で{step(256, 384, "collection0")}と{step(256, 384, "collection100")}である。{settled}</p>"""
    return text + "\n" + texture_table + "\n" + width_table


def prior_values(study: Study) -> dict[str, dict[float, float]]:
    shrinks = sorted(float(name) for name in study.results.ablations("sprite1")[0]["prior"])
    return {
        key: {
            shrink: study.measure.ablation_mean(key, lambda record, shrink=shrink: record["prior"][str(shrink)])
            for shrink in shrinks
        }
        for key in EARLIER_SETS
    }


def prior_section(study: Study) -> str:
    values = prior_values(study)
    shrinks = sorted(values["sprite1"], reverse=True)
    rows = []
    for shrink in shrinks:
        design = float(np.mean([values[key][shrink] for key in DESIGN_SETS]))
        rows.append([plain(shrink), *(f3(values[key][shrink]) for key in EARLIER_SETS), f3(design)])
    columns = list(zip(*[[float(cell) for cell in row[1:]] for row in rows], strict=True))
    best_cells = {(int(np.argmax(column)), index + 1) for index, column in enumerate(columns)}
    table = table_html(
        "t:gamma",
        "事前分布の内側の縮小率 \\(\\gamma\\) を変えたときの切断のIoU（精緻化前の二値マスク、適応（色）のスコア）。各列の最大を太字で示す。",
        ["内側の縮小率 γ", *(COLUMN_NAMES[key] for key in EARLIER_SETS), "開発用の<br>画像の平均"],
        rows,
        align=["l", *[""] * 5],
        best=best_cells,
    )
    best = {key: max(values[key], key=values[key].get) for key in EARLIER_SETS}
    default = DEFAULT_SETTINGS.prior.inside_shrink

    def gap(key: str) -> str:
        return f3(values[key][best[key]] - values[key][default])

    occupancy = {
        "stand": float(np.mean([study.results.occupancy(key) for key in ("sprite1", "sprite2")])),
        "collection": study.results.occupancy("collection0"),
    }
    text = f"""<p>事前分布の内側の縮小率 \\(\\gamma\\)（<ref k="s:prior"></ref>）を変えた結果を<ref k="t:gamma"></ref>に示す。立ち絵1では \\(\\gamma={plain(best["sprite1"])}\\)、立ち絵2では \\(\\gamma={plain(best["sprite2"])}\\) で最大になるのに対し、17体では、転写0%が \\(\\gamma={plain(best["collection0"])}\\)、転写100%が \\(\\gamma={plain(best["collection100"])}\\) で最大になり、内側の正のスコアを弱めるほどIoUが高い。\\(\\gamma={plain(default)}\\) と最良の差は、立ち絵1で{gap("sprite1")}、立ち絵2で{gap("sprite2")}、17体の転写0%で{gap("collection0")}、転写100%で{gap("collection100")}である。誤差モデルMで楕円の内側を前景が占める割合は、立ち絵では約{occupancy["stand"]:.2f}、17体では約{occupancy["collection"]:.2f}であり（<ref k="s:noise"></ref>）、内側の割合が小さいデータセットほど、内側の事前分布を弱めたときのIoUが高い。\\(\\gamma\\) もまた、データセットにより最適値が異なる定数である。</p>"""
    return text + "\n" + table


def finish_values(study: Study) -> dict[tuple[str, str], tuple[float, float]]:
    return {
        (stage, key): (
            study.measure.ablation_mean(key, lambda record, stage=stage: record["finish"][stage]["iou"]),
            study.measure.ablation_mean(key, lambda record, stage=stage: record["finish"][stage]["mae"]),
        )
        for stage, _ in STAGES
        for key in EARLIER_SETS
    }


def finish_section(study: Study) -> str:
    values = finish_values(study)
    rows = [
        [name, *(f"{f3(values[(stage, key)][0])} / {f2(values[(stage, key)][1])}" for key in EARLIER_SETS)]
        for stage, name in STAGES
    ]
    table = table_html(
        "t:finish",
        "精緻化の各段の効果（前景のIoU / MAE）。切断は適応（色）のスコアを使う木の切断で、立ち絵1、立ち絵2、17体の全試行から求めた。",
        ["精緻化", *(COLUMN_NAMES[key] for key in EARLIER_SETS)],
        rows,
        align=["l", *[""] * 4],
        widths=["44%", "14%", "14%", "14%", "14%"],
    )

    def change(first: str, second: str, key: str, index: int = 1, digits: int = 2) -> str:
        return signed(values[(second, key)][index] - values[(first, key)][index], digits)

    text = f"""<p>精緻化の各段を<ref k="t:finish"></ref>に示す。二値マスクを双一次に拡大するだけの場合に対し、ジョイントバイラテラルアップサンプリングによるMAEの変化は、立ち絵1で{change("bilinear", "jbu", "sprite1")}、立ち絵2で{change("bilinear", "jbu", "sprite2")}、17体の転写0%と100%で{change("bilinear", "jbu", "collection0")}と{change("bilinear", "jbu", "collection100")}である。全画素に色の補正をかけると、MAEは立ち絵1で{f2(values[("jbu_lut", "sprite1")][1])}、17体の転写0%で{f2(values[("jbu_lut", "collection0")][1])}と、補正をかけない場合の{f2(values[("jbu", "sprite1")][1])}と{f2(values[("jbu", "collection0")][1])}より大きくなる。補正の対象を境界付近の帯に限ると、MAEは立ち絵1で{f2(values[("jbu_lutband", "sprite1")][1])}、17体の転写0%で{f2(values[("jbu_lutband", "collection0")][1])}と、補正をかけない場合と同程度になる。ガイデッドフィルタを加えたときのIoUの変化は、立ち絵1で{change("jbu_lutband", "final", "sprite1", 0, 3)}、立ち絵2で{change("jbu_lutband", "final", "sprite2", 0, 3)}、17体の転写0%と100%で{change("jbu_lutband", "final", "collection0", 0, 3)}と{change("jbu_lutband", "final", "collection100", 0, 3)}である。</p>"""
    return text + "\n" + table


def model_values(study: Study) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    methods = ("ellipse", "prior", "tex", "oracle")
    values = {}
    for key in EARLIER_SETS:
        for model in MODEL_ORDER:
            if model == "M":
                pairs = {method: (study.mean(key, method), study.mean(key, method, "mae")) for method in methods}
            else:
                pairs = {
                    method: (
                        study.measure.model_mean(key, model, method),
                        study.measure.model_mean(key, model, method, "mae"),
                    )
                    for method in methods
                }
            values[(key, model)] = pairs
    return values


def models_section(study: Study) -> str:
    values = model_values(study)
    methods = ("ellipse", "prior", "tex", "oracle")
    rows = []
    for key in EARLIER_SETS:
        for model in MODEL_ORDER:
            pairs = values[(key, model)]
            rows.append(
                [
                    COLUMN_NAMES[key] if model == "M" else "",
                    model,
                    *(f"{f3(pairs[method][0])} / {f2(pairs[method][1])}" for method in methods),
                ]
            )
    table = table_html(
        "t:models",
        "楕円の誤差モデルを変えたときの前景のIoU / MAE（精緻化後）。誤差モデルMの値は主な結果と同じ試行である。",
        [
            "データセット",
            "誤差<br>モデル",
            "楕円を<br>そのまま",
            "事前分布<br>のみ",
            "テクスチャ<br>のみ",
            "正解の<br>事前分布",
        ],
        rows,
        align=["l", "l", *[""] * 4],
    )
    iou = values
    gain = {cell: iou[cell]["tex"][0] - iou[cell]["prior"][0] for cell in iou}
    ordered = [cell for cell in iou if iou[cell]["tex"][0] > iou[cell]["prior"][0] > iou[cell]["ellipse"][0]]
    exceptions = [cell for cell in iou if cell not in ordered]
    scope = "立ち絵1、立ち絵2、17体の四つのデータセットと四つの誤差モデル"
    sequence = "テクスチャのみ、事前分布のみ、楕円をそのまま使う場合の順にIoUが高い。"
    if not exceptions:
        order = f"{scope}のすべてで、{sequence}"
    else:
        listed = "、".join(
            f"{SHORT_NAMES[key]}の誤差モデル{model}（テクスチャのみ{f3(iou[(key, model)]['tex'][0])}、"
            f"事前分布のみ{f3(iou[(key, model)]['prior'][0])}、楕円をそのまま使う場合{f3(iou[(key, model)]['ellipse'][0])}）"
            for key, model in exceptions
        )
        order = f"{scope}の{len(iou)}通りのうち{len(ordered)}通りで、{sequence}例外は、{listed}である。"
    harder = {key: iou[(key, "L")]["tex"][0] - iou[(key, "T")]["tex"][0] for key in EARLIER_SETS}
    text = f"""<p>楕円の誤差モデルを変えた結果を<ref k="t:models"></ref>に示す（<ref k="s:noise"></ref>）。{order}テクスチャの手がかりによる事前分布のみに対する向上は、{signed(min(gain.values()))}から{signed(max(gain.values()))}の範囲である。誤差が小さいモデルTから大きいモデルLに変えたときのテクスチャのみのIoUの変化は、立ち絵1で{signed(harder["sprite1"])}、立ち絵2で{signed(harder["sprite2"])}、17体の転写0%と100%で{signed(harder["collection0"])}と{signed(harder["collection100"])}である。</p>"""
    return text + "\n" + table
