from paper.numbers import above_zero, below_zero, ceil3, f2, f3, interval_text, signed
from paper.quantities import (
    degenerate_trials,
    mean_auc,
    mean_color_weight,
    mean_overlap,
    ordering_disagreement,
    rank_correlations,
    share_below,
    trial_count,
)
from paper.results import (
    CUE_METHODS,
    DOSES,
    EARLIER_SETS,
    METHODS,
    PUBLIC_SETS,
    SUBSETS,
)
from paper.study import Study
from paper.table import table_html

METHOD_HEADERS = {
    "ellipse": "楕円を<br>そのまま",
    "prior": "事前分布<br>のみ",
    "tex": "テクスチャ<br>のみ",
    "app": "色と<br>パッチのみ",
    "mix": "固定<br>重み",
    "adaptive_c": "適応<br>（色）",
    "adaptive": "適応（色と<br>パッチ）",
    "oracle": "正解<br>スコア",
}
METHOD_LABELS = {
    "ellipse": "楕円をそのまま使用",
    "prior": "事前分布のみ",
    "tex": "テクスチャのみ",
    "app": "色とパッチのみ",
    "mix": "固定重み",
    "adaptive_c": "適応（色）",
    "adaptive": "適応（色とパッチ）",
    "oracle": "正解スコア",
}
MAIN_ROWS = (*DOSES, "others", "sprite1", "sprite2", "collection0", "collection100", "public0", "public100")
SUMMARY_ROW = MAIN_ROWS.index("sprite1")


def main_tables(study: Study) -> tuple[str, str]:
    iou_rows, mae_rows = [], []
    best_iou: set[tuple[int, int]] = set()
    best_mae: set[tuple[int, int]] = set()
    for row, key in enumerate(MAIN_ROWS):
        iou = [study.mean(key, method) for method in METHODS]
        mae = [study.mean(key, method, "mae") for method in METHODS]
        iou_rows.append([SUBSETS[key].name, *(f3(value) for value in iou)])
        mae_rows.append([SUBSETS[key].name, *(f2(value) for value in mae)])
        candidates = range(len(CUE_METHODS) + 1)
        best_iou.add((row, max(candidates, key=lambda index: iou[index]) + 1))
        best_mae.add((row, min(candidates, key=lambda index: mae[index]) + 1))
    header = ["条件", *(METHOD_HEADERS[method] for method in METHODS)]
    align = ["l", *[""] * len(METHODS)]
    widths = ["28%", *["9%"] * len(METHODS)]
    iou_table = table_html(
        "t:iou",
        "前景のIoU（各行の最大を太字で示す。正解スコアは上限として別に示し、太字の対象に含めない）。",
        header,
        iou_rows,
        align=align,
        sub_rows=(SUMMARY_ROW,),
        best=best_iou,
        widths=widths,
    )
    mae_table = table_html(
        "t:mae",
        "背景のぼかし描画のMAE（0〜255。小さいほどよい。各行の最小を太字で示す）。",
        header,
        mae_rows,
        align=align,
        sub_rows=(SUMMARY_ROW,),
        best=best_mae,
        widths=widths,
    )
    return iou_table, mae_table


def main_results(study: Study) -> str:
    m = study.mean
    d = study.difference_text
    earlier = EARLIER_SETS
    gain = {key: study.difference(key, "prior", "ellipse") for key in earlier}
    texture_gain = {key: study.difference(key, "tex", "prior") for key in earlier}
    cues = ("tex", "app", "mix", "adaptive_c", "adaptive")
    regret = max(max(m(key, method) for method in cues) - m(key, "mix") for key in earlier)
    adaptive_size = max(abs(study.difference(key, "adaptive_c", "mix")) for key in earlier)
    return f"""<p><ref k="t:iou"></ref>と<ref k="t:mae"></ref>に、誤差モデルMでの結果を示す。<ref k="f:bars"></ref>に、六つのデータセットのIoUと、場面を単位とするブートストラップ法による95%信頼区間を示す。</p>
<p><b>事前分布と手がかりの効果。</b>楕円をそのまま使うと、IoUは立ち絵1で{f3(m("sprite1", "ellipse"))}、立ち絵2で{f3(m("sprite2", "ellipse"))}、17体の転写0%と100%で{f3(m("collection0", "ellipse"))}と{f3(m("collection100", "ellipse"))}である。デンドログラムと境界項だけを使う事前分布のみの構成は、これを{signed(gain["sprite1"])}、{signed(gain["sprite2"])}、{signed(gain["collection0"])}、{signed(gain["collection100"])}だけ上げる。{significance_sentence(study, "prior", "ellipse", earlier)}テクスチャの手がかりを加えると、事前分布のみに対してさらに{signed(texture_gain["sprite1"])}、{signed(texture_gain["sprite2"])}、{signed(texture_gain["collection0"])}、{signed(texture_gain["collection100"])}上がる。</p>
<p><b>手がかりの順位はデータセットで異なる。</b>立ち絵1の全12条件では、テクスチャのみのIoUが{f3(m("sprite1", "tex"))}で、色とパッチのみの{f3(m("sprite1", "app"))}を上回る。色とパッチのみは色転写の強さが増えるほど悪化し、星雲の背景では、転写0%の{f3(m("dose0", "app"))}から100%の{f3(m("dose100", "app"))}に下がる。テクスチャのみは{f3(m("dose0", "tex"))}から{f3(m("dose100", "tex"))}にとどまる。17体の転写0%では、色とパッチのみが{f3(m("collection0", "app"))}で、テクスチャのみの{f3(m("collection0", "tex"))}を上回り、順位が立ち絵1と逆になる。転写100%では、テクスチャのみが{f3(m("collection100", "tex"))}、色とパッチのみが{f3(m("collection100", "app"))}で、固定重みが{f3(m("collection100", "mix"))}で最良に近い。17体の転写100%では、色の手がかりの識別性能が下がっても、パッチの手がかりはテクスチャより識別性能が高く（<ref k="s:reliab"></ref>）、併用が有効だったと考えられる。</p>
<p>固定重みのテクスチャのみに対するIoUの差は、立ち絵1で{d("sprite1", "mix", "tex")}、設計に使っていない立ち絵2で{d("sprite2", "mix", "tex")}、17体の転写0%で{d("collection0", "mix", "tex")}、転写100%で{d("collection100", "mix", "tex")}である。信頼区間が0を含まないデータセットでは、いずれも固定重みが上回る。固定重みは、四つのデータセットのどれでも、最良の構成との差が{ceil3(regret)}以内に収まる。</p>
<p><b>重みの適応の効果。</b>適応（色）の固定重みに対するIoUの差は、立ち絵1で{d("sprite1", "adaptive_c", "mix")}、立ち絵2で{d("sprite2", "adaptive_c", "mix")}、17体の転写0%で{d("collection0", "adaptive_c", "mix")}、転写100%で{d("collection100", "adaptive_c", "mix")}である。大きさは最大でも{f3(adaptive_size)}にとどまり、符号はデータセットで異なる。パッチの重みも調整した適応（色とパッチ）は、17体で{d("collection0", "adaptive", "mix")}と{d("collection100", "adaptive", "mix")}と、信頼区間が0を含まない向上を示す。立ち絵1では{d("sprite1", "adaptive", "mix")}で、信頼区間が0を含む。立ち絵2では{d("sprite2", "adaptive", "mix")}である。MAEでも、適応（色とパッチ）は17体で小さくなり（転写0%で{d("collection0", "adaptive", "mix", "mae", 2)}）、ほかのデータセットでは差が小さい。</p>
<p><b>上限との差。</b>正解スコアを用いたときのIoUは、立ち絵1で{f3(m("sprite1", "oracle"))}、立ち絵2で{f3(m("sprite2", "oracle"))}、17体の転写0%と100%で{f3(m("collection0", "oracle"))}と{f3(m("collection100", "oracle"))}である。これは、デンドログラムと精緻化が表現できる形の限界を示す。適応（色）との差は、それぞれ{f3(m("sprite1", "oracle") - m("sprite1", "adaptive_c"))}、{f3(m("sprite2", "oracle") - m("sprite2", "adaptive_c"))}、{f3(m("collection0", "oracle") - m("collection0", "adaptive_c"))}、{f3(m("collection100", "oracle") - m("collection100", "adaptive_c"))}であり、手がかりの精度に改善の余地が残る。{headroom_sentence(study)}</p>"""


def blind_paragraph(study: Study) -> str:
    found = degenerate_trials(study.results, "collection100")
    if found is None:
        return ""
    share = 100.0 * found.trials / found.total_trials
    return (
        f"\n<p><b>手がかりが働かない場面。</b>17体の転写100%には、背景がほぼ一様な壁紙で、強さ100%の色転写により前景が背景と同じ色の一様な塗りになる場面が{found.scenes}場面（{found.trials}試行、全{found.total_trials}試行の{share:.0f}%）ある。"
        'この場面では、三つの手がかりの楕円内のAUCがすべて0.5で、バタチャリヤ係数が1になり（<ref k="f:scatter"></ref>の右下の点）、'
        f"手がかりを使う構成のIoUの平均は{f3(found.cue_iou)}である。"
        f"これらの場面を除くと、17体の転写100%の固定重みのIoUは{f3(found.with_mix)}から{f3(found.without_mix)}になる。</p>"
    )


def significance_sentence(study: Study, first: str, second: str, keys: tuple[str, ...]) -> str:
    if all(above_zero(study.interval(key, first, second)) for key in keys):
        return "四つのデータセットのすべてで、信頼区間は0を含まない。"
    return ""


def headroom_sentence(study: Study) -> str:
    gaps = [study.mean(key, "oracle") - study.mean(key, "adaptive_c") for key in ("collection0", "collection100")]
    bound = int(min(gaps) * 100) / 100
    return f"特に17体では差が{bound:.2f}以上あり、本稿の手がかりと楕円だけでは前景の形を十分に決められない。"


def public_table(study: Study) -> str:
    columns = []
    for key in PUBLIC_SETS:
        columns.append([study.mean(key, method) for method in METHODS])
        columns.append([study.mean(key, method, "mae") for method in METHODS])
    best = set()
    for column, values in enumerate(columns):
        choose = max if column % 2 == 0 else min
        best.add((choose(range(len(CUE_METHODS) + 1), key=lambda index: values[index]), column + 1))
    rows = [
        [
            METHOD_LABELS[method],
            f3(columns[0][index]),
            f2(columns[1][index]),
            f3(columns[2][index]),
            f2(columns[3][index]),
        ]
        for index, method in enumerate(METHODS)
    ]
    return table_html(
        "t:new",
        "公開素材での前景のIoUとMAE（精緻化後。各列の最良を太字で示す。正解スコアは上限として別に示し、太字の対象に含めない）。",
        ["構成", "IoU", "MAE", "IoU", "MAE"],
        rows,
        align=["l", "", "", "", ""],
        best=best,
        widths=["32%", "17%", "17%", "17%", "17%"],
        group=[("", 1), ("転写0%", 2), ("転写100%", 2)],
    )


def public_results(study: Study) -> str:
    keys = PUBLIC_SETS
    zero, full = keys

    def means(method: str) -> tuple[float, float]:
        return study.mean(zero, method), study.mean(full, method)

    best = {key: max(CUE_METHODS, key=lambda method: study.mean(key, method)) for key in keys}
    regret = {key: study.mean(key, best[key]) - study.mean(key, "mix") for key in keys}
    scenes = study.results.scenes(zero)
    trials = len(study.results.trials(zero))
    ellipse, adaptive, prior, oracle = means("ellipse"), means("adaptive_c"), means("prior"), means("oracle")
    d = study.difference_text
    text = (
        f'<p>手法の設計と定数の選択に使っていない公開素材の17枚（<ref k="s:data"></ref>）で評価した結果を、<ref k="t:iou"></ref>と<ref k="t:mae"></ref>の最後の2行に示す。'
        "境界項の倍率、グラフカットの \\(\\lambda\\)、手がかりの重みを含め、すべての定数は他のデータセットで定めた値のままである。"
        f"場面数は転写0%と100%のそれぞれで{scenes}、試行数は{trials}である。</p>\n"
        f"<p>楕円をそのまま使うとIoUは転写0%で{f3(ellipse[0])}、100%で{f3(ellipse[1])}になる。適応（色）では{f3(adaptive[0])}と{f3(adaptive[1])}になり、事前分布のみでは{f3(prior[0])}と{f3(prior[1])}である。"
        f"正解スコアを用いると、上限は{f3(oracle[0])}と{f3(oracle[1])}になる。"
        f"手がかりの構成のうち、IoUが最大なのは、転写0%で{METHOD_LABELS[best[zero]]}（{f3(study.mean(zero, best[zero]))}）、100%で{METHOD_LABELS[best[full]]}（{f3(study.mean(full, best[full]))}）である。"
        f"固定重みと最良の構成のIoUの差は、転写0%で{f3(regret[zero])}、100%で{f3(regret[full])}にとどまる。</p>\n"
        "<p>構成どうしの差は、場面を単位とする信頼区間とあわせて次のとおりである。"
        f"固定重みのテクスチャのみに対するIoUの差は、転写0%で{d(zero, 'mix', 'tex')}、100%で{d(full, 'mix', 'tex')}である。"
        f"固定重みの色とパッチのみに対する差は、転写0%で{d(zero, 'mix', 'app')}、100%で{d(full, 'mix', 'app')}である。"
        f"適応（色）の固定重みに対する差は、転写0%で{d(zero, 'adaptive_c', 'mix')}、100%で{d(full, 'adaptive_c', 'mix')}で、"
        f"適応（色とパッチ）では{d(zero, 'adaptive', 'mix')}と{d(full, 'adaptive', 'mix')}である。</p>\n"
    )
    if all(above_zero(study.interval(key, "mix", other)) for key in keys for other in ("tex", "app")):
        text += "<p>固定重みは、テクスチャのみと色とパッチのみのどちらに対しても、転写0%と100%の両方で、信頼区間が0を含まない差で上回る。</p>\n"
    graph = {key: study.interval(key, "adaptive_c", "adaptive_c", first_family="graph") for key in keys}
    graph_error = {key: study.interval(key, "adaptive_c", "adaptive_c", "mae", first_family="graph") for key in keys}
    if all(above_zero(graph[key]) and below_zero(graph_error[key], 2) for key in keys):
        verdict = "転写0%と100%のどちらでも、グラフカットのIoUが高く、MAEが小さい。"
    else:
        verdict = "符号は条件により異なる。"
    text += (
        f"<p>画素単位のグラフカットと木の切断（適応（色）のスコア）の差は、IoUで転写0%が{interval_text(graph[zero])}、100%が{interval_text(graph[full])}、"
        f"MAEで転写0%が{interval_text(graph_error[zero])}、100%が{interval_text(graph_error[full])}である。"
        f"差はグラフカットから木の切断を引いた値で、{verdict}</p>\n"
    )
    return text


def dose_results(study: Study) -> str:
    results = study.results
    overlap = {dose: {cue: mean_overlap(results, dose, cue) for cue in ("color", "texture", "patch")} for dose in DOSES}
    auc = {dose: {cue: mean_auc(results, dose, cue) for cue in ("color", "texture", "patch")} for dose in DOSES}
    weight = {dose: mean_color_weight(results, dose) for dose in DOSES}
    first, last = DOSES[0], DOSES[-1]
    gaps = [study.mean(dose, "mix") - study.mean(dose, "tex") for dose in DOSES]
    largest = max(abs(gap) for gap in gaps)
    if largest <= 0.005:
        mixed = f"いずれも絶対値が{ceil3(largest)}以下で、同程度である。"
    elif gaps[-1] < gaps[0]:
        mixed = "転写が小さい側では同程度か上回り、大きい側では下回る。"
    else:
        mixed = "符号は条件により異なる。"
    differences = "、".join(signed(gap) for gap in gaps)
    return f"""<p>色転写の量を変えたときの、手がかりの性質と結果を<ref k="f:dose"></ref>に示す。星雲の背景に対する立ち絵1の5条件である。</p>
<p>楕円の内側と外側の分布のバタチャリヤ係数は、色で、転写0%の{f2(overlap[first]["color"])}から100%の{f2(overlap[last]["color"])}に増え、分布が重なっていく。テクスチャの係数は{f2(overlap[first]["texture"])}から{f2(overlap[last]["texture"])}でほとんど変わらず、パッチの係数は{f2(overlap[first]["patch"])}から{f2(overlap[last]["patch"])}と小さいままである。楕円内の前景と背景を分けるAUC（正解から求めた値）は、色が{f2(auc[first]["color"])}から{f2(auc[last]["color"])}に、パッチが{f2(auc[first]["patch"])}から{f2(auc[last]["patch"])}に下がるのに対し、テクスチャは{f2(auc[first]["texture"])}から{f2(auc[last]["texture"])}でほぼ一定である。色の係数の増加は、色のAUCの低下と対応する。式(<ref k="e:wcol" bare="1"></ref>)で定めた色の重みは、{f2(weight[first])}から{f2(weight[last])}に下がる。</p>
<p>前景のIoUは、色とパッチのみでは{f3(study.mean(first, "app"))}から{f3(study.mean(last, "app"))}に下がるのに対し、テクスチャのみは{f3(study.mean(first, "tex"))}から{f3(study.mean(last, "tex"))}への低下にとどまる。固定重みのテクスチャのみに対するIoUの差は、転写0%、25%、50%、75%、100%の順に{differences}で、{mixed}適応（色）の固定重みに対する差は、転写100%で{signed(study.mean(last, "adaptive_c") - study.mean(last, "mix"))}、テクスチャのみに対する差は{signed(study.mean(last, "adaptive_c") - study.mean(last, "tex"))}である。</p>"""


def reliability_results(study: Study) -> str:
    results = study.results
    correlations = rank_correlations(results)
    texture, color, patch = (correlations[cue] for cue in ("texture", "color", "patch"))
    total = trial_count(results)
    patch_share = share_below(results, "patch", 0.1)
    auc = {key: {cue: mean_auc(results, key, cue) for cue in ("texture", "color", "patch")} for key in EARLIER_SETS}
    names = {"texture": "テクスチャ", "color": "色", "patch": "パッチ"}
    best_cue = names[max(auc["sprite1"], key=lambda cue: auc["sprite1"][cue])]
    disagreement = 100.0 * ordering_disagreement(results)
    top = "dose100"
    top_overlap = {cue: mean_overlap(results, top, cue) for cue in ("color", "texture")}
    top_auc = {cue: mean_auc(results, top, cue) for cue in ("color", "texture")}
    set_names = (
        ("sprite1", "立ち絵1"),
        ("sprite2", "立ち絵2"),
        ("collection0", "17体の転写0%"),
        ("collection100", "17体の転写100%"),
    )

    def per_set(correlation) -> str:
        return "、".join(f"{name}で{signed(correlation.per_set[key], 2)}" for key, name in set_names)

    return f"""<p>全試行について、バタチャリヤ係数と楕円内のAUCの関係を<ref k="f:scatter"></ref>に示す。係数が小さいほどAUCが高い関係が、テクスチャと色で見られる。立ち絵1、立ち絵2、17体の全{total}試行の順位相関（Spearman）は、テクスチャで{signed(texture.pooled, 2)}、色で{signed(color.pooled, 2)}である。データセットごとの順位相関は、テクスチャでは{per_set(texture)}、色では{per_set(color)}である。一方、パッチの係数は{patch_share * 100:.0f}%の試行で0.1未満に集まり、順位相関は{signed(patch.pooled, 2)}と小さい。パッチの係数は識別性能の差を区別できないため、重みの調整には使わない。</p>
<p>データセットごとの楕円内の平均AUCは、17体の転写100%でテクスチャが{auc["collection100"]["texture"]:.2f}、色が{auc["collection100"]["color"]:.2f}、パッチが{auc["collection100"]["patch"]:.2f}であり、色の識別性能が低くても、パッチはテクスチャを上回る。立ち絵1では、テクスチャが{auc["sprite1"]["texture"]:.2f}、色が{auc["sprite1"]["color"]:.2f}、パッチが{auc["sprite1"]["patch"]:.2f}で、{best_cue}が最も高い。</p>
<p>係数は、手がかりの種類の間では直接比べにくい。色とテクスチャの係数の大小と、AUCの大小が食い違う試行は、{total}試行の{disagreement:.0f}%を占める。星雲の背景で転写100%の条件では、色の係数が{f2(top_overlap["color"])}、色のAUCが{f2(top_auc["color"])}であるのに対し、テクスチャの係数は{f2(top_overlap["texture"])}、AUCは{f2(top_auc["texture"])}である。係数は、楕円の内側と外側という、境界から離れた画素の標本どうしの分布の離れを測る。境界付近の背景の分布は、外側の標本とは異なることがある。標本を取る領域と判定する領域とで特徴量の分布が異なるこの状況は、共変量シフト（covariate shift）と呼ばれる<cite k="shimodaira2000"></cite>。その大きさは、手がかりの種類ごとに異なると考えられる。このため、色の重みは、基準とするテクスチャの係数との比で定める（式(<ref k="e:wcol" bare="1"></ref>)）。</p>
<p>楕円の内側と外側を、前景と背景の誤りを含むラベルとみなすと、この関係の理論的な背景は次のとおりである。クラスごとに一定の割合でラベルが入れ替わる場合、誤ったラベルに対するAUCは、正しいラベルに対するAUCの一次関数になる<cite k="menon2015"></cite>。したがって、同じ標本を使う二つの手がかりの識別性能の順序は、誤ったラベルのもとでも保たれる。実際には、入れ替わりが画素の位置や外観に依存するため、この仮定は成り立たない。<ref k="f:scatter"></ref>の順位相関の絶対値も、テクスチャで{abs(texture.pooled):.2f}、色で{abs(color.pooled):.2f}であり、単調な関係を表す1には届かない。</p>"""


def main_section(study: Study) -> str:
    iou_table, mae_table = main_tables(study)
    return (
        '<h3 id="s:main">主な結果</h3>\n'
        + main_results(study)
        + blind_paragraph(study)
        + '\n<figure id="f:bars"><img src="figs/fig_bars.svg" alt="データセットごとのIoU" style="width:100%"><figcaption><span class="no">図</span>データセットごとの前景のIoU（誤差モデルM、精緻化後）。棒は場面を単位とする平均、縦線は95%信頼区間（ブートストラップ法）、点線は正解スコアを用いた上限である。</figcaption></figure>\n'
        + iou_table
        + "\n"
        + mae_table
        + '\n<h3 id="s:public">公開素材での評価</h3>\n'
        + public_results(study)
        + "\n"
        + public_table(study)
        + '\n<h3 id="s:dose">色転写の量と手がかり</h3>\n'
        + dose_results(study)
        + '\n<figure id="f:dose"><img src="figs/fig_dose.svg" alt="色転写の量と手がかり" style="width:100%"><figcaption><span class="no">図</span>色転写の量に対する、(a)内側と外側の分布のバタチャリヤ係数、(b)手がかりの識別性能（楕円内のAUC）と色の重み、(c)前景のIoU（立ち絵1、星雲、各8試行）。</figcaption></figure>\n'
        + '<h3 id="s:reliab">信頼度の推定</h3>\n'
        + reliability_results(study)
        + '\n<figure id="f:scatter"><img src="figs/fig_scatter.svg" alt="係数とAUC" style="width:100%"><figcaption><span class="no">図</span>バタチャリヤ係数と楕円内のAUC（1点が1試行、立ち絵1、立ち絵2、17体の全試行）。係数は正解を使わずに求め、AUCは正解から求めた。(c)のパッチの係数は0付近に集まる。</figcaption></figure>\n'
    )
