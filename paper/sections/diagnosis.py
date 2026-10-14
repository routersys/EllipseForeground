import numpy as np

from ellipse_foreground.evaluation.experiments.oracle import ERODE_DEPTH
from ellipse_foreground.evaluation.experiments.unmixing import OCCUPANCY_EDGES
from paper.numbers import f2, f3, interval_stack, plain, signed
from paper.results import PHOTO_SET, SETS
from paper.sections.pairs import classified_by, paired, value
from paper.sections.refinement import REFINE_SETS, RefineSet, datasets_phrase, names_of, row_phrase
from paper.study import Study
from paper.table import table_html

PURE_RING = 0.95
REPLACEMENTS = (
    ("ellipse", "楕円"),
    ("prior", "事前分布"),
    ("samples", "標本"),
    ("both", "事前分布と標本"),
)
DECOMPOSITION = (
    ("ring_far", "遠い背景"),
    ("ring_near", "近い背景"),
    ("core_same", "前景（同じ範囲）"),
    ("core_all", "前景（画像全体）"),
)
MIXTURES = (
    ("unmix_truth", "正解の占有率"),
    ("unmix_tree", "切断から求めた占有率"),
    ("unmix_population", "集団の占有率"),
)


def oracle_rows(study: Study, selection: RefineSet) -> list[dict]:
    return [record for record in study.results.oracles(selection.key) if selection.accepts(record["scene"])]


def all_oracle_rows(study: Study) -> dict[str, list[dict]]:
    return {selection.name: oracle_rows(study, selection) for selection in REFINE_SETS}


def mean_purity(rows: list[dict], part: str) -> float | None:
    values = [row["purity"][part] for row in rows if row["purity"][part] is not None]
    return float(np.mean(values)) if values else None


def purity_text(rows: list[dict], part: str) -> str:
    result = mean_purity(rows, part)
    return "−" if result is None else f2(result)


def difference_cells(rows: list[dict], names: tuple[tuple[str, str], ...], reference: str = "tree") -> list[str]:
    return [interval_stack(paired(rows, value(name), value(reference))) for name, _ in names]


def oracle_table(rows: dict[str, list[dict]]) -> str:
    body = []
    for selection in REFINE_SETS:
        data = rows[selection.name]
        mean = float(np.mean([row["tree"]["iou"] for row in data]))
        body.append([selection.short, f3(mean), *difference_cells(data, REPLACEMENTS)])
    return table_html(
        "t:oracle",
        "正解から作った入力に置き換えたときのIoU（精緻化後）。右の四列は、置き換えたIoUから木の切断のIoUを引いた差で、場面を単位とするブートストラップ法の区間を付けた。",
        [
            "データセット",
            "木の切断<br>IoU",
            "楕円を正解に<br>［95%信頼区間］",
            "事前分布だけを正解に<br>［95%信頼区間］",
            "標本を正解に<br>［95%信頼区間］",
            "事前分布と標本を正解に<br>［95%信頼区間］",
        ],
        body,
        align=["l", "", "", "", "", ""],
        widths=["18%", "8%", "18.5%", "18.5%", "18.5%", "18.5%"],
        dense=True,
    )


def decomposition_table(rows: dict[str, list[dict]]) -> str:
    body = []
    for selection in REFINE_SETS:
        data = rows[selection.name]
        body.append(
            [
                selection.short,
                *difference_cells(data, DECOMPOSITION),
                purity_text(data, "core"),
                purity_text(data, "ring"),
            ]
        )
    return table_html(
        "t:oracle2",
        "標本の置き換えの内訳（精緻化後のIoUの木の切断との差）。背景の標本は、正規化半径が1.3より大きい遠い範囲と、1.3以下の近い範囲に分けて、正解の背景に置き換えた。前景の標本は、正規化半径が0.5未満の範囲と、画像全体に分けて、正解の前景に置き換えた。右の二列は、置き換える前の標本の純度（標本の画素のうち、正解が同じ側の画素の割合）の平均である。",
        [
            "データセット",
            "遠い背景<br>［95%信頼区間］",
            "近い背景<br>［95%信頼区間］",
            "前景（同じ範囲）<br>［95%信頼区間］",
            "前景（画像全体）<br>［95%信頼区間］",
            "前景の純度",
            "背景の純度",
        ],
        body,
        align=["l", "", "", "", "", "", ""],
        widths=["18%", "15.5%", "15.5%", "15.5%", "15.5%", "10%", "10%"],
        dense=True,
    )


def mixture_table(rows: dict[str, list[dict]]) -> str:
    body = []
    for selection in REFINE_SETS:
        data = rows[selection.name]
        body.append(
            [
                selection.short,
                interval_stack(paired(data, value("samples_ct"), value("tree"))),
                *difference_cells(data, MIXTURES),
            ]
        )
    return table_html(
        "t:unmix",
        "混合分解で取り戻せる向上（精緻化後のIoUの木の切断との差）。左から、色とテクスチャの標本だけを正解の領域に置き換えた場合と、標本を置き換えずに、環ごとの前景の占有率から色とテクスチャの分布を混合分解で求めた場合である。占有率は、正解、木の切断、集団の平均のそれぞれから求めた。",
        [
            "データセット",
            "色とテクスチャの標本を正解に<br>［95%信頼区間］",
            "正解の占有率<br>［95%信頼区間］",
            "切断から求めた占有率<br>［95%信頼区間］",
            "集団の占有率<br>［95%信頼区間］",
        ],
        body,
        align=["l", "", "", "", ""],
        widths=["18%", "22.5%", "19.5%", "20%", "20%"],
        dense=True,
    )


def mean_gain(rows: dict[str, list[dict]], name: str) -> dict[RefineSet, float]:
    return {selection: paired(rows[selection.name], value(name), value("tree")).mean for selection in REFINE_SETS}


def span(gains: dict[RefineSet, float]) -> str:
    return f"{signed(min(gains.values()))}から{signed(max(gains.values()))}"


def largest(gains: dict[RefineSet, float]) -> RefineSet:
    return max(gains, key=gains.get)


def smallest(gains: dict[RefineSet, float]) -> RefineSet:
    return min(gains, key=gains.get)


def percent_span(values: list[float]) -> str:
    low, high = round(100 * min(values)), round(100 * max(values))
    return f"{low}%" if low == high else f"{low}%から{high}%"


def higher_sentence(label: str, groups: dict[str, list[RefineSet]]) -> str:
    if len(groups["higher"]) == len(REFINE_SETS):
        return f"{label}は、表のすべての行で、信頼区間が0を含まない差で木の切断より高い。"
    if groups["higher"]:
        return f"{label}に、信頼区間が0を含まない差で木の切断より高いのは{row_phrase(groups['higher'])}である。"
    return f"{label}に、信頼区間が0を含まない差で木の切断より高い行はない。"


def significance(rows: dict[str, list[dict]], name: str, label: str) -> str:
    return higher_sentence(label, classified_by(rows, value(name), value("tree")))


def comparison_sentence(larger_label: str, smaller_label: str, larger_rows: list[RefineSet]) -> str:
    if len(larger_rows) == len(REFINE_SETS):
        return f"{larger_label}は、表のすべての行で、{smaller_label}より大きい。"
    if not larger_rows:
        return f"{larger_label}が{smaller_label}より大きい行はない。"
    return f"{larger_label}が{smaller_label}より大きいのは{row_phrase(larger_rows)}である。"


def dataset_phrase(chosen: list[RefineSet]) -> str:
    if len(chosen) == len(REFINE_SETS):
        return f"{datasets_phrase()}のすべて"
    if 2 * len(chosen) > len(REFINE_SETS):
        rest = [selection for selection in REFINE_SETS if selection not in chosen]
        return f"{names_of(rest)}を除くデータセット"
    return names_of(chosen)


def prior_share(rows: dict[str, list[dict]]) -> dict[RefineSet, float]:
    ellipse, prior = mean_gain(rows, "ellipse"), mean_gain(rows, "prior")
    return {selection: prior[selection] / ellipse[selection] for selection in REFINE_SETS if ellipse[selection] > 0.005}


def share_sentence(shares: dict[RefineSet, float]) -> str:
    if not shares:
        return ""
    text = (
        f"楕円を正解にした向上が0.005を超える{len(shares)}行で、事前分布だけを正解にした向上は、"
        f"楕円を正解にした向上の{percent_span(list(shares.values()))}である。"
    )
    over = [selection for selection, ratio in shares.items() if ratio > 1]
    if over:
        text += (
            f"100%を超える{names_of(over)}では、事前分布だけを正解にした向上のほうが、楕円を正解にした向上より大きい。"
        )
    return text


def ceiling_text(study: Study) -> str:
    keys = (*SETS, PHOTO_SET)
    ceilings = [study.mean(key, "oracle") for key in keys]
    margins = [study.mean(key, "oracle") - study.mean(key, "adaptive_c") for key in keys]
    if min(margins) > 0.05:
        verdict = "木の切断の表現力は、精度を制限する主な要因ではない。"
    else:
        verdict = "木の切断の表現力が精度を制限する程度は、データセットにより異なる。"
    return (
        f"参考として、正解のスコアで切断したときの木の切断のIoUは、データセットごとに{f3(min(ceilings))}から{f3(max(ceilings))}で、"
        f"適応（色）の構成との差は{f3(min(margins))}から{f3(max(margins))}になる。{verdict}"
    )


def oracle_text(study: Study, rows: dict[str, list[dict]]) -> str:
    gains = {name: mean_gain(rows, name) for name, _ in REPLACEMENTS}
    samples_higher = [s for s in REFINE_SETS if gains["samples"][s] > gains["prior"][s]]
    share_text = share_sentence(prior_share(rows))
    return (
        f"<p>精度が何に制限されているかを調べるために、木の切断の入力の一部を、正解から作った入力に置き換え、IoUがどれだけ高くなるかを測った。"
        "この置き換えは正解を使うため、手法ではなく診断である。置き換えるのは、次の四つである。"
        "楕円を、正解の外接矩形に内接する楕円にする（事前のスコアと、標本の範囲の両方が変わる）。"
        "事前のスコアだけを、その楕円から作る。"
        f"前景と背景の標本を、正解のマスクとその補集合を{ERODE_DEPTH}画素だけ収縮した領域にする（三つの手がかりと、バタチャリヤ係数から求める重みが、この領域から作り直される）。"
        "事前のスコアと標本の両方を置き換える。"
        '結果を<ref k="t:oracle"></ref>に示す。</p>'
        f"<p>{significance(rows, 'ellipse', '楕円を正解にした場合')}{significance(rows, 'prior', '事前分布だけを正解にした場合')}"
        f"{significance(rows, 'samples', '標本を正解にした場合')}"
        f"向上の大きさ（差の平均）は、楕円が{span(gains['ellipse'])}、事前分布だけが{span(gains['prior'])}、標本が{span(gains['samples'])}、"
        f"両方が{span(gains['both'])}である。{share_text}"
        f"{comparison_sentence('標本を正解にした向上', '事前分布だけを正解にした向上', samples_higher)}"
        f"標本の向上が最大の行は{largest(gains['samples']).short}（{signed(gains['samples'][largest(gains['samples'])])}）、最小の行は{smallest(gains['samples']).short}（{signed(gains['samples'][smallest(gains['samples'])])}）である。"
        f"{ceiling_text(study)}</p>"
    )


def decomposition_text(rows: dict[str, list[dict]]) -> str:
    gains = {name: mean_gain(rows, name) for name, _ in DECOMPOSITION}
    far, near = gains["ring_far"], gains["ring_near"]
    same, whole = gains["core_same"], gains["core_all"]
    near_larger = [s for s in REFINE_SETS if near[s] > far[s]]
    whole_larger = [s for s in REFINE_SETS if whole[s] > same[s]]
    core_purity = {s: mean_purity(rows[s.name], "core") for s in REFINE_SETS}
    ring_purity = {s: mean_purity(rows[s.name], "ring") for s in REFINE_SETS}
    low_core = min((v for v in core_purity.values() if v is not None), default=None)
    high_core = max((v for v in core_purity.values() if v is not None), default=None)
    low_ring = min((v for v in ring_purity.values() if v is not None), default=None)
    purity_sentence = (
        f"置き換える前の標本の純度は、前景の標本が{f2(low_core)}から{f2(high_core)}、背景の標本は最小で{f2(low_ring)}である。"
        if low_core is not None and low_ring is not None
        else ""
    )
    return (
        f'<p>標本の置き換えの内訳を、<ref k="t:oracle2"></ref>に示す。{purity_sentence}'
        f"遠い背景だけを正解にした向上は{span(far)}、近い背景だけを正解にした向上は{span(near)}である。"
        f"{comparison_sentence('近い背景だけを正解にした向上', '遠い背景だけを正解にした向上', near_larger)}"
        f"{background_reading(far, near, near_larger, low_ring)}"
        f"前景の標本を、同じ範囲で正解にした向上は{span(same)}、画像全体で正解にした向上は{span(whole)}である。"
        f"{comparison_sentence('画像全体で正解にした向上', '同じ範囲で正解にした向上', whole_larger)}"
        f"{foreground_reading(same, core_purity)}</p>"
    )


def background_reading(
    far: dict[RefineSet, float], near: dict[RefineSet, float], near_larger: list[RefineSet], low_ring: float | None
) -> str:
    if low_ring is None or low_ring < PURE_RING or len(near_larger) < len(REFINE_SETS):
        return ""
    return (
        f"背景の標本は、置き換える前の純度が{f2(low_ring)}以上と高く、遠い範囲を正解にしても向上は小さいのに対し、"
        f"近い範囲の背景に置き換えると、{largest(near).short}で最大の向上になる。"
        "背景の標本の誤りは、前景の混入よりも、標本が境界に近い背景を含まないことによる部分が大きいと考えられる。"
    )


def foreground_reading(same: dict[RefineSet, float], core_purity: dict[RefineSet, float | None]) -> str:
    known = {selection: purity for selection, purity in core_purity.items() if purity is not None}
    if not known:
        return ""
    lowest = min(known, key=known.get)
    return f"前景の標本の純度が最も低い{lowest.short}（{f2(known[lowest])}）では、同じ範囲で正解にした向上が{signed(same[lowest])}になる。"


def unmixing_shares(rows: dict[str, list[dict]], name: str) -> dict[RefineSet, float]:
    samples = mean_gain(rows, "samples_ct")
    gains = mean_gain(rows, name)
    return {s: gains[s] / samples[s] for s in REFINE_SETS if samples[s] > 0.005}


def mixture_text(rows: dict[str, list[dict]]) -> str:
    truth = mean_gain(rows, "unmix_truth")
    tree = mean_gain(rows, "unmix_tree")
    population = mean_gain(rows, "unmix_population")
    shares = unmixing_shares(rows, "unmix_truth")
    ending = "にとどまる" if shares and max(shares.values()) < 0.5 else "である"
    share_text = (
        f"占有率が正解でも、向上は、標本を正解にした向上（色とテクスチャだけ）の{percent_span(list(shares.values()))}{ending}"
        f"（標本を正解にした向上が0.005を超える{len(shares)}行）。"
        if shares
        else ""
    )
    classes = classified_by(rows, value("unmix_truth"), value("tree"))
    classes_tree = classified_by(rows, value("unmix_tree"), value("tree"))
    return (
        "<p>標本の誤りが、前景と背景の分布が混ざっていることだけから生じるなら、混ざる割合が分かれば補正できる。"
        f"楕円の正規化半径を{len(OCCUPANCY_EDGES) - 1}個の環に分け、環ごとに、色とテクスチャのヒストグラムを数える。"
        "環 \\(j\\) の分布は、前景の分布 \\(f\\) と背景の分布 \\(b\\) を、前景の占有率 \\(a_j\\) で混ぜた \\(a_j f+(1-a_j)b\\) だとして、"
        '\\(f\\) と \\(b\\) を非負の最小二乗で求める。この処理を混合分解と呼ぶ。これは、観測した分布を二つの分布の混合と見なして、元の分布を求める考え方<cite k="elkan2008,scott2013"></cite>に沿う。'
        "占有率には、正解から数えた値、木の切断から数えた値、楕円からの距離だけで決めた集団の平均を使った。"
        '結果を<ref k="t:unmix"></ref>に示す。'
        f"{share_text}"
        f"{higher_sentence('正解の占有率を使った場合', classes)}{higher_sentence('切断から求めた占有率を使った場合', classes_tree)}"
        f"向上の大きさ（差の平均）は、正解の占有率が{span(truth)}、切断から求めた占有率が{span(tree)}、集団の占有率が{span(population)}である。"
        f"{mixture_verdict(rows)}</p>"
    )


def mixture_verdict(rows: dict[str, list[dict]]) -> str:
    share = estimated_share(rows)
    if share is None:
        return ""
    if share < 0.5:
        return f"切断から求めた占有率と集団の占有率で取り戻せるのは、標本を正解にした向上の最大でも{round(100 * share)}%にとどまる。"
    return "切断から求めた占有率か集団の占有率でも、標本を正解にした向上の半分以上を取り戻せる行がある。"


def diagnosis_section(study: Study) -> str:
    rows = all_oracle_rows(study)
    return (
        oracle_text(study, rows)
        + "\n"
        + oracle_table(rows)
        + "\n"
        + decomposition_text(rows)
        + "\n"
        + decomposition_table(rows)
        + "\n"
        + mixture_text(rows)
        + "\n"
        + mixture_table(rows)
    )


def diagnosis_summary(study: Study) -> dict[str, str]:
    rows = all_oracle_rows(study)
    gains = {name: mean_gain(rows, name) for name, _ in REPLACEMENTS}
    shares = prior_share(rows)
    return {
        "oracle_prior": span(gains["prior"]),
        "oracle_samples": span(gains["samples"]),
        "oracle_ellipse": span(gains["ellipse"]),
        "oracle_share_low": f"{round(100 * min(shares.values()))}%" if shares else "−",
        "oracle_share_high": f"{round(100 * max(shares.values()))}%" if shares else "−",
        "oracle_depth": plain(ERODE_DEPTH),
    }


def recovered_share(rows: dict[str, list[dict]]) -> float | None:
    ratios = unmixing_shares(rows, "unmix_truth")
    return max(ratios.values()) if ratios else None


def estimated_share(rows: dict[str, list[dict]]) -> float | None:
    ratios = [*unmixing_shares(rows, "unmix_tree").values(), *unmixing_shares(rows, "unmix_population").values()]
    return max(ratios) if ratios else None


def mixture_sentence(study: Study) -> str:
    share = recovered_share(all_oracle_rows(study))
    if share is None:
        return ""
    return f"楕円の環ごとの分布を、正解から数えた占有率で前景と背景の分布に分解しても、取り戻せるのは、標本を正解にした向上の{round(100 * share)}%以下である。"


def larger_rows(study: Study) -> list[RefineSet]:
    rows = all_oracle_rows(study)
    prior, samples = mean_gain(rows, "prior"), mean_gain(rows, "samples")
    return [s for s in REFINE_SETS if samples[s] > prior[s]]


def change_clause(gains: dict[RefineSet, float]) -> str:
    if min(gains.values()) > 0:
        return f"IoUは木の切断より{span(gains)}高くなる"
    return f"IoUは木の切断に対して{span(gains)}変わる"


def intro_diagnosis(study: Study) -> str:
    rows = all_oracle_rows(study)
    return (
        f"事前のスコアだけを正解の楕円のものにすると、{change_clause(mean_gain(rows, 'prior'))}。"
        f"標本を正解にすると、{change_clause(mean_gain(rows, 'samples'))}。"
        f"{mixture_sentence(study)}"
    )


def abstract_diagnosis(study: Study) -> str:
    summary = diagnosis_summary(study)
    larger = larger_rows(study)
    rows_text = f"{dataset_phrase(larger)}で" if larger else "どのデータセットでも"
    comparison = "大きい" if larger else "大きくない"
    return (
        f"誤差の要因を、正解から作った入力への置き換えで分けると、標本を正解にしたときのIoUの変化は{summary['oracle_samples']}で、"
        f"事前のスコアだけを正解にしたときの{summary['oracle_prior']}より、{rows_text}{comparison}。"
        f"{mixture_sentence(study)}"
    )


def future_diagnosis(study: Study) -> str:
    share = estimated_share(all_oracle_rows(study))
    larger = larger_rows(study)
    if 2 * len(larger) > len(REFINE_SETS) and share is not None and share < 0.5:
        return (
            "標本の誤りを、1枚の画像と楕円だけから減らす方法である。"
            f'<ref k="s:abl_diag"></ref>の診断では、{dataset_phrase(larger)}で、標本を正解にしたときの向上のほうが、事前のスコアだけを正解にしたときより大きかった。'
            f"正解を使わずに求めた占有率による混合分解で取り戻せたのは、標本を正解にした向上の{round(100 * share)}%以下であった。"
        )
    return '標本の誤りを減らす方法である。<ref k="s:abl_diag"></ref>の診断で、標本の誤りがIoUに与える影響を測った。'


def discussion_diagnosis(study: Study) -> str:
    rows = all_oracle_rows(study)
    larger = larger_rows(study)
    if 2 * len(larger) > len(REFINE_SETS):
        lead = f"残る誤差は、{dataset_phrase(larger)}で、事前分布より標本に由来する部分が大きい。"
    else:
        lead = "事前分布と標本のどちらが誤差の主因かは、データセットにより異なる。"
    return (
        f"<p><b>{lead}</b>"
        f"{mixture_sentence(study)}{mixture_verdict(rows)}"
        '標本を正解にした向上には、前景の標本から背景の混入を除く効果と、背景の標本を境界に近い範囲から取る効果が含まれる（<ref k="t:oracle2"></ref>）。'
        "これらを、1枚の画像と楕円だけから行う方法は、本稿では示せていない。"
        "本稿は学習を使わない手法を対象とし、学習済みのモデルの知識を使う方法は扱っていない。</p>\n"
    )


def conclusion_diagnosis(study: Study) -> str:
    larger = larger_rows(study)
    if 2 * len(larger) > len(REFINE_SETS):
        return f"正解から作った入力への置き換えでは、{dataset_phrase(larger)}で、標本を正解にした向上が、事前のスコアだけを正解にした向上より大きかった。"
    return "正解から作った入力への置き換えでは、事前分布と標本のどちらが誤差の主因かは、データセットにより異なった。"
