from paper.figures.qualitative import RENDER_PICKS, Pick, qualitative_picks, representative
from paper.markup import figure
from paper.numbers import f2
from paper.results import TREE
from paper.sections.main import METHOD_LABELS
from paper.study import Study

ROW_NAMES = (
    "立ち絵1の星雲の例",
    "立ち絵1の夜景の例",
    "立ち絵2の例",
    "公開素材の転写0%の例",
    "公開素材の転写100%の例",
    "公開素材の下位の例",
)
COMPARED = ("ellipse", "prior", "tex", "mix", "adaptive_c")
RENDERED = ("prior", "tex", "adaptive_c")

COMPARISON_CAPTION = (
    "条件ごとのIoUが中央値に近い試行の結果。左から、入力と指定した楕円、楕円をそのまま使った結果、事前分布のみ、"
    "テクスチャのみ、固定重み、適応（色）、グラフカット、正解のαを並べた。白は正解と一致した前景、赤は背景を前景とした誤り、"
    "青は前景を背景とした誤りで、左上の数値はIoUである。最後の行は、公開素材の転写100%でIoUが下位1/8にあたる試行である。"
)
RENDER_CAPTION = (
    "背景のぼかし描画（上段）と、正解描画との差の絶対値を6倍にした画像（下段）。正解描画は正解のαによる描画である。"
)


def qualitative_section(text: str) -> str:
    return (
        text
        + "\n"
        + figure("f:qual", "fig_qual.png", "定性的な比較", COMPARISON_CAPTION)
        + "\n"
        + figure("f:render", "fig_render.png", "ぼかし描画の比較", RENDER_CAPTION)
    )


def pick_record(study: Study, pick: Pick) -> dict:
    return next(
        record
        for record in study.results.trials(pick.key)
        if record["scene"] == pick.scene and record["trial"] == pick.trial
    )


def graph_cut_sentence(study: Study) -> str:
    rows = []
    for name, pick in zip(ROW_NAMES, qualitative_picks(study.results), strict=True):
        record = pick_record(study, pick)
        others = [record[TREE][method]["iou"] for method in COMPARED]
        rows.append((name, record["graph"]["adaptive_c"]["iou"], record[TREE]["mix"]["iou"], max(others)))
    best = [name for name, graph, _, others in rows if graph >= others]
    lower = [(name, mix) for name, graph, mix, others in rows if graph < others and graph < mix]
    if not best:
        return "グラフカットがIoUの最大になる例はない。"
    sentence = f"グラフカットは、{'、'.join(best)}でIoUが最大である"
    if lower:
        segments = [f"{name}では固定重みの{f2(mix)}より低" for name, mix in lower]
        sentence += "が、" + "く、".join(segments) + "い"
    return sentence + "。"


def render_sentence(study: Study) -> str:
    segments = []
    for (title, key, scene), name in zip(RENDER_PICKS, ("立ち絵1", "公開素材"), strict=True):
        pick = Pick(title, key, *representative(study.results, key, "median", scene))
        record = pick_record(study, pick)
        errors = {method: record[TREE][method]["mae"] for method in RENDERED}
        smallest = min(errors, key=errors.get)
        segments.append(f"{name}の例で{METHOD_LABELS[smallest]}の{f2(errors[smallest])}")
    return f"最も小さいのは、{'、'.join(segments)}である。"


def qualitative_text(study: Study) -> str:
    return (
        '<p><ref k="f:qual"></ref>に、条件ごとにIoUが中央値に近い試行を一つずつ示す（立ち絵1は星雲とそれ以外の背景から一つずつ、公開素材は転写0%と100%の試行と、転写100%でIoUが下位1/8にあたる試行）。'
        "白は正解と一致した前景、赤は背景を前景とした誤り、青は前景を背景とした誤りである。"
        "楕円をそのまま使うと、楕円に収まらない髪や衣服の端が青の誤りに、楕円の内側の背景が赤の誤りになる。事前分布のみでは、デンドログラムの境界に沿って形が整うが、前景の一部が背景になり、背景の一部が前景に残る。"
        "立ち絵1の二つの例と立ち絵2の例では、テクスチャのみ、固定重み、適応（色）で輪郭の多くが正解と一致し、誤りは耳や髪の先、衣服の内部の小さな領域に残る。立ち絵1の夜景の例では、テクスチャのみが足元の背景の領域を前景に含める。公開素材の転写0%の例でも輪郭の多くが一致し、誤りは髪の先に残る。"
        "公開素材の転写100%の二つの例では、背景の平坦な領域（壁の帯）が前景に入る構成が多く、平坦さを使うテクスチャのみで特に目立つ。"
        f"{graph_cut_sentence(study)}</p>\n"
        '<p><ref k="f:render"></ref>に、同じ試行のうち二つの背景のぼかし描画と、正解のαによる正解描画との差を示す。'
        "立ち絵1の例では、楕円をそのまま使う、あるいは事前分布のみでは、前景の縁が背景と一緒にぼける、あるいは背景の色が前景に残る誤りが縁に沿って目立ち、テクスチャのみと適応（色）では、差は縁の細い範囲にとどまる。"
        "公開素材の転写0%の例では、楕円をそのまま使うと、楕円の内側の背景のテクスチャがぼけずに残る。テクスチャのみと適応（色）では、白い髪の先が背景として扱われて、差が髪に集まる。"
        f"MAEが{render_sentence(study)}</p>"
    )
