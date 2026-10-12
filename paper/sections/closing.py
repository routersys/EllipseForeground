from paper.numbers import ceil3, f2, f3, signed
from paper.quantities import (
    GraphCutComparison,
    graph_cut_comparison,
    headline,
    mean_auc,
    rank_correlations,
)
from paper.results import EARLIER_SETS, SETS, SUBSETS
from paper.sections.ablation import largest_drop_form
from paper.study import Study

SET_LABELS = {key: SUBSETS[key].short for key in SETS}
HELD_OUT_SETS = ("sprite2", "public0", "public100")


def join(keys: list[str]) -> str:
    return "、".join(SET_LABELS[key] for key in keys)


def range_text(values: list[float]) -> str:
    return f"{f3(min(values))}〜{f3(max(values))}"


def graph_cut_abstract(comparison: GraphCutComparison) -> str:
    positive = [key for key in SETS if comparison.difference[key] > 0]
    negative = [key for key in SETS if comparison.difference[key] <= 0]
    significant = [key for key in comparison.higher if key in positive]
    significant_text = join(significant) or "なし"
    if not negative:
        values = [comparison.difference[key] for key in positive]
        return f"画素単位のグラフカットは、六つの集合のすべてで木の切断よりIoUが{range_text(values)}高い（信頼区間が0を含まないのは{significant_text}）"
    names = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六"}
    positive_values = [comparison.difference[key] for key in positive]
    negative_values = [-comparison.difference[key] for key in negative]
    if len(negative) == 1:
        head = f"{SET_LABELS[negative[0]]}を除く{names[len(positive)]}つの集合で"
    else:
        head = f"{join(positive)}で"
    tail = f"{join(negative)}では{range_text(negative_values)}低い"
    return f"画素単位のグラフカットは、{head}木の切断よりIoUが{range_text(positive_values)}高く（信頼区間が0を含まないのは{significant_text}）、{tail}"


def abstract(study: Study) -> str:
    summary = headline(study.measure)
    comparison = graph_cut_comparison(study.measure)
    ratio = ""
    if study.timing:
        ratio = (
            f"。同じ得点のもとでの最適化1回の計算時間は、木の動的計画法のほうが約{study.timing.ratio:.0f}分の1である"
        )
    return (
        "被写体を楕円で大まかに指定したときに、画像からその前景のアルファマットを推定する方法を述べる。画像から単連結法で階層木を作り、楕円に由来する事前分布、画素ごとの手がかり、境界項の和を最大にする切断を、動的計画法で厳密に求める。"
        "手がかりには、色、3×3パッチの最近傍、原寸画像の局所的な平坦さを使う。平坦さは画素ごとの色変換で保たれるため、色転写を受けた合成画像でも前景と背景を分ける。"
        "さらに、楕円の内側と外側の分布のバタチャリヤ係数から手がかりの信頼度を正解なしで推定し、色の重みを調整する。"
        "動画編集ソフトウェアの出力から復元した前景とアルファマットを、CC0の背景に合成した同じキャラクターの髪型違いの立ち絵2種と、17体の立ち絵を16種の背景に合成した102場面（色転写0%と100%）で比較した。"
        "加えて、手法の設計に使っていない、公開素材の立ち絵17枚とCC0の背景16枚から作った102場面（色転写0%と100%）でも評価した。"
        f"楕円をそのまま使うと前景のIoUは{f2(summary.ellipse_low)}から{f2(summary.ellipse_high)}であるのに対し、適応（色）の構成では{f2(summary.adaptive_low)}〜{f2(summary.adaptive_high)}になる。"
        f"テクスチャのみと色とパッチのみの優劣は画像集合で異なり、両者を併用する固定配合は六つの集合のすべてで最良の構成とのIoUの差が{ceil3(summary.mix_regret)}以内であった。信頼度に基づく色の重みの調整の効果は、IoUで最大{f3(summary.adaptive_effect)}と小さい。"
        f"{graph_cut_abstract(comparison)}{ratio}。"
    )


def graph_cut_discussion(study: Study, comparison: GraphCutComparison) -> str:
    diffs = "、".join(f"{SET_LABELS[key]}で{signed(comparison.difference[key])}" for key in SETS)
    higher = join(comparison.higher) or "なし"
    lower = join(comparison.lower) or "なし"
    errors = (
        "ぼかし描画のMAEは、六つの集合のすべてでグラフカットのほうが小さい。"
        if all(value < 0 for value in comparison.error_difference.values())
        else ""
    )
    speed = ""
    if study.timing:
        speed = f'最適化1回の計算時間は、木の動的計画法がグラフカットの約{study.timing.ratio:.0f}分の1である（<ref k="s:abl_gc"></ref>）。'
    constraint = (
        "木の切断は、画像の階層木に含まれる領域しか表現できず、細い連結で背景とつながる細部を分けられないことがある。"
        "この制約が、グラフカットが高くなる集合の一因と考えられる。"
        if comparison.higher
        else "木の切断は、画像の階層木に含まれる領域しか表現できず、細い連結で背景とつながる細部を分けられないことがある。"
    )
    unexplained = (
        f"{join(comparison.lower)}で木の切断が高かった理由は、本稿の実験からは特定できない。"
        if comparison.lower
        else ""
    )
    return (
        f"<p><b>木の切断とグラフカットは、精度と最適化の速さで異なる。</b>グラフカットから木の切断を引いたIoUの差は、{diffs}である。"
        f"信頼区間が0を含まず、グラフカットが高い集合は{higher}、木の切断が高い集合は{lower}である。{errors}{speed}"
        f"{constraint}{unexplained}"
        "楕円を動かすたびに切断を更新する対話的な処理では、最適化の時間が短い木の切断が適すると考えられる。輪郭が明瞭な画像で最終的な描画の精度を優先する場合は、グラフカットが適すると考えられる。</p>\n"
    )


def adaptive_patch_sentence(study: Study) -> str:
    def clear(key: str) -> bool:
        return study.interval(key, "adaptive", "mix").low > 0

    def touching(key: str) -> bool:
        interval = study.interval(key, "adaptive", "mix")
        return interval.mean > 0 and not interval.excludes_zero

    collections = ("collection0", "collection100")
    public = ("public0", "public100")
    parts = []
    if all(clear(key) for key in collections):
        parts.append("17体では向上し")
    if all(touching(key) or clear(key) for key in public):
        parts.append("公開素材でも差は正の向きだが信頼区間は0を含むか0に接する")
    if study.difference("sprite1", "adaptive", "mix") < 0:
        parts.append("立ち絵1では下がる")
    return "、".join(parts)


def discussion(study: Study) -> str:
    summary = headline(study.measure)
    comparison = graph_cut_comparison(study.measure)
    correlations = rank_correlations(study.results)
    auc = {
        key: {cue: mean_auc(study.results, key, cue) for cue in ("texture", "color", "patch")} for key in EARLIER_SETS
    }
    dose = {
        (cue, dose): mean_auc(study.results, dose, cue) for cue in ("color", "texture") for dose in ("dose0", "dose100")
    }
    held_out_below = all(study.mean(key, "app") < study.mean(key, "mix") for key in HELD_OUT_SETS)
    held_out_text = (
        "設計に使っていない立ち絵2と公開素材でも、色とパッチのみが固定配合を下回り、固定配合と最良の構成の差が"
        f"{ceil3(summary.held_out_regret)}以内であるという傾向は保たれた。"
        if held_out_below
        else "設計に使っていない立ち絵2と公開素材では、色とパッチのみが固定配合を下回るという傾向は保たれなかった。固定配合と最良の構成の差は"
        f"{ceil3(summary.held_out_regret)}以内である。"
    )
    patch_detail = adaptive_patch_sentence(study)
    return (
        '<h2 id="s:disc">考察</h2>\n'
        '<h3 id="s:disc_res">結果の読み方</h3>\n'
        "<p><b>手がかりの有効性は集合によって異なる。</b>色とパッチの手がかりは、楕円の内側と外側の標本の分布の差を使う。この差が境界付近の画素にも当てはまるのは、前景と背景の絵柄が異なり、背景の標本が境界付近の背景と同じ分布に従うと見なせる場合である。"
        "立ち絵1は、輪郭線と一様な塗りからなる前景と、写真や絵画の背景の組である。色転写は前景の色の平均と分散を背景に合わせるため、この絵柄の違いのうち色の分布の部分を弱める。一方、一様な塗りは変換後も一様なので、テクスチャの手がかりは保たれる。"
        f"実際、立ち絵1の楕円内の平均AUCは、テクスチャが{f2(auc['sprite1']['texture'])}、色が{f2(auc['sprite1']['color'])}、パッチが{f2(auc['sprite1']['patch'])}である。色転写の強さを0%から100%に変えると、色のAUCは{f2(dose[('color', 'dose0')])}から{f2(dose[('color', 'dose100')])}に下がるが、テクスチャのAUCは{f2(dose[('texture', 'dose0')])}から{f2(dose[('texture', 'dose100')])}にとどまる。"
        f"17体の集合では、前景の絵柄が多様で陰影や細かい模様を持つものが多く、テクスチャの平均AUCは転写0%で{f2(auc['collection0']['texture'])}、100%で{f2(auc['collection100']['texture'])}にとどまる。"
        f"これに対し、色は転写0%で{f2(auc['collection0']['color'])}、パッチは100%でも{f2(auc['collection100']['patch'])}と、テクスチャより高い。この違いが、テクスチャのみと色とパッチのみの優劣が集合で変わる主な理由と考えられる。</p>\n"
        f"<p><b>固定配合は集合を問わず安定している。</b>六つの集合のすべてで、固定配合と最良の構成のIoUの差は{ceil3(summary.mix_regret)}以内である。これに対し、テクスチャのみでは最大{f3(summary.texture_regret)}、色とパッチのみでは最大{f3(summary.appearance_regret)}の差が生じる。"
        "画像の集合が事前にわからない場合は、手がかりを併用する固定配合が無難な選択である。</p>\n"
        f"<p><b>信頼度に基づく重みの効果は小さい。</b>色の重みを調整する適応（色）の効果は、固定配合に対して最大でも{f3(summary.adaptive_effect)}で、符号は集合により異なる。パッチの重みも調整すると、{patch_detail}。"
        "効果が小さい理由は、次のように考えられる。係数は境界から離れた標本の分布の離れであり、境界付近の画素の識別性能とは一致しない。また、色の重みを調整できる範囲は0から0.5と狭く、調整後もテクスチャとパッチの手がかりが残る。"
        f"パッチの重みについては、係数が識別性能との対応が弱く、調整の根拠として使いにくい。係数と手がかりの識別性能の順位相関の絶対値は、テクスチャで{abs(correlations['texture'].pooled):.2f}、色で{abs(correlations['color'].pooled):.2f}である。"
        "この関係を境界付近の画素の識別性能と対応づける方法は、本稿では示せていない。</p>\n"
        + graph_cut_discussion(study, comparison)
        + '<h3 id="s:disc_threat">妥当性への脅威</h3>\n'
        "<p><b>合成による評価。</b>評価に使った画像は、立ち絵と背景を合成したものである。前景と背景は別の出自を持つため、実際の写真に比べて絵柄の違いが大きく、外見の手がかりに有利に働くと考えられる。写真の人物や、前景と背景が連続する画像での結果は本稿では示していない。</p>\n"
        "<p><b>設計用の集合と定数の選択。</b>立ち絵1と17体は、手法の設計と定数の選択に使った。選択した定数のうち、境界項の倍率とグラフカットの \\(\\lambda\\) は、最適値が集合により大きく異なる。既定の事前分布の内側の縮小率も、最適値が集合により異なる。"
        f"{held_out_text}立ち絵2は立ち絵1と同じキャラクターの髪型違いで、絵柄が共通するため、立ち絵1に対する独立性は低い。背景も8種にすぎない。公開素材の立ち絵17枚のうち12枚は同じ作者の7セットで、絵柄が近い。結果の一般性は限られる。17体の集合は、本稿の実験の前に行った探索でも使っており、その影響を完全には切り離せない。</p>\n"
        '<p><b>楕円の誤差モデル。</b>利用者の楕円の誤差は、倍率と中心のずれを一様乱数で引く模擬である。実際の操作で生じる誤差の分布とは異なる可能性がある。誤差モデルを変えた結果は<ref k="s:abl_noise"></ref>に示したが、実際の利用者の操作は評価していない。</p>\n'
        "<p><b>描画の簡略化。</b>ぼかし描画は、円板の点拡がり関数による均一なぼかしである。深度に応じた変化や、明るい光源の広がりは扱っておらず、MAEはこの描画に対する値である。</p>\n"
        '<p><b>学習済みモデルとの比較。</b>本稿は学習を使わない手法の性質を調べるもので、学習済みの分割モデル<cite k="kirillov2023"></cite>との比較は行っていない。これらのモデルは、本稿の手法より高い精度を出す可能性がある。</p>\n'
        '<p><b>計算時間の測定範囲。</b>計算時間は、動的計画法と最小カットの1回あたりの時間だけを測った。楕円を動かすたびに必要な手がかりの再計算は測っておらず、特にパッチの最近傍探索は計算量が他の段より大きい（<ref k="s:cost"></ref>）。</p>\n'
        '<h3 id="s:disc_future">今後の課題</h3>\n'
        "<p>三つの課題が残る。第一は、写真の人物を含む画像での評価である。第二は、手がかりの信頼度を、境界付近の画素の識別性能と対応づける方法である。そのためには、楕円の内側と外側の標本のうち、境界に近い画素だけを使う方法や、手がかり間の一致を使う方法が考えられる。"
        "第三は、木の切断とグラフカットの併用である。木の切断の結果を初期値として、境界付近の帯の中だけを画素単位で再判定すれば、計算時間を抑えながら、細部の精度を上げられる可能性がある。</p>\n"
    )


def conclusion(study: Study) -> str:
    summary = headline(study.measure)
    comparison = graph_cut_comparison(study.measure)
    correlations = rank_correlations(study.results)
    positive = [key for key in ("sprite1", "sprite2", "public0", "public100") if comparison.difference[key] > 0]
    if len(positive) == 4:
        graph_text = (
            f"画素単位のグラフカットは、立ち絵と公開素材で木の切断より{f3(min(comparison.difference[key] for key in positive))}〜"
            f"{f3(max(comparison.difference[key] for key in positive))}高いIoUを示し、木の切断は最適化の計算が速い。"
        )
    else:
        graph_text = "画素単位のグラフカットと木の切断のIoUの差は集合により異なり、木の切断は最適化の計算が速い。"
    return (
        '<h2 id="s:concl">おわりに</h2>\n'
        "<p>楕円で大まかに指定された前景を、単連結法の階層木の上の最適な切断として推定する方法を述べ、手がかり、境界項、最適化のそれぞれの効果を、六つの画像集合で評価した。</p>\n"
        f"<p>階層木の上の切断により、楕円をそのまま使う場合のIoU {f2(summary.ellipse_low)}〜{f2(summary.ellipse_high)} を、適応（色）の構成で {f2(summary.adaptive_low)}〜{f2(summary.adaptive_high)} に上げた。原寸の平坦さを使うテクスチャの手がかりは、色転写で色とパッチの手がかりが崩れる合成画像で有効であり、立ち絵1の全12条件でIoUが{f3(study.mean('sprite1', 'tex'))}になった。"
        f"一方、17体の転写0%では、色とパッチのみがテクスチャのみを上回る。両者の優劣は集合により異なるが、併用する固定配合は、設計に使っていない集合を含むすべての集合で、最良との差を{ceil3(summary.mix_regret)}以内に抑えた。</p>\n"
        f"<p>楕円の内側と外側の分布のバタチャリヤ係数は、テクスチャと色の識別性能と対応した（順位相関 {signed(correlations['texture'].pooled, 2)}、{signed(correlations['color'].pooled, 2)}）。しかし、係数に基づく色の重みの調整が固定配合に与える効果は、IoUで最大{f3(summary.adaptive_effect)}と小さかった。"
        f"境界項の形は、強さを調整すれば精度が同程度になり、{largest_drop_form(study)}は倍率を大きくしたときの低下が最も大きかった。{graph_text}</p>\n"
        "<p>本稿の評価は合成画像に限られ、学習済みモデルとの比較も行っていない。写真での評価、境界付近の画素に対応づけた信頼度、木の切断とグラフカットの併用が、今後の課題である。</p>\n"
    )
