from paper.markup import figure

COMPARISON_CAPTION = (
    "条件ごとのIoUが中央値に近い試行の結果。左から、入力と指定した楕円、楕円をそのまま使った結果、事前分布のみ、"
    "テクスチャのみ、固定配合、適応（色）、グラフカット、正解のαである。白は正解と一致した前景、赤は背景を前景とした誤り、"
    "青は前景を背景とした誤りで、左上の数値はIoUである。最後の行は、公開素材の転写100%でIoUが下位1/8にあたる試行である。"
)
RENDER_CAPTION = (
    "背景のぼかし描画（上段）と、基準描画との差の絶対値を6倍にした画像（下段）。基準描画は正解のαによる描画である。"
)


def qualitative_section(text: str) -> str:
    return (
        text
        + "\n"
        + figure("f:qual", "fig_qual.png", "定性的な比較", COMPARISON_CAPTION)
        + "\n"
        + figure("f:render", "fig_render.png", "ぼかし描画の比較", RENDER_CAPTION)
    )


def qualitative_text(study) -> str:
    return '<p><ref k="f:qual"></ref>に、条件ごとにIoUが中央値に近い試行を一つずつ示す。</p>'
