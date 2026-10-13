from pathlib import Path

import numpy as np

from ellipse_foreground.evaluation.own import CANVAS_HEIGHT as STAND_HEIGHT
from ellipse_foreground.evaluation.own import CANVAS_WIDTH as STAND_WIDTH
from ellipse_foreground.evaluation.own import own_reconstruction
from ellipse_foreground.evaluation.runner import GROUPS
from ellipse_foreground.evaluation.sprites import CANVAS_HEIGHT, CANVAS_WIDTH
from ellipse_foreground.settings import DEFAULT_SETTINGS
from paper.numbers import plain
from paper.results import SETS
from paper.sections.closing import abstract
from paper.study import Study
from paper.table import table_html

TRIALS_PER_SCENE = {
    "sprite1": GROUPS["sprite1"].trials,
    "sprite2": GROUPS["sprite2"].trials,
    "collection0": GROUPS["public0"].trials,
    "collection100": GROUPS["public0"].trials,
    "public0": GROUPS["public0"].trials,
    "public100": GROUPS["public100"].trials,
}
TRUTH_SOURCE = {
    "sprite1": "出力から復元",
    "sprite2": "出力から復元",
    "collection0": "アルファチャンネル",
    "collection100": "アルファチャンネル",
    "public0": "アルファチャンネル",
    "public100": "アルファチャンネル",
}
DESIGN_USE = {
    "sprite1": "はい",
    "sprite2": "いいえ",
    "collection0": "はい",
    "collection100": "はい",
    "public0": "いいえ",
    "public100": "いいえ",
}
SET_TITLES = {
    "sprite1": "立ち絵1",
    "sprite2": "立ち絵2",
    "collection0": "17体・色転写0%",
    "collection100": "17体・色転写100%",
    "public0": "公開素材・色転写0%",
    "public100": "公開素材・色転写100%",
}


def recovery_error(data_root: Path) -> tuple[float, float]:
    return own_reconstruction(data_root, "sprite1")


def data_table(study: Study) -> str:
    rows = []
    for key in SETS:
        scenes = study.results.scenes(key)
        trials = len(study.results.trials(key))
        per_scene = TRIALS_PER_SCENE[key]
        count = f"{trials}（各{per_scene}）" if trials == scenes * per_scene else f"{trials}（各{per_scene}まで）"
        stand = key.startswith("sprite")
        size = f"{STAND_WIDTH}×{STAND_HEIGHT}" if stand else f"{CANVAS_WIDTH}×{CANVAS_HEIGHT}"
        unit = f"{scenes}条件" if stand else f"{scenes}場面"
        rows.append([SET_TITLES[key], size, unit, count, TRUTH_SOURCE[key], DESIGN_USE[key]])
    return table_html(
        "t:data",
        "評価に用いる画像データセット。試行は、誤差モデルMで引いた楕円の一つを指す。色転写の強さは、0%が変換なし、100%が前景の色の平均と標準偏差を背景に完全に合わせる場合である。",
        ["データセット", "画像の大きさ", "条件・場面", "試行数", "正解の取得", "開発用"],
        rows,
        align=["l", "", "", "", "", ""],
    )


def values(study: Study, data_root: Path) -> dict[str, str]:
    results = study.results
    mean, largest = recovery_error(data_root)
    stand_occupancy = np.mean([results.occupancy(key) for key in ("sprite1", "sprite2")])
    stand_spill = np.mean([results.spill(key) for key in ("sprite1", "sprite2")])
    return {
        "occ_ymm": f"{stand_occupancy:.2f}",
        "occ_f": f"{results.occupancy('collection0'):.2f}",
        "spill_ymm": f"{100 * stand_spill:.0f}%",
        "spill_f": f"{100 * results.spill('collection0'):.0f}%",
        "recov_mean": f"{mean:.3f}",
        "recov_max": f"{largest:.1f}",
        "m_tree": plain(DEFAULT_SETTINGS.boundary.strength),
        "lam_gc": plain(DEFAULT_SETTINGS.graph_cut.smoothness),
        "abstract": abstract(study),
    }
