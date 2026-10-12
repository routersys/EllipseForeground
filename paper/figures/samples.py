from pathlib import Path

import matplotlib.pyplot as plt

from ellipse_foreground.evaluation.sprites import SCENES_PER_SPRITE, public_scenes, sprite_files
from paper.figures.style import save

LABELS = {
    "breezy_oak": "Breezy・A",
    "breezy_john": "Breezy・B",
    "cyanea1_prom": "Cyanea・set 1",
    "cyanea5_outfit": "Cyanea・set 5",
    "cyanea7_casual": "Cyanea・set 7",
}
BACKGROUNDS = (0, 2, 3, 5, 7, 8, 10, 11)
CROP = (240, 1040)
STRENGTHS = (0.0, 1.0)


def first_scene_index(data_root: Path, identifier: str) -> int:
    for position, (name, _) in enumerate(sprite_files(data_root)):
        if name != identifier:
            continue
        start = position * SCENES_PER_SPRITE
        for scene in public_scenes(data_root, 0.0, selection=range(start, start + SCENES_PER_SPRITE)):
            background = int(scene.name.split("_bg")[1].split("_")[0])
            if background in BACKGROUNDS:
                return scene.index
    raise ValueError(identifier)


def public_samples(directory: Path, data_root: Path) -> None:
    indices = {identifier: first_scene_index(data_root, identifier) for identifier in LABELS}
    figure, axes = plt.subplots(2, len(LABELS), figsize=(7.6, 2.6))
    for row, strength in enumerate(STRENGTHS):
        for column, (identifier, index) in enumerate(indices.items()):
            scene = next(public_scenes(data_root, strength, selection=range(index, index + 1)))
            axis = axes[row, column]
            axis.imshow(scene.texture_source[:, CROP[0] : CROP[1]])
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
            if row == 0:
                axis.set_title(LABELS[identifier], fontsize=8.2)
    axes[0, 0].set_ylabel("転写0%", fontsize=8.2)
    axes[1, 0].set_ylabel("転写100%", fontsize=8.2)
    figure.tight_layout(w_pad=0.3, h_pad=0.3)
    save(figure, directory, "fig_vn", vector=False)
