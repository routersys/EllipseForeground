from dataclasses import replace

from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.metrics import mask_iou
from ellipse_foreground.evaluation.scene import Scene
from ellipse_foreground.evaluation.trials import draw_trials
from ellipse_foreground.settings import DEFAULT_SETTINGS

WORK_WIDTHS = (128, 256, 384)


def resolution_records(scene: Scene, width: int, count: int) -> list[dict]:
    settings = replace(DEFAULT_SETTINGS, work=replace(DEFAULT_SETTINGS.work, width=width))
    context = SceneContext(scene, settings)
    analysis = context.analysis
    records = []
    for index, ellipse in enumerate(draw_trials(scene, count)):
        if not context.informative(ellipse):
            continue
        regions = analysis.regions(ellipse)
        scores = analysis.prior_map(ellipse) + settings.weights.texture * analysis.texture_map(regions)
        mask = analysis.cut(scores)
        records.append(
            {
                "scene": scene.index,
                "name": scene.name,
                "trial": index,
                "width": width,
                "raw_iou": mask_iou(mask, context.work_truth),
                **context.finished_score(mask),
            }
        )
    return records
