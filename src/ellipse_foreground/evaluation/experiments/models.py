from ellipse_foreground.evaluation.experiments.context import SceneContext
from ellipse_foreground.evaluation.trials import ErrorModel, draw_trials


def model_records(context: SceneContext, model: ErrorModel, count: int) -> list[dict]:
    analysis = context.analysis
    texture_weight = context.settings.weights.texture
    records = []
    for index, ellipse in enumerate(draw_trials(context.scene, count, model)):
        if not context.informative(ellipse):
            continue
        regions = analysis.regions(ellipse)
        prior = analysis.prior_map(ellipse)
        texture = prior + texture_weight * analysis.texture_map(regions)
        oracle = 2.0 * context.work_alpha - 1.0
        records.append(
            {
                "scene": context.scene.index,
                "name": context.scene.name,
                "trial": index,
                "model": model.name,
                "tree": {
                    "ellipse": context.finished_score(regions.inside),
                    "prior": context.finished_score(analysis.cut(prior)),
                    "tex": context.finished_score(analysis.cut(texture)),
                    "oracle": context.finished_score(analysis.cut(oracle)),
                },
            }
        )
    return records
