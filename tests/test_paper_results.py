import pytest

from ellipse_foreground.evaluation.results import append_records
from paper.numbers import interval_text
from paper.quantities import (
    degenerate_trials,
    graph_cut_comparison,
    headline,
    ordering_disagreement,
    rank_correlations,
)
from paper.results import SUBSETS, Measure, Results, sprite_means


def score(iou: float, mae: float = 1.0) -> dict:
    return {"raw_iou": iou, "iou": iou, "mae": mae}


def trial_record(scene: int, trial: int, tree_iou: float, graph_iou: float, overlap: float, auc: float) -> dict:
    methods = ("prior", "tex", "app", "mix", "adaptive_c", "adaptive")
    tree = {method: score(tree_iou) for method in methods}
    tree["ellipse"] = score(tree_iou - 0.2)
    tree["oracle"] = score(tree_iou + 0.05)
    return {
        "scene": scene,
        "trial": trial,
        "overlaps": {"color": overlap, "texture": overlap, "patch": overlap},
        "auc": {"color": auc, "texture": auc, "patch": auc},
        "weights": {"adaptive_color": [1.0, 0.25, 0.3], "adaptive_color_patch": [1.0, 0.25, 0.6]},
        "occupancy": 0.5,
        "spill": 0.1,
        "tree": tree,
        "graph": {method: score(graph_iou) for method in methods},
    }


@pytest.fixture
def results(tmp_path):
    records = [
        trial_record(0, 0, 0.8, 0.9, 0.1, 0.9),
        trial_record(0, 1, 0.6, 0.7, 0.2, 0.8),
        trial_record(1, 0, 0.5, 0.6, 0.3, 0.7),
        trial_record(5, 0, 0.9, 1.0, 0.4, 0.6),
    ]
    append_records(tmp_path / "comparison" / "sprite1.jsonl", records)
    return Results(tmp_path)


def test_subsets_select_scenes_of_a_group(results):
    assert {record["scene"] for record in results.trials("sprite1")} == {0, 1, 5}
    assert {record["scene"] for record in results.trials("dose0")} == {0}
    assert {record["scene"] for record in results.trials("dose25")} == {1}
    assert {record["scene"] for record in results.trials("others")} == {5}
    assert results.trials("sprite2") == []


def test_scene_count_and_occupancy(results):
    assert results.scenes("sprite1") == 3
    assert results.occupancy("sprite1") == pytest.approx(0.5)
    assert results.spill("sprite1") == pytest.approx(0.1)


def test_mean_averages_trials_within_a_scene_first(results):
    measure = Measure(results)
    assert measure.mean("sprite1", "tex") == pytest.approx((0.7 + 0.5 + 0.9) / 3)
    assert measure.mean("sprite1", "tex", family="graph") == pytest.approx((0.8 + 0.6 + 1.0) / 3)
    assert measure.mean("sprite1", "mix", "mae") == pytest.approx(1.0)


def test_interval_of_a_constant_difference_has_no_spread(results):
    measure = Measure(results)
    interval = measure.interval("sprite1", "tex", "ellipse")
    assert interval.mean == pytest.approx(0.2)
    assert interval.low == pytest.approx(0.2)
    assert interval.high == pytest.approx(0.2)
    assert measure.significant("sprite1", "tex", "ellipse")
    assert not measure.significant("sprite1", "tex", "mix")


def test_regret_is_measured_against_the_best_cue_method(results):
    measure = Measure(results)
    assert measure.regret("sprite1", "mix") == pytest.approx(0.0)


def test_headline_and_graph_cut_comparison_cover_the_available_sets(tmp_path):
    for group in ("sprite1", "sprite2", "collection0", "collection100", "public0", "public100"):
        append_records(
            tmp_path / "comparison" / f"{group}.jsonl",
            [trial_record(scene, 0, 0.8, 0.9, 0.1, 0.9) for scene in range(3)],
        )
    measure = Measure(Results(tmp_path))
    summary = headline(measure)
    assert summary.ellipse_low == pytest.approx(0.6)
    assert summary.mix_regret == pytest.approx(0.0)
    comparison = graph_cut_comparison(measure)
    assert set(comparison.higher) == set(comparison.difference)
    assert comparison.lower == []
    assert comparison.difference["public0"] == pytest.approx(0.1)


def test_rank_correlation_is_negative_when_overlap_and_auc_oppose(tmp_path):
    records = [trial_record(scene, 0, 0.8, 0.9, 0.1 * scene, 1.0 - 0.05 * scene) for scene in range(6)]
    for group in ("sprite1", "sprite2", "collection0", "collection100"):
        append_records(tmp_path / "comparison" / f"{group}.jsonl", records)
    results = Results(tmp_path)
    correlations = rank_correlations(results)
    assert correlations["texture"].pooled == pytest.approx(-1.0)
    assert set(correlations["color"].per_set) == {"sprite1", "sprite2", "collection0", "collection100"}
    assert ordering_disagreement(results) == pytest.approx(0.0)


def test_every_subset_names_a_group_and_a_scene_range():
    for subset in SUBSETS.values():
        assert subset.group
        assert subset.includes(0) or subset.scenes.start > 0


def test_interval_text_shows_mean_and_bounds(results):
    text = interval_text(Measure(results).interval("sprite1", "tex", "ellipse"))
    assert text == "+0.200［+0.200，+0.200］"


def test_scenes_where_no_cue_works_are_detected_and_excluded(tmp_path):
    records = [trial_record(scene, 0, 0.8, 0.9, 0.1, 0.9) for scene in range(3)]
    blind = trial_record(3, 0, 0.0, 0.0, 1.0, 0.5)
    blind["occupancy"] = 0.5
    append_records(tmp_path / "comparison" / "collection100.jsonl", [*records, blind])
    found = degenerate_trials(Results(tmp_path), "collection100")
    assert found is not None
    assert (found.scenes, found.trials, found.total_trials) == (1, 1, 4)
    assert found.cue_iou == pytest.approx(0.0)
    assert found.without_mix == pytest.approx(0.8)
    assert found.with_mix == pytest.approx(0.6)
    assert degenerate_trials(Results(tmp_path / "missing"), "collection100") is None


def test_sprite_means_average_scenes_within_a_sprite():
    records = [{"scene": scene, "value": value} for scene, value in ((0, 0.5), (1, 0.7), (2, 0.9), (6, 0.9))]
    assert sprite_means(records, lambda record: record["value"]) == pytest.approx([0.7, 0.9])


def test_sprite_interval_resamples_sprites(tmp_path):
    records = [trial_record(scene, 0, 0.5, 0.6, 0.1, 0.9) for scene in (0, 1, 2, 6, 7)]
    append_records(tmp_path / "comparison" / "public0.jsonl", records)
    interval = Measure(Results(tmp_path)).sprite_interval("public0", "tex", "tex", first_family="graph")
    assert interval.mean == pytest.approx(0.1)
    assert interval.low == pytest.approx(0.1)
    assert interval.high == pytest.approx(0.1)
