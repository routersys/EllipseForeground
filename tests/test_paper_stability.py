import numpy as np
import pytest

from ellipse_foreground.settings import DEFAULT_SETTINGS
from paper.results import Results
from paper.sections.refinement import REFINE_SETS
from paper.sections.stability import (
    GROUP_STRIDE,
    UNSEEN_NAME,
    abstract_adaptation,
    adaptation_summary,
    all_adaptation_rows,
    auc_note,
    conclusion_stability,
    correlation_of,
    discussion_stability,
    gated_value,
    intro_stability,
    pooled_rows,
    quality_auc,
    stability_section,
    unseen_rows,
    unstable,
    value,
    weakest_stable,
)
from paper.study import Study
from tests.paper_fixtures import write_results


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("results")
    write_results(directory)
    return Study(Results(directory), None)


def test_a_trial_is_unstable_below_the_threshold_only():
    threshold = DEFAULT_SETTINGS.stability.threshold
    assert unstable({"stability": threshold - 0.01})
    assert not unstable({"stability": threshold})
    assert not unstable({"stability": 1.0})


def test_the_gated_value_uses_the_adapted_result_only_when_unstable():
    adapted = {"iou": 0.9}
    plain = {"iou": 0.7}
    low = {"stability": 0.5, "refit": adapted, "tree": plain}
    high = {"stability": 0.99, "refit": adapted, "tree": plain}
    assert gated_value("refit", "tree")(low) == 0.9
    assert gated_value("refit", "tree")(high) == 0.7
    assert value("tree")(low) == 0.7


def test_the_correlation_of_a_monotone_relation_is_one():
    rows = [{"stability": 0.8 + 0.01 * index, "tree": {"iou": 0.5 + 0.02 * index}} for index in range(10)]
    assert correlation_of(rows) == pytest.approx(1.0)
    assert correlation_of(rows[:1] * 5) == 0.0


def test_the_auc_ranks_stable_trials_above_failures():
    rows = [{"stability": 0.9 + 0.01 * index, "tree": {"iou": 0.9 if index >= 5 else 0.3}} for index in range(10)]
    assert quality_auc(rows) == pytest.approx(1.0)


def test_the_auc_note_names_the_rows_without_both_classes():
    single = {selection.name: [{"stability": 0.9, "tree": {"iou": 0.9}}] * 2 for selection in REFINE_SETS}
    note = auc_note(single)
    assert "−" in note
    assert REFINE_SETS[0].short in note
    both = [{"stability": 0.9, "tree": {"iou": 0.9}}, {"stability": 0.5, "tree": {"iou": 0.2}}]
    assert auc_note({selection.name: both for selection in REFINE_SETS}) == ""


def test_the_weakest_stable_row_has_the_lowest_mean_among_stable_trials():
    unstable_trial = {"stability": DEFAULT_SETTINGS.stability.threshold - 0.1, "tree": {"iou": 0.1}}
    rows = {selection.name: [{"stability": 1.0, "tree": {"iou": 0.9}}, unstable_trial] for selection in REFINE_SETS}
    rows[REFINE_SETS[2].name] = [{"stability": 1.0, "tree": {"iou": 0.6}}]
    selection, mean = weakest_stable(rows)
    assert selection == REFINE_SETS[2]
    assert mean == pytest.approx(0.6)
    assert weakest_stable({selection.name: [unstable_trial] for selection in REFINE_SETS}) is None


def test_every_row_set_has_adaptation_records(study):
    rows = all_adaptation_rows(study)
    assert set(rows) == {selection.name for selection in REFINE_SETS}
    assert all(rows[name] for name in rows)


def test_pooled_rows_keep_the_scenes_of_different_sets_apart(study):
    rows = all_adaptation_rows(study)
    pooled = pooled_rows(rows)
    keys = {row["scene"] for row in pooled}
    assert len(keys) == sum(len({row["scene"] for row in rows[selection.name]}) for selection in REFINE_SETS)
    assert all(row["scene"] < GROUP_STRIDE * (len(REFINE_SETS) + 1) for row in pooled)


def test_unseen_rows_leave_out_the_scenes_used_for_the_selection(study):
    rows = unseen_rows(study)
    assert rows
    scenes = {row["scene"] % GROUP_STRIDE for row in rows}
    assert all(scene % 3 != 0 or scene % 4 != 0 for scene in scenes)
    groups = {row["scene"] // GROUP_STRIDE for row in rows}
    assert groups == {1, 2, 3}
    photo = {row["scene"] % GROUP_STRIDE for row in rows if row["scene"] // GROUP_STRIDE == 3}
    assert photo and all(scene % 4 != 0 for scene in photo)
    public = {row["scene"] % GROUP_STRIDE for row in rows if row["scene"] // GROUP_STRIDE == 1}
    assert public and all(scene % 3 != 0 for scene in public)


def test_the_section_has_both_tables_and_the_unseen_row(study):
    section = stability_section(study)
    assert 'id="t:stab"' in section
    assert 'id="t:refit"' in section
    assert UNSEEN_NAME in section
    assert "全体" in section
    assert "<<" not in section


def test_the_summary_values_are_filled(study):
    summary = adaptation_summary(study)
    assert summary["stab_threshold"] == str(DEFAULT_SETTINGS.stability.threshold)
    assert summary["refit_iterations"] == str(DEFAULT_SETTINGS.refit.iterations)
    assert summary["stab_share"].endswith("%")
    assert float(summary["stab_rho"]) <= 1.0


def test_the_sentences_are_not_empty_and_are_cached(study):
    for sentence in (
        abstract_adaptation(study),
        intro_stability(study),
        discussion_stability(study),
        conclusion_stability(study),
    ):
        assert sentence.strip()
        assert "<<" not in sentence
    assert "refit_abstract" in study.cache
    assert study.remembered("refit_abstract", lambda: "other") == study.cache["refit_abstract"]


def test_the_pooled_stability_is_a_ratio(study):
    pooled = pooled_rows(all_adaptation_rows(study))
    assert 0.0 <= float(np.mean([row["stability"] for row in pooled])) <= 1.0
