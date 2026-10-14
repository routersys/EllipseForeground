import pytest

from paper.results import Results
from paper.sections.diagnosis import (
    abstract_diagnosis,
    all_oracle_rows,
    background_reading,
    conclusion_diagnosis,
    dataset_phrase,
    diagnosis_section,
    diagnosis_summary,
    discussion_diagnosis,
    intro_diagnosis,
    mean_purity,
    mixture_verdict,
    prior_share,
    share_sentence,
)
from paper.sections.refinement import REFINE_SETS
from paper.study import Study
from tests.paper_fixtures import write_results


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("results")
    write_results(directory)
    return Study(Results(directory), None)


def test_every_row_set_has_oracle_records(study):
    rows = all_oracle_rows(study)
    assert set(rows) == {selection.name for selection in REFINE_SETS}
    assert all(rows[name] for name in rows)


def test_the_mean_purity_skips_empty_samples():
    rows = [{"purity": {"core": 0.5, "ring": None}}, {"purity": {"core": 1.0, "ring": None}}]
    assert mean_purity(rows, "core") == pytest.approx(0.75)
    assert mean_purity(rows, "ring") is None


def test_the_prior_share_is_a_ratio_of_gains(study):
    shares = prior_share(all_oracle_rows(study))
    assert shares
    assert all(0.0 <= share <= 5.0 for share in shares.values())


def test_the_section_has_the_three_tables(study):
    section = diagnosis_section(study)
    for identifier in ("t:oracle", "t:oracle2", "t:unmix"):
        assert f'id="{identifier}"' in section
    assert "<<" not in section
    assert section.count("<tr") >= 3 * (len(REFINE_SETS) + 1)


def test_the_summary_values_are_filled(study):
    summary = diagnosis_summary(study)
    assert summary["oracle_prior"]
    assert summary["oracle_samples"]
    assert summary["oracle_share_low"].endswith("%")


def test_the_sentences_are_not_empty(study):
    for sentence in (
        intro_diagnosis(study),
        abstract_diagnosis(study),
        discussion_diagnosis(study),
        conclusion_diagnosis(study),
    ):
        assert sentence.strip()
        assert "<<" not in sentence


def test_the_dataset_phrase_names_the_exceptions_when_most_rows_are_chosen():
    assert dataset_phrase(list(REFINE_SETS)).endswith("のすべて")
    assert dataset_phrase(list(REFINE_SETS[1:])) == f"{REFINE_SETS[0].short}を除くデータセット"
    assert dataset_phrase(list(REFINE_SETS[:2])) == "、".join(selection.short for selection in REFINE_SETS[:2])


def test_the_share_sentence_names_the_rows_above_one_hundred_percent():
    sentence = share_sentence({REFINE_SETS[0]: 1.28, REFINE_SETS[1]: 0.5})
    assert "50%から128%" in sentence
    assert f"100%を超える{REFINE_SETS[0].short}では" in sentence
    assert "100%を超える" not in share_sentence({REFINE_SETS[0]: 0.9})
    assert share_sentence({}) == ""


def test_the_background_reading_needs_pure_rings_and_larger_near_gains():
    far = dict.fromkeys(REFINE_SETS, 0.001)
    near = dict.fromkeys(REFINE_SETS, 0.05)
    assert "境界に近い背景" in background_reading(far, near, list(REFINE_SETS), 0.98)
    assert background_reading(far, near, list(REFINE_SETS), 0.9) == ""
    assert background_reading(far, near, list(REFINE_SETS[1:]), 0.98) == ""
    assert background_reading(far, near, list(REFINE_SETS), None) == ""


def test_the_mixture_verdict_depends_on_the_best_estimated_share(monkeypatch):
    monkeypatch.setattr("paper.sections.diagnosis.estimated_share", lambda rows: 0.33)
    assert "最大でも33%にとどまる" in mixture_verdict({})
    monkeypatch.setattr("paper.sections.diagnosis.estimated_share", lambda rows: 0.6)
    assert "半分以上を取り戻せる行がある" in mixture_verdict({})
    monkeypatch.setattr("paper.sections.diagnosis.estimated_share", lambda rows: None)
    assert mixture_verdict({}) == ""
