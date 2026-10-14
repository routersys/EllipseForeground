import pytest

from paper.results import Results
from paper.sections.baselines import (
    STRATA,
    coverage_rows,
    coverage_section,
    coverage_summary,
    coverage_table,
    stratum,
)
from paper.study import Study
from tests.paper_fixtures import write_results


@pytest.fixture
def study(tmp_path):
    write_results(tmp_path)
    return Study(Results(tmp_path), None)


def test_every_trial_belongs_to_exactly_one_stratum(study):
    rows = coverage_rows(study)
    assert rows
    assert sum(len(stratum(rows, low, high)) for low, high, _ in STRATA) == len(rows)


def test_scene_identifiers_do_not_collide_between_datasets(study):
    rows = coverage_rows(study)
    scenes = {row["scene"] for row in rows}
    assert any(name.startswith("public0-") for name in scenes)
    assert any(name.startswith("photo-") for name in scenes)


def test_the_table_has_one_row_per_stratum(study):
    table = coverage_table(study)
    assert 'id="t:coverage"' in table
    assert table.count("<tr") == 1 + len(STRATA)
    for _, _, label in STRATA:
        assert label in table


def test_the_summary_values_are_filled(study):
    summary = coverage_summary(study)
    assert summary["cov_n_all"] == str(len(coverage_rows(study)))
    assert int(summary["cov_n_low"]) > 0
    assert "［" in summary["cov_grabcut_low"]


def test_the_section_states_that_the_split_is_exploratory(study):
    section = coverage_section(study)
    assert "探索的な分析" in section
    assert '<ref k="t:coverage"></ref>' in section
