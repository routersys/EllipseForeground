import pytest

from paper.results import Results
from paper.sections.holes import all_hole_rows, hole_lead, hole_share, holes_discussion, holes_section
from paper.sections.refinement import REFINE_SETS
from paper.study import Study
from tests.paper_fixtures import write_results


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("results")
    write_results(directory)
    return Study(Results(directory), None)


def test_every_row_set_has_hole_records(study):
    rows = all_hole_rows(study)
    assert set(rows) == {selection.name for selection in REFINE_SETS}
    assert all(rows[name] for name in rows)


def test_the_hole_share_is_a_ratio(study):
    for data in all_hole_rows(study).values():
        assert 0.0 <= hole_share(data) <= 1.0


def test_the_section_has_the_table(study):
    section = holes_section(study)
    assert 'id="t:holes"' in section
    assert "<<" not in section
    assert section.count("<tr") >= 1 + len(REFINE_SETS)


def test_the_lead_follows_the_range_of_the_shares():
    assert "大半" in hole_lead([0.6, 0.9])
    assert "一部" in hole_lead([0.1, 0.4])
    assert "大きく異なる" in hole_lead([0.0, 0.9])


def test_the_discussion_is_a_paragraph_or_empty(study):
    text = holes_discussion(study)
    assert not text or text.startswith("<p>")
    assert "<<" not in text
