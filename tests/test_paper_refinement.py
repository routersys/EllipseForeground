import pytest

from ellipse_foreground.settings import DEFAULT_SETTINGS
from paper.results import Results
from paper.sections.refinement import (
    REFINE_SETS,
    abstract_refinement,
    agreement_future,
    conclusion_refinement,
    development_photo,
    discussion_refinement,
    other_photo,
    refinement_rows,
    refinement_section,
    refinement_summary,
    row_phrase,
)
from paper.study import Study, Timing
from tests.paper_fixtures import write_results


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    directory = tmp_path_factory.mktemp("results")
    write_results(directory)
    timing = Timing(
        1.0,
        50.0,
        "3.3万",
        "試験用の環境",
        5,
        40,
        1.5,
        "PyMaxflow 1.3.2",
        band_ms=0.3,
        band_full_ms=1.5,
        band_share=0.05,
        band_radius=2,
    )
    return Study(Results(directory), timing)


def test_photos_are_split_by_the_development_modulus():
    assert development_photo(0) and development_photo(4)
    assert not development_photo(1) and not development_photo(7)
    assert all(development_photo(scene) != other_photo(scene) for scene in range(12))


def test_every_row_set_selects_trials_that_exist_in_both_results(study):
    for selection in REFINE_SETS:
        rows = refinement_rows(study, selection)
        assert rows
        assert all("graph" in row and "band" in row and "agreement" in row for row in rows)


def test_the_photo_rows_do_not_share_scenes(study):
    development = {row["scene"] for row in refinement_rows(study, REFINE_SETS[-2])}
    others = {row["scene"] for row in refinement_rows(study, REFINE_SETS[-1])}
    assert development and others
    assert not development & others


def test_row_phrase_names_the_exceptions_when_most_rows_qualify():
    assert row_phrase(list(REFINE_SETS)) == "表のすべての行"
    assert "を除く" in row_phrase(list(REFINE_SETS[:-1]))
    assert row_phrase(list(REFINE_SETS[:2])) == "、".join(selection.short for selection in REFINE_SETS[:2])


def test_the_section_has_the_three_tables_and_the_timing(study):
    section = refinement_section(study)
    for identifier in ("t:band", "t:bandparam", "t:agree"):
        assert f'id="{identifier}"' in section
    assert "4.7分の1" in section or "約5.0分の1" in section
    assert section.count("<tr") >= 3 + 2 * len(REFINE_SETS) + 9 - 1


def test_the_summary_values_are_filled(study):
    summary = refinement_summary(study)
    assert summary["band_radius"] == str(DEFAULT_SETTINGS.band.radius)
    assert summary["band_rows"] == str(len(REFINE_SETS))
    assert summary["band_item"]
    assert summary["band_share"].endswith("%")


def test_the_summary_sentences_are_not_empty(study):
    for sentence in (
        abstract_refinement(study),
        discussion_refinement(study),
        conclusion_refinement(study),
        agreement_future(study),
    ):
        assert sentence.strip()
        assert "<<" not in sentence
