import pytest

from ellipse_foreground.evaluation.statistics import Interval
from paper.results import Results
from paper.sections.materials import data_table
from paper.sections.photos import photo_section, photo_table, relation
from paper.study import Study
from tests.paper_fixtures import DEFAULT_SCENES, write_results

TRIALS = 2


@pytest.fixture
def study(tmp_path):
    write_results(tmp_path)
    return Study(Results(tmp_path), None)


def test_the_photo_section_reports_the_scene_and_trial_counts(study):
    section = photo_section(study)
    assert 'id="s:photo"' in section
    assert 'id="t:photo"' in section
    assert f"場面数は{DEFAULT_SCENES}、試行数は{DEFAULT_SCENES * TRIALS}" in section
    assert '<cite k="ke2022"></cite>' in section


def test_the_photo_table_lists_every_method_with_iou_and_mae(study):
    table = photo_table(study)
    assert table.count("<tr") == 9
    assert "正解スコア" in table
    assert table.count('class="best"') == 2


def test_the_dataset_table_gains_a_photo_row(study):
    table = data_table(study)
    assert "写真（PPM-100）" in table
    assert f"{DEFAULT_SCENES}枚" in table


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        (Interval(0.1, 0.05, 0.15), "上回る"),
        (Interval(-0.1, -0.15, -0.05), "下回る"),
        (Interval(0.0, -0.05, 0.05), "はっきりしない"),
    ],
)
def test_relation_reads_the_interval_against_zero(interval, expected):
    assert expected in relation(interval)
