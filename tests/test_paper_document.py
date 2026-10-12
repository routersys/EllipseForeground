import re
from pathlib import Path

import pytest

from paper.document import render
from paper.results import Results
from paper.study import Study, Timing
from tests.paper_fixtures import write_results

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    directory = tmp_path_factory.mktemp("results")
    write_results(directory)
    study = Study(Results(directory), Timing(0.5, 40.0, "3.3万〜3.7万", "試験用の環境", 5, 40))
    html, order = render(study, DATA)
    return html, order


def test_every_placeholder_and_reference_is_resolved(page):
    html, _ = page
    assert "<<" not in html
    assert "<ref " not in html
    assert "<cite " not in html
    assert "\\(" not in html
    assert "\\[" not in html


def test_sections_are_numbered_in_order(page):
    html, _ = page
    numbers = re.findall(r'<h2[^>]*><span class="sec-num">(\d+)</span>', html)
    assert numbers == [str(index) for index in range(1, len(numbers) + 1)]
    assert len(numbers) >= 7


def test_all_tables_and_figures_are_captioned(page):
    html, _ = page
    assert len(re.findall(r'<span class="no">表\d+</span>', html)) >= 12
    assert len(re.findall(r'<span class="no">図\d+</span>', html)) == 11
    assert 'class="no">表</span>' not in html


def test_citations_are_listed_in_the_bibliography(page):
    html, order = page
    assert len(order) > 20
    assert html.count('<li><span class="n">[') == len(order)


def test_timing_ratio_is_stated_in_text(page):
    html, _ = page
    assert "約80倍" in html
    assert "約80分の1" in html


def test_abstract_and_values_are_filled(page):
    html, _ = page
    assert '<span class="label">概要</span>被写体を楕円で' in html
    assert "256" in html


def test_document_without_timing_omits_the_ratio(tmp_path):
    write_results(tmp_path)
    html, _ = render(Study(Results(tmp_path), None), DATA)
    assert "分の1" not in html
    assert "Dinic" not in html
