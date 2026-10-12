from pathlib import Path

from paper.__main__ import parse
from paper.build import build
from tests.paper_fixtures import write_results

DATA = Path(__file__).resolve().parent.parent / "data"


def test_build_writes_the_page_and_stylesheet(tmp_path):
    write_results(tmp_path / "results")
    page = build(
        tmp_path / "results",
        DATA,
        tmp_path / "missing-timing.json",
        tmp_path / "output",
        figures=False,
        pdf=False,
    )
    assert page == tmp_path / "output" / "paper.html"
    assert "<math" in page.read_text(encoding="utf-8")
    assert (tmp_path / "output" / "style.css").exists()
    assert not (tmp_path / "output" / "paper.pdf").exists()


def test_command_line_defaults_point_into_the_repository():
    build_options = parse(["build"])
    assert build_options.results.name == "results"
    assert build_options.output.parts[-2:] == ("paper", "build")
    assert parse(["timing"]).output.name == "timing.json"
