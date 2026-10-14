from pathlib import Path

import pytest

from paper.__main__ import parse
from paper.build import CONTENT_WIDTH, PROBE, build, parse_overflow, probe_html
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


def test_the_probe_is_placed_before_the_end_of_the_body():
    text = probe_html("<html><body><p>a</p></body></html>")
    assert text.index(PROBE) < text.index("</body>")
    assert f"width:{CONTENT_WIDTH}px" in PROBE
    with pytest.raises(ValueError):
        probe_html("<p>a</p>")


def test_the_overflow_is_read_from_the_title():
    dom = "<html><head><title>OVERFLOW:[&quot;表23 標本&quot;]</title></head></html>"
    assert parse_overflow(dom) == ["表23 標本"]
    assert parse_overflow("<title>OVERFLOW:[]</title>") == []
    with pytest.raises(RuntimeError):
        parse_overflow("<title>other</title>")


def test_a_wide_element_stops_the_build_before_the_pdf(tmp_path, monkeypatch):
    write_results(tmp_path / "results")
    printed = []
    monkeypatch.setattr("paper.build.overflowing", lambda page, chrome: ["表23 標本"])
    monkeypatch.setattr("paper.build.print_to_pdf", lambda *arguments: printed.append(arguments))
    with pytest.raises(RuntimeError, match="表23"):
        build(
            tmp_path / "results", DATA, tmp_path / "missing-timing.json", tmp_path / "output", figures=False, pdf=True
        )
    assert not printed
