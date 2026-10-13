import json
import re
from pathlib import Path

from paper import mathml
from paper.markup import figure
from paper.numbering import (
    bibliography,
    number_floats,
    number_headings,
    resolve_citations,
    resolve_references,
    tidy,
)
from paper.sections.ablation import (
    finish_section,
    forms_section,
    graph_cut_section,
    models_section,
    prior_section,
    resolution_section,
)
from paper.sections.closing import conclusion, discussion
from paper.sections.main import main_section
from paper.sections.materials import data_table, values
from paper.sections.qualitative import qualitative_section, qualitative_text
from paper.study import Study

SOURCE = Path(__file__).resolve().parent / "source"
VALUE_TOKEN = re.compile(r"<<V:([A-Za-z0-9_.]+)>>")
TABLE_TOKEN = re.compile(r"<<T:([A-Za-z0-9_]+)>>")


def read_source(name: str) -> str:
    return (SOURCE / name).read_text(encoding="utf-8")


def boundary_figure() -> str:
    return figure(
        "f:reg",
        "fig_reg.svg",
        "境界項の強さへの感度",
        "境界項の強さに対するIoU（精緻化前の二値マスク、適応（色）のスコア）。上段は木の切断の三つの形、下段はグラフカットで、点線は境界項なしのIoU、水平線は平均類似度型の最良のIoUである。",
    )


def ablation_sections(study: Study) -> str:
    return (
        '<h3 id="s:abl_pen">境界項の形と強さ</h3>\n'
        + forms_section(study)
        + "\n"
        + boundary_figure()
        + '\n<h3 id="s:abl_gc">グラフカットとの比較</h3>\n'
        + graph_cut_section(study)
        + '\n<h3 id="s:abl_res">解像度の影響</h3>\n'
        + resolution_section(study)
        + '\n<h3 id="s:abl_prior">事前分布の内側の強さ</h3>\n'
        + prior_section(study)
        + '\n<h3 id="s:abl_finish">精緻化</h3>\n'
        + finish_section(study)
        + '\n<h3 id="s:abl_noise">楕円の誤差の大きさ</h3>\n'
        + models_section(study)
        + '\n<h3 id="s:qual">定性的な結果</h3>\n'
        + qualitative_section(qualitative_text(study))
    )


def fill_tokens(html: str, text_values: dict[str, str], tables: dict[str, str]) -> str:
    def table(match: re.Match[str]) -> str:
        return tables[match.group(1)]

    def value(match: re.Match[str]) -> str:
        return text_values[match.group(1)]

    return VALUE_TOKEN.sub(value, TABLE_TOKEN.sub(table, html))


def body(study: Study, data_root: Path) -> tuple[str, dict[str, str]]:
    text_values = values(study, data_root)
    tables = {"data": data_table(study)}
    parts = [
        read_source("introduction.html"),
        read_source("related.html"),
        read_source("problem.html"),
        read_source("method.html"),
        read_source("data.html"),
        main_section(study),
        ablation_sections(study),
        discussion(study),
        conclusion(study),
        read_source("acknowledgments.html"),
        read_source("appendix.html"),
    ]
    return fill_tokens("\n".join(parts), text_values, tables), text_values


def render(study: Study, data_root: Path) -> tuple[str, list[str]]:
    html, text_values = body(study, data_root)
    html = tidy(html)
    html, headings = number_headings(html)
    html, floats = number_floats(html)
    html = resolve_references(html, headings, floats)
    html, order = resolve_citations(html)
    references = json.loads(read_source("references.json"))
    page = read_source("front.html").replace("<<BODY>>", html + "\n" + bibliography(order, references))
    page = fill_tokens(page, text_values, {})
    return mathml.convert(page), order
