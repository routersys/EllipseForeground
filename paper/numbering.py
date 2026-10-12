import re
from collections.abc import Mapping

CJK = "ぁ-んァ-ヶ一-龠々ー、。，．・「」（）"
HEADING = re.compile(r'<(h2|h3)( id="([^"]+)")?( class="([^"]+)")?>(.*?)</\1>', re.S)
FIGURE_ID = re.compile(r'<figure id="(f:[^"]+)"')
TABLE_ID = re.compile(r'<div class="tcap" id="(t:[^"]+)"')
EQUATION_ID = re.compile(r'<div class="eq" id="(e:[^"]+)"')
FIGURE_CAPTION = re.compile(r'(<figure id="f:[^"]+".*?<figcaption>)<span class="no">図</span>', re.S)
TABLE_CAPTION = re.compile(r'(<div class="tcap" id="t:[^"]+">)<span class="no">表</span>')
EQUATION = re.compile(r'<div class="eq" id="(e:[^"]+)">(.*?)</div>', re.S)
REFERENCE = re.compile(r'<ref k="([^"]+)"(?: bare="1")?></ref>')
CITATION = re.compile(r'<cite k="([^"]+)"></cite>')
PROTECTED = re.compile(r"(\\\[.*?\\\]|\\\(.*?\\\)|<[^>]+>)", re.S)


def attribute(identifier: str | None) -> str:
    return f' id="{identifier}"' if identifier else ""


def number_headings(html: str) -> tuple[str, dict[str, tuple[str, str]]]:
    identifiers: dict[str, tuple[str, str]] = {}
    sections = subsections = 0
    pieces = []
    position = 0
    for match in HEADING.finditer(html):
        pieces.append(html[position : match.start()])
        tag, _, identifier, _, klass, title = match.groups()
        position = match.end()
        if klass == "appx":
            if identifier:
                identifiers[identifier] = ("付録A", tag)
            pieces.append(f'<{tag}{attribute(identifier)} class="appx">{title}</{tag}>')
            continue
        if tag == "h2":
            sections += 1
            subsections = 0
            number = str(sections)
        else:
            subsections += 1
            number = f"{sections}.{subsections}"
        if identifier:
            identifiers[identifier] = (number, tag)
        pieces.append(f'<{tag}{attribute(identifier)}><span class="sec-num">{number}</span>{title}</{tag}>')
    pieces.append(html[position:])
    return "".join(pieces), identifiers


def number_floats(html: str) -> tuple[str, dict[str, str]]:
    identifiers: dict[str, str] = {}
    for count, match in enumerate(FIGURE_ID.finditer(html), start=1):
        identifiers[match.group(1)] = f"図{count}"
    for count, match in enumerate(TABLE_ID.finditer(html), start=1):
        identifiers[match.group(1)] = f"表{count}"
    for count, match in enumerate(EQUATION_ID.finditer(html), start=1):
        identifiers[match.group(1)] = str(count)

    figures = iter(range(1, 10**6))
    tables = iter(range(1, 10**6))
    equations = iter(range(1, 10**6))
    html = FIGURE_CAPTION.sub(lambda match: f'{match.group(1)}<span class="no">図{next(figures)}</span>', html)
    html = TABLE_CAPTION.sub(lambda match: f'{match.group(1)}<span class="no">表{next(tables)}</span>', html)
    html = EQUATION.sub(
        lambda match: (
            f'<div class="eq" id="{match.group(1)}">{match.group(2)}<span class="no">({next(equations)})</span></div>'
        ),
        html,
    )
    return html, identifiers


def resolve_references(html: str, headings: Mapping[str, tuple[str, str]], floats: Mapping[str, str]) -> str:
    def resolve(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in headings:
            number, tag = headings[key]
            return f"{number}{'章' if tag == 'h2' else '節'}"
        if key in floats:
            return floats[key]
        raise KeyError(f"解決できない参照です。{key}")

    return REFERENCE.sub(resolve, html)


def cited_keys(html: str) -> list[str]:
    order: list[str] = []
    for match in CITATION.finditer(html):
        for key in match.group(1).split(","):
            if key not in order:
                order.append(key)
    return order


def compress(numbers: list[int]) -> str:
    parts = []
    index = 0
    while index < len(numbers):
        end = index
        while end + 1 < len(numbers) and numbers[end + 1] == numbers[end] + 1:
            end += 1
        if end == index:
            parts.append(str(numbers[index]))
        elif end == index + 1:
            parts.append(f"{numbers[index]},{numbers[end]}")
        else:
            parts.append(f"{numbers[index]}–{numbers[end]}")
        index = end + 1
    return ",".join(parts)


def resolve_citations(html: str) -> tuple[str, list[str]]:
    order = cited_keys(html)
    numbers = {key: position + 1 for position, key in enumerate(order)}

    def resolve(match: re.Match[str]) -> str:
        cited = sorted(numbers[key] for key in match.group(1).split(","))
        return f'<span class="cite">[{compress(cited)}]</span>'

    return CITATION.sub(resolve, html), order


def bibliography(order: list[str], references: Mapping[str, str]) -> str:
    items = "".join(
        f'<li><span class="n">[{position}]</span>{references[key]}</li>' for position, key in enumerate(order, 1)
    )
    return f'<h2 class="refh">参考文献</h2><ol class="refs">{items}</ol>'


def tidy(html: str) -> str:
    pieces = PROTECTED.split(html)
    for index in range(0, len(pieces), 2):
        text = pieces[index]
        text = re.sub(rf"([A-Za-z0-9%\]])[ ]+(?=[{CJK}])", r"\1", text)
        text = re.sub(rf"(?<=[{CJK}])[ ]+(?=[A-Za-z0-9])", "", text)
        pieces[index] = re.sub(r"(?<![A-Za-z0-9.\-])-(?=[0-9])", "−", text)
    return "".join(pieces)
