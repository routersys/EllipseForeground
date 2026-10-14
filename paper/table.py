from collections.abc import Sequence


def tag(name: str, content: str, classes: Sequence[str] = (), extra: str = "") -> str:
    attribute = f' class="{" ".join(classes)}"' if classes else ""
    return f"<{name}{attribute}{extra}>{content}</{name}>"


def table_html(
    identifier: str,
    caption: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: Sequence[str] | None = None,
    sub_rows: Sequence[int] = (),
    best: set[tuple[int, int]] | None = None,
    widths: Sequence[str] | None = None,
    group: Sequence[tuple[str, int]] | None = None,
    dense: bool = False,
) -> str:
    left = [bool(align) and align[column] == "l" for column in range(len(header))]
    heads = "".join(tag("th", title, ["l"] if left[column] else []) for column, title in enumerate(header))
    body = []
    for row_index, row in enumerate(rows):
        cells = []
        for column, content in enumerate(row):
            classes = []
            if best and (row_index, column) in best:
                classes.append("best")
            if left[column]:
                classes.append("l")
            cells.append(tag("td", content, classes))
        body.append(tag("tr", "".join(cells), ["sub"] if row_index in sub_rows else []))
    columns = ""
    if widths:
        columns = "<colgroup>" + "".join(f'<col style="width:{width}">' for width in widths) + "</colgroup>"
    groups = ""
    if group:
        spans = "".join(tag("th", title, ["grp"], f' colspan="{span}"' if span > 1 else "") for title, span in group)
        groups = f"<tr>{spans}</tr>"
    caption_block = f'<div class="tcap" id="{identifier}"><span class="no">表</span>{caption}</div>'
    table_class = "t dense" if dense else "t"
    table = f'<table class="{table_class}">{columns}<thead>{groups}<tr>{heads}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    return f'<div class="table-block">{caption_block}{table}</div>'
