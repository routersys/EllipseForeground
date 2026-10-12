import math

from paper.numbers import ceil3, f3_zero, signed
from paper.table import table_html


def test_number_formats():
    assert signed(0.0004) == "0.000"
    assert signed(0.1234) == "+0.123"
    assert signed(-0.1234) == "-0.123"
    assert signed(0.126, 2) == "+0.13"
    assert f3_zero(0.0003) == "0.000"
    assert ceil3(0.0211) == "0.022"
    assert ceil3(0.021) == "0.021"
    assert math.isclose(float(ceil3(0.3)), 0.3)


def test_table_marks_best_cells_and_left_aligned_columns():
    html = table_html(
        "t:x",
        "見出し",
        ["名前", "値"],
        [["a", "1"], ["b", "2"]],
        align=["l", ""],
        best={(1, 1)},
        sub_rows=(0,),
        widths=["60%", "40%"],
        group=[("", 1), ("群", 1)],
    )
    assert '<div class="tcap" id="t:x">' in html
    assert '<th class="l">名前</th>' in html
    assert '<td class="best">2</td>' in html
    assert '<tr class="sub">' in html
    assert '<col style="width:60%">' in html
    assert '<th class="grp">群</th>' in html
