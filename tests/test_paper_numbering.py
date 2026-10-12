import pytest

from paper.numbering import (
    bibliography,
    compress,
    number_floats,
    number_headings,
    resolve_citations,
    resolve_references,
    tidy,
)


def test_headings_are_numbered_by_level_and_appendix_is_exempt():
    html = '<h2 id="a">一</h2><h3 id="b">二</h3><h3 id="c">三</h3><h2 id="d">四</h2><h2 id="e" class="appx">付録</h2>'
    numbered, identifiers = number_headings(html)
    assert identifiers == {
        "a": ("1", "h2"),
        "b": ("1.1", "h3"),
        "c": ("1.2", "h3"),
        "d": ("2", "h2"),
        "e": ("付録A", "h2"),
    }
    assert '<span class="sec-num">1.2</span>三' in numbered
    assert 'class="appx">付録' in numbered


def test_floats_are_numbered_in_order_of_appearance():
    html = (
        '<figure id="f:x"><figcaption><span class="no">図</span>説明</figcaption></figure>'
        '<div class="tcap" id="t:y"><span class="no">表</span>見出し</div>'
        '<div class="eq" id="e:z">式</div>'
    )
    numbered, identifiers = number_floats(html)
    assert identifiers == {"f:x": "図1", "t:y": "表1", "e:z": "1"}
    assert '<span class="no">図1</span>' in numbered
    assert '<span class="no">表1</span>' in numbered
    assert '<span class="no">(1)</span>' in numbered


def test_references_resolve_to_numbers_and_unknown_keys_fail():
    headings = {"s:a": ("2.1", "h3"), "s:b": ("3", "h2")}
    floats = {"f:x": "図2"}
    resolved = resolve_references('<ref k="s:a"></ref>と<ref k="s:b"></ref>と<ref k="f:x"></ref>', headings, floats)
    assert resolved == "2.1節と3章と図2"
    with pytest.raises(KeyError):
        resolve_references('<ref k="missing"></ref>', headings, floats)


def test_compress_joins_runs_of_citation_numbers():
    assert compress([1]) == "1"
    assert compress([1, 2]) == "1,2"
    assert compress([1, 2, 3, 5]) == "1–3,5"
    assert compress([2, 4, 6]) == "2,4,6"


def test_citations_are_numbered_by_first_appearance():
    html = '<cite k="b"></cite><cite k="a,b"></cite>'
    resolved, order = resolve_citations(html)
    assert order == ["b", "a"]
    assert resolved == '<span class="cite">[1]</span><span class="cite">[1,2]</span>'


def test_bibliography_lists_cited_references_in_order():
    html = bibliography(["b", "a"], {"a": "A文献", "b": "B文献"})
    assert html.index("B文献") < html.index("A文献")
    assert "[1]" in html


def test_tidy_removes_spaces_around_japanese_but_keeps_math_and_tags():
    assert tidy("値は 3 です") == "値は3です"
    assert tidy("a and b") == "a and b"
    assert tidy(r"\(a b\) と c d") == r"\(a b\) とc d"
    assert tidy('<span class="x y">本</span>') == '<span class="x y">本</span>'


def test_tidy_turns_hyphen_before_a_digit_into_a_minus_sign():
    assert tidy("差は-3") == "差は−3"
    assert tidy("a-3") == "a-3"
