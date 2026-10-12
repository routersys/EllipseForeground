from paper.mathml import convert, to_mathml


def test_superscript_and_subscript_become_scripts():
    assert "<msubsup>" in to_mathml("x_i^2")
    assert "<msub>" in to_mathml("x_i")
    assert "<msup>" in to_mathml("x^2")


def test_fraction_and_root():
    assert "<mfrac>" in to_mathml(r"\frac{a}{b}")
    assert "<msqrt>" in to_mathml(r"\sqrt{x}")


def test_greek_letters_and_operators_are_converted():
    markup = to_mathml(r"\alpha \le \beta")
    assert "α" in markup
    assert "≤" in markup
    assert "β" in markup


def test_unary_and_binary_minus_differ():
    assert 'form="prefix"' in to_mathml("-1")
    assert 'form="prefix"' not in to_mathml("a-1")


def test_limits_of_large_operators_are_placed_under_and_over():
    assert "<munderover>" in to_mathml(r"\sum_{i}^{n} x")
    assert "<msubsup>" in to_mathml(r"\int_{0}^{1} x")


def test_cases_environment_becomes_a_table():
    markup = to_mathml(r"\begin{cases}1 & x\\0 & y\end{cases}")
    assert "<mtable>" in markup
    assert markup.count("<mtr>") == 2


def test_display_math_is_marked_as_block():
    assert 'display="block"' in to_mathml("x", display=True)
    assert 'display="block"' not in to_mathml("x")


def test_convert_replaces_inline_and_display_delimiters():
    converted = convert(r"a \(x\) b \[y\] c")
    assert converted.count("<math") == 2
    assert r"\(" not in converted
    assert r"\[" not in converted


def test_unknown_commands_are_kept_as_text():
    assert r"<mtext>\unknown</mtext>" in to_mathml(r"\unknown")
