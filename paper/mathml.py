import re

GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "varphi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "ell": "ℓ",
}
OPERATORS = {
    "cdot": "⋅",
    "times": "×",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ne": "≠",
    "neq": "≠",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "cup": "∪",
    "cap": "∩",
    "to": "→",
    "rightarrow": "→",
    "infty": "∞",
    "partial": "∂",
    "nabla": "∇",
    "approx": "≈",
    "propto": "∝",
    "pm": "±",
    "mid": "∣",
    "ll": "≪",
    "gg": "≫",
    "equiv": "≡",
    "sim": "∼",
    "land": "∧",
    "lor": "∨",
    "forall": "∀",
    "exists": "∃",
    "emptyset": "∅",
    "setminus": "∖",
    "circ": "∘",
    "ast": "∗",
    "oplus": "⊕",
    "leftarrow": "←",
    "ni": "∋",
    "top": "⊤",
    "Rightarrow": "⇒",
    "Leftrightarrow": "⇔",
    "ldots": "…",
    "cdots": "⋯",
    "dots": "…",
    "triangleq": "≜",
    "neg": "¬",
}
FUNCTIONS = {
    "ln",
    "log",
    "exp",
    "max",
    "min",
    "arg",
    "sup",
    "inf",
    "sin",
    "cos",
    "det",
    "tr",
    "lim",
    "var",
    "cov",
    "clip",
    "sign",
    "diag",
    "rank",
}
LIMIT_FUNCTIONS = {"max", "min", "arg", "sup", "inf", "lim"}
LARGE_OPERATORS = {"sum": "∑", "prod": "∏", "bigcup": "⋃", "bigcap": "⋂"}
SPACES = {",": "0.17em", ";": "0.28em", ":": "0.22em", "quad": "1em", "qquad": "2em", "!": "-0.17em", " ": "0.25em"}
ACCENTS = {"bar": "¯", "hat": "^", "tilde": "~", "vec": "→", "dot": "˙", "overline": "¯"}
FONTS = {
    "mathbf": "bold",
    "mathrm": "normal",
    "mathcal": "script",
    "mathbb": "double-struck",
    "mathit": "italic",
    "boldsymbol": "bold-italic",
    "mathsf": "sans-serif",
    "mathfrak": "fraktur",
}
DOUBLE_STRUCK = {"1": "𝟙", "R": "ℝ", "N": "ℕ", "E": "𝔼", "P": "ℙ"}
SIZED_DELIMITERS = {
    "big",
    "Big",
    "bigg",
    "Bigg",
    "bigl",
    "Bigl",
    "biggl",
    "Biggl",
    "bigr",
    "Bigr",
    "biggr",
    "Biggr",
}
DELIMITERS = {
    "\\{": "{",
    "\\}": "}",
    "\\|": "‖",
    "\\lvert": "|",
    "\\rvert": "|",
    "\\langle": "⟨",
    "\\rangle": "⟩",
    "\\lVert": "‖",
    "\\rVert": "‖",
}
BRACKETS = {"lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉"}
FUNCTION_ATOM = re.compile(
    r"<mi>(ln|log|exp|max|min|arg|sup|inf|sin|cos|det|tr|lim|var|cov|clip|sign|diag|rank|sigmoid)</mi>$"
)
OPERAND_END = re.compile(r"</(mi|mn|msub|msup|msubsup|mrow|mfrac|msqrt|mover|munder|munderover)>$")
COMMAND = re.compile(r"\\([A-Za-z]+)|\\(.)", re.S)
CONTROL_WORD = re.compile(r"\\[A-Za-z]+|\\.")


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Cursor:
    def __init__(self, source: str) -> None:
        self.source = source
        self.position = 0
        self.after_operand = False

    def peek(self) -> str:
        return self.source[self.position] if self.position < len(self.source) else ""

    def at_end(self) -> bool:
        return self.position >= len(self.source)

    def skip_whitespace(self) -> None:
        while not self.at_end() and self.source[self.position] in " \n\t":
            self.position += 1

    def argument(self) -> str:
        self.skip_whitespace()
        if self.peek() == "{":
            self.position += 1
            depth, start = 1, self.position
            while depth:
                character = self.source[self.position]
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                self.position += 1
            return self.source[start : self.position - 1]
        if self.peek() == "\\":
            match = CONTROL_WORD.match(self.source, self.position)
            self.position = match.end()
            return match.group(0)
        character = self.peek()
        self.position += 1
        return character

    def command(self) -> str:
        match = COMMAND.match(self.source, self.position)
        self.position = match.end()
        return match.group(1) or match.group(2)


def identifier(text: str, variant: str | None = None) -> str:
    attribute = f' mathvariant="{variant}"' if variant else ""
    return f"<mi{attribute}>{escape(text)}</mi>"


def parse_sequence(cursor: Cursor) -> str:
    parts = []
    while not cursor.at_end():
        atom = parse_atom(cursor)
        if atom is None:
            continue
        is_function = bool(FUNCTION_ATOM.match(atom)) or (
            atom.startswith("<mi>")
            and atom.count("<mi>") == 1
            and len(re.sub(r"<[^>]+>", "", atom)) > 1
            and atom.endswith("</mi>")
        )
        parts.append(parse_scripts(cursor, atom))
        cursor.after_operand = bool(OPERAND_END.search(parts[-1])) or 'form="postfix"' in parts[-1]
        if is_function:
            parts.append('<mspace width="0.17em"></mspace>')
    return "".join(parts)


def sequence_of(text: str) -> str:
    return parse_sequence(Cursor(text))


def has_movable_limits(atom: str) -> bool:
    if atom.startswith('<mo largeop="true" movablelimits'):
        return True
    match = re.match(r"<mi>([a-z]+)</mi>$", atom)
    return bool(match and match.group(1) in LIMIT_FUNCTIONS)


def parse_scripts(cursor: Cursor, atom: str) -> str:
    subscript = superscript = None
    while True:
        cursor.skip_whitespace()
        character = cursor.peek()
        if character == "_" and subscript is None:
            cursor.position += 1
            subscript = f"<mrow>{sequence_of(cursor.argument())}</mrow>"
        elif character == "^" and superscript is None:
            cursor.position += 1
            superscript = f"<mrow>{sequence_of(cursor.argument())}</mrow>"
        elif character == "'":
            cursor.position += 1
            superscript = "<mo>′</mo>" if superscript is None else superscript + "<mo>′</mo>"
        else:
            break
    if subscript is None and superscript is None:
        return atom
    limits = has_movable_limits(atom)
    under, over, both = ("munder", "mover", "munderover") if limits else ("msub", "msup", "msubsup")
    if subscript and superscript:
        return f"<{both}>{atom}{subscript}{superscript}</{both}>"
    if subscript:
        return f"<{under}>{atom}{subscript}</{under}>"
    return f"<{over}>{atom}{superscript}</{over}>"


def parse_atom(cursor: Cursor) -> str | None:
    cursor.skip_whitespace()
    if cursor.at_end():
        return None
    character = cursor.peek()
    if character == "{":
        return f"<mrow>{sequence_of(cursor.argument())}</mrow>"
    if character == "\\":
        return parse_command(cursor, cursor.command())
    cursor.position += 1
    if character.isdigit():
        start = cursor.position
        while not cursor.at_end() and (
            cursor.peek().isdigit()
            or (cursor.peek() == "." and cursor.source[cursor.position + 1 : cursor.position + 2].isdigit())
        ):
            cursor.position += 1
        return f"<mn>{escape(character + cursor.source[start : cursor.position])}</mn>"
    if character.isalpha():
        return identifier(character)
    if character == "|":
        return '<mo stretchy="false" lspace="0.08em" rspace="0.08em">|</mo>'
    if character in "+-=<>/,;:!()[].":
        unary = False
        if character == "-":
            character = "−"
            unary = not cursor.after_operand
        form = ' form="prefix" lspace="0" rspace="0"' if unary else ""
        if character in "([":
            form = ' form="prefix" stretchy="false"'
        elif character in ")]":
            form = ' form="postfix" stretchy="false"'
        return f"<mo{form}>{escape(character)}</mo>"
    if character == "*":
        return "<mo>∗</mo>"
    return f"<mtext>{escape(character)}</mtext>"


def fraction(cursor: Cursor) -> str:
    numerator, denominator = cursor.argument(), cursor.argument()
    return f"<mfrac><mrow>{sequence_of(numerator)}</mrow><mrow>{sequence_of(denominator)}</mrow></mfrac>"


def environment(cursor: Cursor) -> str:
    name = cursor.argument()
    end = f"\\end{{{name}}}"
    stop = cursor.source.index(end, cursor.position)
    body = cursor.source[cursor.position : stop]
    cursor.position = stop + len(end)
    rows = [row for row in body.split("\\\\") if row.strip()]
    table_rows = []
    for row in rows:
        cells = "".join(f"<mtd columnalign='left'>{sequence_of(cell)}</mtd>" for cell in row.split("&"))
        table_rows.append(f"<mtr>{cells}</mtr>")
    table = "".join(table_rows)
    if name == "cases":
        return f'<mrow><mo fence="true" stretchy="true" form="prefix">{{</mo><mtable>{table}</mtable></mrow>'
    return f"<mtable>{table}</mtable>"


def font(cursor: Cursor, name: str) -> str:
    argument = cursor.argument()
    if name == "mathbb" and argument in DOUBLE_STRUCK:
        return f"<mi>{DOUBLE_STRUCK[argument]}</mi>"
    inner = sequence_of(argument)
    variant = FONTS[name]
    inner = re.sub(r"<mi>([A-Za-z])</mi>", lambda match: f'<mi mathvariant="{variant}">{match.group(1)}</mi>', inner)
    if name == "mathrm":
        inner = re.sub(r"<mi>([^<]+)</mi>", r'<mi mathvariant="normal">\1</mi>', inner)
    return f"<mrow>{inner}</mrow>"


def delimiter_pair(cursor: Cursor, name: str) -> str:
    delimiter = cursor.argument()
    if delimiter == ".":
        return "<mrow></mrow>"
    delimiter = DELIMITERS.get(delimiter, delimiter)
    form = "prefix" if name == "left" else "postfix"
    return f'<mo fence="true" stretchy="true" form="{form}">{escape(delimiter)}</mo>'


def parse_command(cursor: Cursor, name: str) -> str:
    if name in GREEK:
        return identifier(GREEK[name])
    if name in OPERATORS:
        return f"<mo>{OPERATORS[name]}</mo>"
    if name in LARGE_OPERATORS:
        return f'<mo largeop="true" movablelimits="true">{LARGE_OPERATORS[name]}</mo>'
    if name == "int":
        return '<mo largeop="true">∫</mo>'
    if name in FUNCTIONS:
        return f"<mi>{name}</mi>"
    if name == "operatorname":
        return f"<mi>{escape(cursor.argument())}</mi>"
    if name in SIZED_DELIMITERS:
        argument = cursor.argument()
        delimiter = DELIMITERS.get(argument, argument)
        return f'<mo fence="true" stretchy="true" lspace="0" rspace="0">{escape(delimiter)}</mo>'
    if name in ("lVert", "rVert", "Vert"):
        return '<mo stretchy="false" lspace="0.05em" rspace="0.05em">‖</mo>'
    if name in ("tfrac", "frac"):
        return fraction(cursor)
    if name == "sqrt":
        return f"<msqrt>{sequence_of(cursor.argument())}</msqrt>"
    if name == "text":
        return f"<mtext>{escape(cursor.argument())}</mtext>"
    if name in FONTS:
        return font(cursor, name)
    if name in ACCENTS:
        return f'<mover accent="true"><mrow>{sequence_of(cursor.argument())}</mrow><mo>{ACCENTS[name]}</mo></mover>'
    if name in ("left", "right"):
        return delimiter_pair(cursor, name)
    if name in ("lvert", "rvert"):
        return "<mo>|</mo>"
    if name in ("{", "}"):
        return f"<mo>{name}</mo>"
    if name == "|":
        return "<mo>‖</mo>"
    if name in BRACKETS:
        return f"<mo>{BRACKETS[name]}</mo>"
    if name in ("langle", "rangle"):
        return "<mo>⟨</mo>" if name == "langle" else "<mo>⟩</mo>"
    if name in SPACES:
        return f'<mspace width="{SPACES[name]}"></mspace>'
    if name == "begin":
        return environment(cursor)
    return f"<mtext>\\{name}</mtext>"


def to_mathml(source: str, display: bool = False) -> str:
    attribute = ' display="block"' if display else ""
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML"{attribute}><mrow>{sequence_of(source)}</mrow></math>'


def convert(html: str) -> str:
    html = re.sub(r"\\\[(.+?)\\\]", lambda match: to_mathml(match.group(1), True), html, flags=re.S)
    return re.sub(r"\\\((.+?)\\\)", lambda match: to_mathml(match.group(1), False), html, flags=re.S)
