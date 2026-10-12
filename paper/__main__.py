import argparse
import sys
from pathlib import Path

from paper.build import CHROME, build
from paper.timing import measure

REPOSITORY = Path(__file__).resolve().parent.parent


def parse(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m paper")
    commands = parser.add_subparsers(dest="command", required=True)
    page = commands.add_parser("build")
    page.add_argument("--results", type=Path, default=REPOSITORY / "results")
    page.add_argument("--data", type=Path, default=REPOSITORY / "data")
    page.add_argument("--timing", type=Path, default=REPOSITORY / "results" / "timing.json")
    page.add_argument("--output", type=Path, default=REPOSITORY / "paper" / "build")
    page.add_argument("--chrome", type=Path, default=CHROME)
    page.add_argument("--no-figures", action="store_true")
    page.add_argument("--no-pdf", action="store_true")
    timing = commands.add_parser("timing")
    timing.add_argument("--data", type=Path, default=REPOSITORY / "data")
    timing.add_argument("--output", type=Path, default=REPOSITORY / "results" / "timing.json")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    options = parse(arguments)
    if options.command == "timing":
        summary = measure(options.data, options.output)
        print(f"木 {summary['tree_ms']:.3f} ms、グラフカット {summary['graph_cut_ms']:.1f} ms")
        return 0
    page = build(
        options.results,
        options.data,
        options.timing,
        options.output,
        options.chrome,
        figures=not options.no_figures,
        pdf=not options.no_pdf,
    )
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
