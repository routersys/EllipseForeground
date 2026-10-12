import argparse
import sys
from pathlib import Path

from ellipse_foreground.evaluation.runner import EXPERIMENTS, GROUPS, run_group

REPOSITORY = Path(__file__).resolve().parents[3]
OPEN_END = 10**6


def parse(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ellipse_foreground.evaluation")
    parser.add_argument("group", choices=sorted(GROUPS))
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=OPEN_END)
    parser.add_argument("--data", type=Path, default=REPOSITORY / "data")
    parser.add_argument("--results", type=Path, default=REPOSITORY / "results")
    return parser.parse_args(arguments)


def main(arguments: list[str]) -> int:
    options = parse(arguments)
    experiments = options.experiments.split(",")
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        print(f"不明な実験です。{', '.join(unknown)}", file=sys.stderr)
        return 2
    run_group(GROUPS[options.group], options.data, options.results, experiments, range(options.start, options.end))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
