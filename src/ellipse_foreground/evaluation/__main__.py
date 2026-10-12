import argparse
import sys
from pathlib import Path

from PIL import Image

from ellipse_foreground.evaluation.listing import listed_paths
from ellipse_foreground.evaluation.runner import (
    COLLECTION_SEED,
    EXPERIMENTS,
    GROUPS,
    GroupSpec,
    collection_group,
    run_group,
)

REPOSITORY = Path(__file__).resolve().parents[3]
OPEN_END = 10**6
COLLECTIONS = {"collection0": 0.0, "collection100": 1.0}


def parse(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ellipse_foreground.evaluation")
    parser.add_argument("group", choices=sorted([*GROUPS, *COLLECTIONS]))
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=OPEN_END)
    parser.add_argument("--data", type=Path, default=REPOSITORY / "data")
    parser.add_argument("--results", type=Path, default=REPOSITORY / "results")
    parser.add_argument("--sprites", type=Path)
    parser.add_argument("--backgrounds", type=Path)
    parser.add_argument("--seed", type=int, default=COLLECTION_SEED)
    parser.add_argument("--grounded", action="store_true")
    return parser.parse_args(arguments)


def select_group(options: argparse.Namespace) -> GroupSpec | None:
    if options.group in GROUPS:
        return GROUPS[options.group]
    if options.sprites is None or options.backgrounds is None:
        return None
    Image.MAX_IMAGE_PIXELS = None
    return collection_group(
        COLLECTIONS[options.group],
        listed_paths(options.sprites),
        listed_paths(options.backgrounds),
        options.seed,
        options.grounded,
    )


def main(arguments: list[str]) -> int:
    options = parse(arguments)
    experiments = options.experiments.split(",")
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        print(f"不明な実験です。{', '.join(unknown)}", file=sys.stderr)
        return 2
    spec = select_group(options)
    if spec is None:
        print("手元の素材の集合には --sprites と --backgrounds が必要です。", file=sys.stderr)
        return 2
    run_group(spec, options.data, options.results, experiments, range(options.start, options.end))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
