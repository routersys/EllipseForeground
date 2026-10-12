import sys
from pathlib import Path

from harness.arguments import USAGE, HarnessArguments, HarnessMode
from harness.benchmark import benchmark
from harness.cases import CaseContext
from harness.golden import HarnessError
from harness.runner import DEFAULT_OUTPUT, compare, render, transition, verify, write_golden


def run(arguments: HarnessArguments) -> int:
    if arguments.mode is HarnessMode.COMPARE:
        return compare(Path(arguments.before), Path(arguments.after))
    context = CaseContext()
    output = Path(arguments.output_directory) if arguments.output_directory else DEFAULT_OUTPUT
    if arguments.mode is HarnessMode.GOLDEN:
        return write_golden(context)
    if arguments.mode is HarnessMode.VERIFY:
        return verify(context)
    if arguments.mode is HarnessMode.TRANSITION:
        return transition(context, output)
    if arguments.mode is HarnessMode.BENCHMARK:
        return benchmark(context)
    return render(context, output)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        return run(HarnessArguments.parse(argv))
    except HarnessError as error:
        print(error, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
