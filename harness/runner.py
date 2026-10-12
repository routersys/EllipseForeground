import sys
from collections import Counter
from pathlib import Path

from ellipse_foreground import Mix
from ellipse_foreground.evaluation.trials import draw_trials
from harness.cases import FAST, MAIN_SCENE, Case, CaseContext, CaseResult, all_cases, evaluate
from harness.comparison import compare_arrays
from harness.golden import Golden, GoldenCase, current_environment
from harness.output import load_alpha, save_alpha

DEFAULT_OUTPUT = Path("harness-output")
TRANSITION_TRIALS = 4
TRANSITION_MIXES = (Mix.FIXED_MIX, Mix.ADAPTIVE_COLOR_PATCH)


def report(message: str) -> None:
    print(message, file=sys.stderr)


def evaluate_all(context: CaseContext, cases: tuple[Case, ...] | None = None) -> list[CaseResult]:
    return [evaluate(case, context) for case in (cases or all_cases())]


def count_duplicates(results: list[CaseResult]) -> int:
    names = [name for name, count in Counter(result.name for result in results).items() if count > 1]
    for name in names:
        report(f"{name}: ケース名が重複しています。")
    return len(names)


def count_identical_outputs(results: list[CaseResult]) -> int:
    by_hash: dict[str, list[str]] = {}
    for result in results:
        by_hash.setdefault(result.hash, []).append(result.name)
    identical = [names for names in by_hash.values() if len(names) > 1]
    for names in identical:
        report(f"{', '.join(names)}: 出力が同じです。違いを検出できるケースにしてください。")
    return len(identical)


def render(context: CaseContext, output_directory: Path) -> int:
    output_directory.mkdir(parents=True, exist_ok=True)
    failures = 0
    results = []
    for case in all_cases():
        result = evaluate(case, context)
        results.append(result)
        print(f"{result.name}: {result.hash}")
        if result.alpha is None:
            continue
        file_name = f"{result.name}.png"
        save_alpha(output_directory / file_name, result.alpha)
        opaque = int((result.alpha > 0.5).sum())
        print(f"  {file_name} opaque={opaque}")
        if opaque == 0:
            report(f"{file_name}: 出力が完全に透明です。")
            failures += 1
    print(f"-> {output_directory}")
    return 1 if failures + count_duplicates(results) else 0


def write_golden(context: CaseContext) -> int:
    results = evaluate_all(context)
    if count_duplicates(results) + count_identical_outputs(results):
        return 1
    golden = Golden(
        current_environment(), context.identity, tuple(GoldenCase(item.name, item.hash) for item in results)
    )
    golden.save()
    print(f"environment: {golden.environment}")
    print(f"input: {golden.input}")
    for item in results:
        print(f"{item.name}: {item.hash}")
    return 0


def verify(context: CaseContext) -> int:
    golden = Golden.load()
    if golden.input != context.identity:
        report(f"基準値は入力 {golden.input} で書かれています。今回は入力 {context.identity} です。")
        return 1
    if golden.environment != current_environment():
        report(
            f"基準値は {golden.environment} で書かれています。今回は {current_environment()} です。"
            "出力の差はこの違いから来ることがあります。"
        )
    current = {item.name: item for item in evaluate_all(context)}
    failures = 0
    for case in golden.cases:
        actual = current.pop(case.name, None)
        if actual is None:
            report(f"{case.name}: ケースがありません。基準値を書き直してください。")
            failures += 1
        elif actual.hash != case.hash:
            report(f"{case.name}: 一致しません。基準値 {case.hash}、今回 {actual.hash}")
            failures += 1
        else:
            print(f"{case.name}: 一致")
    for name in current:
        report(f"{name}: 基準値にありません。基準値を書き直してください。")
        failures += 1
    return 1 if failures else 0


def transition(context: CaseContext, output_directory: Path) -> int:
    failures = 0
    scene = context.scenes[MAIN_SCENE]
    ellipses = draw_trials(scene, TRANSITION_TRIALS)
    for mix in TRANSITION_MIXES:
        shared = context.fresh_context(MAIN_SCENE, FAST).analysis
        for index, ellipse in enumerate(ellipses + ellipses[:1]):
            reused = shared.extract(ellipse, mix)
            fresh = context.fresh_context(MAIN_SCENE, FAST).analysis.extract(ellipse, mix)
            name = f"transition-{mix.value}-{index}"
            difference = compare_arrays(fresh, reused)
            if difference.is_empty:
                print(f"{name}: 一致")
                continue
            failures += 1
            output_directory.mkdir(parents=True, exist_ok=True)
            save_alpha(output_directory / f"{name}-fresh.png", fresh)
            save_alpha(output_directory / f"{name}-reused.png", reused)
            report(f"{name}: {difference}。使い回した解析の結果が作り直した場合と違います。-> {output_directory}")
    return 1 if failures else 0


def compare(before: Path, after: Path) -> int:
    failures = 0
    names = sorted(path.name for path in before.glob("*.png"))
    if not names:
        report(f"比べる PNG がありません。{before}")
        return 1
    for name in names:
        counterpart = after / name
        if not counterpart.exists():
            report(f"{name}: 後の出力先にありません。")
            failures += 1
            continue
        difference = compare_arrays(load_alpha(before / name), load_alpha(counterpart))
        if difference.is_empty:
            print(f"{name}: 一致")
        else:
            report(f"{name}: {difference}")
            failures += 1
    for extra in sorted({path.name for path in after.glob("*.png")} - set(names)):
        report(f"{extra}: 前の出力先にありません。")
        failures += 1
    return 1 if failures else 0
