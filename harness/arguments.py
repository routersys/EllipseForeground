from dataclasses import dataclass
from enum import Enum

from harness.golden import HarnessError


class HarnessMode(Enum):
    RENDER = "render"
    GOLDEN = "golden"
    VERIFY = "verify"
    TRANSITION = "transition"
    COMPARE = "compare"
    BENCHMARK = "benchmark"


USAGE = """使い方:
  [<出力先>]                      ケースを描き、PNG とハッシュを出します
  --golden                        基準値を golden.json に書きます
  --verify                        基準値と照合します
  --transition                    同じ解析を使い回した結果を、作り直した解析の結果と照合します
  --compare <前> <後>             2 つの出力先の PNG を画素ごとに比べます
  --benchmark                     段ごとの処理時間を計測します"""

_FLAGS = {
    "--golden": HarnessMode.GOLDEN,
    "--verify": HarnessMode.VERIFY,
    "--transition": HarnessMode.TRANSITION,
    "--compare": HarnessMode.COMPARE,
    "--benchmark": HarnessMode.BENCHMARK,
}


@dataclass(frozen=True)
class HarnessArguments:
    mode: HarnessMode
    output_directory: str | None = None
    before: str | None = None
    after: str | None = None

    @classmethod
    def parse(cls, arguments: list[str]) -> "HarnessArguments":
        mode: HarnessMode | None = None
        positionals: list[str] = []
        for argument in arguments:
            if argument in _FLAGS:
                if mode is not None:
                    raise HarnessError("モードは 1 つだけ指定してください。")
                mode = _FLAGS[argument]
            elif argument.startswith("-"):
                raise HarnessError(f"不明なオプションです。{argument}")
            else:
                positionals.append(argument)
        selected = mode or HarnessMode.RENDER
        if selected is HarnessMode.COMPARE:
            if len(positionals) != 2:
                raise HarnessError("--compare には前と後の出力先を続けてください。")
            return cls(selected, None, positionals[0], positionals[1])
        if selected in (HarnessMode.RENDER, HarnessMode.TRANSITION):
            if len(positionals) > 1:
                raise HarnessError(f"余分な引数です。{positionals[1]}")
            return cls(selected, positionals[0] if positionals else None)
        if positionals:
            raise HarnessError(f"余分な引数です。{positionals[0]}")
        return cls(selected)
