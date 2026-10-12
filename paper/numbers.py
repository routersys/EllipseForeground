import math

from ellipse_foreground.evaluation.statistics import Interval


def f2(value: float) -> str:
    return f"{value:.2f}"


def f3(value: float) -> str:
    return f"{value:.3f}"


def f3_zero(value: float) -> str:
    return "0.000" if abs(value) < 0.0005 else f3(value)


def ceil3(value: float) -> str:
    return f"{math.ceil(value * 1000 - 1e-9) / 1000:.3f}"


def signed(value: float, digits: int = 3) -> str:
    text = f"{abs(value):.{digits}f}"
    if float(text) == 0:
        return text
    return ("+" if value > 0 else "-") + text


def above_zero(interval: Interval, digits: int = 3) -> bool:
    return round(interval.low, digits) > 0


def below_zero(interval: Interval, digits: int = 3) -> bool:
    return round(interval.high, digits) < 0


def separates_from_zero(interval: Interval, digits: int = 3) -> bool:
    return above_zero(interval, digits) or below_zero(interval, digits)


def percent(value: float) -> str:
    return f"{100 * value:.0f}%"


def interval_text(interval: Interval, digits: int = 3) -> str:
    return f"{signed(interval.mean, digits)}［{signed(interval.low, digits)}，{signed(interval.high, digits)}］"


def plain(value: float) -> str:
    return f"{value:g}"
