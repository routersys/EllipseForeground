import numpy as np

_GOLDEN = 0x9E3779B97F4A7C15
_FIRST_MIX = np.uint64(0xBF58476D1CE4E5B9)
_SECOND_MIX = np.uint64(0x94D049BB133111EB)
_MODULUS = 1 << 64


def pixel_priorities(count: int, seed: int) -> np.ndarray:
    start = np.uint64((seed * _GOLDEN) % _MODULUS)
    with np.errstate(over="ignore"):
        state = np.arange(count, dtype=np.uint64) * np.uint64(_GOLDEN) + start
        state = (state ^ (state >> np.uint64(30))) * _FIRST_MIX
        state = (state ^ (state >> np.uint64(27))) * _SECOND_MIX
        return state ^ (state >> np.uint64(31))


def choose_samples(mask: np.ndarray, priorities: np.ndarray, limit: int) -> np.ndarray:
    candidates = np.flatnonzero(mask.ravel())
    if len(candidates) <= limit:
        return candidates
    order = np.argsort(priorities[candidates], kind="stable")[:limit]
    return np.sort(candidates[order])
