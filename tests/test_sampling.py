import numpy as np

from ellipse_foreground.cues.sampling import choose_samples, pixel_priorities


def test_priorities_are_reproducible_and_depend_on_the_seed():
    assert np.array_equal(pixel_priorities(1000, 5), pixel_priorities(1000, 5))
    assert not np.array_equal(pixel_priorities(1000, 5), pixel_priorities(1000, 6))


def test_priorities_are_distinct():
    assert len(np.unique(pixel_priorities(100000, 0))) == 100000


def test_priorities_are_spread_over_the_whole_range():
    scaled = pixel_priorities(100000, 3).astype(np.float64) / 2.0**64
    histogram = np.histogram(scaled, bins=10, range=(0, 1))[0]
    assert histogram.min() > 9000


def test_choose_samples_returns_everything_below_the_limit():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:6] = True
    chosen = choose_samples(mask, pixel_priorities(100, 0), 20)
    assert np.array_equal(chosen, np.flatnonzero(mask.ravel()))


def test_choose_samples_respects_limit_mask_and_order():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:30, 5:30] = True
    chosen = choose_samples(mask, pixel_priorities(1600, 1), 100)
    assert len(chosen) == 100
    assert np.all(np.diff(chosen) > 0)
    assert mask.ravel()[chosen].all()


def test_choose_samples_keeps_common_pixels_when_the_mask_grows():
    priorities = pixel_priorities(1600, 2)
    small = np.zeros((40, 40), dtype=bool)
    small[10:20, 10:20] = True
    large = np.zeros((40, 40), dtype=bool)
    large[8:22, 8:22] = True
    from_large = set(choose_samples(large, priorities, 60).tolist())
    from_small = set(choose_samples(small, priorities, 60).tolist())
    assert from_large & set(np.flatnonzero(small.ravel()).tolist()) <= from_small
