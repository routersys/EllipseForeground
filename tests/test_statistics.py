import numpy as np
import pytest

from ellipse_foreground.evaluation.statistics import Interval, bootstrap_interval, paired_interval, scene_means


def rows():
    return [
        {"scene": 0, "a": 1.0, "b": 0.5},
        {"scene": 0, "a": 3.0, "b": 1.5},
        {"scene": 1, "a": 2.0, "b": 1.0},
        {"scene": 2, "a": 4.0, "b": 3.0},
    ]


def test_scene_means_average_trials_within_each_scene():
    assert scene_means(rows(), lambda row: row["a"]).tolist() == [2.0, 2.0, 4.0]


def test_scene_means_are_ordered_by_scene_index():
    shuffled = [rows()[3], rows()[0], rows()[2], rows()[1]]
    assert scene_means(shuffled, lambda row: row["a"]).tolist() == [2.0, 2.0, 4.0]


def test_interval_is_reproducible_and_contains_the_mean_for_a_symmetric_sample():
    values = np.random.default_rng(0).normal(0.5, 1.0, 200)
    first, second = bootstrap_interval(values), bootstrap_interval(values)
    assert first == second
    assert first.low < first.mean < first.high


def test_interval_matches_the_manual_percentile_bootstrap():
    values = np.random.default_rng(1).normal(0.0, 1.0, 30)
    rng = np.random.default_rng(0)
    means = [rng.choice(values, len(values)).mean() for _ in range(2000)]
    low, high = np.percentile(means, [2.5, 97.5])
    assert bootstrap_interval(values) == Interval(float(values.mean()), float(low), float(high))


def test_interval_width_shrinks_with_more_scenes():
    rng = np.random.default_rng(2)
    small = bootstrap_interval(rng.normal(0, 1, 20))
    large = bootstrap_interval(rng.normal(0, 1, 400))
    assert (large.high - large.low) < (small.high - small.low)


def test_excludes_zero_reports_significance():
    assert Interval(0.1, 0.02, 0.2).excludes_zero
    assert Interval(-0.1, -0.2, -0.02).excludes_zero
    assert not Interval(0.1, -0.02, 0.2).excludes_zero


def test_paired_interval_uses_the_per_scene_difference():
    interval = paired_interval(rows(), lambda row: row["a"], lambda row: row["b"])
    differences = scene_means(rows(), lambda row: row["a"] - row["b"])
    assert interval == bootstrap_interval(differences)
    assert interval.mean == pytest.approx(differences.mean())
