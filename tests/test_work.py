import numpy as np
import pytest

from ellipse_foreground.settings import WorkSettings
from ellipse_foreground.work import normalize_range, reduce_image, work_shape


def test_work_shape_keeps_the_aspect_ratio():
    assert work_shape((360, 640, 3), WorkSettings()) == (144, 256)
    assert work_shape((640, 640, 3), WorkSettings(width=100)) == (100, 100)


def test_normalize_range_maps_the_percentiles_to_zero_and_one():
    ramp = np.linspace(0.2, 0.6, 1001, dtype=np.float32)
    small = np.stack([np.tile(ramp, (4, 1))] * 3, axis=2)
    work = normalize_range(small, WorkSettings())
    low, high = np.percentile(ramp, [2, 98])
    assert work[0, np.searchsorted(ramp, low), 0] == pytest.approx(0.0, abs=2e-3)
    assert work[0, np.searchsorted(ramp, high), 0] == pytest.approx(1.0, abs=2e-3)


def test_normalize_range_clips_to_the_ceiling_and_floor():
    small = np.zeros((10, 100, 3), dtype=np.float32)
    small[:, 99:] = 5.0
    work = normalize_range(small, WorkSettings())
    assert work.min() == 0.0
    assert work.max() == pytest.approx(1.5)


def test_normalize_range_keeps_flat_channels_flat():
    work = normalize_range(np.full((8, 8, 3), 0.4, dtype=np.float32), WorkSettings())
    assert np.all(work == 0.0)


def test_reduce_image_returns_float32_arrays_of_the_work_shape():
    image = np.random.default_rng(0).random((90, 160, 3)).astype(np.float32)
    small, work = reduce_image(image, WorkSettings(width=64))
    assert small.shape == work.shape == (36, 64, 3)
    assert small.dtype == work.dtype == np.float32
