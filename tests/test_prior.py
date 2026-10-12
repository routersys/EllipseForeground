import numpy as np

from ellipse_foreground.geometry import Ellipse
from ellipse_foreground.prior import ellipse_prior, elliptical_prior
from ellipse_foreground.settings import PriorSettings


def test_elliptical_prior_is_clipped_to_unit_range():
    prior = elliptical_prior((40, 60), Ellipse(30.0, 20.0, 12.0, 8.0), 0.35)
    assert prior.min() == -1.0
    assert prior.max() == 1.0


def test_elliptical_prior_is_zero_on_the_boundary():
    prior = elliptical_prior((41, 61), Ellipse(30.0, 20.0, 12.0, 8.0), 0.35)
    assert prior[20, 42] == 0.0


def test_inside_is_shrunk_and_outside_is_not():
    settings = PriorSettings(scales=(1.0,))
    ellipse = Ellipse(30.0, 20.0, 12.0, 8.0)
    plain = elliptical_prior((41, 61), ellipse, settings.softness)
    shrunk = ellipse_prior((41, 61), ellipse, settings)
    assert np.allclose(shrunk[plain > 0], settings.inside_shrink * plain[plain > 0])
    assert np.allclose(shrunk[plain <= 0], plain[plain <= 0])


def test_prior_decreases_away_from_the_center():
    prior = ellipse_prior((41, 61), Ellipse(30.0, 20.0, 12.0, 8.0), PriorSettings())
    along_axis = prior[20, 30:61]
    assert np.all(np.diff(along_axis) <= 1e-12)
