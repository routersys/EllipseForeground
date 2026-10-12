import numpy as np

from ellipse_foreground.geometry import Ellipse


def test_inscribed_uses_bounding_box():
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:12, 6:20] = True
    ellipse = Ellipse.inscribed(mask)
    assert (ellipse.center_x, ellipse.center_y) == (13.0, 8.0)
    assert (ellipse.radius_x, ellipse.radius_y) == (7.0, 4.0)


def test_scaled_changes_only_radii():
    ellipse = Ellipse(10.0, 5.0, 4.0, 2.0).scaled(1.5)
    assert ellipse == Ellipse(10.0, 5.0, 6.0, 3.0)


def test_resized_scales_axes_independently():
    assert Ellipse(10.0, 6.0, 4.0, 2.0).resized(0.5, 2.0) == Ellipse(5.0, 12.0, 2.0, 4.0)


def test_normalized_radius_is_zero_at_center_and_one_on_axis():
    ellipse = Ellipse(10.0, 6.0, 4.0, 2.0)
    radius = ellipse.normalized_radius((13, 21))
    assert radius[6, 10] == 0.0
    assert radius[6, 14] == 1.0
    assert radius[8, 10] == 1.0


def test_inside_matches_normalized_radius():
    ellipse = Ellipse(10.3, 6.1, 5.2, 3.3)
    shape = (13, 21)
    radius = ellipse.normalized_radius(shape)
    for factor in (0.5, 1.0, 1.3):
        assert np.array_equal(ellipse.inside(shape, factor), radius <= factor)


def test_inside_grows_with_factor():
    ellipse = Ellipse(10.0, 6.0, 4.0, 2.0)
    assert ellipse.inside((13, 21), 0.5).sum() < ellipse.inside((13, 21), 1.0).sum()
