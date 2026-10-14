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


def test_from_moments_recovers_a_filled_ellipse():
    rows, columns = np.mgrid[0:60, 0:80]
    mask = ((columns - 40.0) / 24.0) ** 2 + ((rows - 28.0) / 14.0) ** 2 <= 1.0
    ellipse = Ellipse.from_moments(mask)
    assert abs(ellipse.center_x - 40.5) < 0.6
    assert abs(ellipse.center_y - 28.5) < 0.6
    assert abs(ellipse.radius_x - 24.0) < 0.8
    assert abs(ellipse.radius_y - 14.0) < 0.8


def test_from_moments_keeps_a_minimum_radius():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    ellipse = Ellipse.from_moments(mask)
    assert (ellipse.radius_x, ellipse.radius_y) == (2.0, 2.0)
    assert (ellipse.center_x, ellipse.center_y) == (4.5, 4.5)


def test_shifted_moves_the_center_in_units_of_the_radii():
    assert Ellipse(10.0, 6.0, 4.0, 2.0).shifted(0.5, -1.0) == Ellipse(12.0, 4.0, 4.0, 2.0)


def test_stretched_scales_the_axes_independently():
    assert Ellipse(10.0, 6.0, 4.0, 2.0).stretched(2.0, 0.5) == Ellipse(10.0, 6.0, 8.0, 1.0)


def test_blended_interpolates_centers_linearly_and_radii_geometrically():
    first, second = Ellipse(10.0, 4.0, 4.0, 9.0), Ellipse(20.0, 8.0, 16.0, 4.0)
    assert first.blended(second, 0.0) == first
    assert first.blended(second, 1.0) == second
    middle = first.blended(second, 0.5)
    assert (middle.center_x, middle.center_y) == (15.0, 6.0)
    assert (middle.radius_x, middle.radius_y) == (8.0, 6.0)
