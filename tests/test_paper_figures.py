import numpy as np
import pytest

from ellipse_foreground.geometry import Ellipse
from ellipse_foreground.tree import build_tree
from paper.figures.diagrams import leaves_under, solve_example
from paper.figures.overview import region_labels, tree_mosaic
from paper.figures.plots import bars_plot, boundary_plot, dose_plot, scatter_plot
from paper.figures.qualitative import crop_box, error_picture
from paper.figures.style import configure
from paper.results import Results
from paper.study import Study
from tests.helpers import random_work
from tests.paper_fixtures import write_results


def test_example_tree_selects_the_expected_cut():
    scores, best, cut = solve_example()
    assert set(cut) == {"n1", "e"}
    assert best["r"] == pytest.approx(1.4)
    assert scores["n1"] == pytest.approx(1.1)
    assert sorted(leaves_under("r")) == ["a", "b", "c", "d", "e", "f"]


def test_region_labels_merge_below_the_height_threshold():
    work = random_work(6, 8, seed=2)
    tree = build_tree(work)
    assert len(np.unique(region_labels(tree, 0.0))) == tree.leaf_count
    assert len(np.unique(region_labels(tree, float(tree.height.max())))) == 1


def test_mosaic_has_no_more_regions_than_requested_when_possible():
    work = random_work(10, 12, seed=3)
    mosaic, count = tree_mosaic(build_tree(work), work, target=20)
    assert mosaic.shape == work.shape
    assert 1 <= count <= 20


def test_error_picture_colors_each_kind_of_agreement():
    truth = np.array([[True, True], [False, False]])
    alpha = np.array([[1.0, 0.0], [1.0, 0.0]])
    picture = error_picture(alpha, truth)
    assert picture[0, 0].tolist() == [1, 1, 1]
    assert picture[0, 1].tolist() == [0.25, 0.5, 1.0]
    assert picture[1, 0].tolist() == [0.95, 0.2, 0.2]
    assert picture[1, 1].tolist() == [0, 0, 0]


def test_crop_box_covers_the_foreground_and_the_ellipse_inside_the_image():
    truth = np.zeros((100, 200), dtype=bool)
    truth[20:80, 90:110] = True
    left, right, top, bottom = crop_box(truth, Ellipse(100, 50, 40, 60), truth.shape)
    assert left <= 60 and right >= 140
    assert top == 0 and bottom == 100


def test_result_plots_are_written(tmp_path):
    write_results(tmp_path / "results")
    study = Study(Results(tmp_path / "results"), None)
    configure()
    for plot in (dose_plot, scatter_plot, boundary_plot, bars_plot):
        plot(study, tmp_path / "figs")
    names = {path.name for path in (tmp_path / "figs").iterdir()}
    assert {"fig_dose.png", "fig_scatter.svg", "fig_reg.svg", "fig_bars.png"} <= names
