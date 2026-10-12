from pathlib import Path

from paper.figures.diagrams import dynamic_programming, recovery
from paper.figures.overview import overview_figure, texture_cues
from paper.figures.plots import bars_plot, boundary_plot, dose_plot, scatter_plot
from paper.figures.qualitative import comparison_figure, render_figure
from paper.figures.samples import public_samples
from paper.figures.style import configure
from paper.study import Study


def build_figures(study: Study, data_root: Path, directory: Path) -> None:
    configure()
    dynamic_programming(directory)
    recovery(directory, data_root)
    overview_figure(directory, data_root)
    texture_cues(directory, data_root)
    public_samples(directory, data_root)
    dose_plot(study, directory)
    scatter_plot(study, directory)
    boundary_plot(study, directory)
    bars_plot(study, directory)
    comparison_figure(directory, data_root, study.results)
    render_figure(directory, data_root, study.results)
