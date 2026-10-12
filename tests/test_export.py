import numpy as np
import pytest
from PIL import Image

from ellipse_foreground.evaluation.export import main, recover_sources, write_sources
from ellipse_foreground.evaluation.own import CANVAS_HEIGHT, CANVAS_WIDTH, own_reconstruction, own_sources


def write_exports(folder, group, seed=0):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    background = rng.uniform(40, 255, (CANVAS_HEIGHT, CANVAS_WIDTH, 3))
    color = rng.uniform(0, 255, (CANVAS_HEIGHT, CANVAS_WIDTH, 3))
    alpha = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH))
    alpha[200:440, 500:700] = 1.0
    alpha[200:440, 500] = 0.4
    black = alpha[..., None] * color
    composite = black + (1 - alpha[..., None]) * background
    names = {
        "composite.png": composite,
        "foreground_only.png": black,
        "background_only.png": background,
        "original_colors_composite.png": composite,
        "original_colors_foreground.png": black,
    }
    for name, array in names.items():
        Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(folder / name)
    return alpha


@pytest.mark.parametrize("group", ["sprite1", "sprite2"])
def test_recovered_alpha_follows_the_exports(tmp_path, group):
    alpha = write_exports(tmp_path / "raw", group)
    sources = recover_sources(group, tmp_path / "raw")
    assert sources.alpha.dtype == np.float32
    assert np.abs(sources.alpha[:, :50]).max() == 0.0
    assert np.abs(sources.alpha[220:420, 520:680] - 1.0).max() < 0.02
    assert sources.foreground.dtype == np.float32
    assert sources.reconstruction_mean < 1.0
    assert (alpha > 0.5).sum() > 0


def test_written_sources_keep_the_character_and_drop_everything_else(tmp_path):
    write_exports(tmp_path / "raw", "sprite1")
    destination = write_sources("sprite1", tmp_path / "raw", tmp_path / "data")
    assert destination == tmp_path / "data" / "own" / "sprite1.npz"
    alpha, foreground = own_sources(tmp_path / "data", "sprite1")
    assert alpha.shape == (CANVAS_HEIGHT, CANVAS_WIDTH)
    assert foreground.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)
    assert not foreground[alpha == 0].any()
    assert foreground[300, 600].any()
    mean, largest = own_reconstruction(tmp_path / "data", "sprite1")
    assert 0.0 <= mean <= largest


def test_command_line_writes_the_sources(tmp_path, capsys):
    write_exports(tmp_path / "raw", "sprite2")
    assert main(["sprite2", str(tmp_path / "raw"), "--data", str(tmp_path / "data")]) == 0
    assert (tmp_path / "data" / "own" / "sprite2.npz").exists()
    assert "sprite2.npz" in capsys.readouterr().out
