import numpy as np

from ellipse_foreground.evaluation.recovery import character_columns, fill_foreground, recover_alpha


def synthetic_exports(height=40, width=60, seed=0):
    rng = np.random.default_rng(seed)
    background = rng.uniform(40, 255, (height, width, 3))
    foreground = rng.uniform(0, 255, (height, width, 3))
    alpha = np.zeros((height, width))
    alpha[8:32, 20:40] = 1.0
    alpha[8:32, 20] = 0.4
    composite = alpha[..., None] * foreground + (1 - alpha[..., None]) * background
    return alpha, foreground, background, alpha[..., None] * foreground, composite


def test_recover_alpha_inverts_the_composite():
    alpha, _, background, black_composite, composite = synthetic_exports()
    recovered = recover_alpha(composite, black_composite, background)
    assert np.allclose(recovered, np.where(alpha < 0.01, 0.0, alpha), atol=1e-6)


def test_recover_alpha_clips_to_the_unit_interval():
    _, _, background, black_composite, composite = synthetic_exports(seed=1)
    recovered = recover_alpha(composite + 30.0, black_composite, background)
    assert recovered.min() >= 0.0
    assert recovered.max() <= 1.0


def test_character_columns_follow_the_main_connected_range_with_a_margin():
    alpha = np.zeros((20, 80))
    alpha[:, 30:40] = 1.0
    alpha[:, 70:72] = 1.0
    keep = character_columns(alpha)
    assert keep[:, 23:47].all()
    assert not keep[:, :23].any()
    assert not keep[:, 47:].any()


def test_recover_alpha_discards_pillars_outside_the_character():
    alpha, _, background, black_composite, composite = synthetic_exports(width=100)
    pillar = np.zeros_like(alpha)
    pillar[:10, 90:93] = 1.0
    composite = composite * (1 - pillar[..., None])
    black = black_composite * (1 - pillar[..., None])
    recovered = recover_alpha(composite, black, background)
    assert recovered[:, 80:].max() == 0.0


def test_fill_foreground_extends_colors_into_transparent_pixels():
    alpha = np.zeros((20, 20))
    alpha[5:15, 5:15] = 1.0
    color = np.zeros((20, 20, 3))
    color[5:15, 5:15] = [0.8, 0.4, 0.2]
    filled = fill_foreground(color, alpha)
    assert np.allclose(filled[5:15, 5:15], [0.8, 0.4, 0.2], atol=1e-6)
    assert np.allclose(filled[2, 10], [0.8, 0.4, 0.2], atol=1e-6)
    assert filled.dtype == np.float32
