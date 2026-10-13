import dataclasses

import numpy as np
import pytest

from ellipse_foreground.cues.patch import (
    choose_patch_samples,
    exhaustive_distances,
    nearest_sample_distances,
    patch_evidence,
    patch_vectors,
    pruned_distances,
)
from ellipse_foreground.cues.sampling import pixel_priorities
from ellipse_foreground.settings import PatchSettings
from tests.helpers import noisy_scene, random_work


def test_patch_vectors_hold_the_window_in_row_major_order():
    work = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    vectors = patch_vectors(work, 1)
    assert vectors.shape == (6, 27)
    assert np.array_equal(vectors[4, 12:15], work[1, 1])
    assert np.array_equal(vectors[0, 0:3], work[0, 0])


def brute_force_distances(vectors, sample_indices, shape, exclusion):
    width = shape[1]
    result = np.empty(len(vectors))
    for pixel in range(len(vectors)):
        row, column = divmod(pixel, width)
        best = np.inf
        for sample in sample_indices.tolist():
            sample_row, sample_column = divmod(sample, width)
            if abs(sample_row - row) <= exclusion and abs(sample_column - column) <= exclusion:
                continue
            best = min(best, np.linalg.norm(vectors[pixel] - vectors[sample]))
        result[pixel] = min(best, 1e6)
    return result


def test_nearest_sample_distances_match_brute_force_with_exclusion():
    work = random_work(9, 11, seed=21)
    vectors = patch_vectors(work, 1)
    shape = work.shape[:2]
    samples = np.sort(np.random.default_rng(2).choice(len(vectors), 25, replace=False))
    settings = dataclasses.replace(PatchSettings(), block_rows=16)
    found = nearest_sample_distances(vectors, samples, shape, settings)
    assert np.allclose(found, brute_force_distances(vectors, samples, shape, settings.exclusion), atol=1e-6)


def pruning_inputs(kind):
    if kind == "noisy":
        work = np.clip(noisy_scene(72, 128, seed=3)[0], 0.0, 1.0)
    elif kind == "random":
        work = random_work(72, 128, seed=8)
    else:
        work = np.full((72, 128, 3), 0.4, dtype=np.float32)
    vectors = patch_vectors(work, 1)
    samples = np.sort(np.random.default_rng(6).choice(len(vectors), 1800, replace=False))
    return vectors, samples, work.shape[:2]


@pytest.mark.parametrize("kind", ["noisy", "random", "constant"])
def test_pruned_distances_equal_exhaustive_distances(kind):
    vectors, samples, shape = pruning_inputs(kind)
    settings = dataclasses.replace(PatchSettings(), block_rows=256)
    pruned = pruned_distances(vectors, samples, shape, settings)
    exhaustive = exhaustive_distances(vectors, samples, shape, settings)
    assert np.abs(pruned - exhaustive).max() < 1e-7


def test_pruning_is_used_for_large_sample_sets_and_changes_nothing_else():
    vectors, samples, shape = pruning_inputs("noisy")
    settings = PatchSettings()
    assert np.array_equal(
        nearest_sample_distances(vectors, samples, shape, settings), pruned_distances(vectors, samples, shape, settings)
    )
    few = samples[:200]
    assert np.array_equal(
        nearest_sample_distances(vectors, few, shape, settings), exhaustive_distances(vectors, few, shape, settings)
    )


def test_duplicate_patches_have_distance_zero():
    work = np.zeros((12, 12, 3), dtype=np.float32)
    vectors = patch_vectors(work, 1)
    samples = np.array([0, 143])
    settings = PatchSettings()
    distances = nearest_sample_distances(vectors, samples, (12, 12), settings)
    assert distances[70] == 0.0


def test_patch_evidence_is_positive_near_the_core_appearance():
    work = np.zeros((20, 20, 3), dtype=np.float32)
    work[:, :10] = random_work(20, 10, seed=4) * 0.3 + 0.7
    work[:, 10:] = random_work(20, 10, seed=5) * 0.3
    core = np.zeros((20, 20), dtype=bool)
    core[:, :5] = True
    ring = np.zeros((20, 20), dtype=bool)
    ring[:, 15:] = True
    settings = PatchSettings()
    core_samples, ring_samples = choose_patch_samples(core, ring, pixel_priorities(400, 0), settings)
    evidence = patch_evidence(patch_vectors(work, 1), core_samples, ring_samples, (20, 20), settings)
    assert evidence[:, 6:9].mean() > 0.5
    assert evidence[:, 11:14].mean() < -0.5


def test_ring_sample_limit_follows_the_core_sample_count():
    core = np.zeros((60, 60), dtype=bool)
    core[:10, :10] = True
    ring = np.ones((60, 60), dtype=bool)
    ring[:10, :10] = False
    core_samples, ring_samples = choose_patch_samples(core, ring, pixel_priorities(3600, 0), PatchSettings())
    assert len(core_samples) == 100
    assert len(ring_samples) == 800


@pytest.mark.parametrize("empty", ["core", "ring"])
def test_an_empty_sample_set_gives_no_evidence(empty):
    from ellipse_foreground.cues.patch import patch_evidence

    vectors = np.random.default_rng(0).random((20, 27))
    samples = np.arange(5)
    none = np.array([], dtype=int)
    core, ring = (none, samples) if empty == "core" else (samples, none)
    evidence = patch_evidence(vectors, core, ring, (4, 5), PatchSettings())
    assert evidence.shape == (4, 5)
    assert not evidence.any()
