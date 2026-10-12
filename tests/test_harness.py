import json

import numpy as np
import pytest
from harness.arguments import HarnessArguments, HarnessMode
from harness.comparison import ImageDifference, compare_arrays
from harness.golden import Golden, GoldenCase, HarnessError, current_environment, digest


def test_arguments_default_to_rendering():
    assert HarnessArguments.parse([]) == HarnessArguments(HarnessMode.RENDER)
    assert HarnessArguments.parse(["out"]) == HarnessArguments(HarnessMode.RENDER, "out")


@pytest.mark.parametrize(
    ("flag", "mode"),
    [
        ("--golden", HarnessMode.GOLDEN),
        ("--verify", HarnessMode.VERIFY),
        ("--transition", HarnessMode.TRANSITION),
        ("--benchmark", HarnessMode.BENCHMARK),
    ],
)
def test_arguments_select_one_mode(flag, mode):
    assert HarnessArguments.parse([flag]).mode is mode


def test_arguments_compare_takes_two_directories():
    assert HarnessArguments.parse(["--compare", "a", "b"]) == HarnessArguments(HarnessMode.COMPARE, None, "a", "b")
    with pytest.raises(HarnessError):
        HarnessArguments.parse(["--compare", "a"])


@pytest.mark.parametrize("arguments", [["--golden", "--verify"], ["--unknown"], ["--golden", "extra"], ["a", "b"]])
def test_arguments_reject_invalid_combinations(arguments):
    with pytest.raises(HarnessError):
        HarnessArguments.parse(arguments)


def test_identical_arrays_have_no_difference():
    array = np.arange(12).reshape(3, 4)
    assert compare_arrays(array, array.copy()).is_empty
    assert str(compare_arrays(array, array.copy())) == "一致"


def test_difference_reports_count_maximum_and_box():
    first = np.zeros((6, 8))
    second = first.copy()
    second[2, 3] = 0.5
    second[4, 6] = -2.0
    difference = compare_arrays(first, second)
    assert difference == ImageDifference(2, 2.0, 3, 2, 6, 4)
    assert "differing=2" in str(difference)


def test_difference_of_color_images_counts_pixels_not_channels():
    first = np.zeros((4, 4, 3))
    second = first.copy()
    second[1, 1] = [1, 1, 1]
    assert compare_arrays(first, second).differing == 1


def test_difference_rejects_different_shapes():
    with pytest.raises(ValueError, match="寸法"):
        compare_arrays(np.zeros((2, 2)), np.zeros((3, 3)))


def test_digest_depends_on_content_shape_and_dtype():
    array = np.arange(6, dtype=np.float32)
    assert digest(array) == digest(array.copy())
    assert digest(array) != digest(array.reshape(2, 3))
    assert digest(array) != digest(array.astype(np.float64))
    assert digest(array, array + 1) != digest(array + 1, array)


def test_golden_round_trip(tmp_path):
    golden = Golden("env", "input", (GoldenCase("a", "H1"), GoldenCase("b", "H2")))
    path = tmp_path / "golden.json"
    golden.save(path)
    assert Golden.load(path) == golden
    assert json.loads(path.read_text(encoding="utf-8"))["cases"][0] == {"name": "a", "hash": "H1"}


def test_golden_rejects_missing_broken_and_duplicate_files(tmp_path):
    with pytest.raises(HarnessError, match="基準値がありません"):
        Golden.load(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(HarnessError, match="読めません"):
        Golden.load(broken)
    duplicate = tmp_path / "duplicate.json"
    Golden("env", "input", (GoldenCase("a", "1"), GoldenCase("a", "2"))).save(duplicate)
    with pytest.raises(HarnessError, match="重複"):
        Golden.load(duplicate)


def test_current_environment_names_the_libraries():
    environment = current_environment()
    assert "numpy" in environment
    assert "scipy" in environment
