import numpy as np
import pytest

from ellipse_foreground import Settings
from ellipse_foreground.evaluation.results import append_records, finished_scenes, read_records, result_files
from ellipse_foreground.evaluation.runner import GroupSpec, output_path, run_group
from ellipse_foreground.evaluation.scene import Scene, scene_from_composite
from ellipse_foreground.settings import WorkSettings
from tests.helpers import noisy_scene

SETTINGS = Settings(work=WorkSettings(width=64))


def make_scene(index: int = 0) -> Scene:
    _, source, body = noisy_scene(144, 256, seed=3)
    return scene_from_composite(f"scene{index}", "group", index, source, body.astype(np.float32))


def test_results_round_trip_and_merge_partitions(tmp_path):
    append_records(tmp_path / "a.jsonl", [{"scene": 0, "trial": 0, "value": 1}, {"scene": 0, "trial": 1, "value": 2}])
    append_records(
        tmp_path / "a.5-9.jsonl", [{"scene": 5, "trial": 0, "value": 3}, {"scene": 0, "trial": 1, "value": 9}]
    )
    assert [path.name for path in result_files(tmp_path, "a")] == ["a.jsonl", "a.5-9.jsonl"]
    merged = read_records(tmp_path, "a")
    assert [(item["scene"], item["trial"], item["value"]) for item in merged] == [(0, 0, 1), (0, 1, 9), (5, 0, 3)]
    assert finished_scenes(tmp_path, "a") == {0, 5}


def test_output_path_names_partitions(tmp_path):
    assert output_path(tmp_path, "comparison", "g", range(0, 10**6)) == tmp_path / "comparison" / "g.jsonl"
    assert output_path(tmp_path, "comparison", "g", range(3, 7)) == tmp_path / "comparison" / "g.3-7.jsonl"


def test_run_group_writes_records_and_resumes(tmp_path):
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene(0), make_scene(1)]), 2, None)
    selection = range(0, 10**6)
    assert run_group(spec, tmp_path, tmp_path / "out", ["trials"], selection, SETTINGS) == 2
    records = read_records(tmp_path / "out" / "comparison", "tiny")
    assert {record["scene"] for record in records} == {0, 1}
    assert run_group(spec, tmp_path, tmp_path / "out", ["trials"], selection, SETTINGS) == 0


def test_resolution_runs_only_on_strided_scenes(tmp_path, monkeypatch):
    calls = []

    def fake_records(scene, width, count):
        calls.append((scene.index, width))
        return [{"scene": scene.index, "trial": 0, "width": width}]

    monkeypatch.setattr("ellipse_foreground.evaluation.runner.resolution_records", fake_records)
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene(index) for index in range(4)]), 2, 3)
    run_group(spec, tmp_path, tmp_path / "out", ["resolution"], range(0, 10**6), SETTINGS)
    assert sorted({index for index, _ in calls}) == [0, 3]
    assert {width for _, width in calls} == {128, 256, 384}


def test_missing_experiments_of_a_finished_scene_are_run_again(tmp_path):
    spec = GroupSpec("tiny", lambda data, selection: iter([make_scene(0)]), 2, None)
    selection = range(0, 10**6)
    run_group(spec, tmp_path, tmp_path / "out", ["trials"], selection, SETTINGS)
    comparison = tmp_path / "out" / "comparison" / "tiny.jsonl"
    before = comparison.read_text(encoding="utf-8")
    (tmp_path / "out" / "ablation" / "tiny.jsonl").unlink()
    assert run_group(spec, tmp_path, tmp_path / "out", ["trials"], selection, SETTINGS) == 1
    assert (tmp_path / "out" / "ablation" / "tiny.jsonl").exists()
    assert comparison.read_text(encoding="utf-8") == before


def test_a_failing_producer_leaves_no_partial_records(tmp_path):
    def produce():
        yield {"scene": 0, "trial": 0}
        raise RuntimeError

    with pytest.raises(RuntimeError):
        append_records(tmp_path / "a.jsonl", produce())
    assert not (tmp_path / "a.jsonl").exists()
