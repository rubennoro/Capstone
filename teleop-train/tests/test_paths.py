"""teleop_train.paths: names, folder layout, the marker and the refusals of PLAN.md §4.1."""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import pytest
import yaml

import synthetic_episode as se
from teleop_train import paths as P
from teleop_train.paths import PathsError


@pytest.fixture
def roots(tmp_path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    se.make_task(raw_root, "pick_test")
    return raw_root, out_root


# ---- names ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,ok", [
    ("pick_test", True), ("a", True), ("x" * 64, True), ("task_01", True),
    ("x" * 65, False), ("", False), ("Pick", False), ("pick-test", False), ("pick test", False), ("../x", False),
    (None, False), (3, False),
])
def test_is_task_name(name, ok):
    assert P.is_task_name(name) is ok


@pytest.mark.parametrize("name,ok", [
    ("episode_000", True), ("episode_007", True), ("episode_1000", True),
    ("episode_01", False), ("episode_abc", False), ("Episode_000", False), ("episode_000x", False), ("", False),
    (None, False), (7, False),
])
def test_is_episode_name(name, ok):
    assert P.is_episode_name(name) is ok


def test_episode_number():
    assert P.episode_number("episode_000") == 0
    assert P.episode_number("episode_007") == 7
    assert P.episode_number("episode_1234") == 1234
    for bad in ("episode_7", "ep_007", "episode_007/"):
        with pytest.raises(ValueError):
            P.episode_number(bad)


@pytest.mark.parametrize("name", ["teleop-v1", "A_b-9", "x" * 64])
def test_check_dataset_name_ok(name):
    P.check_dataset_name(name)


@pytest.mark.parametrize("name", ["", "x" * 65, "a b", "a/b", "..", "a.zip", None])
def test_check_dataset_name_refused(name):
    with pytest.raises(PathsError) as e:
        P.check_dataset_name(name)
    assert "\n" not in str(e.value)


def test_constants():
    assert P.POSE_SOURCES == ("target", "measured")
    assert P.DEFAULT_POSE_SOURCE in P.POSE_SOURCES
    assert P.DEFAULT_RAW_ROOT == Path.home() / "CapstoneTeleop"
    assert P.DEFAULT_OUT_ROOT == Path.home() / "Capstone" / "data"
    assert issubclass(PathsError, ValueError)


# ---- task_paths and TaskPaths ---------------------------------------------------------------------------------

def test_task_paths_layout(roots):
    raw_root, out_root = roots
    tp = P.task_paths("pick_test", raw_root, out_root)
    assert tp.task == "pick_test"
    assert tp.raw_task_dir == raw_root.resolve() / "pick_test"
    assert tp.out_task_dir == out_root.resolve() / "pick_test"
    o = tp.out_task_dir
    assert tp.raw_episode_dir("episode_003") == tp.raw_task_dir / "episode_003"
    assert tp.processed_dir("episode_003") == o / "episode_003" / "processed"
    assert tp.episode_zarr("episode_003") == o / "episode_003" / "processed" / "episode.zarr"
    assert tp.status_json("episode_003") == o / "episode_003" / "processed" / "status.json"
    assert tp.package_dir("ds1") == o / "package" / "ds1"
    assert tp.zip_path("ds1") == o / "package" / "ds1" / "ds1.zarr.zip"
    assert tp.package_json("ds1") == o / "package" / "ds1" / "package.json"
    assert tp.marker == o / "teleop_source.yaml"


def test_task_paths_bad_names_in_methods(roots):
    tp = P.task_paths("pick_test", *roots)
    for method in (tp.raw_episode_dir, tp.processed_dir, tp.episode_zarr, tp.status_json):
        with pytest.raises(PathsError):
            method("../episode_000")
    for method in (tp.package_dir, tp.zip_path, tp.package_json):
        with pytest.raises(PathsError):
            method("../x")


def test_task_paths_expands_and_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    se.make_task(tmp_path / "CapstoneTeleop", "pick_test")
    tp = P.task_paths("pick_test", "~/CapstoneTeleop", "~/a/../Capstone/data")
    assert tp.raw_task_dir == tmp_path.resolve() / "CapstoneTeleop" / "pick_test"
    assert tp.out_task_dir == tmp_path.resolve() / "Capstone" / "data" / "pick_test"
    monkeypatch.chdir(tmp_path)
    tp2 = P.task_paths("pick_test", "CapstoneTeleop", "out")
    assert tp2.raw_task_dir.is_absolute() and tp2.out_task_dir == tmp_path.resolve() / "out" / "pick_test"


@pytest.mark.parametrize("task", ["Pick", "pick-test", "", "../pick_test"])
def test_task_paths_bad_task_name(roots, task):
    with pytest.raises(PathsError, match="task name"):
        P.task_paths(task, *roots)


def test_task_paths_task_yaml_missing(roots):
    raw_root, out_root = roots
    (raw_root / "other").mkdir()
    with pytest.raises(PathsError, match="task.yaml: missing"):
        P.task_paths("other", raw_root, out_root)
    with pytest.raises(PathsError, match="task.yaml: missing"):
        P.task_paths("nothing_here", raw_root, out_root)


def test_task_paths_refuses_overlap_with_raw_root(roots, tmp_path):
    raw_root, _ = roots
    for out_root in (raw_root, raw_root / "sub", raw_root / "pick_test"):
        with pytest.raises(PathsError, match="overlaps the raw root"):
            P.task_paths("pick_test", raw_root, out_root)
    # a raw root inside the output task folder
    nested_raw = tmp_path / "o" / "pick_test" / "raw"
    se.make_task(nested_raw, "pick_test")
    with pytest.raises(PathsError, match="overlaps the raw root"):
        P.task_paths("pick_test", nested_raw, tmp_path / "o")


# ---- raw_episodes ---------------------------------------------------------------------------------------------

def test_raw_episodes_order_and_filter(roots):
    tp = P.task_paths("pick_test", *roots)
    d = tp.raw_task_dir
    for n in ("episode_010", "episode_002", "episode_1000", "episode_999"):
        (d / n).mkdir()
    (d / "episode_003").write_text("a file")
    (d / "episode_01").mkdir()
    (d / "episode_abc").mkdir()
    (d / "notes").mkdir()
    os.symlink(d / "episode_002", d / "episode_004")
    assert P.raw_episodes(tp) == ["episode_002", "episode_010", "episode_999", "episode_1000"]


def test_raw_episodes_includes_incomplete(roots):
    tp = P.task_paths("pick_test", *roots)
    se.make_episode(tp.raw_task_dir, 0)
    se.make_episode(tp.raw_task_dir, 1, complete=False)
    assert P.raw_episodes(tp) == ["episode_000", "episode_001"]


def test_raw_episodes_empty(roots):
    assert P.raw_episodes(P.task_paths("pick_test", *roots)) == []


# ---- prepare_out_task -----------------------------------------------------------------------------------------

def test_prepare_out_task_creates_marker_and_copy(roots):
    raw_root, out_root = roots
    se.make_episode(raw_root / "pick_test", 0)
    before = se.tree_hash(raw_root)
    tp = P.task_paths("pick_test", raw_root, out_root)
    assert not out_root.exists()
    P.prepare_out_task(tp)
    assert se.tree_hash(raw_root) == before
    marker = yaml.safe_load(tp.marker.read_text())
    assert list(marker) == ["source", "raw_task_dir", "created"]
    assert marker["source"] == "teleop_collect"
    assert marker["raw_task_dir"] == str(tp.raw_task_dir)
    assert Path(marker["raw_task_dir"]).is_absolute()
    created = datetime.datetime.fromisoformat(marker["created"])
    assert created.tzinfo is not None
    assert (tp.out_task_dir / "task.yaml").read_bytes() == (tp.raw_task_dir / "task.yaml").read_bytes()
    assert sorted(p.name for p in tp.out_task_dir.iterdir()) == ["task.yaml", "teleop_source.yaml"]


def test_prepare_out_task_idempotent_and_refreshes_task_copy(roots):
    raw_root, out_root = roots
    tp = P.task_paths("pick_test", raw_root, out_root)
    P.prepare_out_task(tp)
    marker_bytes = tp.marker.read_bytes()
    P.prepare_out_task(tp)
    assert tp.marker.read_bytes() == marker_bytes
    # teleop_collect advances next_episode in task.yaml at every recording: the copy follows, the marker stays
    raw_task = tp.raw_task_dir / "task.yaml"
    data = yaml.safe_load(raw_task.read_text())
    data["next_episode"] = 5
    raw_task.write_text(yaml.safe_dump(data, sort_keys=False))
    before = se.tree_hash(raw_root)
    P.prepare_out_task(tp)
    assert se.tree_hash(raw_root) == before
    assert (tp.out_task_dir / "task.yaml").read_bytes() == raw_task.read_bytes()
    assert tp.marker.read_bytes() == marker_bytes
    assert not [p for p in tp.out_task_dir.iterdir() if p.name.endswith(".tmp")]


def test_prepare_out_task_keeps_other_content(roots):
    tp = P.task_paths("pick_test", *roots)
    P.prepare_out_task(tp)
    status = tp.status_json("episode_000")
    status.parent.mkdir(parents=True)
    status.write_text("{}")
    P.prepare_out_task(tp)
    assert status.read_text() == "{}"


def test_prepare_out_task_refuses_handheld_folder(roots):
    raw_root, out_root = roots
    tp = P.task_paths("pick_test", raw_root, out_root)
    tp.out_task_dir.mkdir(parents=True)
    (tp.out_task_dir / "episodes.csv").write_text("handheld")
    before_raw, before_out = se.tree_hash(raw_root), se.tree_hash(out_root)
    with pytest.raises(PathsError, match="without teleop_source.yaml") as e:
        P.prepare_out_task(tp)
    assert "\n" not in str(e.value)
    assert se.tree_hash(raw_root) == before_raw and se.tree_hash(out_root) == before_out
    assert not tp.marker.exists()


def test_prepare_out_task_refuses_empty_existing_folder(roots):
    tp = P.task_paths("pick_test", *roots)
    tp.out_task_dir.mkdir(parents=True)
    with pytest.raises(PathsError, match="without teleop_source.yaml"):
        P.prepare_out_task(tp)


def test_prepare_out_task_refuses_other_raw_folder(roots, tmp_path):
    raw_root, out_root = roots
    P.prepare_out_task(P.task_paths("pick_test", raw_root, out_root))
    other_raw = tmp_path / "raw2"
    se.make_task(other_raw, "pick_test")
    tp2 = P.task_paths("pick_test", other_raw, out_root)
    before = se.tree_hash(out_root)
    with pytest.raises(PathsError, match="was made from") as e:
        P.prepare_out_task(tp2)
    assert str(raw_root.resolve() / "pick_test") in str(e.value)
    assert se.tree_hash(out_root) == before


def test_prepare_out_task_accepts_same_raw_folder_by_another_spelling(roots, tmp_path):
    raw_root, out_root = roots
    P.prepare_out_task(P.task_paths("pick_test", raw_root, out_root))
    link = tmp_path / "raw_link"
    os.symlink(raw_root, link)
    P.prepare_out_task(P.task_paths("pick_test", link, out_root))          # resolves to the same folder


@pytest.mark.parametrize("content", [
    "source: umi_processing\nraw_task_dir: /x\n",
    "raw_task_dir: /x\n",
    "source: teleop_collect\n",
    "- a list\n",
    "{unclosed: [\n",
])
def test_prepare_out_task_refuses_bad_marker(roots, content):
    tp = P.task_paths("pick_test", *roots)
    tp.out_task_dir.mkdir(parents=True)
    tp.marker.write_text(content)
    with pytest.raises(PathsError) as e:
        P.prepare_out_task(tp)
    assert "\n" not in str(e.value)
    assert not (tp.out_task_dir / "task.yaml").exists()


def test_prepare_out_task_refuses_file_or_symlink(roots, tmp_path):
    tp = P.task_paths("pick_test", *roots)
    tp.out_task_dir.parent.mkdir(parents=True)
    tp.out_task_dir.write_text("not a folder")
    with pytest.raises(PathsError, match="not a real folder"):
        P.prepare_out_task(tp)
    tp.out_task_dir.unlink()
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.symlink(target, tp.out_task_dir)
    with pytest.raises(PathsError, match="not a real folder"):
        P.prepare_out_task(tp)
    assert list(target.iterdir()) == []


def test_prepare_out_task_refuses_overlap(roots):
    raw_root, _ = roots
    tp = P.TaskPaths("pick_test", raw_root.resolve() / "pick_test", raw_root.resolve() / "pick_test")
    before = se.tree_hash(raw_root)
    with pytest.raises(PathsError, match="overlaps the raw root"):
        P.prepare_out_task(tp)
    assert se.tree_hash(raw_root) == before


def test_read_marker(roots):
    tp = P.task_paths("pick_test", *roots)
    with pytest.raises(PathsError, match="without teleop_source.yaml"):
        P.read_marker(tp)
    P.prepare_out_task(tp)
    assert P.read_marker(tp)["raw_task_dir"] == str(tp.raw_task_dir)
