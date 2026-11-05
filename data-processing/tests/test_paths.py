import os
from pathlib import Path

import pytest

from dataproc.paths import (
    DEFAULT_ROOT, data_root, episode_dirs, episode_index, next_episode_index, resolve_intake_target, resolve_task,
    validate_episode_dir_name, validate_name,
)


def test_root_default_env_and_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("DATAPROC_ROOT", raising=False)
    assert data_root().path == DEFAULT_ROOT
    monkeypatch.setenv("DATAPROC_ROOT", "")
    assert data_root().path == DEFAULT_ROOT
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path))
    r = data_root()
    assert r.path == tmp_path
    assert (r.incoming, r.tasks, r.rigs, r.quarantine) == tuple(tmp_path / n for n in
                                                                 ("incoming", "tasks", "rigs", "quarantine"))
    assert data_root(tmp_path / "x").path == tmp_path / "x"
    assert r.rig_file("rig_v1") == tmp_path / "rigs" / "rig_v1.yaml"
    with pytest.raises(ValueError):
        r.rig_file("../etc")


@pytest.mark.parametrize("name", ["block_raise", "a", "A-1_b", "block_raise_1"])
def test_valid_names(name):
    assert validate_name(name) == name


@pytest.mark.parametrize("name", ["", "a b", "a/b", "..", "ä", "a.b", None])
def test_invalid_names(name):
    with pytest.raises(ValueError):
        validate_name(name)


def test_episode_index():
    assert episode_index("x/episode_0") == 0
    assert episode_index("episode_12") == 12
    for bad in ("episode_", "episode_01", "episode_-1", "episode_3x", "ep_3"):
        assert episode_index(bad) is None


def test_episode_dirs_numeric_sort_and_next_index(tmp_path):
    assert episode_dirs(tmp_path / "nope") == []
    assert next_episode_index(tmp_path / "nope") == 0
    assert next_episode_index(tmp_path) == 0
    for n in (0, 2, 10, 1):
        (tmp_path / f"episode_{n}").mkdir()
    (tmp_path / "calibration").mkdir()
    (tmp_path / "episode_01").mkdir()          # not a valid episode name
    (tmp_path / "episode_7").write_text("")    # a file, not a dir (but its number is taken)
    assert [p.name for p in episode_dirs(tmp_path)] == ["episode_0", "episode_1", "episode_2", "episode_10"]
    assert next_episode_index(tmp_path) == 11


def test_next_index_does_not_refill_gaps(tmp_path):
    (tmp_path / "episode_3").mkdir()
    assert next_episode_index(tmp_path) == 4


def test_resolve_task_names(tmp_path):
    root = data_root(tmp_path)
    t = resolve_task("block_raise", root)
    assert (t.folder, t.task_dir, t.episode, t.episode_dir) == ("block_raise", root.tasks / "block_raise", None, None)
    t = resolve_task("block_raise/episode_3", root)
    assert (t.folder, t.episode, t.episode_dir) == ("block_raise", 3, root.tasks / "block_raise" / "episode_3")
    assert resolve_task("block_raise/", root).episode is None


def test_resolve_task_paths(tmp_path, monkeypatch):
    root = data_root(tmp_path)
    t = resolve_task(str(root.tasks / "block_raise_1"), root)
    assert (t.folder, t.task_dir, t.episode) == ("block_raise_1", root.tasks / "block_raise_1", None)
    t = resolve_task(str(root.tasks / "block_raise" / "episode_12"), root)
    assert (t.folder, t.episode) == ("block_raise", 12)
    monkeypatch.chdir(root.path)
    (root.tasks).mkdir()
    t = resolve_task("./tasks/block_raise", root)
    assert t.task_dir == root.tasks / "block_raise"


@pytest.mark.parametrize("spec", ["/elsewhere/block_raise", "a/b/c", "block_raise/episode_01", "bad name", "x/../y"])
def test_resolve_task_rejects(tmp_path, spec):
    with pytest.raises(ValueError):
        resolve_task(spec, data_root(tmp_path))


def test_resolve_task_rejects_bad_folder_name_in_path(tmp_path):
    root = data_root(tmp_path)
    with pytest.raises(ValueError):
        resolve_task(str(root.tasks / "bad.name"), root)


def test_intake_target(tmp_path):
    tasks = tmp_path / "tasks"
    # no folder (not even tasks/) -> the name, with or without --new
    assert resolve_intake_target(tasks, "block_raise", new=False) == "block_raise"
    assert resolve_intake_target(tasks, "block_raise", new=True) == "block_raise"
    assert not tasks.exists()                                         # pure: nothing created
    (tasks / "block_raise").mkdir(parents=True)
    assert resolve_intake_target(tasks, "block_raise", new=False) == "block_raise"
    assert resolve_intake_target(tasks, "block_raise", new=True) == "block_raise_1"
    (tasks / "block_raise_1").mkdir()
    (tasks / "block_raise_3").mkdir()
    assert resolve_intake_target(tasks, "block_raise", new=True) == "block_raise_2"   # first free
    (tasks / "block_raise_2").write_text("")                          # a file also blocks the name
    assert resolve_intake_target(tasks, "block_raise", new=True) == "block_raise_4"
    assert sorted(p.name for p in tasks.iterdir()) == ["block_raise", "block_raise_1", "block_raise_2",
                                                       "block_raise_3"]
    with pytest.raises(ValueError):
        resolve_intake_target(tasks, "bad/name", new=False)


def test_intake_target_on_suffixed_name(tmp_path):
    (tmp_path / "block_raise_1").mkdir()
    assert resolve_intake_target(tmp_path, "block_raise_1", new=True) == "block_raise_1_1"
    assert isinstance(resolve_intake_target(Path(tmp_path), "block_raise_1", new=False), str)


# --------------------------------------------------------------------------------------------- fix round (R-intake)

@pytest.mark.parametrize("name", ["s01_episode_000001", "2026-11-05_s01.ep3", "A", "0x"])
def test_valid_episode_dir_names(name):
    assert validate_episode_dir_name(name) == name


@pytest.mark.parametrize("name", ["", ".", "..", ".hidden", "-x", "_x", "a b", "a/b", "é", None])
def test_invalid_episode_dir_names(name):
    with pytest.raises(ValueError):
        validate_episode_dir_name(name)


def test_dangling_and_file_episode_entries_are_taken(tmp_path):
    os.symlink(tmp_path / "missing", tmp_path / "episode_7")
    (tmp_path / "episode_9").write_text("")
    (tmp_path / "x").mkdir()
    os.symlink(tmp_path / "x", tmp_path / "episode_4")
    assert [p.name for p in episode_dirs(tmp_path)] == ["episode_4"]     # a symlink to a dir is a dir
    assert next_episode_index(tmp_path) == 10


def test_huge_episode_numbers(tmp_path):
    for n in (0, 2, 10 ** 30):
        (tmp_path / f"episode_{n}").mkdir()
    assert next_episode_index(tmp_path) == 10 ** 30 + 1


def test_intake_target_counts_dangling_symlink(tmp_path):
    (tmp_path / "br").mkdir()
    os.symlink(tmp_path / "nowhere", tmp_path / "br_1")
    assert resolve_intake_target(tmp_path, "br", new=True) == "br_2"
    os.symlink(tmp_path / "nowhere", tmp_path / "solo")
    assert resolve_intake_target(tmp_path, "solo", new=True) == "solo_1"


def test_intake_target_existing_non_directory(tmp_path):
    (tmp_path / "br").write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        resolve_intake_target(tmp_path, "br", new=False)
    assert resolve_intake_target(tmp_path, "br", new=True) == "br_1"


def test_resolve_task_normalises_dotdot(tmp_path):
    root = data_root(tmp_path)
    t = resolve_task(str(root.tasks / "x" / ".." / "block_raise"), root)
    assert (t.folder, t.task_dir) == ("block_raise", root.tasks / "block_raise")
    with pytest.raises(ValueError):
        resolve_task(str(root.tasks / ".." / "block_raise"), root)


def test_resolve_task_symlinked_root(tmp_path):
    real = tmp_path / "real"
    (real / "tasks").mkdir(parents=True)
    os.symlink(real, tmp_path / "link")
    root = data_root(tmp_path / "link")
    for spec in (str(real / "tasks" / "a"), str(tmp_path / "link" / "tasks" / "a" / "episode_2")):
        t = resolve_task(spec, root)
        assert t.folder == "a" and t.task_dir == root.tasks / "a"
    root_real = data_root(real)
    assert resolve_task(str(tmp_path / "link" / "tasks" / "a"), root_real).task_dir == real / "tasks" / "a"


@pytest.mark.parametrize("spec", ["block_raise/episode_01", "/x/tasks/br/episode_007"])
def test_resolve_task_leading_zero_message(tmp_path, spec):
    with pytest.raises(ValueError, match="no leading zeros"):
        resolve_task(spec, data_root(tmp_path))
