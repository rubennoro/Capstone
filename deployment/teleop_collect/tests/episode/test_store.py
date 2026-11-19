"""TaskStore, Task, EpisodeDir: names, task.yaml, numbering never reused, finish, delete, discard, clean_incomplete."""
from __future__ import annotations

import os

import pytest
import yaml

from teleop_collect.teleop.episode import (COMPLETE_FILE, META_FILE, TASK_FILE, EpisodeDir, EpisodeInfo, TaskError,
                                           TaskStore, episode_name)


def meta(outcome="success", duration=12.5, frames=300):
    return {"outcome": outcome, "duration_s": duration, "counts": {"frames_main": frames}}


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "data")


def test_create_writes_task_yaml(store):
    t = store.create("pick_block", "pick the red block", {"motion": {"v_max_m_s": 0.05}})
    assert (t.name, t.description, t.dir) == ("pick_block", "pick the red block", store.root / "pick_block")
    data = yaml.safe_load((t.dir / TASK_FILE).read_text())
    assert list(data) == ["name", "description", "created", "next_episode", "settings"]
    assert data["name"] == "pick_block" and data["next_episode"] == 0
    assert data["settings"] == {"motion": {"v_max_m_s": 0.05}}
    assert "T" in data["created"]
    assert store.tasks() == ["pick_block"]
    assert store.open("pick_block").description == "pick the red block"


@pytest.mark.parametrize("name", ["", "Pick", "pick-block", "a" * 65, "../x", "pick block", "é"])
def test_bad_task_names(store, name):
    with pytest.raises(TaskError):
        store.create(name, "", {})


def test_task_exists_and_description_one_line(store):
    store.create("a", "", {})
    with pytest.raises(TaskError, match="already exists"):
        store.create("a", "", {})
    with pytest.raises(TaskError, match="one line"):
        store.create("b", "two\nlines", {})


def test_open_and_tasks_ignore_invalid_folders(store):
    store.create("good", "", {})
    (store.root / "no_yaml").mkdir()
    (store.root / "wrong_name").mkdir()
    (store.root / "wrong_name" / TASK_FILE).write_text("name: other\nnext_episode: 0\n")
    (store.root / "Bad-Name").mkdir()
    assert store.tasks() == ["good"]
    for name in ("no_yaml", "wrong_name", "missing"):
        with pytest.raises(TaskError):
            store.open(name)
    with pytest.raises(TaskError):
        store.open("../good")


def test_tasks_without_data_root(tmp_path):
    assert TaskStore(tmp_path / "nothing").tasks() == []


def test_numbering_counts_up_and_never_reuses(store):
    t = store.create("t", "", {})
    eps = [t.new_episode() for _ in range(3)]
    assert [e.name for e in eps] == ["episode_000", "episode_001", "episode_002"]
    assert all(e.path.is_dir() for e in eps)
    eps[2].finish(meta())
    t.delete("episode_002")                                    # the highest, deleted
    eps[1].discard()
    assert t.new_episode().name == "episode_003"
    assert yaml.safe_load((t.dir / TASK_FILE).read_text())["next_episode"] == 4
    assert [p.name for p in t.dir.iterdir() if p.name.startswith(".")] == []    # no temp file left


def test_numbering_never_below_an_existing_folder(store):
    t = store.create("t", "", {})
    (t.dir / "episode_007").mkdir()                            # e.g. task.yaml restored from a backup
    assert t.new_episode().name == "episode_008"


def test_more_digits_beyond_999(store):
    t = store.create("t", "", {})
    data = yaml.safe_load((t.dir / TASK_FILE).read_text())
    data["next_episode"] = 999
    (t.dir / TASK_FILE).write_text(yaml.safe_dump(data))
    assert [t.new_episode().name, t.new_episode().name] == ["episode_999", "episode_1000"]
    assert episode_name(12345) == "episode_12345"


def test_finish_meta_then_complete_and_listing(store):
    t = store.create("t", "", {})
    e0, e1, e2 = (t.new_episode() for _ in range(3))
    e1.finish(meta("fail", 31.0, 900))
    e0.finish(meta("success", 24.1, 700))
    assert (e0.path / META_FILE).exists() and (e0.path / COMPLETE_FILE).exists()
    assert os.stat(e0.path / COMPLETE_FILE).st_mtime_ns >= os.stat(e0.path / META_FILE).st_mtime_ns
    assert t.episodes() == [EpisodeInfo("episode_000", "success", 24.1, 700),
                            EpisodeInfo("episode_001", "fail", 31.0, 900)]     # e2 not complete: not listed
    with pytest.raises(TaskError, match="already complete"):
        e0.finish(meta())
    e2.discard()
    with pytest.raises(TaskError, match="missing"):
        e2.finish(meta())


def test_broken_meta_not_listed(store):
    t = store.create("t", "", {})
    e = t.new_episode()
    e.finish({"outcome": "maybe"})
    assert t.episodes() == []


def test_delete_rules(store):
    t = store.create("t", "", {})
    e = t.new_episode()
    with pytest.raises(TaskError, match="not complete"):
        t.delete(e.name)
    with pytest.raises(TaskError, match="no such"):
        t.delete("episode_042")
    with pytest.raises(TaskError):
        t.delete("../t")
    e.finish(meta())
    t.delete(e.name)
    assert not e.path.exists() and t.dir.exists()


def test_discard_only_inside_its_task(store, tmp_path):
    t = store.create("t", "", {})
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "file").write_text("keep")
    link = t.dir / "episode_050"
    link.symlink_to(outside)
    with pytest.raises(TaskError, match="symlink"):
        EpisodeDir(t.dir, "episode_050").discard()
    e = t.new_episode()
    e.path = outside                                           # a path that is not its own folder
    with pytest.raises(TaskError, match="not an episode folder"):
        e.discard()
    other = store.create("other", "", {}).new_episode()
    e2 = EpisodeDir(t.dir, "episode_000")
    e2.path = other.path                                       # another task's episode
    with pytest.raises(TaskError):
        e2.discard()
    assert (outside / "file").read_text() == "keep" and other.path.exists()
    with pytest.raises(TaskError):
        EpisodeDir(t.dir, "notes")
    ok = t.new_episode()
    (ok.path / "robot_states.csv").write_text("x")
    ok.discard()
    assert not ok.path.exists()
    ok.discard()                                               # already gone: nothing to do


def test_clean_incomplete(store, tmp_path):
    a = store.create("a", "", {})
    b = store.create("b", "", {})
    a0, a1 = a.new_episode(), a.new_episode()
    b0 = b.new_episode()
    a0.finish(meta())
    (a.dir / "notes").mkdir()                                  # not an episode: kept
    outside = tmp_path / "outside"
    outside.mkdir()
    (b.dir / "episode_009").symlink_to(outside)                # a symlink is never followed or deleted
    deleted = store.clean_incomplete()
    assert sorted(deleted) == sorted([a1.path, b0.path])
    assert a0.path.exists() and not a1.path.exists() and not b0.path.exists()
    assert (a.dir / "notes").exists() and outside.exists()
    assert store.clean_incomplete() == []
    assert a.new_episode().name == "episode_002"


def test_episodes_cached_cheap_for_200_episodes(store):
    """The 20 ms loop may call episodes(): each meta.yaml is parsed once, a repeat call only stats."""
    import time

    t = store.create("t", "", {})
    for i in range(200):
        t.new_episode().finish({**meta("success", float(i), i), "padding": {"x": list(range(200))}})
    first = t.episodes()
    assert len(first) == 200
    start = time.perf_counter()
    second = t.episodes()
    elapsed = time.perf_counter() - start
    assert second == first
    assert elapsed < 0.020, f"second episodes() took {elapsed * 1e3:.1f} ms"


def test_episodes_cache_follows_changes(store, monkeypatch):
    from teleop_collect.teleop.episode import store as store_mod

    t = store.create("t", "", {})
    e0, e1 = t.new_episode(), t.new_episode()
    e0.finish(meta("success", 1.0, 10))
    calls = []
    real = store_mod.read_episode_info
    monkeypatch.setattr(store_mod, "read_episode_info", lambda p: calls.append(p.name) or real(p))
    assert [e.name for e in t.episodes()] == ["episode_000"] and calls == ["episode_000"]
    assert len(t.episodes()) == 1 and calls == ["episode_000"]          # cached: not parsed again
    e1.finish(meta("fail", 2.0, 20))                                     # a new episode: parsed once
    assert [e.outcome for e in t.episodes()] == ["success", "fail"] and calls == ["episode_000", "episode_001"]
    t.delete("episode_000")                                              # a deleted one disappears
    assert [e.name for e in t.episodes()] == ["episode_001"]
    os.remove(e1.path / META_FILE)                                       # a changed meta is re-read
    (e1.path / META_FILE).write_text(yaml.safe_dump(meta("success", 3.0, 30)))
    assert t.episodes() == [EpisodeInfo("episode_001", "success", 3.0, 30)]
