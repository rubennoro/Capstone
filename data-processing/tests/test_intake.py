"""``dataproc intake`` (dataproc/intake.py; rules: docs/DESIGN.md "intake" + docs/INTAKE.md)."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from dataproc import cli
from dataproc.config import load_task_config
from dataproc.intake import LEDGER_COLUMNS, LEDGER_NAME, LOCK_NAME, PENDING_NAME, IntakeError, LockHeldError, \
    read_ledger, run_intake
from dataproc.paths import data_root
from dataproc.raw_episode import COMPLETE, MANIFEST, VERIFIED
from fixtures import UNIX0_NS, build_episode, corrupt_hash, drop_csv_row, make_non_monotonic, read_meta, \
    remove_file


@pytest.fixture
def root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path))
    return data_root()


def _quiet(*a, **k):
    pass


def ep(root, name, *, offset_s: float = 0, **kw) -> Path:
    """A valid verified episode in incoming/ whose button press is ``offset_s`` after the base time."""
    return build_episode(root.incoming / name, start_offset_ns=int(offset_s * 1e9), **kw)


def intake(root, task="block_raise", **kw):
    return run_intake(root, task, log=_quiet, **kw)


def rows(root, folder="block_raise"):
    return read_ledger(root.tasks / folder / LEDGER_NAME)


def tree_hashes(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


def reason(root, name) -> dict:
    return json.loads((root.quarantine / name / "reason.json").read_text())


# --------------------------------------------------------------------------------------------- basic flow

def test_first_intake_creates_task_and_numbers_from_zero(root):
    ep(root, "s01_e2", episode_number=2, offset_s=2)
    ep(root, "s01_e1", episode_number=1, offset_s=1)
    ep(root, "s01_e3", episode_number=3, offset_s=3)
    res = intake(root)
    assert res.created_folder and res.folder == "block_raise"
    t = root.tasks / "block_raise"
    assert [o.slot for o in res.by_status("ingested")] == ["episode_0", "episode_1", "episode_2"]
    # oldest first
    assert [r["pi_name"] for r in rows(root)] == ["s01_e1", "s01_e2", "s01_e3"]
    for n in range(3):
        assert (t / f"episode_{n}" / "raw" / "meta.json").is_file()
    assert os.listdir(root.incoming) == []
    cfg = load_task_config(t / "task.yaml")
    assert cfg.label == "block_raise" and cfg.rig == "rig_v1" and cfg.notes == ""
    assert cfg.created.endswith("+00:00")
    assert not list(t.glob(".task.yaml.tmp*"))


def test_later_intake_continues_numbering(root):
    ep(root, "a", episode_number=1)
    intake(root)
    ep(root, "b", episode_number=2, offset_s=5)
    ep(root, "c", episode_number=3, offset_s=6)
    res = intake(root)
    assert not res.created_folder
    assert [o.slot for o in res.by_status("ingested")] == ["episode_1", "episode_2"]
    assert [r["slot"] for r in rows(root)] == ["episode_0", "episode_1", "episode_2"]


def test_numbering_skips_gaps_and_any_entry(root):
    ep(root, "a", episode_number=1)
    intake(root)
    (root.tasks / "block_raise" / "episode_7").write_text("")       # any episode_N entry counts as taken
    ep(root, "b", episode_number=2)
    res = intake(root)
    assert res.outcomes[0].slot == "episode_8"


def test_new_creates_suffixed_folders(root):
    ep(root, "a", episode_number=1)
    intake(root)
    ep(root, "b", episode_number=2)
    r1 = intake(root, new=True)
    assert r1.folder == "block_raise_2" and r1.outcomes[0].slot == "episode_0"
    ep(root, "c", episode_number=3)
    r2 = intake(root, new=True)
    assert r2.folder == "block_raise_3"
    for f in ("block_raise_2", "block_raise_3"):
        assert load_task_config(root.tasks / f / "task.yaml").label == "block_raise"


def test_new_with_nothing_waiting_creates_nothing(root, capsys):
    ep(root, "a", episode_number=1)
    intake(root)
    before = sorted(os.listdir(root.tasks))
    res = intake(root, new=True)
    assert res.nothing_waiting and res.folder is None
    assert sorted(os.listdir(root.tasks)) == before
    assert not root.quarantine.exists()


def test_nothing_waiting_creates_nothing(root, capsys):
    root.incoming.mkdir(parents=True)
    assert cli.main(["intake", "--task", "block_raise"]) == 0
    assert "nothing waiting" in capsys.readouterr().out
    assert not root.tasks.exists() and not root.quarantine.exists()


def test_incoming_missing(root, capsys):
    root.path.mkdir(parents=True, exist_ok=True)
    assert cli.main(["intake", "--task", "block_raise"]) == 0
    assert "incoming/ does not exist" in capsys.readouterr().out
    assert not root.tasks.exists()


def test_root_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATAPROC_ROOT", str(tmp_path / "nope"))
    assert cli.main(["intake", "--task", "block_raise"]) == 0
    assert not (tmp_path / "nope").exists()


# --------------------------------------------------------------------------------------------- left in incoming

def test_unverified_left_in_place(root):
    ep(root, "a", verified=False)
    res = intake(root)
    assert res.nothing_waiting
    assert (root.incoming / "a").is_dir() and not root.tasks.exists()
    assert "not verified" in res.outcomes[0].describe()


def test_dot_temp_file_means_transfer_in_progress(root, capsys):
    d = ep(root, "a", episode_number=1)
    (d / ".cam0.mp4.Xy12ab").write_bytes(b"partial")
    ep(root, "b", episode_number=2)
    assert cli.main(["intake", "--task", "block_raise"]) == 0
    out = capsys.readouterr().out
    assert "a  ->  skipped: transfer in progress" in out
    assert (d / ".cam0.mp4.Xy12ab").exists() and (d / VERIFIED).exists()
    assert [r["pi_name"] for r in rows(root)] == ["b"]


def test_dot_dirs_in_incoming_ignored(root):
    (root.incoming / ".a.tmp").mkdir(parents=True)
    (root.incoming / ".a.tmp" / VERIFIED).touch()
    res = intake(root)
    assert res.nothing_waiting and res.outcomes == []
    assert (root.incoming / ".a.tmp").is_dir()


def test_plain_file_in_incoming_left(root):
    root.incoming.mkdir(parents=True)
    (root.incoming / "notes.txt").write_text("x")
    res = intake(root)
    assert res.nothing_waiting and (root.incoming / "notes.txt").exists()


# --------------------------------------------------------------------------------------------- quarantine

@pytest.mark.parametrize("corrupt, code", [
    (lambda d: corrupt_hash(d), "hash_mismatch"),
    (lambda d: drop_csv_row(d), "count_mismatch"),
    (lambda d: make_non_monotonic(d), "t_ns_not_monotonic"),
    (lambda d: remove_file(d, "imu.csv"), "missing_file"),
    (lambda d: remove_file(d, COMPLETE), "incomplete"),
    (lambda d: (d / "extra.log").write_text("x"), "extra_file"),
    (lambda d: (d / "cam1.mp4").write_bytes(b"not a video") or _remanifest(d), "video_unreadable"),
])
def test_validation_failure_quarantined(root, corrupt, code):
    d = ep(root, "bad", episode_number=1)
    corrupt(d)
    ep(root, "good", episode_number=2, offset_s=1)
    res = intake(root)
    q = res.by_status("quarantined")
    assert [o.pi_name for o in q] == ["bad"] and code in q[0].reasons
    r = reason(root, "bad")
    assert code in r["reasons"] and r["report"]["ok"] is False
    assert code in {p["code"] for p in r["report"]["problems"]}
    assert r["dataproc_version"] and r["quarantined_at"] and r["pi_name"] == "bad"
    assert (root.quarantine / "bad" / "raw" / "meta.json").exists()
    assert [x["pi_name"] for x in rows(root)] == ["good"]
    assert res.by_status("ingested")[0].slot == "episode_0"


def _remanifest(d):
    from fixtures import write_manifest
    write_manifest(d)


def test_invalid_meta_quarantined_first(root):
    d = ep(root, "z_badmeta", episode_number=1)
    (d / "meta.json").write_text("{not json")
    _remanifest(d)
    ep(root, "a_good", episode_number=2)
    res = intake(root)
    assert [o.pi_name for o in res.outcomes] == ["z_badmeta", "a_good"]
    assert "meta_invalid" in res.outcomes[0].reasons


def test_all_quarantined_creates_no_task_folder(root, capsys):
    corrupt_hash(ep(root, "bad"))
    assert cli.main(["intake", "--task", "block_raise"]) == 0
    assert "bad  ->  quarantined: hash_mismatch" in capsys.readouterr().out
    assert not root.tasks.exists() or not (root.tasks / "block_raise").exists()


def test_every_reason_recorded(root):
    ep(root, "a", episode_number=1)
    intake(root)
    d = ep(root, "a2", episode_number=1, rig_id="rig_v2")          # same pair, other rig, bad hash
    corrupt_hash(d)
    intake(root)
    r = reason(root, "a2")
    assert r["reasons"] == ["hash_mismatch", "duplicate", "rig_mismatch"]
    assert r["details"]["duplicate"]["session_episode"]["taken_by"] == "block_raise/episode_0"
    assert r["details"]["rig_mismatch"] == {"episode_rig": "rig_v2", "task_rig": "rig_v1"}


def test_duplicate_name_across_tasks(root):
    ep(root, "a", session_id="s01", episode_number=1)
    intake(root, task="other_task")
    ep(root, "a", session_id="s99", episode_number=7)
    res = intake(root)
    assert res.outcomes[0].reasons == ("duplicate",)
    assert reason(root, "a")["details"]["duplicate"]["pi_name"]["taken_by"] == "other_task/episode_0"
    assert not (root.tasks / "block_raise").exists()


def test_duplicate_pair_across_tasks(root):
    ep(root, "a", session_id="s01", episode_number=4)
    intake(root, task="other_task")
    ep(root, "b", session_id="s01", episode_number=4)
    assert intake(root).outcomes[0].reasons == ("duplicate",)


def test_duplicate_within_one_run(root):
    ep(root, "a", session_id="s01", episode_number=4, offset_s=0)
    ep(root, "b", session_id="s01", episode_number=4, offset_s=1)
    res = intake(root)
    assert [o.describe() for o in res.outcomes] == ["episode_0", "quarantined: duplicate"]
    assert reason(root, "b")["details"]["duplicate"]["session_episode"]["taken_by"] == "block_raise/episode_0"


def test_rig_mismatch_against_task_yaml(root):
    ep(root, "a", episode_number=1)
    intake(root)
    ep(root, "b", episode_number=2, rig_id="rig_v2")
    assert intake(root).outcomes[0].reasons == ("rig_mismatch",)


def test_rig_mismatch_within_first_run(root):
    ep(root, "a", episode_number=1, offset_s=0)                     # oldest: defines the rig
    ep(root, "b", episode_number=2, offset_s=1, rig_id="rig_v2")
    res = intake(root)
    assert [o.describe() for o in res.outcomes] == ["episode_0", "quarantined: rig_mismatch"]
    assert load_task_config(root.tasks / "block_raise" / "task.yaml").rig == "rig_v1"


def test_quarantine_name_collision(root):
    for k in range(3):
        corrupt_hash(ep(root, "x", episode_number=1))
        intake(root)
    assert sorted(os.listdir(root.quarantine)) == ["x", "x__2", "x__3"]
    for q in ("x", "x__2", "x__3"):
        assert (root.quarantine / q / "raw" / MANIFEST).exists()
        assert (root.quarantine / q / "reason.json").exists()


def test_symlink_episode_quarantined(root, tmp_path):
    real = build_episode(tmp_path / "elsewhere" / "real")
    root.incoming.mkdir(parents=True)
    (root.incoming / "link").symlink_to(real)
    (root.incoming / "dangling").symlink_to(tmp_path / "missing")
    res = intake(root)
    by = {o.pi_name: o for o in res.outcomes}
    assert "episode_is_symlink" in by["link"].reasons and "episode_is_symlink" in by["dangling"].reasons
    assert (root.quarantine / "link" / "raw").is_symlink()
    assert (real / MANIFEST).exists() and not (real / "reason.json").exists()   # target untouched


def test_bad_pi_dir_name_quarantined(root):
    ep(root, "bad name!", episode_number=1)
    res = intake(root)
    assert "bad_episode_name" in res.outcomes[0].reasons
    assert (root.quarantine / "bad name!" / "raw").is_dir()


# --------------------------------------------------------------------------------------------- calibration

def test_calibration_takes_exactly_one(root):
    corrupt_hash(ep(root, "c0_bad", episode_number=0, offset_s=0))
    ep(root, "c1", episode_number=1, offset_s=1)
    ep(root, "c2", episode_number=2, offset_s=2)
    corrupt_hash(ep(root, "c3_bad", episode_number=3, offset_s=3))
    res = intake(root, calibration=True)
    assert [o.describe() for o in res.outcomes][:2] == ["quarantined: hash_mismatch", "calibration/c1"]
    assert [o.status for o in res.outcomes[2:]] == ["waiting", "waiting"]
    t = root.tasks / "block_raise"
    assert (t / "calibration" / "c1" / "raw" / "meta.json").exists()
    assert (t / "task.yaml").exists() and not list(t.glob("episode_*"))
    assert sorted(os.listdir(root.incoming)) == ["c2", "c3_bad"]
    assert sorted(os.listdir(root.quarantine)) == ["c0_bad"]
    (r,) = rows(root)
    assert r["kind"] == "calibration" and r["slot"] == "calibration/c1" and r["pi_name"] == "c1"
    # the next intake (demonstrations) picks up the rest and numbers from 0
    res = intake(root)
    assert [o.describe() for o in res.outcomes] == ["episode_0", "quarantined: hash_mismatch"]


def test_calibration_checks_duplicate_and_rig(root):
    ep(root, "a", episode_number=1)
    intake(root)
    ep(root, "a2", episode_number=1, offset_s=0)                    # duplicate pair
    ep(root, "b", episode_number=2, offset_s=1, rig_id="rig_v2")    # rig mismatch
    ep(root, "c", episode_number=3, offset_s=2)
    res = intake(root, calibration=True)
    assert [o.describe() for o in res.outcomes] == ["quarantined: duplicate", "quarantined: rig_mismatch",
                                                    "calibration/c"]


def test_calibration_name_collision(root):
    ep(root, "c", episode_number=1)
    intake(root, calibration=True)
    cal = root.tasks / "block_raise" / "calibration"
    (cal / "d").mkdir()                                             # a stray dir that is not in the ledger
    ep(root, "d", episode_number=2)
    res = intake(root, calibration=True)
    assert res.outcomes[0].slot == "calibration/d__2"


# --------------------------------------------------------------------------------------------- ledger, moves

def test_ledger_columns_and_values(root):
    d = ep(root, "s01_e5", session_id="s01", episode_number=5, offset_s=3)
    manifest_sha = hashlib.sha256((d / MANIFEST).read_bytes()).hexdigest()
    start = read_meta(d)
    intake(root)
    with open(root.tasks / "block_raise" / LEDGER_NAME, newline="") as f:
        lines = list(csv.reader(f))
    assert tuple(lines[0]) == LEDGER_COLUMNS
    r = dict(zip(lines[0], lines[1]))
    assert r["slot"] == "episode_0" and r["pi_name"] == "s01_e5" and r["session_id"] == "s01"
    assert r["episode_number"] == "5" and r["kind"] == "demonstration" and r["rig_id"] == "rig_v1"
    expected_start = UNIX0_NS + start["button_start_ns"] - start["wall_anchor"]["clock_ns"]
    assert int(r["start_unix_ns"]) == expected_start
    assert abs(int(r["intake_unix_ns"]) - __import__("time").time_ns()) < 60e9
    assert r["manifest_sha256"] == manifest_sha


def test_raw_files_byte_identical_after_move(root):
    d = ep(root, "a", episode_number=1)
    before = tree_hashes(d)
    st = {p.name: os.stat(p).st_ino for p in d.iterdir()}
    intake(root)
    raw = root.tasks / "block_raise" / "episode_0" / "raw"
    assert tree_hashes(raw) == before
    assert {p.name: os.stat(p).st_ino for p in raw.iterdir()} == st     # renamed, not copied
    assert sorted(os.listdir(raw.parent)) == ["raw"]


def test_exdev_errors_without_copy(root, monkeypatch, capsys):
    import errno
    ep(root, "a", episode_number=1)
    real_rename = os.rename

    def fake(src, dst):
        if Path(dst).name == "raw":
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", fake)
    assert cli.main(["intake", "--task", "block_raise"]) == 1
    assert "EXDEV" in capsys.readouterr().err
    assert (root.incoming / "a" / "meta.json").exists()
    assert not (root.tasks / "block_raise" / "episode_0" / "raw").exists()


def test_reconcile_after_crash(root, capsys):
    ep(root, "a", episode_number=1)
    intake(root)
    # simulate a crash after the rename of "b" but before its ledger row
    d = ep(root, "b", session_id="s02", episode_number=9)
    t = root.tasks / "block_raise"
    (t / PENDING_NAME).write_text(json.dumps({"slot": "episode_1", "pi_name": "b", "kind": "demonstration"}))
    (t / "episode_1").mkdir()
    os.rename(d, t / "episode_1" / "raw")
    # and a calibration move with no row and no pending record
    build_episode(t / "calibration" / "cal_x" / "raw", session_id="s03", episode_number=0)
    assert cli.main(["intake", "--task", "block_raise"]) == 0
    out = capsys.readouterr().out
    assert "reconciled: block_raise/episode_1" in out and "reconciled: block_raise/calibration/cal_x" in out
    rs = {r["slot"]: r for r in rows(root)}
    assert rs["episode_1"]["pi_name"] == "b" and rs["episode_1"]["session_id"] == "s02"
    assert rs["episode_1"]["episode_number"] == "9" and rs["episode_1"]["kind"] == "demonstration"
    assert rs["calibration/cal_x"]["kind"] == "calibration" and rs["calibration/cal_x"]["pi_name"] == "cal_x"
    assert len(rs["episode_1"]["manifest_sha256"]) == 64
    # reconcile is idempotent and the rebuilt rows count for dedupe
    ep(root, "b", session_id="s77", episode_number=1)
    res = intake(root)
    assert res.reconciled == [] and res.outcomes[0].reasons == ("duplicate",)


def test_reconcile_in_other_task_folders(root):
    ep(root, "a", episode_number=1)
    intake(root, task="other")
    t = root.tasks / "other"
    (t / "episode_1").mkdir()
    os.rename(ep(root, "b", episode_number=2), t / "episode_1" / "raw")
    (t / PENDING_NAME).write_text(json.dumps({"slot": "episode_1", "pi_name": "b"}))
    res = intake(root)
    assert res.reconciled == [("other", "episode_1")]


def test_lock_held_exits_1(root, capsys):
    root.path.mkdir(parents=True, exist_ok=True)
    ep(root, "a", episode_number=1)
    fd = os.open(root.path / LOCK_NAME, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert cli.main(["intake", "--task", "block_raise"]) == 1
        assert "another intake is running" in capsys.readouterr().err
        with pytest.raises(LockHeldError):
            intake(root)
    finally:
        os.close(fd)
    assert (root.incoming / "a").is_dir() and not root.tasks.exists()
    assert intake(root).outcomes[0].slot == "episode_0"                 # released -> runs


def test_target_is_a_file_exits_1(root, capsys):
    root.tasks.mkdir(parents=True)
    (root.tasks / "block_raise").write_text("")
    ep(root, "a")
    assert cli.main(["intake", "--task", "block_raise"]) == 1
    assert "not a directory" in capsys.readouterr().err
    assert (root.incoming / "a").is_dir()


def test_target_with_bad_task_yaml_exits_1(root):
    (root.tasks / "block_raise").mkdir(parents=True)
    (root.tasks / "block_raise" / "task.yaml").write_text("label: [\n")
    ep(root, "a")
    with pytest.raises(IntakeError):
        intake(root)


def test_existing_empty_folder_gets_task_yaml(root):
    (root.tasks / "block_raise").mkdir(parents=True)                # e.g. crash between mkdir and task.yaml
    ep(root, "a")
    res = intake(root)
    assert res.outcomes[0].slot == "episode_0"
    assert load_task_config(root.tasks / "block_raise" / "task.yaml").rig == "rig_v1"


def test_label_is_the_given_name(root):
    ep(root, "a")
    intake(root, task="pick-cube")
    assert load_task_config(root.tasks / "pick-cube" / "task.yaml").label == "pick-cube"


# --------------------------------------------------------------------------------------------- CLI

def test_cli_end_to_end(root, capsys):
    ep(root, "s01_e1", episode_number=1, offset_s=0)
    corrupt_hash(ep(root, "s01_e2", episode_number=2, offset_s=1))
    ep(root, "s01_e3", episode_number=3, offset_s=2, verified=False)
    assert cli.main(["intake", "--task", "block_raise"]) == 0
    out = capsys.readouterr().out
    assert "s01_e1  ->  episode_0" in out
    assert "s01_e2  ->  quarantined: hash_mismatch" in out
    assert "s01_e3  ->  skipped: not verified" in out
    assert "1 ingested into tasks/block_raise (created), 1 quarantined, 1 left in incoming" in out
    (root.incoming / "s01_e3" / VERIFIED).touch()
    assert cli.main(["intake", "--task", str(root.tasks / "block_raise"), "--new"]) == 0
    assert "s01_e3  ->  episode_0" in capsys.readouterr().out
    assert (root.tasks / "block_raise_2" / "episode_0" / "raw").is_dir()


def test_cli_calibration(root, capsys):
    ep(root, "c1", episode_number=1)
    assert cli.main(["intake", "--task", "block_raise", "--calibration"]) == 0
    assert "c1  ->  calibration/c1" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------- review round 1
# (ported from the R-intake review: crash injection, permissions, races, ordering)

import subprocess
import sys

import dataproc.intake as I
from dataproc.raw_episode import RawEpisode

needs_non_root = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")


def _state(root):
    inc = sorted(os.listdir(root.incoming)) if root.incoming.exists() else []
    led, raws = {}, {}
    if root.tasks.exists():
        for t in sorted(root.tasks.iterdir()):
            led[t.name] = [(r["slot"], r["pi_name"]) for r in read_ledger(t / LEDGER_NAME)]
            raws[t.name] = sorted(str(p.relative_to(t)) for p in t.rglob("raw"))
    q = sorted(os.listdir(root.quarantine)) if root.quarantine.exists() else []
    return inc, led, raws, q


class _Boom(Exception):
    pass


@pytest.mark.parametrize("point, which", [(p, w) for p in ("pending", "mkdir", "rename", "ledger", "task_yaml")
                                          for w in (0, 1) if (p, w) != ("task_yaml", 1)])
def test_crash_then_rerun_consistent(root, monkeypatch, point, which):
    """A crash at any step of a move, then a rerun: every episode ends up ingested exactly once."""
    for i, n in enumerate(["a", "b", "c"]):
        ep(root, n, offset_s=i, episode_number=i)
    calls = {"n": 0}
    real_wa, real_mkdir, real_rename, real_ledger, real_osr = (I.fsutil.write_atomic, os.mkdir, I._move,
                                                               I.Ledger.append, os.rename)

    def maybe(kind):
        if kind == point:
            calls["n"] += 1
            if calls["n"] == which + 1:
                raise _Boom(point)

    def wa(path, text):
        if path.name == PENDING_NAME:
            maybe("pending")
        return real_wa(path, text)

    def mk(p, *a, **k):
        if Path(p).name.startswith("episode_"):
            maybe("mkdir")
        return real_mkdir(p, *a, **k)

    def rn(src, dst):
        if dst.name == "raw" and "tasks" in str(dst):
            maybe("rename")
        return real_rename(src, dst)

    def al(ledger, row):
        maybe("ledger")
        return real_ledger(ledger, row)

    def osr(src, dst):
        if Path(dst).name == "task.yaml":
            maybe("task_yaml")
        return real_osr(src, dst)

    monkeypatch.setattr(I.fsutil, "write_atomic", wa)
    monkeypatch.setattr(I.os, "mkdir", mk)
    monkeypatch.setattr(I, "_move", rn)
    monkeypatch.setattr(I.Ledger, "append", al)
    monkeypatch.setattr(I.os, "rename", osr)
    with pytest.raises(_Boom):
        intake(root)
    monkeypatch.undo()
    intake(root)
    inc, led, raws, q = _state(root)
    assert sorted(n for _, n in led["block_raise"]) == ["a", "b", "c"]
    assert q == [] and inc == [] and len(raws["block_raise"]) == 3
    manifests = [hashlib.sha256(m.read_bytes()).hexdigest() for m in root.path.rglob(MANIFEST)]
    assert len(manifests) == len(set(manifests)) == 3


def test_failed_quarantine_rename_leaves_no_orphan(root, monkeypatch):
    corrupt_hash(ep(root, "bad"))
    real = I._move

    def rn(src, dst):
        if "quarantine" in str(dst):
            raise OSError(13, "Permission denied")
        return real(src, dst)

    monkeypatch.setattr(I, "_move", rn)
    for _ in range(2):
        with pytest.raises(OSError):
            intake(root)
    assert os.listdir(root.quarantine) == []
    monkeypatch.undo()
    intake(root)
    assert os.listdir(root.quarantine) == ["bad"]
    assert reason(root, "bad")["reasons"] == ["hash_mismatch"]


def test_failed_ingest_rename_does_not_burn_a_number(root, monkeypatch):
    ep(root, "a")
    real = I._move

    def rn(src, dst):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(I, "_move", rn)
    with pytest.raises(OSError):
        intake(root)
    monkeypatch.undo()
    assert intake(root).outcomes[0].slot == "episode_0"


def test_interrupted_quarantine_gets_reason_on_reconcile(root):
    q = root.quarantine / "x"
    build_episode(q / "raw")                                          # moved, reason.json never written
    root.incoming.mkdir()
    intake(root)
    r = reason(root, "x")
    assert r["reasons"] == ["interrupted"] and r["episode"] == "raw/"


@needs_non_root
def test_readonly_episode_dir_skipped(root, capsys):
    ep(root, "a", episode_number=1, offset_s=0)
    ep(root, "b", episode_number=2, offset_s=1)
    os.chmod(root.incoming / "a", 0o555)
    try:
        assert cli.main(["intake", "--task", "block_raise"]) == 0
        out = capsys.readouterr().out
        assert "a  ->  skipped: permission denied (owner uid" in out and "mode dr-xr-xr-x" in out
        assert _state(root)[2] == {"block_raise": ["episode_0/raw"]}
        assert [r["pi_name"] for r in rows(root)] == ["b"]
    finally:
        os.chmod(root.incoming / "a", 0o755)
    assert intake(root).outcomes[0].slot == "episode_1"


@needs_non_root
def test_readonly_episode_dir_alone_creates_nothing(root):
    ep(root, "a")
    os.chmod(root.incoming / "a", 0o555)
    try:
        for _ in range(2):
            res = intake(root)
            assert res.nothing_waiting and "permission denied" in res.outcomes[0].note
        assert not root.tasks.exists() and not root.quarantine.exists()
    finally:
        os.chmod(root.incoming / "a", 0o755)


@needs_non_root
@pytest.mark.parametrize("which", ["incoming", "tasks", "quarantine", "root"])
def test_readonly_root_dirs_exit_1_and_create_nothing(root, capsys, which):
    ep(root, "a")
    corrupt_hash(ep(root, "bad", episode_number=7))
    d = {"incoming": root.incoming, "tasks": root.tasks, "quarantine": root.quarantine, "root": root.path}[which]
    d.mkdir(parents=True, exist_ok=True)
    (root.path / LOCK_NAME).touch()
    before = sorted(str(p.relative_to(root.path)) for p in root.path.rglob("*"))
    os.chmod(d, 0o555)
    try:
        for _ in range(2):
            assert cli.main(["intake", "--task", "block_raise"]) == 1
            assert "permission denied" in capsys.readouterr().err
    finally:
        os.chmod(d, 0o755)
    assert sorted(str(p.relative_to(root.path)) for p in root.path.rglob("*")) == before


def test_dot_file_appearing_during_validation_skips(root, monkeypatch):
    corrupt_hash(ep(root, "a"))                                       # even a failing episode is not quarantined
    ep(root, "b", episode_number=2, offset_s=1)
    real = RawEpisode.validate

    def v(self, *a, **k):
        if self.path.name in ("a", "b"):
            (self.path / ".cam0.mp4.XyZ123").write_bytes(b"partial")
        return real(self, *a, **k)

    monkeypatch.setattr(RawEpisode, "validate", v)
    res = intake(root)
    assert [o.describe() for o in res.outcomes] == ["skipped: transfer in progress"] * 2
    assert sorted(os.listdir(root.incoming)) == ["a", "b"]
    assert not root.tasks.exists() and not root.quarantine.exists()


def test_error_mid_run_prints_partial_summary(root, monkeypatch, capsys):
    ep(root, "a", episode_number=1, offset_s=0)
    ep(root, "b", episode_number=2, offset_s=1)
    ep(root, "c", episode_number=3, offset_s=2)
    real = I._move

    def rn(src, dst):
        if Path(src).name == "b":
            raise IntakeError("cannot move b (EXDEV)")
        return real(src, dst)

    monkeypatch.setattr(I, "_move", rn)
    assert cli.main(["intake", "--task", "block_raise"]) == 1
    out, err = capsys.readouterr()
    assert "a  ->  episode_0" in out and "before the error: 1 ingested" in out
    assert "stopped at b; nothing after it was processed" in err
    assert sorted(os.listdir(root.incoming)) == ["b", "c"]


def test_unexpected_error_mid_run(root, monkeypatch, capsys):
    ep(root, "a")
    monkeypatch.setattr(I.Ledger, "append", lambda *a: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    assert cli.main(["intake", "--task", "block_raise"]) == 1
    err = capsys.readouterr().err
    assert "unexpected error: disk on fire" in err and "stopped at a" in err


def test_concurrent_processes(root):
    for i in range(6):
        ep(root, f"e{i}", offset_s=i, episode_number=i)
    code = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); "
            "from dataproc.cli import main; sys.exit(main(['intake', '--task', 'block_raise']))")
    ps = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          for _ in range(4)]
    rcs = [p.wait() for p in ps]
    for p in ps:
        p.stdout.close()
        p.stderr.close()
    inc, led, _, q = _state(root)
    assert sorted(n for _, n in led["block_raise"]) == [f"e{i}" for i in range(6)]
    assert q == [] and inc == [] and set(rcs) <= {0, 1} and 0 in rcs


def test_folder_with_unledgered_episode_counts_for_dedupe(root):
    from dataproc.config import TaskConfig, dump_task_config
    t = root.tasks / "block_raise"
    build_episode(t / "episode_0" / "raw", session_id="old", episode_number=0)
    dump_task_config(TaskConfig("block_raise", "rig_v1", "2026-01-01T00:00:00"), t / "task.yaml")
    ep(root, "a", session_id="old", episode_number=0)
    res = intake(root)
    assert res.reconciled == [("block_raise", "episode_0")] and res.outcomes[0].reasons == ("duplicate",)


def test_ties_ordering(root):
    for n, k in (("z", 1), ("a", 3), ("m", 2)):
        ep(root, n, session_id="s", episode_number=k)
    assert {o.pi_name: o.slot for o in intake(root).outcomes} == {"z": "episode_0", "m": "episode_1",
                                                                  "a": "episode_2"}


def test_calibration_all_fail_creates_no_folder(root):
    corrupt_hash(ep(root, "x"))
    res = intake(root, calibration=True)
    assert res.outcomes[0].status == "quarantined" and not root.tasks.exists()


def test_bad_ledger_in_other_folder_exits_1(root, capsys):
    (root.tasks / "other").mkdir(parents=True)
    (root.tasks / "other" / LEDGER_NAME).write_text("garbage\n")
    ep(root, "a")
    assert cli.main(["intake", "--task", "block_raise"]) == 1
    assert "unexpected header" in capsys.readouterr().err
    assert (root.incoming / "a").is_dir()


# --------------------------------------------------------------------------------------------- review round 2

def _tree(root):
    return sorted(str(p.relative_to(root.path)) for p in root.path.rglob("*"))


@needs_non_root
def test_readonly_existing_target_folder(root, capsys):
    ep(root, "a", episode_number=1)
    intake(root)
    ep(root, "b", episode_number=2, offset_s=1)
    corrupt_hash(ep(root, "bad", episode_number=3, offset_s=2))
    t = root.tasks / "block_raise"
    before = _tree(root)
    os.chmod(t, 0o555)
    try:
        assert cli.main(["intake", "--task", "block_raise"]) == 1
        err = capsys.readouterr().err
        assert "permission denied" in err and f"effective uid {os.geteuid()}" in err
    finally:
        os.chmod(t, 0o755)
    assert _tree(root) == before


@needs_non_root
def test_readonly_calibration_subdir(root, capsys):
    ep(root, "a", episode_number=1)
    intake(root)
    cal = root.tasks / "block_raise" / "calibration"
    cal.mkdir()
    ep(root, "c", episode_number=2, offset_s=1)
    os.chmod(cal, 0o555)
    try:
        assert cli.main(["intake", "--task", "block_raise", "--calibration"]) == 1
        assert "calibration/ folder" in capsys.readouterr().err
    finally:
        os.chmod(cal, 0o755)
    assert (root.incoming / "c").is_dir() and os.listdir(cal) == []


@needs_non_root
def test_readonly_quarantine_checked_before_reconcile(root, capsys):
    build_episode(root.quarantine / "x" / "raw")                     # reconcile would write reason.json here
    ep(root, "a")
    os.chmod(root.quarantine / "x", 0o555)
    os.chmod(root.quarantine, 0o555)
    try:
        assert cli.main(["intake", "--task", "block_raise"]) == 1
        err = capsys.readouterr().err
    finally:
        os.chmod(root.quarantine, 0o755)
        os.chmod(root.quarantine / "x", 0o755)
    assert "permission denied" in err and "Traceback" not in err


@needs_non_root
def test_reconcile_permission_error_is_clean(root, capsys):
    build_episode(root.quarantine / "x" / "raw")
    ep(root, "a")
    os.chmod(root.quarantine / "x", 0o555)                           # quarantine/ itself writable
    try:
        assert cli.main(["intake", "--task", "block_raise"]) == 1
        err = capsys.readouterr().err
    finally:
        os.chmod(root.quarantine / "x", 0o755)
    assert "while reconciling" in err and "Traceback" not in err


def test_quarantine_reason_write_fails_then_reconcile(root, monkeypatch):
    corrupt_hash(ep(root, "bad"))
    real = I.fsutil.write_atomic

    def wa(path, text):
        if path.name == I.REASON_NAME:
            raise OSError(28, "No space left")
        return real(path, text)

    monkeypatch.setattr(I.fsutil, "write_atomic", wa)
    with pytest.raises(OSError):
        intake(root)
    assert os.listdir(root.quarantine / "bad") == ["raw"]
    monkeypatch.undo()
    intake(root)
    assert reason(root, "bad")["reasons"] == ["interrupted"] and os.listdir(root.quarantine) == ["bad"]


def test_interrupt_after_successful_rename_keeps_data(root, monkeypatch):
    ep(root, "a")
    real = I._move

    def rn(src, dst):
        real(src, dst)
        raise KeyboardInterrupt

    monkeypatch.setattr(I, "_move", rn)
    with pytest.raises(KeyboardInterrupt):
        intake(root)
    monkeypatch.undo()
    t = root.tasks / "block_raise"
    assert (t / "episode_0" / "raw" / "meta.json").is_file() and (t / "task.yaml").is_file()
    res = intake(root)
    assert res.reconciled == [("block_raise", "episode_0")] and rows(root)[0]["pi_name"] == "a"


def test_episode_mkdir_fails(root, monkeypatch):
    ep(root, "a")
    real = os.mkdir

    def mk(p, *a, **k):
        if Path(p).name.startswith("episode_"):
            raise OSError(28, "No space left")
        return real(p, *a, **k)

    monkeypatch.setattr(I.os, "mkdir", mk)
    with pytest.raises(OSError):
        intake(root)
    monkeypatch.undo()
    assert not root.tasks.exists() or os.listdir(root.tasks) == []
    assert intake(root).outcomes[0].slot == "episode_0"


@pytest.mark.parametrize("new, calibration", [(False, False), (True, False), (False, True), (True, True)])
def test_failed_first_ingest_rolls_back_task_folder(root, monkeypatch, capsys, new, calibration):
    """A failed first move leaves no task folder/task.yaml created by this run (also a --new _k folder)."""
    t = root.tasks / "block_raise"
    if new:
        t.mkdir(parents=True)
        (t / "task.yaml").write_text("label: block_raise\nrig: rig_v1\ncreated: x\nnotes: ''\n")
    before = _tree(root)
    ep(root, "a")
    (root.incoming / "u").mkdir()                                    # unverified: gives the summary a skip line
    monkeypatch.setattr(I, "_move", lambda s, d: (_ for _ in ()).throw(OSError(5, "EIO")))
    argv = ["intake", "--task", "block_raise"] + (["--new"] if new else []) + (["--calibration"] if calibration
                                                                               else [])
    assert cli.main(argv) == 1
    out, err = capsys.readouterr()
    assert [x for x in _tree(root) if not x.startswith("incoming") and x != LOCK_NAME] == \
        [x for x in before if not x.startswith("incoming")]
    assert "nothing waiting" not in out and "no task folder created" not in out
    assert "a  ->  error: [Errno 5] EIO (left in incoming)" in out
    assert "(created)" not in out and "stopped at a" in err


def test_failed_ingest_into_existing_folder_keeps_it(root, monkeypatch):
    ep(root, "a", episode_number=1)
    intake(root)
    ep(root, "b", episode_number=2)
    monkeypatch.setattr(I, "_move", lambda s, d: (_ for _ in ()).throw(OSError(5, "EIO")))
    with pytest.raises(OSError):
        intake(root)
    t = root.tasks / "block_raise"
    assert (t / "task.yaml").is_file() and sorted(p.name for p in t.glob("episode_*")) == ["episode_0"]


def test_partial_summary_when_only_skipped_before_error(root, monkeypatch, capsys):
    (root.incoming / "u").mkdir(parents=True)
    ep(root, "a")
    monkeypatch.setattr(I.Ledger, "append", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main(["intake", "--task", "block_raise"]) == 1
    out, err = capsys.readouterr()
    assert "stopped at a" in err and "nothing waiting" not in out
    assert "a  ->  error: boom (already moved out of incoming" in out
    assert "before the error: 0 ingested into tasks/block_raise (created), 0 quarantined, 1 left in incoming, " \
           "1 failed" in out


def test_dot_appears_during_validation_only_for_one(root, monkeypatch):
    ep(root, "a", episode_number=1)
    ep(root, "b", episode_number=2, offset_s=1)
    real = RawEpisode.validate

    def v(self, *a, **k):
        r = real(self, *a, **k)
        if self.path.name == "b":
            (self.path / ".imu.csv.Ab12").write_bytes(b"x")
        return r

    monkeypatch.setattr(RawEpisode, "validate", v)
    res = intake(root)
    assert [o.describe() for o in res.outcomes] == ["episode_0", "skipped: transfer in progress"]


def test_dot_appears_during_validation_calibration_not_consumed(root, monkeypatch):
    ep(root, "a", episode_number=1)
    ep(root, "b", episode_number=2, offset_s=1)
    real = RawEpisode.validate

    def v(self, *a, **k):
        r = real(self, *a, **k)
        if self.path.name == "a":
            (self.path / ".x").write_bytes(b"x")
        return r

    monkeypatch.setattr(RawEpisode, "validate", v)
    res = intake(root, calibration=True)
    assert [o.describe() for o in res.outcomes] == ["skipped: transfer in progress", "calibration/b"]


def test_dir_vanishes_during_validation_no_orphan(root, monkeypatch, tmp_path):
    corrupt_hash(ep(root, "a"))
    real = RawEpisode.validate

    def v(self, *a, **k):
        r = real(self, *a, **k)
        os.rename(self.path, tmp_path / "elsewhere")
        return r

    monkeypatch.setattr(RawEpisode, "validate", v)
    with pytest.raises(FileNotFoundError):
        intake(root)
    assert os.listdir(root.quarantine) == []
