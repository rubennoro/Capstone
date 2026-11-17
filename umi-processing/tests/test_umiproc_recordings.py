"""umiproc.recordings: taking Pi recordings into the data root by kind (map / calibration / demonstration), the
session folder a map recording creates, the pi_sources.csv lookups ``run`` uses, and what is refused.

Recordings are tiny Pi-format episodes (tests/umiproc_testlib.py); the source folder is checked to be only read.
"""

import csv
import json
import os

import pytest

from umiproc import recordings as R
from umiproc.dp.config import load_task_config
from umiproc.dp.intake import IntakeError, TaskTarget, read_ledger, reconcile
from umiproc.dp.paths import DataRoot
from umiproc.dp.raw_episode import COMPLETE, RawEpisode
from umiproc.dp.run.plan import Recording, _classify_taken
from umiproc_testlib import pi_recording, tree_digest

RIG = "rig_pi_v3"
quiet = lambda *a, **k: None  # noqa: E731


@pytest.fixture
def env(tmp_path):
    src = tmp_path / "pi_incoming"
    src.mkdir()
    for n in range(1, 7):
        pi_recording(src, n)
    return DataRoot(tmp_path / "root"), src


def _sources(root):
    with open(root.path / "pi_sources.csv", newline="") as f:
        return list(csv.DictReader(f))


def _session(root, src, task="block_raise"):
    """map 1 -> new folder, calibration 2, demonstrations 3 and 4."""
    m = R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task=task, log=quiet)
    c = R.take_recording(root, src, "episode_000002", R.CALIBRATION, rig_id=RIG, folder=m.folder, log=quiet)
    d = [R.take_recording(root, src, f"episode_00000{n}", R.DEMONSTRATION, rig_id=RIG, folder=m.folder, log=quiet)
         for n in (3, 4)]
    return m, c, d


def test_a_session_by_kind(env):
    root, src = env
    before = tree_digest(src)
    m, c, (d3, d4) = _session(root, src)
    assert tree_digest(src) == before                                     # /data/incoming is only read
    f = root.tasks / "block_raise"
    assert (m.folder, m.kind, m.status, m.task_dir) == ("block_raise", R.MAP, "ingested", f)
    assert m.slot == f"map/{m.pi_name}" and (m.slot_dir / "raw" / COMPLETE).is_file()
    assert m.pi_name.startswith("pi-") and m.pi_name.endswith("_episode_000001")
    assert c.slot == f"calibration/{c.pi_name}" and (c.slot_dir / "raw").is_dir()
    assert d3.slot == f".staging/{d3.pi_name}" and (d4.slot_dir / "raw").is_dir()
    task = load_task_config(f / "task.yaml")
    assert (task.label, task.rig) == ("block_raise", RIG)
    assert RawEpisode.open(m.slot_dir / "raw").validate(decode_videos=True).ok
    # both ledgers say the kind
    rows = _sources(root)
    assert [(r["source_name"], r["kind"], r["task"], r["status"]) for r in rows] == [
        ("episode_000001", "map", "block_raise", "ingested"),
        ("episode_000002", "calibration", "block_raise", "ingested"),
        ("episode_000003", "demonstration", "block_raise", "ingested"),
        ("episode_000004", "demonstration", "block_raise", "ingested")]
    assert [(r["slot"], r["kind"]) for r in read_ledger(f / "episodes.csv")] == [
        (m.slot, "map"), (c.slot, "calibration"), (d3.slot, "demonstration"), (d4.slot, "demonstration")]
    assert m.row["kind"] == "map" and m.row["converted_name"] == m.pi_name
    assert not [p for p in root.incoming.iterdir() if not p.name.startswith(".")]   # staging left empty


def test_second_map_makes_the_next_folder_with_the_same_label(env):
    root, src = env
    a = R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    b = R.take_recording(root, src, "episode_000002", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    (root.tasks / "block_raise_3").write_text("a file also takes the name\n")
    c = R.take_recording(root, src, "episode_000003", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    assert (a.folder, b.folder, c.folder) == ("block_raise", "block_raise_2", "block_raise_4")
    for t in (a, b, c):
        assert load_task_config(t.task_dir / "task.yaml").label == "block_raise"
        assert R.map_slots(t.task_dir) == [t.slot_dir]


def test_lookups_for_run(env):
    root, src = env
    m, c, (d3, _) = _session(root, src)
    by_name = {r["source_name"]: r for r in R.ledger_rows(root)}
    sha = {n: by_name[n]["source_manifest_sha256"] for n in by_name}
    assert R.kind_of(root, sha["episode_000001"]) == R.MAP
    assert R.kind_of(root, sha["episode_000002"]) == R.CALIBRATION
    assert R.kind_of(root, sha["episode_000003"]) == R.DEMONSTRATION
    assert R.kind_of(root, "0" * 64) is None and R.taken_row(root, "0" * 64) is None
    assert R.count_by_kind(root) == {"map": 1, "calibration": 1, "demonstration": 2}
    assert R.not_a_demonstration(by_name["episode_000001"]) == "the map recording of task folder block_raise"
    assert (R.not_a_demonstration(by_name["episode_000002"])
            == "the gripper calibration recording of task folder block_raise")
    assert R.not_a_demonstration(by_name["episode_000003"]) is None and R.not_a_demonstration(None) is None
    # the copied run plan skips them too
    for name, why in (("episode_000001", "is the map recording"), ("episode_000002", "is a calibration")):
        rec = Recording(name, "skip", taken=by_name[name])
        _classify_taken(rec, "block_raise", m.task_dir, set(), set(), {}, False, False)
        assert rec.action == "skip" and rec.why.startswith(why), rec.why


def test_inspect_refuses_what_cannot_be_taken(env, tmp_path):
    root, src = env
    R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    with pytest.raises(R.RecordingError) as e:
        R.inspect_recording(root, src, "episode_000001")
    assert e.value.code == "already_taken" and "the map recording of task folder block_raise" in str(e.value)
    assert e.value.details["row"]["kind"] == "map"
    (src / "episode_000005" / COMPLETE).unlink()
    with pytest.raises(R.RecordingError) as e:
        R.inspect_recording(root, src, "episode_000005")
    assert e.value.code == "still_syncing"
    (src / "episode_000006" / "imu.csv").write_text("changed\n")
    with pytest.raises(R.RecordingError) as e:
        R.inspect_recording(root, src, "episode_000006")
    assert e.value.code == "manifest_failed"
    with pytest.raises(R.RecordingError) as e:
        R.inspect_recording(root, src, "episode_000099")
    assert e.value.code == "source_missing"
    for name, source in (("../x", src), ("episode_000002", tmp_path / "nope"), ("episode_000002", root.incoming)):
        root.incoming.mkdir(parents=True, exist_ok=True)
        with pytest.raises(R.RecordingError) as e:
            R.inspect_recording(root, source, name)
        assert e.value.code == "usage"


def test_stage_then_discard_records_nothing(env):
    root, src = env
    with R.session_lock(root):
        s = R.inspect_recording(root, src, "episode_000001")
        staged = R.stage(root, s, RIG)
        assert staged.path == root.incoming / staged.pi_name and (staged.path / ".verified").is_file()
        assert json.loads((staged.path / "meta.json").read_text())["rig_id"] == RIG
        R.discard(staged)
    assert not staged.path.exists() and not (root.path / "pi_sources.csv").exists() and not root.tasks.exists()
    t = R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    assert t.status == "ingested"                                         # it can be taken after all


def test_stage_needs_an_empty_incoming(env):
    root, src = env
    (root.incoming / "left_over").mkdir(parents=True)
    with pytest.raises(R.RecordingError, match="left_over"):
        R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    assert not root.tasks.exists() and not (root.path / "pi_sources.csv").exists()


def test_calibration_and_demonstrations_need_the_map_folder(env):
    root, src = env
    for kind in (R.CALIBRATION, R.DEMONSTRATION):
        with pytest.raises(R.RecordingError) as e:
            R.take_recording(root, src, "episode_000002", kind, rig_id=RIG, folder="block_raise", log=quiet)
        assert e.value.code == "folder" and "umiproc map" in str(e.value)
    assert not root.tasks.exists() and not (root.path / "pi_sources.csv").exists()
    assert not [p for p in root.incoming.iterdir() if not p.name.startswith(".")]
    with pytest.raises(R.RecordingError):
        R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, folder="block_raise", log=quiet)
    with pytest.raises(R.RecordingError):
        R.take_recording(root, src, "episode_000001", "mapping", rig_id=RIG, task="block_raise", log=quiet)


def test_one_map_per_folder(tmp_path):
    root = DataRoot(tmp_path / "root")
    t = TaskTarget(root, "block_raise", "block_raise")
    t.ensure(RIG)
    slot_dir, slot, kind = t.next_slot("pi-x_episode_000001", False, map_slot=True)
    assert (slot, kind) == ("map/pi-x_episode_000001", "map") and (t.dir / "map").is_dir()
    os.mkdir(slot_dir)
    with pytest.raises(IntakeError, match="one map per folder"):
        t.next_slot("pi-x_episode_000002", False, map_slot=True)


def test_reconcile_knows_map_slots(env):
    root, src = env
    m = R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    ledger = m.task_dir / "episodes.csv"
    ledger.write_text(ledger.read_text().splitlines()[0] + "\n")          # the row lost in a crash
    assert reconcile(root, log=quiet) == [("block_raise", m.slot)]
    assert [(r["slot"], r["kind"], r["pi_name"]) for r in read_ledger(ledger)] == [(m.slot, "map", m.pi_name)]


def test_duplicate_is_quarantined_and_recorded(env, tmp_path):
    root, src = env
    m = R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    other = tmp_path / "other_pi"
    other.mkdir()
    pi_recording(other, 3, seed=99)            # another recording numbered 3: same converted name as episode_000003
    R.take_recording(root, src, "episode_000003", R.DEMONSTRATION, rig_id=RIG, folder=m.folder, log=quiet)
    t = R.take_recording(root, other, "episode_000003", R.DEMONSTRATION, rig_id=RIG, folder=m.folder, log=quiet)
    assert t.status == "quarantined" and "duplicate" in t.reasons and t.slot is None
    assert (t.quarantine_dir / "reason.json").is_file()
    assert _sources(root)[-1]["status"] == "quarantined" and _sources(root)[-1]["kind"] == "demonstration"


def test_session_lock_is_exclusive(tmp_path):
    root = DataRoot(tmp_path / "root")
    with R.session_lock(root):
        with pytest.raises(R.RecordingError) as e:
            with R.session_lock(root):
                pass
        assert e.value.code == "busy"
    with R.session_lock(root):                                            # released
        pass


def test_task_folders(env):
    root, src = env
    assert R.task_folders(root) == []
    _session(root, src)
    (root.tasks / ".hidden").mkdir()
    (root.tasks / "a_file").write_text("x")
    assert [p.name for p in R.task_folders(root)] == ["block_raise"]
