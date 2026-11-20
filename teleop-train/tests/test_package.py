"""teleop_train.package: selection, consistency, the zip, package.json, the command (PLAN.md §4.1, §6, §7, §11).

The "converted episodes" are built here directly (an ``episode.zarr`` written with ``write_episode_zarr`` and a
``status.json`` with the keys convert writes), so these tests do not depend on ``convert.py``; ``convert_task`` is
monkeypatched where the command would call it.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr

import synthetic_episode as se
from teleop_train import cli, convert, package, paths
from teleop_train._copied._vendor.replay_buffer import ReplayBuffer
from teleop_train._copied.zarr_writer import (MAIN_KEY, POS_KEY, ROT_KEY, UW_KEY, WIDTH_KEY, EpisodeArrays,
                                              load_episode_zarr, write_episode_zarr)
from teleop_train.package import PackageError, package_dataset

TASK = "pick_test"
STATUS_SCHEMA = "teleop_train.episode.v1"
RIG_SHA = convert.rig_sha256(se.load_rig())            # the fixture rig = the deployment rig
_DEFAULT_RIG = package.DEPLOY_RIG_YAML
F_T_TCP = se.F_T_TCP.tolist()
#: convert.convert_task as E4 wrote it (None while convert.py is a stub); the autouse fixture replaces it
_REAL_CONVERT_TASK = getattr(convert, "convert_task", None)
#: package.DEPLOY_FRAMES_YAML before the autouse fixture replaces it
_DEFAULT_FRAMES = package.DEPLOY_FRAMES_YAML
RECORD_KEYS = ["dataset", "label_dir", "labels", "folders", "rig", "episodes", "excluded", "totals", "zip", "source",
               "pose_source", "F_T_TCP", "warnings", "teleop_train_version", "created"]


def _read_status_ref(path) -> dict | None:
    """The documented contract of convert.read_status: the dict, or None when missing / unreadable / other schema."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == STATUS_SCHEMA else None


@pytest.fixture(autouse=True)
def fake_convert(monkeypatch):
    """convert.py may still be a stub: a reference read_status if it has none, and a convert_task that must not run
    unless a test replaces it."""
    monkeypatch.setattr(convert, "CONVERT_VERSION", 1)
    if not callable(getattr(convert, "read_status", None)):
        monkeypatch.setattr(convert, "read_status", _read_status_ref, raising=False)

    def no_convert(*a, **k):
        raise AssertionError("convert_task called with convert_missing=False")
    monkeypatch.setattr(convert, "convert_task", no_convert, raising=False)


@pytest.fixture(autouse=True)
def deploy_tcp(tmp_path, monkeypatch):
    """The deployment's frames.yaml for these tests: the synthetic episodes' TCP (se.F_T_TCP)."""
    f = tmp_path / "frames.yaml"
    f.write_text(yaml.safe_dump({"F_T_TCP": F_T_TCP, "settled": False}), encoding="utf-8")
    monkeypatch.setattr(package, "DEPLOY_FRAMES_YAML", f)
    return f


@pytest.fixture(autouse=True)
def deploy_rig(monkeypatch):
    """The deployment rig for these tests: the fixture rig (a copy of deployment/contracts/rigs/rig_pi_v3_640.yaml)."""
    monkeypatch.setattr(package, "DEPLOY_RIG_YAML", se.RIG_FILE)
    return se.RIG_FILE


@pytest.fixture
def tp(tmp_path):
    raw_root, out_root = tmp_path / "raw", tmp_path / "out"
    se.make_task(raw_root, TASK)
    t = paths.task_paths(TASK, raw_root, out_root)
    paths.prepare_out_task(t)
    return t


def _arrays(tp, ep: str, n: int, m: int) -> EpisodeArrays:
    rng = np.random.default_rng(paths.episode_number(ep))
    up = np.repeat(np.arange(m), -(-n // m))[:n]
    up[-1] = m - 1
    k = np.arange(n, dtype=np.float64)
    return EpisodeArrays(
        name=f"{tp.task}_{ep}", task_label=tp.task,
        main_rgb=rng.integers(0, 256, (n, 224, 224, 3), dtype=np.uint8),
        uw_rgb=rng.integers(0, 256, (m, 224, 224, 3), dtype=np.uint8),
        uw_upsample_local=up,
        eef_pos=np.stack([0.5 + 0.001 * k, np.zeros(n), 0.4 + 0 * k], axis=1),
        eef_rot_axis_angle=np.stack([np.pi - 0.01 * k, 0.01 * k + 0.1, np.zeros(n)], axis=1),
        gripper_width=0.08 - 0.001 * k)


def add_episode(tp, number: int, *, state: str = "converted", outcome: str | None = "success", n: int = 4, m: int = 2,
                kept_s: float = 10.0, write_zarr: bool = True, write_status: bool = True, reason: str = "",
                **overrides) -> str:
    """A raw episode folder (empty), plus a converted (or excluded) output episode."""
    ep = f"episode_{number:03d}"
    (tp.raw_task_dir / ep).mkdir()
    tp.processed_dir(ep).mkdir(parents=True)
    if state == "converted" and write_zarr:
        write_episode_zarr(_arrays(tp, ep, n, m), tp.episode_zarr(ep), num_workers=1)
    status = {
        "schema": STATUS_SCHEMA, "state": state, "reason": reason, "episode": ep, "outcome": outcome,
        "raw_episode_dir": str(tp.raw_episode_dir(ep)), "meta_sha256": "b" * 64,
        "window_ns": [0, int(kept_s * 1e9) + 2_000_000_000], "kept_ns": [1_000_000_000, int(kept_s * 1e9) + 1_000_000_000],
        "raw_s": kept_s + 2.0, "kept_s": kept_s, "n_steps": n if state == "converted" else 0,
        "n_uw": m if state == "converted" else 0, "pose_source": "target", "F_T_TCP": F_T_TCP,
        "rig_id": "rig_pi_v3_640", "rig_sha256": RIG_SHA, "settings": dict(convert.SETTINGS),
        "code_sha256": convert.code_sha256(), "convert_version": 1,
        "teleop_train": {"version": "0.1.0", "commit": "0" * 40, "dirty": False}, "dropped": {},
        "created": "2026-11-19T12:00:00+00:00",
    }
    status.update(overrides)
    if write_status:
        tp.status_json(ep).write_text(json.dumps(status), encoding="utf-8")
    return ep


def read_zip(path: Path) -> ReplayBuffer:
    store = zarr.ZipStore(str(path), mode="r")
    return ReplayBuffer(root=zarr.open_group(store=store, mode="r"))


def json_get(path: Path, key: str):
    """train/lib.sh json_get: dotted path, an integer part indexes a list."""
    v = json.loads(Path(path).read_text())
    for k in key.split("."):
        v = v[int(k)] if isinstance(v, list) else v[k]
    return "" if v is None else v


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- selection ---------------------------------------------------------------------------------------------------

def test_selection_order_and_excluded(tp):
    for i in (10, 2, 0, 1):                                  # created out of order
        add_episode(tp, i)
    add_episode(tp, 3, state="excluded", reason="main frame gap 0.31 s > 0.1 s")
    add_episode(tp, 4, outcome="fail")
    add_episode(tp, 5, write_status=False)                  # raw folder without a conversion
    add_episode(tp, 6, write_zarr=False)                    # status says converted, zarr gone
    add_episode(tp, 7, outcome=None)
    res = package_dataset(tp, "ds1", "target", convert_missing=False)
    rec = res.record
    assert [e["slot"] for e in rec["episodes"]] == ["episode_000", "episode_001", "episode_002", "episode_010"]
    assert [e["pi_name"] for e in rec["episodes"]] == [f"{TASK}_episode_000", f"{TASK}_episode_001",
                                                       f"{TASK}_episode_002", f"{TASK}_episode_010"]
    assert all(e["folder"] == TASK and e["label"] == TASK and e["outcome"] == "success" for e in rec["episodes"])
    ex = {x["entry"]: x for x in rec["excluded"]}
    assert [x["entry"] for x in rec["excluded"]] == ["episode_003", "episode_004", "episode_005", "episode_006",
                                                     "episode_007"]
    assert ex["episode_003"]["state"] == "excluded" and ex["episode_003"]["reason"] == "main frame gap 0.31 s > 0.1 s"
    assert ex["episode_004"] == {"folder": TASK, "entry": "episode_004", "state": "fail",
                                 "reason": "outcome fail (use --include-fail)"}
    assert ex["episode_005"]["state"] == "excluded" and "not converted" in ex["episode_005"]["reason"]
    assert ex["episode_006"]["state"] == "excluded" and "episode.zarr missing" in ex["episode_006"]["reason"]
    assert ex["episode_007"]["state"] == "fail" and "outcome not set" in ex["episode_007"]["reason"]
    assert rec["totals"] == {"episodes": 4, "n_steps": 16, "per_label": {TASK: 4}, "excluded": 5}


def test_episodes_n_keeps_the_first_n(tp):
    for i in range(4):
        add_episode(tp, i)
    add_episode(tp, 1 + 10, outcome="fail")
    rec = package_dataset(tp, "ds2", "target", episodes_n=2, convert_missing=False).record
    assert [e["slot"] for e in rec["episodes"]] == ["episode_000", "episode_001"]
    assert rec["excluded"] == [
        {"folder": TASK, "entry": "episode_002", "state": "left out", "reason": "--episodes 2 (only the first 2)"},
        {"folder": TASK, "entry": "episode_003", "state": "left out", "reason": "--episodes 2 (only the first 2)"},
        {"folder": TASK, "entry": "episode_011", "state": "fail", "reason": "outcome fail (use --include-fail)"},
    ]
    assert rec["totals"]["episodes"] == 2 and rec["totals"]["excluded"] == 3
    rec1 = package_dataset(tp, "ds1", "target", episodes_n=1, convert_missing=False).record
    assert [e["slot"] for e in rec1["episodes"]] == ["episode_000"]        # nested subsets


@pytest.mark.parametrize("n,msg", [(0, "at least 1"), (3, "only 2 usable episodes")])
def test_episodes_n_refused(tp, n, msg):
    add_episode(tp, 0)
    add_episode(tp, 1)
    with pytest.raises(PackageError, match=msg):
        package_dataset(tp, "ds", "target", episodes_n=n, convert_missing=False)
    assert not (tp.out_task_dir / paths.PACKAGE_DIR).exists()


def test_include_fail(tp):
    add_episode(tp, 0)
    add_episode(tp, 1, outcome="fail")
    add_episode(tp, 2)
    rec = package_dataset(tp, "ds", "target", include_fail=True, convert_missing=False).record
    assert [(e["slot"], e["outcome"]) for e in rec["episodes"]] == [
        ("episode_000", "success"), ("episode_001", "fail"), ("episode_002", "success")]
    assert rec["excluded"] == []


def test_nothing_usable(tp):
    add_episode(tp, 0, outcome="fail")
    add_episode(tp, 1, state="excluded", reason="no motion")
    with pytest.raises(PackageError, match=r"nothing to package.*0 of 2.*1 fail, 1 excluded.*--include-fail"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    assert not (tp.out_task_dir / paths.PACKAGE_DIR).exists()


def test_status_of_another_schema_left_out(tp):
    add_episode(tp, 0)
    add_episode(tp, 1, schema="dataproc.episode.v1")
    rec = package_dataset(tp, "ds", "target", convert_missing=False).record
    assert [e["slot"] for e in rec["episodes"]] == ["episode_000"]
    assert rec["excluded"][0]["entry"] == "episode_001" and "another schema" in rec["excluded"][0]["reason"]


def test_no_episodes_at_all(tp):
    with pytest.raises(PackageError, match="no episode folders"):
        package_dataset(tp, "ds", "target", convert_missing=False)


# ---- refusals ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key,value", [
    ("rig_sha256", "c" * 64),
    ("rig_id", "rig_pi_v3"),
    ("F_T_TCP", np.eye(4).tolist()),
])
def test_consistency_refused(tp, key, value):
    add_episode(tp, 0)
    add_episode(tp, 1)
    add_episode(tp, 2, **{key: value})
    add_episode(tp, 3, **{key: value})
    with pytest.raises(PackageError, match=r"^episode_002 has another .* than episode_000"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    assert not (tp.out_task_dir / paths.PACKAGE_DIR).exists()
    # the first two alone are fine
    assert package_dataset(tp, "ds", "target", episodes_n=2, convert_missing=False).record["totals"]["episodes"] == 2


def test_deploy_tcp_refused(tp, deploy_tcp):
    add_episode(tp, 0)
    add_episode(tp, 1)
    other = [row[:] for row in F_T_TCP]
    other[2][3] += 2e-9
    deploy_tcp.write_text(yaml.safe_dump({"F_T_TCP": other}), encoding="utf-8")
    with pytest.raises(PackageError, match=r"^the episodes' F_T_TCP differs from the deployment's .*frames.yaml "
                                           r"\(max \|diff\| 2e-09 > 1e-09\)"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    assert not (tp.out_task_dir / paths.PACKAGE_DIR).exists()
    other[2][3] -= 1.5e-9                                   # 5e-10 off: within the hello check's tolerance
    deploy_tcp.write_text(yaml.safe_dump({"F_T_TCP": other}), encoding="utf-8")
    assert package_dataset(tp, "ds", "target", convert_missing=False).record["F_T_TCP"] == F_T_TCP


def test_deploy_tcp_unreadable_refused(tp, deploy_tcp):
    add_episode(tp, 0)
    deploy_tcp.write_text("F_T_TCP: [[1, 0], [0, 1]]\n", encoding="utf-8")
    with pytest.raises(PackageError, match="frames.yaml: cannot read F_T_TCP"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    deploy_tcp.unlink()
    with pytest.raises(PackageError, match="frames.yaml: cannot read F_T_TCP"):
        package_dataset(tp, "ds", "target", convert_missing=False)


def test_real_deploy_frames_yaml(tp, monkeypatch):
    """This branch's deployment/contracts/frames.yaml is the default and parses; a status in its TCP passes."""
    real = Path(__file__).resolve().parents[2] / "deployment" / "contracts" / "frames.yaml"
    assert _DEFAULT_FRAMES == real and real.is_file()
    monkeypatch.setattr(package, "DEPLOY_FRAMES_YAML", real)
    m = package.deploy_F_T_TCP()
    assert m.shape == (4, 4) and m[3].tolist() == [0.0, 0.0, 0.0, 1.0]
    add_episode(tp, 0, F_T_TCP=m.tolist())
    assert package_dataset(tp, "ds", "target", convert_missing=False).record["F_T_TCP"] == m.tolist()
    if not np.allclose(m, np.array(F_T_TCP)):
        add_episode(tp, 1, F_T_TCP=F_T_TCP)
        with pytest.raises(PackageError, match="has another F_T_TCP"):
            package_dataset(tp, "ds2", "target", convert_missing=False)


def test_pose_source_mismatch_refused(tp):
    add_episode(tp, 0)
    add_episode(tp, 1, pose_source="measured")
    with pytest.raises(PackageError, match=r"^episode_001: converted with pose source measured, not target"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    with pytest.raises(PackageError, match=r"^episode_000: converted with pose source target, not measured"):
        package_dataset(tp, "ds", "measured", convert_missing=False)


def test_convert_version_mismatch_refused(tp, monkeypatch):
    add_episode(tp, 0)
    add_episode(tp, 1, convert_version=0)
    with pytest.raises(PackageError, match=r"^episode_001: converted by convert version 0, this is 1"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    monkeypatch.setattr(convert, "CONVERT_VERSION", 2)
    with pytest.raises(PackageError, match=r"^episode_000: converted by convert version 1, this is 2"):
        package_dataset(tp, "ds", "target", convert_missing=False)


def test_zarr_not_matching_status_refused(tp):
    add_episode(tp, 0)
    add_episode(tp, 1, n_steps=5)                           # the zarr has 4
    with pytest.raises(PackageError, match=r"episode_001/processed/episode.zarr: 4 steps but its status.json says 5"):
        package_dataset(tp, "ds", "target", convert_missing=False)


def test_zarr_of_another_task_refused(tp):
    ep = add_episode(tp, 0)
    root = zarr.open_group(store=zarr.DirectoryStore(str(tp.episode_zarr(ep))), mode="r+")
    root.attrs["task_label"] = "other_task"
    with pytest.raises(PackageError, match="task label 'other_task'"):
        package_dataset(tp, "ds", "target", convert_missing=False)


def test_status_missing_keys_refused(tp):
    ep = add_episode(tp, 0)
    st = json.loads(tp.status_json(ep).read_text())
    del st["rig_sha256"], st["F_T_TCP"]
    tp.status_json(ep).write_text(json.dumps(st))
    with pytest.raises(PackageError, match="missing rig_sha256, F_T_TCP"):
        package_dataset(tp, "ds", "target", convert_missing=False)


@pytest.mark.parametrize("name", ["", "a/b", "x" * 65, "-lead", "_lead", "sp ace"])
def test_bad_dataset_name_refused(tp, name):
    add_episode(tp, 0)
    with pytest.raises(PackageError):
        package_dataset(tp, name, "target", convert_missing=False)


def test_bad_pose_source_refused(tp):
    add_episode(tp, 0)
    with pytest.raises(PackageError, match="pose source"):
        package_dataset(tp, "ds", "both", convert_missing=False)


def test_handheld_folder_refused(tmp_path):
    raw_root, out_root = tmp_path / "raw", tmp_path / "out"
    se.make_task(raw_root, TASK)
    t = paths.task_paths(TASK, raw_root, out_root)
    t.out_task_dir.mkdir(parents=True)                     # a handheld folder of the same name: no marker
    with pytest.raises(PackageError, match="without teleop_source.yaml"):
        package_dataset(t, "ds", "target", convert_missing=False)


def test_nothing_converted_refused(tmp_path):
    raw_root, out_root = tmp_path / "raw", tmp_path / "out"
    se.make_task(raw_root, TASK)
    t = paths.task_paths(TASK, raw_root, out_root)
    with pytest.raises(PackageError, match="nothing converted yet"):
        package_dataset(t, "ds", "target", convert_missing=False)
    assert not out_root.exists()


def test_existing_dataset_refused_and_overwrite(tp):
    add_episode(tp, 0)
    add_episode(tp, 1)
    first = package_dataset(tp, "ds", "target", episodes_n=1, convert_missing=False)
    sha_before = sha256(first.zip_path)
    with pytest.raises(PackageError, match="already exists.*--overwrite"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    assert sha256(first.zip_path) == sha_before
    second = package_dataset(tp, "ds", "target", overwrite=True, convert_missing=False)
    assert second.zip_path == first.zip_path
    assert second.record["totals"]["episodes"] == 2
    assert json.loads(second.package_json.read_text())["zip"]["sha256"] == sha256(second.zip_path) != sha_before
    assert sorted(p.name for p in (tp.out_task_dir / paths.PACKAGE_DIR).iterdir()) == ["ds"]   # no tmp / old left
    assert sorted(p.name for p in tp.package_dir("ds").iterdir()) == ["ds.zarr.zip", "package.json"]


def test_existing_dataset_refused_before_convert(tp, monkeypatch):
    add_episode(tp, 0)
    package_dataset(tp, "ds", "target", convert_missing=False)
    calls = []
    monkeypatch.setattr(convert, "convert_task", lambda *a, **k: calls.append(1) or [])
    with pytest.raises(PackageError, match="already exists"):
        package_dataset(tp, "ds", "target")
    assert calls == []


# ---- atomicity ---------------------------------------------------------------------------------------------------

def _failing_build(episodes, out_zip_path, **kw):
    Path(out_zip_path).write_bytes(b"half a zip")
    raise RuntimeError("disk full\nsecond line")


def test_failed_zip_leaves_no_dataset(tp, monkeypatch):
    add_episode(tp, 0)
    monkeypatch.setattr(package, "build_task_zarr_zip", _failing_build)
    with pytest.raises(PackageError, match=r"^building ds.zarr.zip failed \(RuntimeError: disk full\)$"):
        package_dataset(tp, "ds", "target", convert_missing=False)
    assert not (tp.out_task_dir / paths.PACKAGE_DIR).exists()


def test_failed_overwrite_keeps_the_old_dataset(tp, monkeypatch):
    add_episode(tp, 0)
    add_episode(tp, 1)
    old = package_dataset(tp, "ds", "target", episodes_n=1, convert_missing=False)
    old_zip, old_json = sha256(old.zip_path), old.package_json.read_bytes()
    monkeypatch.setattr(package, "build_task_zarr_zip", _failing_build)
    with pytest.raises(PackageError, match="building ds.zarr.zip failed"):
        package_dataset(tp, "ds", "target", overwrite=True, convert_missing=False)
    assert sha256(old.zip_path) == old_zip and old.package_json.read_bytes() == old_json
    assert sorted(p.name for p in (tp.out_task_dir / paths.PACKAGE_DIR).iterdir()) == ["ds"]


def test_failure_in_package_json_leaves_no_dataset(tp, monkeypatch):
    add_episode(tp, 0)

    def boom(*a, **k):
        raise OSError("no space")
    monkeypatch.setattr(package, "_record", boom)
    with pytest.raises(OSError):
        package_dataset(tp, "ds", "target", convert_missing=False)
    assert not (tp.out_task_dir / paths.PACKAGE_DIR).exists()


def test_raw_folder_unchanged(tp):
    add_episode(tp, 0)
    add_episode(tp, 1)
    before = se.tree_hash(tp.raw_task_dir)
    package_dataset(tp, "ds", "target", convert_missing=False)
    assert se.tree_hash(tp.raw_task_dir) == before


# ---- warnings ----------------------------------------------------------------------------------------------------

def test_long_episode_warning(tp):
    add_episode(tp, 0, kept_s=60.0)
    add_episode(tp, 1, kept_s=61.5)
    res = package_dataset(tp, "ds", "target", convert_missing=False)
    assert len(res.warnings) == 1 and res.warnings[0].startswith("episode_001: 61.5 s")
    assert "max_sequence_length" in res.warnings[0]
    assert res.record["warnings"] == res.warnings
    assert json.loads(res.package_json.read_text())["warnings"] == res.warnings


# ---- the zip and package.json ------------------------------------------------------------------------------------

def test_zip_reads_back_with_replay_buffer(tp):
    add_episode(tp, 0, n=5, m=2)
    add_episode(tp, 1, n=4, m=3)
    add_episode(tp, 2, n=6, m=2)
    res = package_dataset(tp, "ds", "target", convert_missing=False)
    rb = read_zip(res.zip_path)
    assert set(rb.data.keys()) == {MAIN_KEY, UW_KEY, POS_KEY, ROT_KEY, WIDTH_KEY}
    assert rb.data[MAIN_KEY].shape == (15, 224, 224, 3) and rb.data[MAIN_KEY].dtype == np.uint8
    assert rb.data[UW_KEY].shape == (7, 224, 224, 3) and rb.data[UW_KEY].dtype == np.uint8
    assert rb.data[POS_KEY].shape == (15, 3) and rb.data[POS_KEY].dtype == np.float32
    assert rb.data[ROT_KEY].shape == (15, 3) and rb.data[ROT_KEY].dtype == np.float32
    assert rb.data[WIDTH_KEY].shape == (15, 1) and rb.data[WIDTH_KEY].dtype == np.float32
    assert list(rb.meta["episode_ends"][:]) == [5, 9, 15]
    assert list(rb.meta[f"episode_ends_{UW_KEY}"][:]) == [2, 5, 7]
    assert len(rb.meta[f"upsample_index_{UW_KEY}"]) == 15 and len(rb.meta[f"downsample_index_{UW_KEY}"]) == 7
    assert list(rb.task_names[:]) == [TASK] * 3
    assert list(rb.episode_names[:]) == [f"{TASK}_episode_000", f"{TASK}_episode_001", f"{TASK}_episode_002"]
    # images copied byte for byte: decoding gives the episode zarrs' pixels, in episode order
    main = np.concatenate([load_episode_zarr(tp.episode_zarr(f"episode_{i:03d}")).main_rgb for i in range(3)])
    uw = np.concatenate([load_episode_zarr(tp.episode_zarr(f"episode_{i:03d}")).uw_rgb for i in range(3)])
    assert np.array_equal(rb.data[MAIN_KEY][:], main) and np.array_equal(rb.data[UW_KEY][:], uw)
    pos = np.concatenate([load_episode_zarr(tp.episode_zarr(f"episode_{i:03d}")).eef_pos for i in range(3)])
    assert np.array_equal(rb.data[POS_KEY][:], pos)


def test_package_json(tp):
    add_episode(tp, 0, kept_s=12.25)
    add_episode(tp, 1, outcome="fail")
    res = package_dataset(tp, "pick-v1", "target", convert_missing=False)
    assert res.zip_path == tp.zip_path("pick-v1") and res.package_json == tp.package_json("pick-v1")
    rec = json.loads(res.package_json.read_text(encoding="utf-8"))
    assert rec == res.record
    assert list(rec) == RECORD_KEYS
    assert rec["dataset"] == "pick-v1" and rec["label_dir"] == TASK and rec["labels"] == [TASK]
    assert rec["folders"] == [{"folder": TASK, "label": TASK, "rig": "rig_pi_v3_640"}]
    assert rec["rig"] == "rig_pi_v3_640"
    assert rec["episodes"] == [{"folder": TASK, "slot": "episode_000", "pi_name": f"{TASK}_episode_000",
                                "label": TASK, "n_steps": 4, "kept_s": 12.25, "outcome": "success"}]
    assert rec["zip"] == {"name": "pick-v1.zarr.zip", "size_bytes": os.path.getsize(res.zip_path),
                          "sha256": sha256(res.zip_path)}
    assert rec["source"] == {"kind": "teleop_collect", "raw_task_dir": str(tp.raw_task_dir)}
    assert rec["pose_source"] == "target" and rec["F_T_TCP"] == F_T_TCP
    assert rec["teleop_train_version"] == "0.1.0"
    assert rec["created"].endswith("+00:00") and len(rec["created"]) == 25
    # what train/upload.sh and train/submit.sh read with json_get
    pj = res.package_json
    assert json_get(pj, "label_dir") == TASK
    assert json_get(pj, "zip.sha256") == sha256(res.zip_path)
    assert json_get(pj, "totals.n_steps") == 4
    assert json_get(pj, "totals.episodes") == 1
    assert json_get(pj, "episodes.0.slot") == "episode_000"


def test_package_dir_permissions(tp):
    add_episode(tp, 0)
    res = package_dataset(tp, "ds", "target", convert_missing=False)
    um = os.umask(0)
    os.umask(um)
    assert (tp.package_dir("ds").stat().st_mode & 0o777) == (0o777 & ~um)
    assert (res.zip_path.stat().st_mode & 0o777) == (0o666 & ~um)


# ---- convert_missing ---------------------------------------------------------------------------------------------

def test_convert_missing_calls_convert_task(tp, monkeypatch):
    add_episode(tp, 0)
    seen = []

    def fake_convert_task(t, pose_source, episodes=None, force=False, progress=None):
        seen.append((t, pose_source, episodes, force, progress))
        return [types.SimpleNamespace(episode="episode_000", state="up to date")]
    monkeypatch.setattr(convert, "convert_task", fake_convert_task)
    prog = object()
    res = package_dataset(tp, "ds", "target", progress=prog)
    assert seen == [(tp, "target", None, False, prog)]
    assert [r.state for r in res.convert_results] == ["up to date"]


def test_convert_refusal_becomes_package_error(tmp_path, monkeypatch):
    raw_root, out_root = tmp_path / "raw", tmp_path / "out"
    se.make_task(raw_root, TASK)
    t = paths.task_paths(TASK, raw_root, out_root)
    t.out_task_dir.mkdir(parents=True)                     # handheld folder

    def fake_convert_task(t, pose_source, episodes=None, force=False, progress=None):
        paths.prepare_out_task(t)
        return []
    monkeypatch.setattr(convert, "convert_task", fake_convert_task)
    with pytest.raises(PackageError, match="without teleop_source.yaml"):
        package_dataset(t, "ds", "target")


# ---- the command -------------------------------------------------------------------------------------------------

def _cli(tp, *extra):
    return cli.main(["package", TASK, *extra, "--raw-root", str(tp.raw_task_dir.parent),
                     "--out-root", str(tp.out_task_dir.parent)])


def test_cli_package(tp, monkeypatch, capsys):
    add_episode(tp, 0)
    add_episode(tp, 1, kept_s=75.0)
    add_episode(tp, 2, outcome="fail")
    calls = []
    monkeypatch.setattr(convert, "convert_task",
                        lambda t, ps, episodes=None, force=False, progress=None:
                        calls.append(ps) or [types.SimpleNamespace(state="up to date")] * 3)
    assert _cli(tp, "ds1") == 0
    assert calls == ["target"]
    out = capsys.readouterr().out.splitlines()
    assert "convert: 3 up to date" in out
    assert any(line.startswith("packaged 2 episodes of pick_test") for line in out)
    assert any("episode_002" in line and "outcome fail (use --include-fail)" in line for line in out)
    assert any(line.startswith("warning: episode_001: 75.0 s") for line in out)
    assert any(line.startswith(f"wrote {tp.zip_path('ds1')} (") for line in out)
    assert "  cd ~/Capstone && train/all.sh pick_test ds1 --config umi_policy_dunet" in out
    assert "  cd ~/Capstone && train/all.sh pick_test ds1 --config umi_policy_dunetp" in out
    assert any("LOCAL_DATA_ROOT in train/config.local.env must point at " + str(tp.out_task_dir.parent) in line
               for line in out)


def test_cli_options(tp, monkeypatch, capsys):
    for i in range(3):
        add_episode(tp, i, pose_source="measured")
    add_episode(tp, 3, outcome="fail", pose_source="measured")
    monkeypatch.setattr(convert, "convert_task", lambda *a, **k: [])
    monkeypatch.setattr(paths, "DEFAULT_OUT_ROOT", tp.out_task_dir.parent)
    assert _cli(tp, "ds", "--pose-source", "measured", "--episodes", "3", "--include-fail") == 0
    out = capsys.readouterr().out
    assert "LOCAL_DATA_ROOT" not in out                    # default out root: no note
    rec = json.loads(tp.package_json("ds").read_text())
    assert [e["slot"] for e in rec["episodes"]] == ["episode_000", "episode_001", "episode_002"]
    assert rec["excluded"][0]["entry"] == "episode_003" and rec["excluded"][0]["state"] == "left out"
    assert rec["pose_source"] == "measured"
    assert _cli(tp, "ds", "--pose-source", "measured", "--overwrite") == 0
    assert json.loads(tp.package_json("ds").read_text())["totals"]["episodes"] == 3


def test_cli_refusals(tp, monkeypatch, capsys):
    add_episode(tp, 0)
    monkeypatch.setattr(convert, "convert_task", lambda *a, **k: [])
    assert _cli(tp, "ds") == 0
    capsys.readouterr()
    assert _cli(tp, "ds") == 1                              # existing name
    err = capsys.readouterr().err
    assert err.startswith("teleop_train package: dataset ") and "already exists" in err and err.count("\n") == 1
    assert cli.main(["package", "no_such_task", "ds", "--raw-root", str(tp.raw_task_dir.parent),
                     "--out-root", str(tp.out_task_dir.parent)]) == 1
    err = capsys.readouterr().err
    assert "task.yaml: missing" in err and err.count("\n") == 1


def test_cli_with_real_convert(tmp_path, monkeypatch, capsys):
    """End to end on synthetic raw episodes with convert.py's own convert_task (skipped while it is a stub)."""
    if _REAL_CONVERT_TASK is None:
        pytest.skip("convert.convert_task not written yet")
    monkeypatch.setattr(convert, "convert_task", _REAL_CONVERT_TASK)
    raw_root, out_root = tmp_path / "raw", tmp_path / "out"
    task_dir = se.make_task(raw_root, TASK)
    se.make_episode(task_dir, 0)
    se.make_episode(task_dir, 1, outcome="fail")
    before = se.tree_hash(task_dir)
    assert cli.main(["package", TASK, "e2e", "--raw-root", str(raw_root), "--out-root", str(out_root)]) == 0
    out = capsys.readouterr().out
    assert "  cd ~/Capstone && train/all.sh pick_test e2e --config umi_policy_dunet" in out.splitlines()
    assert se.tree_hash(task_dir) == before
    t = paths.task_paths(TASK, raw_root, out_root)
    rec = json.loads(t.package_json("e2e").read_text())
    assert [e["slot"] for e in rec["episodes"]] == ["episode_000"]
    assert rec["excluded"] == [{"folder": TASK, "entry": "episode_001", "state": "fail",
                                "reason": "outcome fail (use --include-fail)"}]
    assert rec["F_T_TCP"] == F_T_TCP and rec["pose_source"] == "target"
    assert rec["zip"]["sha256"] == sha256(t.zip_path("e2e"))
    rb = read_zip(t.zip_path("e2e"))
    assert int(rb.meta["episode_ends"][-1]) == rec["totals"]["n_steps"] == rec["episodes"][0]["n_steps"]
    assert list(rb.task_names[:]) == [TASK]


# --------------------------------------------------------------------------------------------- review follow-ups

def test_the_real_deployment_rig_is_the_fixture_rig():
    """The rig this branch's deployment preps with has the fixture's content (so RIG_SHA is the deployment's)."""
    assert package.deploy_rig_sha256(_DEFAULT_RIG) == RIG_SHA


def test_refuses_a_rig_other_than_the_deployments(tp):
    for n in range(2):
        add_episode(tp, n, rig_sha256="c" * 64, rig_id="rig_pi_v2_640")
    with pytest.raises(PackageError, match="not the deployment's rig_pi_v3_640.yaml") as e:
        package_dataset(tp, "ds", "target", convert_missing=False)
    assert "\n" not in str(e.value)
    assert not tp.package_dir("ds").exists()


def test_unreadable_deployment_rig_is_refused(tp, tmp_path, monkeypatch):
    add_episode(tp, 0)
    monkeypatch.setattr(package, "DEPLOY_RIG_YAML", tmp_path / "missing.yaml")
    with pytest.raises(PackageError, match="cannot read the deployment rig"):
        package_dataset(tp, "ds", "target", convert_missing=False)


@pytest.mark.parametrize("key, value", [("settings", {"grid_hz": 60}), ("code_sha256", "0" * 64)])
def test_refuses_status_from_other_settings_or_code(tp, key, value):
    add_episode(tp, 0, **{key: value})
    with pytest.raises(PackageError, match="other settings or another teleop_train version"):
        package_dataset(tp, "ds", "target", convert_missing=False)


def test_package_reconverts_a_stale_episode_before_zipping(tmp_path, monkeypatch):
    """With the real convert_task: a status written with other settings is converted again, then packaged."""
    if _REAL_CONVERT_TASK is None:
        pytest.skip("convert.convert_task not written yet")
    monkeypatch.setattr(convert, "convert_task", _REAL_CONVERT_TASK)
    raw_root, out_root = tmp_path / "raw", tmp_path / "out"
    task_dir = se.make_task(raw_root, TASK)
    se.make_episode(task_dir, 0)
    t = paths.task_paths(TASK, raw_root, out_root)
    package_dataset(t, "v1", "target")
    st_path = t.status_json("episode_000")
    old = json.loads(st_path.read_text())
    stale = dict(old, settings=dict(old["settings"], margin_s=0.3), created="2000-01-01T00:00:00+00:00")
    st_path.write_text(json.dumps(stale))
    res = package_dataset(t, "v2", "target")
    new = json.loads(st_path.read_text())
    assert new["settings"] == convert.SETTINGS and new["created"] != stale["created"]
    assert [r.state for r in res.convert_results] == ["converted"]
    assert res.record["totals"]["episodes"] == 1
