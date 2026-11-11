"""``dataproc package``: selection (exported only; failed/ and .staging/ never), checks, output layout,
package.json, and the zip read back with the vendored ReplayBuffer.

Most tests build small task folders by hand (episode.zarr + status.json, the exact files ``run`` leaves in an
``episode_N``); one test packages the episodes of the real sim -> intake -> process (oracle) pipeline.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from dataproc.config import TaskConfig, dump_task_config
from dataproc.export.zarr_writer import UW_KEY, load_episode_zarr, write_episode_zarr
from dataproc.files import save_status
from dataproc.fsutil import sha256_file
from dataproc.intake import read_ledger
from dataproc.package import PackageError, package_dataset
from dataproc.package.cli import cli_package
from dataproc.paths import data_root
from dataproc.report import STAGE_NAMES
from dataproc._vendor.replay_buffer import ReplayBuffer
from test_export_zarr_schema import random_episode
from test_process_common import TASK, copy_root, processed_root, quiet

RIG = "rig_pi_v1"


# --------------------------------------------------------------------------------------------- helpers

def make_folder(root, folder, label, episodes, rig=RIG, calibration="cal_000"):
    """``tasks/<folder>`` with a task.yaml and one exported ``episode_<k+1>`` per ``(pi_name, n_steps)``."""
    td = root.tasks / folder
    td.mkdir(parents=True)
    dump_task_config(TaskConfig(label=label, rig=rig, created="2026-11-10T00:00:00+00:00"), td / "task.yaml")
    rng = np.random.default_rng(zlib.crc32(folder.encode()))
    for k, (name, n) in enumerate(episodes, start=1):
        add_episode(td, f"episode_{k}", random_episode(rng, name, n, label=label), calibration)
    return td


def add_episode(td, slot, ep, calibration="cal_000", exported=True):
    d = td / slot
    (d / "raw").mkdir(parents=True)
    (d / "processed").mkdir()
    status = {s: {"status": "done", "outputs": {}} for s in STAGE_NAMES}
    status["width"]["outputs"] = {"calibration": calibration}
    if exported:
        write_episode_zarr(ep, d / "processed" / "episode.zarr", num_workers=1)
        status["export"]["outputs"] = {"dir": "episode.zarr", "n_steps": ep.n_steps, "n_uw": ep.n_uw, "name": ep.name}
    else:
        status["slam"] = {"status": "failed", "reasons": ["tracking_lost"], "reason": "tracking_lost"}
        for s in STAGE_NAMES[2:]:
            status[s] = {"status": "skipped", "reason": "slam failed"}
    save_status(d / "processed", status)


def add_failed(td, name, stage="validate", reasons=("too_short",)):
    d = td / "failed" / name
    (d / "raw").mkdir(parents=True)
    (d / "processed").mkdir()
    (d / "reason.json").write_text(json.dumps({"stage": stage, "reasons": list(reasons), "message": "x"}))


def add_staging(td, name):
    (td / ".staging" / name / "raw").mkdir(parents=True)


def check_zip(zip_path, sources, labels):
    """The zip read with the vendored ReplayBuffer: episode_ends, names, task names (one task segment per
    episode), the lowdim data, and the ultrawide upsample / episode_ends / downsample maps."""
    rb = ReplayBuffer.create_from_path(str(zip_path))
    m = rb.meta
    n = np.array([e.n_steps for e in sources])
    mu = np.array([e.n_uw for e in sources])
    ends, uw_ends = np.cumsum(n), np.cumsum(mu)
    assert rb.n_episodes == len(sources)
    assert rb.episode_ends[:].tolist() == ends.tolist()
    assert m["episode_names"][:].tolist() == [e.name for e in sources]
    assert m["task_names"][:].tolist() == labels
    assert m["task_lengths"][:].tolist() == (n - 1).tolist()
    assert m[f"episode_ends_{UW_KEY}"][:].tolist() == uw_ends.tolist()
    up, ds = m[f"upsample_index_{UW_KEY}"][:], m[f"downsample_index_{UW_KEY}"][:]
    assert len(up) == ends[-1] and len(ds) == uw_ends[-1] == rb.data[UW_KEY].shape[0]
    assert rb.data["camera_right_main_rgb"].shape == (ends[-1], 224, 224, 3)
    s = su = 0
    for e, ee, ue in zip(sources, ends, uw_ends):
        assert np.array_equal(up[s:ee], e.uw_upsample_local + su)
        dl = ds[su:ue]
        assert np.all((dl >= s) & (dl < ee)) and np.array_equal(up[dl], np.arange(su, ue))
        s, su = ee, ue
    for k, attr in (("gripper_right_eef_pos", "eef_pos"), ("gripper_right_eef_rot_axis_angle", "eef_rot_axis_angle"),
                    ("gripper_right_gripper_width", "gripper_width")):
        assert np.array_equal(rb.data[k][:], np.concatenate([getattr(e, attr) for e in sources]))
    return rb


def sources_of(res):
    return [load_episode_zarr(e.zarr_path) for e in res.selection.episodes]


@pytest.fixture
def root(tmp_path):
    return data_root(tmp_path / "root")


# --------------------------------------------------------------------------------------------- the real pipeline

@pytest.fixture(scope="module")
def real_root(tmp_path_factory):
    """A copy of the shared processed root (1 calibration + 3 demos, oracle SLAM) for this module."""
    src, res = processed_root(tmp_path_factory)
    assert res.exit_code == 0
    return copy_root(src, tmp_path_factory.mktemp("package_real") / "root")


def test_real_pipeline_one_folder(real_root, capsys):
    td = real_root.tasks / TASK
    assert cli_package(real_root, "practice", [TASK]) == 0
    out = capsys.readouterr().out
    zp = real_root.path / "datasets" / TASK / "practice" / "practice.zarr.zip"
    assert zp.is_file()
    demos = [r for r in read_ledger(td / "episodes.csv") if r["kind"] == "demonstration"]
    rb = ReplayBuffer.create_from_path(str(zp))
    assert rb.episode_ends[:].tolist() == [150, 300, 450]
    assert rb.meta["episode_names"][:].tolist() == [r["pi_name"] for r in demos]
    assert rb.meta["task_names"][:].tolist() == [TASK] * 3
    rec = json.loads((zp.parent / "package.json").read_text())
    assert [e["slot"] for e in rec["episodes"]] == ["episode_0", "episode_1", "episode_2"]
    assert {e["calibration"] for e in rec["episodes"]} == {"cal_episode_000000"}
    assert rec["rig"] == "rig_v1_fast" and rec["zip"]["sha256"] == sha256_file(zp)
    assert f"mkdir -p CapstoneProject/datasets/{TASK}/practice" in out
    assert (f"rsync -av {zp} <you>@xfer.discovery.neu.edu:CapstoneProject/datasets/{TASK}/practice/"
            "practice.zarr.zip") in out
    assert f"scripts/run_pipeline.sh train --task {TASK} --dataset practice" in out
    assert "warning" not in out


# --------------------------------------------------------------------------------------------- selection

def test_one_folder(root):
    make_folder(root, "block_raise", "block_raise", [("ep_a", 30), ("ep_b", 41), ("ep_c", 25)])
    res = package_dataset(root, "d1", ["block_raise"], log=quiet)
    assert res.zip_path == root.path / "datasets" / "block_raise" / "d1" / "d1.zarr.zip"
    assert res.label == "block_raise" and not res.selection.warnings
    check_zip(res.zip_path, sources_of(res), ["block_raise"] * 3)


def test_two_folders_same_label(root):
    make_folder(root, "block_raise", "block_raise", [("s1_a", 30), ("s1_b", 22)])
    make_folder(root, "block_raise_2", "block_raise", [("s2_a", 35), ("s2_b", 28), ("s2_c", 20)])
    res = package_dataset(root, "both", ["block_raise_2", "block_raise"], log=quiet)
    assert res.zip_path.parent == root.path / "datasets" / "block_raise" / "both"
    assert [e.pi_name for e in res.selection.episodes] == ["s2_a", "s2_b", "s2_c", "s1_a", "s1_b"]   # folder order
    check_zip(res.zip_path, sources_of(res), ["block_raise"] * 5)
    assert res.record["totals"]["per_label"] == {"block_raise": 5}


def test_two_labels_is_multi(root):
    make_folder(root, "block_raise", "block_raise", [("b1", 30), ("b2", 22)])
    make_folder(root, "cup_push", "cup_push", [("c1", 26), ("c2", 31)])
    res = package_dataset(root, "mix", ["block_raise", "cup_push"], log=quiet)
    assert res.label == "multi" and res.zip_path == root.path / "datasets" / "multi" / "mix" / "mix.zarr.zip"
    check_zip(res.zip_path, sources_of(res), ["block_raise", "block_raise", "cup_push", "cup_push"])
    assert res.record["labels"] == ["block_raise", "cup_push"]


def test_episode_order_is_numeric(root):
    td = make_folder(root, "t", "t", [])
    rng = np.random.default_rng(1)
    for k in (10, 2, 1):
        add_episode(td, f"episode_{k}", random_episode(rng, f"ep{k}", 20, label="t"))
    res = package_dataset(root, "d", ["t"], log=quiet)
    assert [e.slot for e in res.selection.episodes] == ["episode_1", "episode_2", "episode_10"]


def test_failed_staging_and_unexported_are_excluded(root, capsys):
    td = make_folder(root, "block_raise", "block_raise", [("good1", 30), ("good2", 25)])
    rng = np.random.default_rng(2)
    add_episode(td, "episode_3", random_episode(rng, "slamfail", 20), exported=False)
    add_failed(td, "bad1")
    write_episode_zarr(random_episode(rng, "bad1", 20), td / "failed" / "bad1" / "processed" / "episode.zarr")
    add_staging(td, "waiting1")
    write_episode_zarr(random_episode(rng, "waiting1", 20), td / ".staging" / "waiting1" / "episode.zarr")
    (td / "episode_4").mkdir()                           # an empty episode_N (interrupted move): ignored
    assert cli_package(root, "d", ["block_raise"]) == 0
    out = capsys.readouterr().out
    zp = root.path / "datasets" / "block_raise" / "d" / "d.zarr.zip"
    rb = ReplayBuffer.create_from_path(str(zp))
    assert rb.meta["episode_names"][:].tolist() == ["good1", "good2"]
    rec = json.loads((zp.parent / "package.json").read_text())
    assert [(x["entry"], x["state"]) for x in rec["excluded"]] == [
        ("episode_3", "failed"), ("failed/bad1", "failed"), (".staging/waiting1", "not finished")]
    assert rec["excluded"][1]["reason"] == "validate: too_short"
    assert rec["totals"]["excluded"] == 3
    for s in ("failed/bad1", ".staging/waiting1", "not finished", "validate: too_short", "slam: tracking_lost"):
        assert s in out


def test_fewer_than_two_per_label_warns(root, capsys):
    make_folder(root, "block_raise", "block_raise", [("b1", 30), ("b2", 22)])
    make_folder(root, "cup_push", "cup_push", [("c1", 26)])
    assert cli_package(root, "mix", ["block_raise", "cup_push"]) == 0
    out = capsys.readouterr().out
    assert "warning: label cup_push has 1 episode(s)" in out and "label block_raise" not in out
    rec = json.loads((root.path / "datasets" / "multi" / "mix" / "package.json").read_text())
    assert len(rec["warnings"]) == 1 and "cup_push" in rec["warnings"][0]


def test_package_json(root):
    make_folder(root, "block_raise", "block_raise", [("a", 30), ("b", 22)], calibration="cal_x")
    make_folder(root, "block_raise_2", "block_raise", [("c", 25)], calibration="cal_y")
    res = package_dataset(root, "pj", ["block_raise", "block_raise_2"], log=quiet)
    rec = json.loads((res.zip_path.parent / "package.json").read_text())
    assert rec == res.record
    assert rec["dataset"] == "pj" and rec["label_dir"] == "block_raise" and rec["rig"] == RIG
    assert rec["folders"] == [{"folder": "block_raise", "label": "block_raise", "rig": RIG},
                              {"folder": "block_raise_2", "label": "block_raise", "rig": RIG}]
    assert rec["episodes"] == [
        {"folder": "block_raise", "slot": "episode_1", "pi_name": "a", "label": "block_raise", "n_steps": 30,
         "calibration": "cal_x"},
        {"folder": "block_raise", "slot": "episode_2", "pi_name": "b", "label": "block_raise", "n_steps": 22,
         "calibration": "cal_x"},
        {"folder": "block_raise_2", "slot": "episode_1", "pi_name": "c", "label": "block_raise", "n_steps": 25,
         "calibration": "cal_y"}]
    assert rec["totals"] == {"episodes": 3, "n_steps": 77, "per_label": {"block_raise": 3}, "excluded": 0}
    assert rec["zip"] == {"name": "pj.zarr.zip", "size_bytes": res.zip_path.stat().st_size,
                          "sha256": sha256_file(res.zip_path)}
    assert rec["dataproc_version"] and rec["created"]


# --------------------------------------------------------------------------------------------- refuse / overwrite

def test_refuse_then_overwrite(root, capsys):
    td = make_folder(root, "block_raise", "block_raise", [("a", 30), ("b", 22)])
    assert cli_package(root, "d", ["block_raise"]) == 0
    zp = root.path / "datasets" / "block_raise" / "d" / "d.zarr.zip"
    before = sha256_file(zp), (zp.parent / "package.json").read_text()
    add_episode(td, "episode_3", random_episode(np.random.default_rng(3), "c", 26))
    capsys.readouterr()
    assert cli_package(root, "d", ["block_raise"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert (sha256_file(zp), (zp.parent / "package.json").read_text()) == before
    assert cli_package(root, "d", ["block_raise"], overwrite=True) == 0
    rb = ReplayBuffer.create_from_path(str(zp))
    assert rb.meta["episode_names"][:].tolist() == ["a", "b", "c"]
    rec = json.loads((zp.parent / "package.json").read_text())
    assert rec["totals"]["episodes"] == 3 and rec["zip"]["sha256"] == sha256_file(zp)
    assert sorted(p.name for p in zp.parent.iterdir()) == ["d.zarr.zip", "package.json"]   # no temp leftovers


# --------------------------------------------------------------------------------------------- errors

def _no_datasets(root):
    return not (root.path / "datasets").exists()


@pytest.mark.parametrize("case", ["missing_folder", "different_rigs", "no_episodes", "bad_name", "duplicate_name",
                                  "same_folder_twice", "no_task_yaml"])
def test_errors_write_nothing(root, capsys, case):
    make_folder(root, "block_raise", "block_raise", [("a", 30), ("b", 22)])
    dataset, folders = "d", ["block_raise"]
    if case == "missing_folder":
        folders, msg = ["block_raise", "nope"], "does not exist"
    elif case == "different_rigs":
        make_folder(root, "other", "block_raise", [("c", 20)], rig="rig_v1_fast")
        folders, msg = ["block_raise", "other"], "different rigs"
    elif case == "no_episodes":
        td = make_folder(root, "empty", "block_raise", [])
        add_failed(td, "x")
        add_staging(td, "y")
        folders, msg = ["empty"], "no exported episode"
    elif case == "bad_name":
        dataset, msg = "bad/name", "invalid dataset name"
    elif case == "duplicate_name":
        make_folder(root, "copy", "block_raise", [("a", 20)])
        folders, msg = ["block_raise", "copy"], "more than once"
    elif case == "same_folder_twice":
        folders, msg = ["block_raise", "block_raise"], "more than once"
    else:
        (root.tasks / "notask").mkdir()
        folders, msg = ["notask"], "task.yaml is missing"
    assert cli_package(root, dataset, folders) == 1
    cap = capsys.readouterr()
    assert msg in cap.err and cap.err.startswith("dataproc package: ")
    assert _no_datasets(root)
    with pytest.raises(PackageError):
        package_dataset(root, dataset, folders, log=quiet)


def test_failed_build_leaves_no_dataset_dir(root, monkeypatch):
    make_folder(root, "block_raise", "block_raise", [("a", 30), ("b", 22)])

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("dataproc.export.zarr_writer.build_task_zarr_zip", boom)
    with pytest.raises(PackageError, match="disk full"):
        package_dataset(root, "d", ["block_raise"], log=quiet)
    assert not (root.path / "datasets" / "block_raise").exists()
