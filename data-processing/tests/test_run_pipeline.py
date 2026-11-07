"""``dataproc run`` end to end: sim episodes written in the Pi's format -> adapt -> intake -> process.

First with ``--slam none`` (ORB-SLAM3 not installed yet: stops at slam), then resumed with the oracle (the sim's
ground truth standing in for a real SLAM backend): the zip is built without redoing what came before slam.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataproc.paths import data_root
from dataproc.pi_adapter import SOURCES_FILE, read_sources
from dataproc.run import EXIT_OK, EXIT_WAITING_FOR_SLAM, RunError, run_pipeline
from dataproc.sim.episode import run_sim
from pi_fixtures import contract_to_pi
from test_process_common import quiet, tree_digest

TASK = "block_raise"
RIG = "rig_v1_fast"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base = tmp_path_factory.mktemp("run_pipeline")
    root = data_root(base / "root")
    sim = base / "sim"
    run_sim(sim, episodes=1, calibration=True, fast=True, rigs_dir=root.rigs, session_id="cal", log=quiet)
    run_sim(sim, episodes=3, fast=True, rigs_dir=root.rigs, session_id="demo", log=quiet)
    src = base / "pi_incoming"
    for n, d in enumerate(sorted(sim.iterdir())):        # cal_episode_000000 first, then the demos
        contract_to_pi(d, src / f"episode_{n:06d}", n, with_exposure=True)
    return root, src


def _statuses(root):
    rep = json.loads((root.tasks / TASK / "output" / "report.json").read_text())
    return rep, {e["slot"]: e for e in rep["episodes"]}


def test_needs_a_calibration_the_first_time(setup):
    root, src = setup
    with pytest.raises(RunError, match="calibration"):
        run_pipeline(root, TASK, source=src, rig_id=RIG, slam="none", log=quiet)


def test_no_calibration_takes_everything_as_demos(setup):
    root, src = setup
    res = run_pipeline(root, TASK, source=src, rig_id=RIG, slam="none", dry_run=True, no_calibration=True,
                       log=quiet)
    assert res.calibration is None
    assert res.demos == ["episode_000000", "episode_000001", "episode_000002", "episode_000003"]
    with pytest.raises(RunError, match="contradict"):
        run_pipeline(root, TASK, calibration_episode="episode_000000", source=src, rig_id=RIG, slam="none",
                     dry_run=True, no_calibration=True, log=quiet)


def test_episodes_selects_a_subset(setup):
    root, src = setup
    res = run_pipeline(root, TASK, source=src, rig_id=RIG, slam="none", dry_run=True, no_calibration=True,
                       episodes=["episode_000003", "episode_000001"], log=quiet)
    assert res.demos == ["episode_000001", "episode_000003"] and "episode_000002" not in res.skipped
    with pytest.raises(RunError, match="not a directory"):
        run_pipeline(root, TASK, source=src, rig_id=RIG, slam="none", dry_run=True, no_calibration=True,
                     episodes=["episode_999999"], log=quiet)


def test_dry_run_changes_nothing(setup):
    root, src = setup
    res = run_pipeline(root, TASK, calibration_episode="episode_000000", source=src, rig_id=RIG, slam="none",
                       dry_run=True, log=quiet)
    assert res.calibration == "episode_000000"
    assert res.demos == ["episode_000001", "episode_000002", "episode_000003"]
    assert not root.tasks.exists() and not root.incoming.exists()


def test_without_slam_then_resume(setup):
    root, src = setup
    before = tree_digest(src)

    # 1. no SLAM installed: everything up to slam runs, every episode stops there
    res = run_pipeline(root, TASK, calibration_episode="episode_000000", source=src, rig_id=RIG, slam="none",
                       log=quiet)
    assert res.exit_code == EXIT_WAITING_FOR_SLAM, res.outcomes
    assert res.process.zip_path is None
    rep, eps = _statuses(root)
    assert rep["calibration"]["state"] == "done"
    assert sorted(eps) == ["episode_0", "episode_1", "episode_2"]
    for e in eps.values():
        assert e["stages"]["check"] == "done" and e["stages"]["slam"] == "failed"
        assert e["reasons"][0] == "slam: slam_unavailable"
    assert tree_digest(src) == before                            # the Pi's folder is never touched
    rows = read_sources(root.path / SOURCES_FILE)
    assert sorted((r["source_name"], r["kind"], r["status"]) for r in rows) == [
        ("episode_000000", "calibration", "ingested"), ("episode_000001", "demonstration", "ingested"),
        ("episode_000002", "demonstration", "ingested"), ("episode_000003", "demonstration", "ingested")]
    assert [p.name for p in root.incoming.iterdir()] == []

    # 2. SLAM available (oracle stands in): the same command resumes, nothing is re-taken, check is not redone
    res = run_pipeline(root, TASK, source=src, rig_id=RIG, slam="oracle", log=quiet)
    assert res.exit_code == EXIT_OK
    assert res.calibration is None and res.demos == []
    assert set(res.skipped) == {"episode_000000", "episode_000001", "episode_000002", "episode_000003"}
    assert res.zip_path.is_file() and res.zip_path.name == f"{TASK}.zarr.zip"
    rep, eps = _statuses(root)
    assert rep["totals"]["exported"] == 3
    for e in eps.values():
        assert "check" not in e["ran_this_call"] and "slam" in e["ran_this_call"]
        assert (root.tasks / TASK / e["slot"] / "processed" / "episode.zarr").is_dir()
    assert tree_digest(src) == before

    # the zip opens as a ReplayBuffer with one task label per episode
    import zarr
    with zarr.ZipStore(str(res.zip_path), mode="r") as st:
        g = zarr.group(store=st)
        assert list(g["meta/task_names"][:]) == [TASK] * 3
        assert len(g["meta/episode_ends"]) == 3
        assert np.isfinite(g["data/gripper_right_eef_pos"][:]).all()
