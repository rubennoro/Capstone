"""Stage 3 (B9 part 1): pure metrics + the driver end to end on sim-iphone-fake with the in-process StubSim (no MuJoCo, no GPU).
Process tests run under the suite lock with the Track B port offset:
cd controlSW/franka_sim && flock $LOCK env -u PYTHONPATH FD_TEST_PORT_OFFSET=2000 MUJOCO_GL=egl $RE -m pytest -q -s tests/test_eval_stage3.py tests/test_eval_runner.py
Needs data/raw/2026-10-01 (the video replay sender); the sweep test also needs the firstreal zarr."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from franka_sim.eval import report as rpt  # noqa: E402
from franka_sim.eval import runner as rn  # noqa: E402
from franka_sim.eval import stage3 as s3  # noqa: E402

SIM = Path(__file__).resolve().parents[1]
DEPLOY = SIM.parent / "franka_deploy"
S = Path(os.environ.get("S", "/tmp/claude-1000/-home-ruben-Capstone/b5f58f43-1b87-4846-beab-dc85de4129a4/scratchpad"))
ZARR = Path(os.environ.get("ZARR") or S / "deploy_work" / "prompt_zarr" / "firstreal-single.zarr.zip")
RAW = SIM.parents[1] / "data" / "raw" / "2026-10-01"
needs_raw = pytest.mark.skipif(not RAW.is_dir(), reason="needs data/raw/2026-10-01")
GATES = {f"stage3.{k}_max" for k in ("collisions", "reflex", "tracking_err_mm_p95", "gripper_width_lag_steps_p95", "inference_latency_s_p95",
                                      "receiver_age_s_p95", "stale_image_errors", "obs_motion_mm_p95")}


# ----------------------------------------------------------------------------------------------- pure
def test_gripper_lag_steps_intervals_and_grasp_exclusion():
    dt = 0.01
    t = np.arange(0, 3, dt)
    cmd = np.where(t < 1.0, 0.08, 0.02)
    meas = np.where(t < 1.0, 0.08, np.where(t < 1.3, 0.08, 0.02))          # follows 0.3 s late
    lag = s3.gripper_lag_steps(t, meas, cmd, 0.05)
    assert lag == [6]                                                       # floor(0.31 / 0.05)
    assert s3.gripper_lag_steps(t, meas, cmd, 0.05, grasped_fn=lambda tt: True) == []            # the object blocks the fingers: excluded
    assert s3.gripper_lag_steps(t, cmd, cmd, 0.05) == []                    # tracked perfectly: no interval at all
    stuck = np.full_like(t, 0.08)
    assert s3.gripper_lag_steps(t, stuck, cmd, 0.05) == [int((t[-1] - 1.0) // 0.05)]     # never arrives: counts to the end
    small = np.where(t < 1.0, 0.08, 0.077)                                  # 3 mm: below the 4 mm opening threshold
    assert s3.gripper_lag_steps(t, np.full_like(t, 0.08), small, 0.05) == []


def test_tracking_and_obs_motion():
    st = np.zeros((4, 7))
    fk = np.zeros((4, 7))
    fk[:, 0] = [0, 0.001, 0.002, 0.003]
    assert s3.tracking_err_mm(st, fk) == pytest.approx([0, 1, 2, 3])
    t = np.array([0.0, 1.0, 2.0])
    fk3 = np.zeros((3, 7))
    fk3[:, 0] = [0.0, 0.01, 0.02]
    assert s3.obs_motion_mm(t, fk3, 1.5, 1.0) == pytest.approx(5.0)         # 5 mm between the two timestamps


def test_poses_centre_and_offsets():
    cfg = {**s3.DEFAULTS}
    p = s3.make_poses(cfg, (0.45, 0.0))
    assert len(p) == 3 and p[0] == (0.45, 0.0, 0.0) and p[1][0] == pytest.approx(0.48) and p[2][1] == pytest.approx(0.02)
    assert p[1][2] == pytest.approx(np.radians(15)) and p[2][2] == pytest.approx(np.radians(-15))
    assert len(s3.make_poses({**cfg, "n_poses": 2}, (0, 0))) == 2


def test_log_tail_parses_task_events(tmp_path):
    p = tmp_path / "task.log"
    t = s3.LogTail(p)
    assert t.poll() == []
    p.write_text('noise\n[franka_task] episode_start {"prompt_index": 0, "episode_dir": "/x/e0"}\n[franka_task] refused_start {"key": "c", "rea')
    assert t.poll() == [("episode_start", {"prompt_index": 0, "episode_dir": "/x/e0"})]
    with open(p, "a") as f:
        f.write('son": "FK far"}\n')
    assert t.poll() == [("refused_start", {"key": "c", "reason": "FK far"})]


def test_stage3_needs_a_sim_or_mujoco(tmp_path):
    import argparse
    a = argparse.Namespace(deploy_config=None, profile="sim-iphone-fake", policy="fake", set=[], out=str(tmp_path), seed=0, stage="3",
                           sim_stub=None, episodes_per_pose=None, poses=None, port_offset=0, stage1=None)
    rep = rn.run_one_stage(rn.resolve(a), "3", {})
    assert rep["verdict"] == "INCOMPLETE" and any("--sim-stub" in n for n in rep["notes"])


# ----------------------------------------------------------------------------------------------- process tests
def _eval(out, *args, timeout=900, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=f"{SIM}:{DEPLOY}", MUJOCO_GL="egl", **(env_extra or {}))
    r = subprocess.run([sys.executable, "-m", "franka_sim", "eval", "--out", str(out), *args], capture_output=True, text=True, env=env,
                       cwd=str(SIM), timeout=timeout)
    print(r.stdout[-3000:])
    return r


def _s3(out, stub, *extra, **kw):
    return _eval(out, "--profile", "sim-iphone-fake", "--stage", "3", "--policy", "fake", "--sim-stub", stub, *extra, **kw)


def _run_dir(out, prefix="3_fake_"):
    ds = sorted(p for p in Path(out).iterdir() if p.is_dir() and p.name.startswith(prefix))
    assert len(ds) == 1, ds
    return ds[0]


def _truths(rd):
    return sorted(rd.glob("work/session_*/task/episodes/*/*/*/episode_*/sim_truth.json"))


@needs_raw
def test_stub_lift_success_two_poses(tmp_path):
    r = _s3(tmp_path, "lift_success", "--episodes-per-pose", "1", "--poses", "2", "--set", "criteria.success.timeout_s=15")
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    rd = _run_dir(tmp_path)
    rep = json.loads((rd / "report.json").read_text())
    assert rpt.validate(rep) == [] and rep["stage"] == "3" and rep["verdict"] == "PASS"
    m = rep["metrics"]
    assert m["labels"] == "yy" and m["success_rate"] == 1.0 and m["n_episodes"] == 2 and m["n_poses"] == 2
    assert m["sim_source"] == "stub" and m["image_source"] == "video_replay"
    assert set(rep["gates"]) == GATES and all(g["value"] is not None and g["pass"] is True for g in rep["gates"].values()), rep["gates"]
    assert len(_truths(rd)) == 2
    t0 = json.loads(_truths(rd)[0].read_text())
    assert t0["truth"]["success"] is True and t0["truth"]["label_written"] == "y" and t0["scenario"] == "lift_success"
    labels = [json.loads(p.with_name("label.json").read_text())["label"] for p in _truths(rd)]
    assert labels == ["y", "y"]
    assert any("stub" in n and "NOT a physics simulation" in n for n in rep["notes"])
    assert any("open user decisions" in n and "criteria.success.lift_min_m" in n for n in rep["notes"])
    assert "stub" in (rd / "report.html").read_text()
    # the table the WORKLOG pastes
    for n, g in rep["gates"].items():
        print(f"GATE {n:50s} {g['value']!s:>22} {g['op']} {g['threshold']} {g['pass']}")


@needs_raw
def test_stub_no_lift_labels_n_at_the_timeout(tmp_path):
    r = _s3(tmp_path, "no_lift", "--episodes-per-pose", "1", "--poses", "1", "--set", "criteria.success.timeout_s=6")
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-2000:]
    rd = _run_dir(tmp_path)
    rep = json.loads((rd / "report.json").read_text())
    assert rep["metrics"]["labels"] == "n" and rep["metrics"]["success_rate"] == 0.0 and rep["verdict"] == "PASS"
    t = json.loads(_truths(rd)[0].read_text())["truth"]
    assert t["reason"] == "timeout" and t["success"] is False


@needs_raw
def test_stub_reflex_is_a_fail_and_the_run_continues(tmp_path):
    r = _s3(tmp_path, "reflex", "--episodes-per-pose", "2", "--poses", "1", "--set", "criteria.success.timeout_s=12")
    assert r.returncode == rn.EXIT_FAIL, r.stderr[-2000:] + r.stdout[-2000:]
    rd = _run_dir(tmp_path)
    rep = json.loads((rd / "report.json").read_text())
    assert rep["verdict"] == "FAIL" and rep["gates"]["stage3.reflex_max"]["pass"] is False
    assert rep["metrics"]["reflex"] >= 1 and rep["metrics"]["n_episodes"] == 2 and "y" not in rep["metrics"]["labels"]   # first episode is labelled exception/aborted by the task
    assert not any(n.startswith("INCOMPLETE") for n in rep["notes"])            # recovered (bridge recover + r) and continued


@needs_raw
def test_stub_collision_is_a_fail(tmp_path):
    r = _s3(tmp_path, "collision", "--episodes-per-pose", "1", "--poses", "1", "--set", "criteria.success.timeout_s=15")
    assert r.returncode == rn.EXIT_FAIL, r.stderr[-2000:] + r.stdout[-2000:]
    rep = json.loads((_run_dir(tmp_path) / "report.json").read_text())
    assert rep["gates"]["stage3.collisions_max"]["value"] == 1 and rep["gates"]["stage3.collisions_max"]["pass"] is False
    assert rep["metrics"]["labels"] == "y"


@needs_raw
def test_sender_death_aborts_with_incomplete(tmp_path):
    r = _s3(tmp_path, "lift_success", "--episodes-per-pose", "3", "--poses", "1", "--set", "criteria.success.timeout_s=15",
            env_extra={"EVAL_3_INJECT": "13:kill:sender"})
    assert r.returncode == rn.EXIT_INCOMPLETE, r.stderr[-2000:] + r.stdout[-2000:]
    rep = json.loads((_run_dir(tmp_path) / "report.json").read_text())
    assert rep["verdict"] == "INCOMPLETE" and any("sender_died" in n and "episode aborted" in n for n in rep["notes"]), rep["notes"]
    assert rep["metrics"]["sender_died"] is True and rep["metrics"]["n_episodes"] < 3


@needs_raw
@pytest.mark.skipif(not ZARR.is_file(), reason="needs the firstreal zarr (demo_replay policy)")
def test_sweep_smoke_on_the_stub_runs_resumes_and_compares(tmp_path):
    """4 points (link x policy); the first invocation is capped at 2 runs (a process killed after 2 points), the rerun runs only the
    other 2; summary.html has one column per point with fake vs demo_replay."""
    base = ["--profile", "sim-iphone-fake", "--sweep", "block_raise_smoke", "--sim-stub", "lift_success", "--set", "criteria.success.timeout_s=15"]
    r1 = _eval(tmp_path, *base, "--max-runs", "2", timeout=1200)
    assert r1.returncode in (1, 3), r1.stderr[-2000:] + r1.stdout[-2000:]      # demo_replay FAILs (3); 2 points are still unrun (INCOMPLETE 1)
    idx = json.loads((tmp_path / "sweep_index.json").read_text())
    done = [p["stages"]["3"] for p in idx["points"] if p["stages"]["3"].get("verdict") in ("PASS", "FAIL")]
    assert len(idx["points"]) == 4 and len(done) == 2 and not any(d["skipped"] for d in done)
    r2 = _eval(tmp_path, *base, timeout=1200)
    assert r2.returncode == rn.EXIT_FAIL, r2.stderr[-2000:] + r2.stdout[-2000:]      # all points ran: demo_replay's obs_motion FAIL outranks
    assert "2 new run(s)" in r2.stdout
    idx = json.loads((tmp_path / "sweep_index.json").read_text())
    st = [p["stages"]["3"] for p in idx["points"]]
    # fake never moves the arm: PASS; demo_replay moves it at demo speed: FAIL on obs_motion_mm_p95 ONLY (speed x image age,
    # a user decision on the gate, see block_raise.yaml), nothing else
    assert [s["verdict"] for s in st] == ["PASS", "FAIL", "PASS", "FAIL"] and [s["skipped"] for s in st] == [True, True, False, False]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["n_runs"] == 4 and len(s["comparison"]["columns"]) == 4
    for p in idx["points"]:
        rep = json.loads((tmp_path / p["stages"]["3"]["run_dir"] / "report.json").read_text())
        failed = sorted(n for n, g in rep["gates"].items() if g["pass"] is False)
        assert failed == ([] if rep["policy_id"] == "fake" else ["stage3.obs_motion_mm_p95_max"]), (rep["policy_id"], failed)
    h = (tmp_path / "summary.html").read_text()
    assert "Sweep comparison" in h and "demo_replay" in h and "wifi_good" in h and "fake" in h
