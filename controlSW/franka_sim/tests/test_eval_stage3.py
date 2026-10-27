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
S = Path(os.environ.get("S", "/tmp/claude-1000/-home-ruben-Capstone/299e5dab-9afb-41ab-aa6c-ac52f8fb5552/scratchpad"))
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


# ----------------------------------------------------------------------------------------------- Part 2 (pure)
SUCC = {"lift_min_m": 0.05, "hold_s": 2.0, "timeout_s": 60, "slip_tol_m": 0.002}


def _sim_ep(tmp_path, t_ep=100.0, hz=25.0, lift_at=3.0):
    """An episode dir with the task's events.jsonl and 25 Hz bridge sim lines: +6 cm from lift_at, held."""
    ep = tmp_path / "episode_0"
    ep.mkdir()
    (ep / "events.jsonl").write_text(json.dumps({"t": t_ep, "kind": "episode_start", "episode_dir": str(ep)}) + "\n")
    lines = []
    for i in range(int(8 * hz)):
        t = t_ep + i / hz
        z = 0.02 + (0.06 if t - t_ep >= lift_at else 0.0)
        lines.append({"t": t, "kind": "sim", "obj": [0.45, 0, z, 1, 0, 0, 0], "gap": 0.03, "grasped": t - t_ep >= 1.0,
                      "contacts": [{"a": "pad_left", "b": "table", "force_n": 5.0}] if i == 3 else [], "reflex": None})
    (ep / "bridge.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return ep


def test_episode_start_t_comes_from_events_jsonl(tmp_path):
    ep = _sim_ep(tmp_path, t_ep=123.25)
    assert s3.episode_start_t(ep) == 123.25
    assert s3.episode_start_t(tmp_path / "missing", timeout_s=0.05) is None


def test_judge_vs_ground_truth_agree_and_disagree(tmp_path):
    ep = _sim_ep(tmp_path, t_ep=100.0, lift_at=3.0)
    cfg = dict(s3.DEFAULTS)
    judge = {"success": True, "t_success_s": 5.01}                  # the judge at 100 Hz decides 10 ms later than a 25 Hz sample
    c = s3.ground_truth_compare(ep, SUCC, 100.0, judge, cfg, tries=1)
    assert c["gt"]["success"] is True and c["gt"]["t_success_s"] == pytest.approx(5.0)
    assert c["agree"] and c["t_success_dt_s"] == pytest.approx(0.01) and c["gt_t_success_mono"] == pytest.approx(105.0)
    late = s3.ground_truth_compare(ep, SUCC, 100.0, {"success": True, "t_success_s": 5.3}, cfg, tries=1)
    assert late["success_agree"] and not late["agree"]                # same success, t_success off by 0.3 s
    # anchoring at the driver's later t0 (A6 finding 1) shifts the relative time: the comparison then uses the same t_start for both
    shifted = s3.ground_truth_compare(ep, SUCC, 100.2, {"success": True, "t_success_s": 4.8}, cfg, tries=1)
    assert shifted["agree"]
    no = s3.ground_truth_compare(ep, SUCC, 100.0, {"success": False, "t_success_s": None}, cfg, tries=1)
    assert not no["success_agree"] and not no["agree"]
    assert s3.ground_truth_compare(tmp_path / "nolines", SUCC, 0.0, judge, cfg, tries=1) is None


def test_aggregate_takes_sim_gt_as_authoritative_and_counts_pad_table():
    import types
    R = types.SimpleNamespace(image_source="render")
    m0 = {"tracking_mm": np.array([1.0]), "gripper_lag": [], "infer_s": [0.1], "obs_motion_mm": [], "reflexes": 0, "stale_image": 0,
          "label": "y", "n_calls": 1, "holds": 0}
    cmp_ok = {"gt": {"success": False}, "agree": False, "success_agree": False, "t_success_dt_s": None}
    eps = [{"label": "y", "pose": (0, 0, 0), "pose_i": 0, "ep_i": 0, "truth": {"success": True}, "collisions": [], "m": m0,
            "pad_table": [{"t_s": 1.0}], "pad_table_force_max_n": 41.0, "cmp": cmp_ok, "auth": {"success": False, "source": "sim_gt"},
            "t0_source": "events", "wall_s": 1.0}]
    m, rows = s3.aggregate(eps, R, "mujoco")
    assert m["success_rate"] == 0.0 and m["success_source"] == "sim_gt" and m["judge_gt_compared"] == 1 and m["judge_gt_agree"] == 0
    assert m["pad_table_contacts"] == 1 and m["pad_table_force_max_n"] == 41.0 and m["collisions"] == 0
    assert rows[0]["gt"] == "n" and rows[0]["pad_table"] == 1 and m["judge_t0_from_events"] == 1


def test_gripper_lag_uses_the_hand_target_not_the_policy_command(tmp_path):
    """W10B item 8: an empty-hand grasp closes to max(0, gap - eps_inner); against the policy's train width it is a
    6.9 mm "error" for ever, against the hand's own width_servo_target it is no lag at all."""
    t = np.arange(0, 4.0, 0.005)
    cmd = np.where(t < 1.0, 0.08, 0.050)                       # policy: close to train 0.050 at t=1 s
    target = np.where(t < 1.0, 0.08, 0.0431)                   # the hand: grasp -> gap 0 -> train b = 0.0431
    meas = np.where(t < 1.0, 0.08, np.maximum(0.0431, 0.08 - (t - 1.0) * 0.4))   # reaches the target in ~0.1 s
    np.savez(tmp_path / "controller_trace.npz", t=t, width_measured=meas, width_commanded=cmd, width_servo_target=target,
             commanded=np.zeros((len(t), 7)), fk=np.zeros((len(t), 7)))
    m = s3.episode_metrics(tmp_path, None, 0.05)
    assert m["gripper_lag_ref"] == "width_servo_target"
    assert m["gripper_lag"] == [1]                             # ~0.09 s -> 1 step, then closed
    old = s3.gripper_lag_steps(t, meas, cmd, 0.05)
    assert old[-1] >= int((t[-1] - 1.2) // 0.05)               # the policy command: open (6.9 mm) until the end (the old defect)
    np.savez(tmp_path / "controller_trace.npz", t=t, width_measured=meas, width_commanded=cmd,
             width_servo_target=np.full_like(t, np.nan), commanded=np.zeros((len(t), 7)), fk=np.zeros((len(t), 7)))
    m = s3.episode_metrics(tmp_path, None, 0.05)
    assert m["gripper_lag_ref"] == "width_commanded" and m["gripper_lag"] == old   # no hand target (fake gripper): fallback
