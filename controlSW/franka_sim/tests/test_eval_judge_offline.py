"""W10B item 10: the live judge rule (eval.judge.SuccessTracker) replayed offline on recorded sim sessions agrees with
the A6 ground truth (ground_truth.evaluate_episode): same success, and t_success within GT_T_SUCCESS_TOL_S (0.1 s).

Sessions: the rows of docs/sim_plan/results/stage1b/campaign.json (the A8 Stage 1b campaign, kept in the scratchpad),
or the campaign.json named by $W10B_CAMPAIGN_JSON. The test skips cleanly when none of the recorded sessions exists.
The judge sees the bridge.jsonl `sim` lines from the task's episode_start, resampled at 100 Hz (the state PUB rate), as
the B9 Part 2 reviewer's offline check did ($S/b9p2rev/offline.py)."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO_CTRL = HERE.parents[1]
CAMPAIGN = Path(os.environ.get("W10B_CAMPAIGN_JSON", REPO_CTRL / "docs" / "sim_plan" / "results" / "stage1b" / "campaign.json"))
CRITERIA = REPO_CTRL / "configs" / "run" / "criteria" / "block_raise.yaml"


def _quat_wxyz_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _episodes():
    if not CAMPAIGN.is_file():
        return []
    rows = json.loads(CAMPAIGN.read_text()).get("rows", [])
    out = []
    for r in rows:
        eps = sorted(glob.glob(os.path.join(str(r.get("session") or ""), "task", "episodes", "*", "*", "*", "episode_*")))
        if eps and os.path.isfile(os.path.join(eps[0], "bridge.jsonl")) and os.path.isfile(os.path.join(eps[0], "events.jsonl")):
            out.append((r["run"], eps[0]))
    return out


def replay_judge(ep: str, success: dict) -> dict | None:
    from franka_sim.eval.judge import SuccessTracker

    t0 = [json.loads(ln) for ln in open(os.path.join(ep, "events.jsonl")) if "episode_start" in ln]
    if not t0:
        return None
    t0 = float(t0[0]["t"])
    ev = []
    for ln in open(os.path.join(ep, "bridge.jsonl")):
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("kind") == "sim" and e.get("t", -1) >= t0 and e.get("obj") is not None:
            ev.append(e)
    if len(ev) < 2:
        return None
    ts = np.array([e["t"] for e in ev])
    tt = np.arange(ts[0], ts[-1], 0.01)
    idx = np.searchsorted(ts, tt, side="right") - 1
    tr = SuccessTracker(success, t0)
    for t, i in zip(tt, idx):
        e = ev[i]
        T = np.eye(4)
        T[:3, :3] = _quat_wxyz_R(e["obj"][3:7])
        T[:3, 3] = e["obj"][:3]
        if tr.update({"t_mono": float(t), "eraser_T_base": T, "grasped": e.get("grasped"), "reflex": e.get("reflex")}):
            break
    return tr.r


def test_offline_judge_agrees_with_ground_truth_on_recorded_sessions():
    eps = _episodes()
    if not eps:
        pytest.skip(f"no recorded sim sessions from {CAMPAIGN} on this machine")
    from franka_sim import ground_truth as G
    from franka_sim.eval.stage3 import GT_T_SUCCESS_TOL_S

    crit = yaml.safe_load(CRITERIA.read_text())["success"]
    success = {k: crit[k] for k in ("lift_min_m", "hold_s", "timeout_s", "slip_tol_m")}
    n, dts, bad = 0, [], []
    for run, ep in eps:
        try:
            gt = G.evaluate_episode(ep, {"success": success})
        except (OSError, ValueError, KeyError):
            gt = None
        j = replay_judge(ep, success)
        if gt is None or j is None:
            continue
        n += 1
        if bool(j["success"]) != bool(gt["success"]):     # None (undecided at the end of the log) counts as not a success
            bad.append(f"{run}: judge {j['success']} ({j['reason']}) vs gt {gt['success']} ({gt['reason']})")
        elif gt["success"]:
            dt = abs(float(j["t_success_s"]) - float(gt["t_success_s"]))
            dts.append(dt)
            if dt > GT_T_SUCCESS_TOL_S:
                bad.append(f"{run}: t_success judge {j['t_success_s']} vs gt {gt['t_success_s']} (dt {dt:.3f} s)")
    if n == 0:
        pytest.skip("recorded sessions have no sim lines")
    print(f"\n[offline judge] {n} episodes, {len(dts)} successes, max dt {max(dts) if dts else None}")
    assert dts, "no success episode to compare t_success on"
    assert not bad, bad
