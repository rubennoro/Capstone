"""The rollout record `sim_deploy.rollout/v2` (sim_deploy/evaltool/SCHEMA.md): what the driver adds to every eval so
`evaltool score` can score it later. eval.yaml's protocol block, rollouts/NNN/gt.jsonl (20 Hz sim ground truth) and
rollouts/NNN/rollout.json.

Where each gt.jsonl field comes from:
- eraser_pose, tcp_pose, gap_m, contacts_eraser, contacts_table, reflex: the sim PC's scene REP :5651 `ground_truth`
  (one call per row; the C++ sim is unchanged);
- min_z_fingers_rig_m: computed here from the newest sim.state/v1 qpos (:5650, the StateMirror, rows of another model
  sha dropped) with the driver's own MuJoCo copy of the same build (the driver checks scene.xml + meshes against
  model.json and the sim PC's `hello` sha): the lowest world z over the VERTICES of every collision geom
  (contype or conaffinity != 0) of the bodies left_finger, right_finger and rig (the rig body carries the mount);
- c2_mode: the newest C2 ArmState's mode (C2Watcher).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from sim_deploy.evaltool import constants as C

RECORD_SCHEMA = "sim_deploy.rollout/v2"
GT_BODIES = ("left_finger", "right_finger", "rig")
PAD = re.compile(r"^(pad|finger_holder)_(l|r)")
CLAMP = "clamped into the workspace box"
LOG_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) ")

# SCHEMA.md "Scoring constants" (user decisions 2026-11-21; success = lift + clear of the table 2026-11-22), values from
# sim_deploy/evaltool/constants.py: the live judge stops on these; evaltool scores on them. Hashed into protocol_sha.
SCORING = {"grasp": "left and right pad or finger_holder geoms touch the eraser in the same gt row (reported, not "
                    "part of success)",
           "success": C.SUCCESS_RULE, "success_lift_m": C.LIFT_M, "success_hold_s": C.HOLD_S,
           "success_hold_tol_s": C.HOLD_TOL_S, "success_clear_m": C.CLEAR_M,
           "success_needs_grasp": C.SUCCESS_NEEDS_GRASP,
           "timeout_s": C.TIMEOUT_S, "early_stop_on_success": True, "gt_hz": C.GT_RATE_HZ}


def sides_touching(contacts: list[dict], obj: str = "eraser") -> set[str]:
    """{'l', 'r'}: which fingers' pads / holders touch obj in one ground_truth reply."""
    return {PAD.match(n).group(2) for c in contacts
            for n, o in ((c.get("a", ""), c.get("b", "")), (c.get("b", ""), c.get("a", "")))
            if o == obj and PAD.match(n)}


def touching(contacts: list[dict], obj: str) -> list[str]:
    """The geom names in contact with obj (the scene REP already leaves out eraser-table)."""
    out = set()
    for c in contacts:
        a, b = c.get("a"), c.get("b")
        if a == obj and b:
            out.add(b)
        elif b == obj and a:
            out.add(a)
    return sorted(out)


class MinZ:
    """Lowest world z of the finger / rig collision geoms' vertices for a qpos of the same model build."""

    def __init__(self, scene_xml: Path) -> None:
        import mujoco

        self._mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.d = mujoco.MjData(self.m)
        bodies = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in GT_BODIES} - {-1}
        self.geoms: list[tuple[int, np.ndarray]] = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] not in bodies or not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                continue
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                a = self.m.mesh_vertadr[mid]
                v = np.array(self.m.mesh_vert[a:a + self.m.mesh_vertnum[mid]], float)
            else:                                         # primitives: the corners of their local AABB
                c, h = self.m.geom_aabb[g][:3], self.m.geom_aabb[g][3:]
                v = np.array([c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            self.geoms.append((g, v))
        if not self.geoms:
            raise RuntimeError(f"{scene_xml}: no collision geoms on bodies {GT_BODIES}")

    def __call__(self, qpos: np.ndarray | None) -> float | None:
        if qpos is None or len(qpos) != self.m.nq:
            return None
        self.d.qpos[:] = qpos
        self._mj.mj_kinematics(self.m, self.d)
        z = np.inf
        for g, v in self.geoms:
            R = self.d.geom_xmat[g].reshape(3, 3)
            z = min(z, float(np.min(v @ R[2]) + self.d.geom_xpos[g][2]))
        return z


def gt_row(t: float, gt: dict, min_z: float | None, c2_mode: str | None, c2_error: str | None) -> dict:
    """One gt.jsonl row from a scene `ground_truth` reply."""
    contacts = gt.get("contacts", [])
    arm = gt.get("arm")
    if arm is None or "reflex" not in arm:
        reflex = None
    elif arm["reflex"]:
        reflex = c2_error if c2_error and "reflex" in c2_error else True
    else:
        reflex = False

    def r(v, k=5):
        return None if v is None else [round(float(x), k) for x in v]

    return {"t": round(t, 3), "eraser_pose": r(gt["objects"]["eraser"]["pose"]), "tcp_pose": r(gt.get("tcp")),
            "gap_m": None if gt.get("finger_gap_m") is None else round(float(gt["finger_gap_m"]), 5),
            "contacts_eraser": touching(contacts, "eraser"),
            "contacts_table": [n for n in touching(contacts, "table") if n != "eraser"],
            "min_z_fingers_rig_m": None if min_z is None else round(min_z, 5),
            "c2_mode": c2_mode, "reflex": reflex}


# ------------------------------------------------------------------------------------------------ eval.yaml


def protocol_sha(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


SEED_KEY_RANDOM = "rng([seed, n]) per rollout n"
SEED_KEY_SET = "rollout n -> start n of the start-set file (eval/start_set.py)"


def protocol(*, seed: int, placement_source: str, placement_rules: dict | None, timeout_s: float, precision, diffusion_steps, exec_h, look: str,
             model_sha256: str, q_start_rad, safety: dict, seed_key: str = SEED_KEY_RANDOM) -> dict:
    """SCHEMA.md eval.yaml `protocol`: what must be equal for two evals' rates to be compared. A start set
    (placement_source ``set:<name>``) hashes every rollout's start (demo / real trial) and eraser pose in
    placement_rules."""
    return {"seed_rule": {"seed": int(seed), "key": seed_key, "placement_source": placement_source,
                          "placement_rules": placement_rules},
            "timeout_s": float(timeout_s), "precision": precision, "diffusion_steps": diffusion_steps,
            "exec_h": exec_h, "look": look, "model_sha256": model_sha256,
            "q_start_rad": None if q_start_rad is None else [float(x) for x in q_start_rad],
            "judge": {**SCORING, "timeout_s": float(timeout_s)}, "min_z_method": "vertex (collision geoms)",
            "limits": {"v_max_m_per_s": safety.get("v_max_m_per_s"), "w_max_rad_per_s": safety.get("w_max_rad_per_s"),
                       "cone": {k: safety.get(k) for k in ("cone_axis", "cone_max_deg", "cone_margin_deg")},
                       "box": {k: safety.get(k) for k in ("workspace_min_m", "workspace_max_m", "box_margin_m")}}}


# ------------------------------------------------------------------------------------------------ rollout.json


def _log_time(line: str) -> dt.datetime | None:
    m = LOG_TS.match(line)
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


def count_clamps(session_dir: str | None, pd_rollout_id: int | None, wall_start: str | None,
                 wall_end: str | None) -> int | None:
    """manager.log lines "clamped into the workspace box" in this rollout: between the manager's own "episode N
    started" and "episode N: <outcome>" lines (policy_deploy calls a rollout an episode) when both are there, else the
    driver's wall-clock window.
    None when there is no log."""
    if not session_dir or session_dir == "None":
        return None
    f = Path(session_dir) / "logs" / "manager.log"
    if not f.exists():
        return None
    lines = f.read_text(errors="replace").splitlines()
    a = b = None
    if pd_rollout_id is not None:
        for ln in lines:
            if a is None and f"episode {pd_rollout_id} started" in ln:
                a = _log_time(ln)
            elif a is not None and b is None and re.search(rf"\bepisode {pd_rollout_id}: ", ln):
                b = _log_time(ln)
    if a is None and wall_start:
        a = dt.datetime.fromisoformat(wall_start)
    if b is None and wall_end:
        b = dt.datetime.fromisoformat(wall_end)
    if a is None or b is None:
        return None
    n = 0
    for ln in lines:
        if CLAMP in ln:
            t = _log_time(ln)
            if t is not None and a <= t <= b:
                n += 1
    return n


STOP_REASON = {"success": "success", "timeout": "timeout", "fault": "fault", "stopped": "fault",
               "fail": "fault", "interrupted": "user", "crashed": "error", "sim_error": "error"}


def stop_reason(rec: dict) -> str:
    """SCHEMA stop_reason from the driver's outcome: success / timeout / fault (C2 fault or stopped, the manager
    ending the rollout itself, the eraser off the table) / user (Ctrl-C) / error (policy_deploy exited, the scene
    REP failed, or the rollout never started: home / start pose / c refused, not run)."""
    return STOP_REASON.get(rec.get("outcome"), "error")


def _by_kind(reasons: dict | None) -> dict | None:
    """'camera_stale: main camera stale, last frame 222 ms ago' -> 'camera_stale' (the reason's kind)."""
    if reasons is None:
        return None
    out: dict = {}
    for k, v in reasons.items():
        kind = str(k).split(":", 1)[0].strip() or "?"
        out[kind] = out.get(kind, 0) + int(v)
    return out


def rollout_doc(rec: dict, *, launcher: str, batch_id: str | None) -> dict:
    c2 = rec.get("c2") or {}
    gate = c2.get("gate") or {}
    steps = rec.get("steps")
    pl = rec.get("placement")
    placement = None
    if rec.get("eraser_start_pose") is not None:
        p = rec["eraser_start_pose"]
        w, x, y, z = (float(v) for v in p[3:7])
        yaw = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
        src = "replay" if pl is None else pl.get("source", "random")
        placement = {"xy": [float(p[0]), float(p[1])], "yaw_deg": round(yaw, 3), "pose": p, "source": src,
                     **({"held_out": pl.get("held_out"), "start": {k: v for k, v in pl.items() if k != "source"}}
                        if src in ("demo", "real_trial") else {"sampled": pl} if pl else {})}
    return {
        "record_schema": RECORD_SCHEMA, "n": rec.get("n"), "source": "sim", "launcher": launcher,
        "batch_id": batch_id, "seed_key": rec.get("seed_key"), "placement": placement,
        "held_out": rec.get("held_out"),
        "eraser_settled_pose": rec.get("eraser_settled_pose"),
        "start_pose_q": (rec.get("start_pose") or {}).get("q"),
        "outcome": rec.get("outcome"), "stop_reason": stop_reason(rec), "note": rec.get("note"),
        "clamps": rec.get("clamps"),
        "c2": {"accepted": gate.get("accepted"), "shortened": gate.get("shortened"),
               "pulled_back": gate.get("pulled_back"), "rejected": gate.get("rejected"),
               "loop_late": c2.get("loop_late"), "modes": c2.get("modes"), "errors": c2.get("errors")},
        "step_summary": None if not steps else {
            "policy_ms_p50": steps.get("policy_ms_p50"), "policy_ms_p95": steps.get("policy_ms_p95"),
            "margin_ms_min": steps.get("margin_ms_min"), "margin_ms_p50": steps.get("margin_ms_p50"),
            "sent": steps.get("sent"), "skipped": steps.get("skipped"), "skipped_by_reason": _by_kind(steps.get("skip_reasons"))},
        "loadavg_start": rec.get("loadavg_start"), "loadavg_end": rec.get("loadavg_end"),
        "loop_late": c2.get("loop_late"),
        "wall_start": rec.get("wall_start"), "wall_end": rec.get("wall_end"),
        "gt": rec.get("gt"),
    }
