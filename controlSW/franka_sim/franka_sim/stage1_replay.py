"""Stage 1: offline kinematic feasibility of the recorded demos on the simulated Franka (no dynamics).

Why: before building the physical station, decide with numbers whether Franka + Franka Hand + printed fingers + phone
mount can perform the 45 recorded demos, and where to put the eraser, the home pose, the finger extension and the camera
side. Each demo's TCP path (ARKit y-up world) is rotated into z-up, anchored so that its grasp TCP lands at (gx, gy,
table_top + grasp_tcp_height) with a yaw about z only, converted to the flange frame (``base_T_tcp @ inv(EE_T_TCP)``)
and tracked with warm-started IK. Per (demo, placement): reach, joint margin, manipulability, joint velocity /
acceleration ratios (Savitzky-Golay), distances (tooling vs links 1-6, hand/arm/tooling vs table, self), safety box and
cone, Cartesian speed, gripper-width map. The bridge's leaky-bucket speed clamp is run at 200 Hz through the real
``SafetyFilter`` (box and cone disabled there, so the clamp statistics isolate the speed limit; they do not depend on the
placement, only on the flange path, hence they are computed once per design).

Search: coarse (10 Hz, early exit at the first violation) over a (gx, gy, yaw) grid, then 60 Hz on the top-k with full
metrics. Score (higher is better, in order): number of fully feasible demos, worst joint margin, mean per-demo minimum
margin, -clamp fraction, mean progress (fraction of frames tracked before the first violation; only guides the search
when nothing is feasible). A demo is feasible when reach = 100 %, joint margin >= criteria.stage1.joint_margin_deg_min,
no collision (distance < 0), tooling clearance >= 0.010 m and no joint-velocity jump (> ``jump_dq_ratio`` x dq_max).
Joint margin: IK keeps 0.02 rad from the limits; a frame whose margin is below the criterion is re-solved once with the
IK limit margin raised to the criterion (warm start), so a feasible null-space choice is not missed.

Design sweep: robots x finger lengths x camera sides; each combination rebuilds the model through the station
``sets``. The length axis depends on ``tooling.finger_adapter.mode`` (A9): ``box`` sweeps
``tooling.finger_adapter.extension_m`` (unchanged), ``kit`` sweeps ``tooling.finger_adapter.adapter_length_m``,
``stock`` has no length axis (camera side only). The side is ``tooling.tcp_x_along_hand``. Every combo row and the
recommendation also report ``flange_to_tip_m`` (flange to the fingertip along flange z, from the built model), so box
and kit designs compare directly. ``--lengths`` maps to the mode's key; ``--finger-extensions`` is its old alias.

Usage: python -m franka_sim stage1 --zarr Z --ids robot=panda,tooling=...,sensor=...,scene=... \\
           [--run criteria=block_raise,sim=stage1] [--robots panda,fr3] [--finger-mode stock|box|kit] \\
           [--lengths 0,0.05,0.1 | --finger-extensions 0,0.05,0.1] \\
           [--camera-sides +y,-y] [--demos all|0,3,7] [--jobs 4] [--quick] --out DIR
Outputs (DIR, outside the repo): <combo>/stage1.json (schema stage1/v1), design_summary.json, summary.txt; <combo> =
<robot>_ext<e>_<side> (box, unchanged), <robot>_len<L>_<side> (kit), <robot>_stock_<side> (stock).
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from franka_sim import frames as fr
from franka_sim import model as _model  # noqa: F401  (puts franka_deploy on sys.path before prepare_demo needs it)

SCHEMA = "stage1/v1"
REACH_LIMIT_M = 1.10   # flange further than this from the base is out of reach (Panda/FR3 reach ~0.96 m)
DEMO_HZ = 60.0
DEFAULTS = {"n_grid": [7, 7], "x_range_m": [0.30, 0.70], "y_range_m": [-0.30, 0.30], "yaw_step_deg": 45,
            "coarse_hz": 10, "n_fine_top_k": 5, "ik_tol_m": 0.001, "ik_tol_deg": 0.5, "n_sg_window": 11,
            "n_sg_order": 3, "tooling_clearance_min_m": 0.010, "jump_dq_ratio": 10.0, "filter_hz": 200.0,
            "dist_cap_m": 0.10, "n_init_seeds": 16, "n_screen_every": 5, "n_null_iters": 5, "n_coarse_full": 30,
            "n_grasp_branches": 3, "branch_sep_rad": 0.10, "gate_cone": False, "gate_cone_fraction_max": 0.0,
            "gate_box": False, "gate_box_fraction_max": 0.0}


# ------------------------------------------------------------------------------------------------------ demos


def load_demos(path, ids=None) -> list[dict]:
    """Low-dimensional demo data from a zarr zip (needs only zarr) or an npz fixture (pos, rotvec, width, ends).
    Each demo: name, T (N,4,4 ARKit y-up world TCP poses), w (N,) gripper width, t (N,) seconds at 60 Hz."""
    path = str(path)
    if path.endswith(".npz"):
        z = np.load(path)
        pos, rv, w, ends = z["pos"], z["rotvec"], z["width"], z["ends"]
        names = [f"ep{i}" for i in range(len(ends))]
    else:
        import zarr

        root = zarr.open(zarr.ZipStore(path, mode="r"), mode="r")
        pos = root["data/gripper_right_eef_pos"][:].astype(float)
        rv = root["data/gripper_right_eef_rot_axis_angle"][:].astype(float)
        w = root["data/gripper_right_gripper_width"][:, 0].astype(float)
        ends = root["meta/episode_ends"][:]
        names = [str(x) for x in root["meta/episode_names"][:]]
    R = Rotation.from_rotvec(rv).as_matrix()
    out, s = [], 0
    for i, e in enumerate(ends):
        e = int(e)
        T = np.tile(np.eye(4), (e - s, 1, 1))
        T[:, :3, :3], T[:, :3, 3] = R[s:e], pos[s:e]
        out.append({"idx": i, "name": names[i], "T": T, "w": np.asarray(w[s:e], float),
                    "t": np.arange(e - s) / DEMO_HZ})
        s = e
    if ids is not None:
        out = [out[i] for i in ids]
    return out


def prepare_demo(d: dict) -> dict:
    """z-up TCP poses, grasp index (``metrics.phases``; width minimum as fallback), grasp point, TCP speeds."""
    T = fr.arkit_to_zup(d["T"])
    height = (d["T"][:, :3, 3] - d["T"][0, :3, 3]) @ np.array([0.0, 1.0, 0.0])
    # No ImportError fallback: a silent switch to the width minimum moved the anchor (A3 review finding 1).
    from franka_deploy.eval.metrics import phases

    ig = None
    ph = phases(d["w"], height, d["t"])
    if ph is not None:
        ig = int(round(ph["t_grasp"] * DEMO_HZ))
    grasp_from = "phases"
    if ig is None:
        ig, grasp_from = int(np.argmin(d["w"])), "min_width"
    ig = int(np.clip(ig, 0, len(T) - 1))
    v = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * DEMO_HZ
    return {**d, "T_zup": T, "i_grasp": ig, "grasp_from": grasp_from, "T_grasp": T[ig],
            "v_tcp": v, "duration_s": float(d["t"][-1] - d["t"][0]) or 1.0 / DEMO_HZ}


# ------------------------------------------------------------------------------------------------ config


def load_cfg(station) -> dict:
    cfg = dict(DEFAULTS)
    sim = station.groups.get("sim") or {}
    cfg.update({k: v for k, v in sim.items() if k in DEFAULTS})
    return cfg


def placement_grid(cfg: dict, quick: bool = False) -> list[tuple[float, float, float]]:
    nx, ny = (3, 3) if quick else cfg["n_grid"]
    xs = np.linspace(*cfg["x_range_m"], nx)
    ys = np.linspace(*cfg["y_range_m"], ny)
    yaws = np.deg2rad(np.arange(0.0, 360.0, 90.0 if quick else float(cfg["yaw_step_deg"])))
    return [(float(x), float(y), float(w)) for x in xs for y in ys for w in yaws]


# tooling.finger_adapter.mode -> the swept length key (None: no length axis)
SWEEP_KEY = {"box": "extension_m", "kit": "adapter_length_m", "stock": None}


def station_spec(ids: dict, run: dict, robot: str, ext, side: str, sets=(), key: str | None = "extension_m") -> dict:
    """Station spec for one combo; ``ext`` is the value of ``tooling.finger_adapter.<key>`` (None = station value)."""
    s = list(sets)
    if ext is not None:
        if key is None:
            raise ValueError("finger mode stock has no length axis: drop --lengths / --finger-extensions")
        s.append(f"tooling.finger_adapter.{key}={ext}")
    if side is not None:
        s.append(f"tooling.tcp_x_along_hand={side}")
    return {"station": {**ids, "robot": robot}, "run": dict(run), "sets": s}


# ----------------------------------------------------------------------------------------------- context


class Ctx:
    """Per-process model context: station, MuJoCo model, ``Kin`` on the EE frame, thresholds."""

    def __init__(self, spec: dict, root=None):
        from franka_sim.kinematics import Kin
        from franka_sim.model import STATION_ROOT, build_model, load_station

        self.spec = spec
        self.st = load_station({"station": spec["station"], "run": spec["run"]},
                               root=Path(root) if root else STATION_ROOT, sets=spec["sets"])
        self.cfg = load_cfg(self.st)
        self.sm = build_model(self.st)
        self.kin = Kin(self.sm, "ee")
        self.EE_T_TCP = self.sm.EE_T_TCP
        self.TCP_T_EE = fr.inv_T(self.EE_T_TCP)
        c = self.st.get("criteria.stage1")
        self.margin_req = np.deg2rad(float(c["joint_margin_deg_min"]))
        self.reach_min = float(c["reachable_fraction_min"])
        self.rob = self.st.groups["robot"]
        j = self.rob["joints"]
        self.dq_max, self.ddq_max = np.array(j["dq_max_rad_s"], float), np.array(j["ddq_max_rad_s2"], float)
        self.dddq_max = np.array(j["dddq_max_rad_s3"], float)
        self.v_cart_max = float(self.rob["cartesian"]["v_max_m_s"])
        self.top_z = float(self.st.get("scene.table.top_z_base_m"))
        self.grasp_h = float(self.st.get("scene.grasp_tcp_height_m"))
        self.wm = {k: float(v) for k, v in self.st.get("tooling.width_map").items()}
        self.stroke = float(self.st.get("tooling.hand.stroke_m"))
        self.safety = self.st.bridge_fragment()["safety"]
        self.home_q = np.array(self.rob["home_q_rad"], float)
        self._build_pairs()
        self.rng = np.random.default_rng(0)
        self.seeds = np.vstack([self.home_q, self.rng.uniform(self.kin.q_lo, self.kin.q_hi,
                                                              (self.cfg["n_init_seeds"] - 1, 7))])
        self.tol = (float(self.cfg["ik_tol_m"]), float(np.deg2rad(self.cfg["ik_tol_deg"])))

    def _build_pairs(self) -> None:
        """Collision pair lists from ``Kin.groups`` (link0..link7 by joint body, fixed in A2fix ca84c9d).

        tooling: housing/cradle vs links 0-6 (no adjacency filter, so link 6 is included); table: links 1-7, hand and
        tooling vs the table, finger pads excluded (reported separately as ``pad_table``) and the base (link 0, which
        stands on it) excluded; self: links 0-7 + hand, adjacent/excluded bodies and the hand's own parts skipped."""
        k, m = self.kin, self.kin.m
        g = k.groups
        self.link_geoms = {i: list(g[f"link{i}"]) for i in range(8)}
        arm = list(g["arm"])                                     # links 1-7
        pads = set(g["pads"])
        hand = list(g["hand"])
        tool, tb = list(g["tooling"]), list(g["table"])
        self._tool_pairs = [(a, b) for a in tool for i in range(0, 7) for b in self.link_geoms[i]]
        # Finger pads are excluded from the table check: they reach below the TCP, so at the placeholder
        # grasp_tcp_height_m they dip into the table (reported separately as ``pad_table``, informational).
        self._table_pairs = [(a, b) for a in arm + hand + tool if a not in pads for b in tb]
        self._pad_pairs = [(a, b) for a in g["pads"] for b in tb]
        hb = {int(m.geom_bodyid[x]) for x in hand}
        ah = self.link_geoms[0] + arm + hand
        self._self_pairs = [(a, b) for n, a in enumerate(ah) for b in ah[n + 1:]
                            if a != b and not k._skip(a, b, True) and not (m.geom_bodyid[a] in hb and m.geom_bodyid[b] in hb)]

    def distances(self, q) -> tuple[float, float, float]:
        """(tooling vs links 1-6, hand/arm/tooling vs table, self), metres, capped at ``dist_cap_m``. The tooling
        distance is computed over all pairs explicitly (no adjacency filter), so it includes link 6."""
        k, cap = self.kin, self.cfg["dist_cap_m"]
        k._set(q)
        dt = min((k._dist(a, b, cap) for a, b in self._tool_pairs), default=cap)
        db = min((k._dist(a, b, cap) for a, b in self._table_pairs), default=cap)
        ds = min((k._dist(a, b, cap) for a, b in self._self_pairs), default=cap)
        return float(min(dt, cap)), float(min(db, cap)), float(min(ds, cap))

    def _first(self, T, kw):
        """Frame 0: IK from every seed; prefer solutions with clearance (tooling/table/self >= the tooling
        threshold), then the largest joint margin; otherwise the lowest residual."""
        thr = self.cfg["tooling_clearance_min_m"]
        runs = [(*self.kin.ik(T, s, **kw), i) for i, s in enumerate(self.seeds)]
        oks = [r for r in runs if r[1]["err_p"] <= self.tol[0] and r[1]["err_r"] <= self.tol[1]]
        if not oks:
            q, info, _ = min(runs, key=lambda r: r[1]["err_p"] / self.tol[0] + r[1]["err_r"] / self.tol[1])
            return q, info

        def key(r):
            return (min(min(self.distances(r[0])), thr), min(self.kin.joint_margin(r[0]).min(), self.margin_req * 2))

        q, info, _ = max(oks, key=key)
        return q, info

    def grasp_branches(self, T) -> list[np.ndarray]:
        """Distinct IK solutions (branches) at the grasp pose, best first, at most ``n_grasp_branches``.

        IK from every seed with the criterion limit margin (plain IK only if none converges), each null-space centred;
        solutions closer than ``branch_sep_rad`` (max joint difference) to a better one are dropped. Rank: clearance
        (min of tooling/table/self, capped at the tooling threshold), then joint margin (capped at 2x the criterion),
        then distance to the home q; the seed index breaks exact ties. The caller tracks the whole demo from each branch
        and keeps the best whole-demo result, so the outcome does not hinge on which seed converged first."""
        if float(np.linalg.norm(T[:3, 3])) > REACH_LIMIT_M:
            return []
        (tp, tr), thr = self.tol, self.cfg["tooling_clearance_min_m"]
        good = lambda i: i["err_p"] <= tp and i["err_r"] <= tr  # noqa: E731
        kw = dict(tol_p=tp * 0.2, tol_r=tr * 0.2, restarts=0)
        sols = []
        for kwi in ({**kw, "limit_margin": self.margin_req}, kw):
            for i, s in enumerate(self.seeds):
                q, info = self.kin.ik(T, s, **kwi)
                if good(info):
                    sols.append((self._refine(T, q) if self.cfg["n_null_iters"] > 0 else q, i))
            if sols:
                break

        def key(r):
            q, i = r
            return (-min(min(self.distances(q)), thr), -min(float(self.kin.joint_margin(q).min()), self.margin_req * 2),
                    float(np.linalg.norm(q - self.home_q)), i)

        out: list[np.ndarray] = []
        for q, _ in sorted(sols, key=key):
            if all(np.max(np.abs(q - b)) >= self.cfg["branch_sep_rad"] for b in out):
                out.append(q)
            if len(out) >= int(self.cfg["n_grasp_branches"]):
                break
        return out

    def solve(self, T, q_prev, first: bool = False):
        """IK to the EE pose; acceptance = the spec residual (1 mm, 0.5 deg). Returns (q, ok, err_p, err_r).

        The joint limit margin of the IK is the criterion (10 deg), so a solution that exists with that margin is
        found without a branch jump; if none exists the plain IK (0.02 rad) is used so the true margin is reported
        (and the caller flags ``joint_margin``)."""
        k, (tp, tr) = self.kin, self.tol
        dist = float(np.linalg.norm(T[:3, 3]))
        if dist > REACH_LIMIT_M:                       # far outside any arm configuration: skip the IK
            return (self.home_q if q_prev is None else q_prev), False, dist - REACH_LIMIT_M, float(np.pi)
        good = lambda i: i["err_p"] <= tp and i["err_r"] <= tr  # noqa: E731
        kw = dict(tol_p=tp * 0.2, tol_r=tr * 0.2, restarts=0)
        kw_m = {**kw, "limit_margin": self.margin_req}
        if first or q_prev is None:
            q, info = self._first(T, kw_m)
            if not good(info):
                q, info = self._first(T, kw)
        else:
            q, info = k.ik(T, q_prev, **kw_m)
            if not good(info):
                q2, info2 = k.ik_multistart(T, np.vstack([q_prev, self.home_q, self.seeds[1:3]]), **kw_m)
                if good(info2) or info2["err_p"] < info["err_p"]:
                    q, info = q2, info2
            if not good(info):
                q2, info2 = k.ik(T, q_prev, **kw)
                if good(info2) or info2["err_p"] < info["err_p"]:
                    q, info = q2, info2
        if good(info) and self.cfg["n_null_iters"] > 0:
            q = self._refine(T, q)
            info = {"err_p": float(np.linalg.norm(self._err(T, q)[:3])), "err_r": float(np.linalg.norm(self._err(T, q)[3:]))}
        return q, bool(good(info)), float(info["err_p"]), float(info["err_r"])

    def _err(self, T, q):
        from franka_sim.kinematics import rot_err

        Tc = self.kin.fk(q)
        return np.r_[T[:3, 3] - Tc[:3, 3], rot_err(Tc[:3, :3], T[:3, :3])]

    def _refine(self, T, q):
        """Null-space joint-centering: move toward the middle of the joint ranges without changing the pose
        (a few projected steps, then a re-convergence), so the trajectory keeps away from limits and the local IK
        does not run into a limit and flip branch. Kept only if the pose error stays within tolerance."""
        k = self.kin
        mid = 0.5 * (k.q_lo + k.q_hi)
        half = 0.5 * (k.q_hi - k.q_lo)
        q0, qq = q, q.copy()
        for _ in range(int(self.cfg["n_null_iters"])):
            J = k.jac(qq)
            N = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J
            dq = N @ ((mid - qq) / half) * 0.15
            qq = np.clip(qq + np.clip(dq, -0.03, 0.03), k.q_lo + self.margin_req, k.q_hi - self.margin_req)
        q2, _ = k.ik(T, qq, tol_p=self.tol[0] * 0.2, tol_r=self.tol[1] * 0.2, restarts=0, limit_margin=self.margin_req,
                     iters=20)
        e = self._err(T, q2)
        return q2 if (np.linalg.norm(e[:3]) <= self.tol[0] and np.linalg.norm(e[3:]) <= self.tol[1]) else q0


# ---------------------------------------------------------------------------------------------- tracking


def place(ctx: Ctx, dem: dict, gx: float, gy: float, yaw: float):
    """(base_T_world, base_T_EE (N,4,4), base_T_tcp (N,4,4)) for a demo at grid point (gx, gy, yaw)."""
    bTw = fr.anchor_placement(dem["T_grasp"], [gx, gy, ctx.top_z + ctx.grasp_h], yaw)
    bTt = bTw @ dem["T_zup"]
    return bTw, bTt @ ctx.TCP_T_EE, bTt


def track_demo(ctx: Ctx, T_ee: np.ndarray, frame_idx: np.ndarray, hz: float, *, early_exit: bool,
               i_grasp: int | None = None, full: bool = False, q_grasp: np.ndarray | None = None) -> dict:
    """Track the EE poses T_ee[frame_idx]; returns scalars (and per-frame arrays when ``full``).

    The IK is seeded (multistart, clearance-aware) at the frame nearest ``i_grasp`` (the task-critical pose) and
    warm-started outward, forwards to the end and backwards to frame 0 (deviation from "multistart at frame 0": the
    start pose is arbitrary, the grasp is what constrains the placement). With ``early_exit`` each direction stops at
    the first violation (reach, margin, collision, tooling clearance, jump); ``first_fail`` is the lowest violating
    60 Hz frame, ``reason`` its cause; progress = processed fraction of the sampled frames."""
    n = len(frame_idx)
    q_arr = np.zeros((n, 7))
    reach = np.zeros(n, bool)
    seen = np.zeros(n, bool)
    marg = np.zeros((n, 7))
    dists = np.zeros((n, 3))
    manip = np.zeros(n)
    fails: dict[int, str] = {}
    dt = 1.0 / hz
    g = 0 if i_grasp is None else int(np.argmin(np.abs(np.asarray(frame_idx) - i_grasp)))

    def one(j, q_prev, first):
        if first and q_grasp is not None:          # a branch from Ctx.grasp_branches (already within tolerance)
            q, ok = np.array(q_grasp, float), True
        else:
            q, ok, ep, er = ctx.solve(T_ee[frame_idx[j]], q_prev, first=first)
        q_arr[j], reach[j], seen[j] = q, ok, True
        marg[j] = ctx.kin.joint_margin(q)
        d = ctx.distances(q)
        dists[j] = d
        why = None
        if not ok:
            why = "unreachable"
        elif marg[j].min() < ctx.margin_req - 1e-6:
            why = "joint_margin"
        elif min(d) < 0.0:
            why = "collision"
        elif d[0] < ctx.cfg["tooling_clearance_min_m"]:
            why = "tooling_clearance"
        elif q_prev is not None and np.max(np.abs(q - q_prev) / dt / ctx.dq_max) > ctx.cfg["jump_dq_ratio"]:
            why = "jump"
        if why:
            fails[j] = why
        if full:
            manip[j] = ctx.kin.manipulability(q)
        return q, ok, why

    q0 = None
    for seg in (range(g, n), range(g - 1, -1, -1)):
        q_prev, first = (q0 if seg.start != g else None), seg.start == g
        for j in seg:
            q, ok, why = one(j, q_prev, first)
            first = False
            if j == g:
                q0 = q
            if ok:
                q_prev = q
            if why and early_exit:
                break
        if early_exit and fails:
            break
    done = int(seen.sum())
    jf = min(fails) if fails else None
    first_fail = int(frame_idx[jf]) if jf is not None else None
    sl = seen
    out = {"n_frames": n, "n_done": done, "progress": done / n, "reach_fraction": float(reach[sl].sum() / n) if full
           else float(reach[sl].mean()),
           "first_fail": first_fail, "reason": fails[jf] if jf is not None else None, "feasible": not fails,
           "min_margin_deg": float(np.rad2deg(marg[sl].min())),
           "min_margin_joint": int(np.unravel_index(np.argmin(marg[sl]), marg[sl].shape)[1]) + 1,
           "min_tooling_clearance_m": float(dists[sl, 0].min()), "min_table_m": float(dists[sl, 1].min()),
           "min_self_m": float(dists[sl, 2].min()), "n_collisions": int((dists[sl] < 0).any(axis=1).sum())}
    if full:
        out["pad_table_min_m"] = min(_pad_table(ctx, q_arr[j]) for j in range(n))
        out.update(_dynamics(ctx, q_arr, reach, manip, hz))
        out["_q"], out["_reach"] = q_arr, reach
    return out


def _gate_safety(ctx: Ctx, T_ee: np.ndarray, frame_idx, r: dict) -> dict:
    """Optional feasibility gates on the bridge approach cone / workspace box (``gate_cone`` / ``gate_box``, default
    off; the cone value is a user decision). A demo fails when the violating fraction of its sampled frames exceeds
    ``gate_*_fraction_max``; ``first_fail`` becomes the earliest violating frame if that is earlier."""
    c = ctx.cfg
    if not (c["gate_cone"] or c["gate_box"]):
        return r
    viol = _safety_flags(ctx, T_ee[np.asarray(frame_idx)])
    for name, on, fmax in (("cone", c["gate_cone"], c["gate_cone_fraction_max"]),
                           ("box", c["gate_box"], c["gate_box_fraction_max"])):
        v = viol[name]
        if on and v.mean() > float(fmax) + 1e-12:
            f = int(np.asarray(frame_idx)[np.argmax(v)])
            r["feasible"] = False
            if r["first_fail"] is None or f < r["first_fail"]:
                r["first_fail"], r["reason"] = f, name
    return r


def _demo_key(r: dict) -> tuple:
    """Whole-demo quality of one tracked branch (higher is better)."""
    return (bool(r["feasible"]), float(r["progress"]), float(r["min_margin_deg"]), float(r["min_tooling_clearance_m"]),
            float(min(r["min_table_m"], r["min_self_m"])))


def track_best(ctx: Ctx, T_ee: np.ndarray, frame_idx: np.ndarray, hz: float, *, early_exit: bool,
               i_grasp: int | None = None, full: bool = False) -> dict:
    """``track_demo`` from each grasp-frame IK branch (``Ctx.grasp_branches``); keeps the best whole-demo result
    (``_demo_key``; the lower branch rank wins exact ties). Branches are compared with early exit at the tracking rate;
    with ``full`` / no early exit the chosen branch is re-tracked over all frames. Adds ``grasp_branch`` (rank) and
    ``n_grasp_branches``."""
    frame_idx = np.asarray(frame_idx)
    g = 0 if i_grasp is None else int(np.argmin(np.abs(frame_idx - i_grasp)))
    br = ctx.grasp_branches(T_ee[frame_idx[g]])
    if not br:                                   # nothing converges at the grasp: report the best-residual attempt
        r = track_demo(ctx, T_ee, frame_idx, hz, early_exit=early_exit, i_grasp=i_grasp, full=full)
        return {**_gate_safety(ctx, T_ee, frame_idx, r), "grasp_branch": None, "n_grasp_branches": 0}
    runs = [_gate_safety(ctx, T_ee, frame_idx, track_demo(ctx, T_ee, frame_idx, hz, early_exit=True, i_grasp=i_grasp,
                                                          q_grasp=q)) for q in br]
    b = max(range(len(br)), key=lambda i: (_demo_key(runs[i]), -i))
    r = runs[b]
    if full or not early_exit:
        r = _gate_safety(ctx, T_ee, frame_idx, track_demo(ctx, T_ee, frame_idx, hz, early_exit=early_exit,
                                                          i_grasp=i_grasp, full=full, q_grasp=br[b]))
    return {**r, "grasp_branch": b, "n_grasp_branches": len(br)}


def _pad_table(ctx: Ctx, q) -> float:
    ctx.kin._set(q)
    return float(min(ctx.kin._dist(a, b, ctx.cfg["dist_cap_m"]) for a, b in ctx._pad_pairs))


def _dynamics(ctx: Ctx, q, reach, manip, hz) -> dict:
    n = len(q)
    w = min(int(ctx.cfg["n_sg_window"]) | 1, n if n % 2 else n - 1)
    o = min(int(ctx.cfg["n_sg_order"]), w - 1)
    out = {"min_manipulability": float(manip.min())}
    if w < 5:
        return out | {"peak_dq_ratio": 0.0, "peak_ddq_ratio": 0.0, "peak_jerk_ratio": 0.0}
    d = [savgol_filter(q, w, o, deriv=k, delta=1.0 / hz, axis=0) for k in (1, 2, 3)]
    for key, arr, lim in (("dq", d[0], ctx.dq_max), ("ddq", d[1], ctx.ddq_max), ("jerk", d[2], ctx.dddq_max)):
        out[f"peak_{key}_ratio"] = float(np.max(np.abs(arr) / lim))
    out["jerk_note"] = "noise-dominated (finite differences of tracked IK); informational"
    return out


def _safety_flags(ctx: Ctx, T_ee: np.ndarray) -> dict:
    """Per frame: flange outside the bridge workspace box; flange z-axis angle to the bridge cone axis (deg) and
    whether it exceeds ``cone_max_deg`` (what ``SafetyFilter`` would clamp on hardware)."""
    s = ctx.safety
    lo, hi = np.array(s["workspace_min"]), np.array(s["workspace_max"])
    p = T_ee[:, :3, 3]
    ax = np.array(s["cone_axis"], float)
    ax /= np.linalg.norm(ax)
    ang = np.degrees(np.arccos(np.clip(T_ee[:, :3, 2] @ ax, -1, 1)))
    return {"box": np.any((p < lo) | (p > hi), axis=1), "angle": ang, "cone": ang > float(s["cone_max_deg"])}


def safety_metrics(ctx: Ctx, T_ee: np.ndarray) -> dict:
    """Workspace box and approach cone violations of the flange path (fractions of frames, max cone angle)."""
    f = _safety_flags(ctx, T_ee)
    return {"box_violation_fraction": float(f["box"].mean()), "cone_max_angle_deg": float(f["angle"].max()),
            "cone_violation_fraction": float(f["cone"].mean()), "cone_max_deg": float(ctx.safety["cone_max_deg"])}


def width_metrics(ctx: Ctx, w: np.ndarray) -> dict:
    """Training width w = a * gap + b  ->  commanded physical gap = (w - b) / a: the fraction outside the hand's
    [gap_min, gap_max] and the p99 of the gap against the stroke."""
    gap = (w - ctx.wm["b"]) / ctx.wm["a"]
    out = (gap < ctx.wm["gap_min_m"]) | (gap > ctx.wm["gap_max_m"])
    p99 = float(np.percentile(gap, 99))
    return {"gap_outside_fraction": float(out.mean()), "gap_min_m": float(gap.min()), "gap_max_m": float(gap.max()),
            "gap_p99_m": p99, "mapped_p99_le_stroke": bool(p99 <= ctx.stroke)}


def clamp_stats(ctx: Ctx, T_ee: np.ndarray, duration_s: float) -> dict:
    """Run the flange path through the real SafetyFilter at ``filter_hz`` (box/cone disabled): clamp fraction,
    maximum backlog (m) and slowdown = time until the filtered target reaches the final pose / demo duration."""
    from franka_deploy.bridge.safety import SafetyConfig, SafetyFilter

    cfg = SafetyConfig.from_dict(ctx.safety)
    cfg.workspace_min, cfg.workspace_max, cfg.cone_max_deg = (-1e3,) * 3, (1e3,) * 3, 180.0
    f_hz, n = float(ctx.cfg["filter_hz"]), len(T_ee)
    t_src = np.arange(n) / DEMO_HZ
    tt = np.minimum(np.arange(0.0, t_src[-1] + 1e-9, 1.0 / f_hz), t_src[-1])
    pos = np.stack([np.interp(tt, t_src, T_ee[:, i, 3]) for i in range(3)], axis=1)
    rot = Slerp(t_src, Rotation.from_matrix(T_ee[:, :3, :3]))(tt)
    R = rot.as_matrix()

    def pose(p, r):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = r, p
        return T

    filt = SafetyFilter(cfg)
    filt.reset(pose(pos[0], R[0]), t=0.0)
    n_clamp, backlog = 0, 0.0
    for k in range(1, len(tt)):
        out, flags = filt.apply(pose(pos[k], R[k]), float(tt[k]))
        n_clamp += bool(flags.get("step_pos") or flags.get("step_rot"))
        backlog = max(backlog, float(np.linalg.norm(pos[k] - out[:3, 3])))
    n_ticks = len(tt) - 1
    t_end, goal = float(tt[-1]), pose(pos[-1], R[-1])
    extra_max = 20.0 * duration_s
    while t_end - tt[-1] < extra_max:
        dp = np.linalg.norm(goal[:3, 3] - filt.last[:3, 3])
        dr = (Rotation.from_matrix(filt.last[:3, :3]).inv() * Rotation.from_matrix(goal[:3, :3])).magnitude()
        if dp < 1e-6 and dr < 1e-4:
            break
        t_end += 1.0 / f_hz
        out, flags = filt.apply(goal, t_end)
        n_ticks += 1
        n_clamp += bool(flags.get("step_pos") or flags.get("step_rot"))
    return {"clamp_fraction": n_clamp / max(n_ticks, 1), "max_backlog_m": backlog,
            "slowdown": float(t_end / duration_s)}


def cartesian_speed(ctx: Ctx, dem: dict, T_ee: np.ndarray) -> dict:
    """Speeds at 60 Hz. ``time_above_vmax_fraction`` is on the flange path (the path the bridge clamp limits, same
    as ``clamp_stats``); the TCP-path number (what the dataset statistics use) is ``tcp_time_above_vmax_fraction``."""
    v = dem["v_tcp"]
    vf = np.linalg.norm(np.diff(T_ee[:, :3, 3], axis=0), axis=1) * DEMO_HZ
    lim = float(ctx.st.get("criteria.limits.v_max_m_s"))
    return {"tcp_speed_p99_m_s": float(np.percentile(v, 99)), "tcp_speed_max_m_s": float(v.max()),
            "flange_speed_p99_m_s": float(np.percentile(vf, 99)), "flange_speed_max_m_s": float(vf.max()),
            "tcp_speed_datasheet_max_m_s": ctx.v_cart_max, "time_above_vmax_fraction": float((vf > lim).mean()),
            "tcp_time_above_vmax_fraction": float((v > lim).mean())}


# ----------------------------------------------------------------------------------------- search workers

_CTX: Ctx | None = None
_DEMOS: list[dict] = []


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def limit_threads() -> None:
    """One BLAS/OpenMP thread per process (workers are processes; 7x7 algebra gains nothing from threads, and
    oversubscription pushed the load past 100 on 20 cores). Effective only before numpy loads BLAS: the
    ``franka_sim stage1`` entry in ``franka_sim.__main__`` sets these before any import; the call in the pool
    initializer only matters for children that load BLAS themselves (spawn)."""
    for k in THREAD_ENV:
        os.environ.setdefault(k, "1")


def _init_worker(spec, root, demos):
    global _CTX, _DEMOS
    limit_threads()
    _CTX, _DEMOS = Ctx(spec, root), demos


def _coarse_task(args):
    """One placement over all demos at the coarse rate."""
    pi, (gx, gy, yaw), dis = args
    ctx = _CTX
    step = max(1, int(round(DEMO_HZ / ctx.cfg["coarse_hz"])))
    rows = []
    for di in dis:
        dem = _DEMOS[di]
        _, T_ee, _ = place(ctx, dem, gx, gy, yaw)
        idx = np.unique(np.r_[np.arange(0, len(T_ee), step), len(T_ee) - 1])
        r = track_best(ctx, T_ee, idx, DEMO_HZ / step, early_exit=True, i_grasp=dem['i_grasp'])
        rows.append({k: r[k] for k in ("feasible", "progress", "first_fail", "reason", "min_margin_deg",
                                       "min_tooling_clearance_m")})
    return pi, rows


def _fine_task(args):
    pi, di, (gx, gy, yaw), q_start = args
    ctx, dem = _CTX, _DEMOS[di]
    bTw, T_ee, bTt = place(ctx, dem, gx, gy, yaw)
    r = track_best(ctx, T_ee, np.arange(len(T_ee)), DEMO_HZ, early_exit=False, i_grasp=dem['i_grasp'], full=True)
    q, reach = r.pop("_q"), r.pop("_reach")
    r.update(safety_metrics(ctx, T_ee))
    r["start_q"] = q[0].tolist() if reach[0] else None
    r["start_tcp_pose7"] = fr.T_to_pose7(bTt[0]).tolist()
    r["grasp_index"] = dem["i_grasp"]
    r["feasible"] = bool(r["feasible"] and r["reach_fraction"] >= 1.0 and r["first_fail"] is None)
    r["base_T_world"] = bTw.tolist()
    return pi, di, r


def _static_task(di):
    """Placement-independent metrics of one demo (clamp, TCP speed, width)."""
    ctx, dem = _CTX, _DEMOS[di]
    # EE path in the demo's own z-up world (a rigid placement does not change speeds).
    T_ee = dem["T_zup"] @ ctx.TCP_T_EE
    return di, {**clamp_stats(ctx, T_ee, dem["duration_s"]), **cartesian_speed(ctx, dem, T_ee),
                **width_metrics(ctx, dem["w"])}


def _score(rows: list[dict]) -> tuple:
    feas = [r for r in rows if r["feasible"]]
    worst = min((r["min_margin_deg"] for r in feas), default=-1e9)
    mean_m = float(np.mean([r["min_margin_deg"] for r in rows]))
    prog = float(np.mean([r["progress"] for r in rows]))
    return (len(feas), worst, mean_m, prog)


class _Map:
    """Pool.imap_unordered with a serial fallback for jobs == 1 (tests, debugging)."""

    def __init__(self, jobs, spec, root, demos):
        self.jobs = jobs
        if jobs <= 1:
            _init_worker(spec, root, demos)
            self.pool = None
        else:
            self.pool = mp.get_context("fork").Pool(jobs, _init_worker, (spec, root, demos))

    def imap(self, fn, items):
        return map(fn, items) if self.pool is None else self.pool.imap_unordered(fn, items, chunksize=1)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# ---------------------------------------------------------------------------------------- one combination


def run_combo(spec: dict, demos: list[dict], *, root=None, jobs: int = 1, quick: bool = False,
              grid: list | None = None, top_k: int | None = None, log=print) -> dict:
    """Search one (robot, finger extension, camera side) design. Returns the ``stage1/v1`` dict."""
    t0 = time.time()
    m = _Map(jobs, spec, root, demos)
    try:
        ctx = _CTX if jobs <= 1 else Ctx(spec, root)
        grid = grid or placement_grid(ctx.cfg, quick)
        k_top = 1 if quick and top_k is None else (top_k or ctx.cfg["n_fine_top_k"])
        n_d = len(demos)
        screen = list(range(0, n_d, int(ctx.cfg["n_screen_every"]))) if n_d > 12 else list(range(n_d))
        coarse = {}
        for pi, rows in m.imap(_coarse_task, [(i, g, screen) for i, g in enumerate(grid)]):
            coarse[pi] = rows
        if len(screen) < n_d:      # two-tier coarse: the best screened placements get every demo
            order = sorted(coarse, key=lambda i: (_score(coarse[i]), -i), reverse=True)[:int(ctx.cfg["n_coarse_full"])]
            rest = [d for d in range(n_d) if d not in screen]
            for pi, rows in m.imap(_coarse_task, [(i, grid[i], rest) for i in order]):
                by = dict(zip(screen, coarse[pi]))
                by.update(zip(rest, rows))
                coarse[pi] = [by[d] for d in range(n_d)]
            for i in set(coarse) - set(order):      # screened only: pad with the screen result (marked partial)
                coarse[i] = coarse[i] + [{"feasible": False, "progress": 0.0, "first_fail": None, "reason": "not_run",
                                          "min_margin_deg": 0.0, "min_tooling_clearance_m": 0.0}] * (n_d - len(screen))
        scored = sorted(((_score(coarse[i]), i) for i in coarse), key=lambda x: (x[0], -x[1]), reverse=True)
        placements = [{"gx": grid[i][0], "gy": grid[i][1], "yaw_deg": float(np.rad2deg(grid[i][2])),
                       "coarse_score": {"n_feasible": s[0], "worst_margin_deg": s[1] if s[0] else None,
                                        "mean_min_margin_deg": s[2], "mean_progress": s[3]}}
                      for s, i in scored]
        log(f"  coarse {len(grid)} placements in {time.time() - t0:.0f} s; best {scored[0][0][0]}/{len(demos)} feasible")
        static = dict(m.imap(_static_task, list(range(len(demos)))))
        fine = {}
        top = [i for _, i in scored[:k_top]]
        tasks = [(pi, di, grid[pi], None) for pi in top for di in range(len(demos))]
        for pi, di, r in m.imap(_fine_task, tasks):
            fine.setdefault(pi, {})[di] = r
        fine_rows = {}
        for pi in top:
            rows = []
            for di in range(len(demos)):
                r = {**fine[pi][di], **static[di]}
                rows.append(r)
            fine_rows[pi] = rows
        fine_scored = sorted(
            ((_score(fine_rows[pi]) + (-float(np.mean([r["clamp_fraction"] for r in fine_rows[pi]])),), pi)
             for pi in top), key=lambda x: (x[0], -x[1]), reverse=True)
        for p in placements:
            p["fine_score"] = None
        for s, pi in fine_scored:
            for p in placements:
                if (p["gx"], p["gy"], p["yaw_deg"]) == (grid[pi][0], grid[pi][1], float(np.rad2deg(grid[pi][2]))):
                    p["fine_score"] = {"n_feasible": s[0], "worst_margin_deg": s[1] if s[0] else None,
                                       "mean_min_margin_deg": s[2], "mean_progress": s[3], "neg_clamp_fraction": s[4]}
        best_pi = fine_scored[0][1]
        gx, gy, yaw = grid[best_pi]
        rows = fine_rows[best_pi]
        rec = _recommend(ctx, demos, rows, (gx, gy, yaw))
    finally:
        m.close()
    feas = [r for r in rows if r["feasible"]]
    summary = {
        "n_demos": len(demos), "n_feasible": len(feas),
        "n_feasible_dq_ok": sum(r["peak_dq_ratio"] <= 1.0 for r in feas),
        "peak_dq_ratio_median": float(np.median([r["peak_dq_ratio"] for r in rows])),
        "cone_max_deg": float(ctx.safety["cone_max_deg"]),
        "cone_violation_fraction_mean": float(np.mean([r["cone_violation_fraction"] for r in rows])),
        "cone_violation_fraction_max": max(r["cone_violation_fraction"] for r in rows),
        "n_demos_over_cone": sum(r["cone_violation_fraction"] > 0 for r in rows),
        "n_feasible_demos_over_cone": sum(r["cone_violation_fraction"] > 0 for r in feas),
        "cone_max_angle_deg": max(r["cone_max_angle_deg"] for r in rows),
        "box_violation_fraction_mean": float(np.mean([r["box_violation_fraction"] for r in rows])),
        "n_demos_outside_box": sum(r["box_violation_fraction"] > 0 for r in rows),
        "gates": {"cone": bool(ctx.cfg["gate_cone"]), "box": bool(ctx.cfg["gate_box"])},
        "n_grasp_from_min_width": sum(d["grasp_from"] != "phases" for d in demos),
        "pad_table_min_m": min(r["pad_table_min_m"] for r in rows),
        "worst_tooling_clearance_m": min(r["min_tooling_clearance_m"] for r in rows),
        "worst_tooling_clearance_feasible_m": min((r["min_tooling_clearance_m"] for r in feas), default=None),
        "worst_joint_margin_feasible_deg": min((r["min_margin_deg"] for r in feas), default=None),
        "worst_joint_margin_all_demos_deg": min(r["min_margin_deg"] for r in rows),
        "clamp_fraction": float(np.mean([r["clamp_fraction"] for r in rows])),
        "time_above_vmax_fraction": float(np.mean([r["time_above_vmax_fraction"] for r in rows])),
        "tcp_time_above_vmax_fraction": float(np.mean([r["tcp_time_above_vmax_fraction"] for r in rows])),
        "slowdown_mean": float(np.mean([r["slowdown"] for r in rows])),
        "slowdown_max": max(r["slowdown"] for r in rows),
        "infeasible": {demos[i]["name"]: {"reason": r["reason"], "first_fail": r["first_fail"],
                                          "reach_fraction": r["reach_fraction"]}
                       for i, r in enumerate(rows) if not r["feasible"]},
    }
    design = {"extension_m": float(ctx.st.get("tooling.finger_adapter.extension_m")),
              "camera_side": str(ctx.st.get("tooling.tcp_x_along_hand")), **design_fields(ctx)}
    return {"schema": SCHEMA, "robot": ctx.sm.robot, "design": design, "station_ids": ctx.st.ids, "station_sets": spec["sets"],
            "station_fingerprint": ctx.sm.station_fingerprint, "quick": quick, "coarse_hz": ctx.cfg["coarse_hz"],
            "config": ctx.cfg, "criteria": {"joint_margin_deg_min": float(np.rad2deg(ctx.margin_req)),
                                            "tooling_clearance_min_m": ctx.cfg["tooling_clearance_min_m"],
                                            "v_max_m_s": ctx.safety["v_max"]},
            "grasp_tcp_height_m": ctx.grasp_h, "F_T_TCP": ctx.sm.F_T_TCP.tolist(),
            "placements": placements, "recommendation": rec, "summary": summary,
            "demos": [{"name": demos[i]["name"], "grasp_from": demos[i]["grasp_from"], **{
                k: v for k, v in rows[i].items() if k != "base_T_world"}} for i in range(len(demos))],
            "wall_s": round(time.time() - t0, 1)}


def _recommend(ctx: Ctx, demos, rows, gp) -> dict:
    gx, gy, yaw = gp
    bTw = [np.array(r["base_T_world"]) for r in rows]
    # Home TCP pose: median of the demo start TCP poses (grasp point shared by all demos), orientation = mean rotation.
    P = np.array([r["start_tcp_pose7"] for r in rows])
    Ts = np.array([fr.pose7_to_T(p) for p in P])
    pos = np.median(Ts[:, :3, 3], axis=0)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    T_home = fr.make_T(R, pos)
    T_ee = T_home @ ctx.TCP_T_EE
    q, info = ctx.kin.ik_multistart(T_ee, ctx.seeds, tol_p=1e-4, tol_r=1e-3)
    d = ctx.distances(q)
    return {"grid_point": {"gx": gx, "gy": gy}, "yaw_deg": float(np.rad2deg(yaw)),
            "grasp_xyz_base": [gx, gy, ctx.top_z + ctx.grasp_h],
            "base_T_world": {demos[i]["name"]: bTw[i].tolist() for i in range(len(demos))},
            "start_q": {demos[i]["name"]: rows[i]["start_q"] for i in range(len(demos))},
            "start_tcp_pose7": {demos[i]["name"]: rows[i]["start_tcp_pose7"] for i in range(len(demos))},
            "home": {"tcp_pose7": fr.T_to_pose7(T_home).tolist(), "tcp_xyz_m": pos.tolist(), "q": q.tolist(),
                     "ik_ok": bool(info["ok"]), "joint_margin_deg": float(np.rad2deg(ctx.kin.joint_margin(q).min())),
                     "tooling_clearance_m": d[0], "table_m": d[1], "self_m": d[2],
                     "start_pos_spread_m": float(np.linalg.norm(Ts[:, :3, 3] - pos, axis=1).max())}}


# ------------------------------------------------------------------------------------------------ sweep


def flange_to_tip_m(sm) -> float:
    """Flange to the fingertip along flange z (m), from the built model, for every finger mode: kit =
    ``info["finger_kit"]["flange_to_tip_m"]`` (TCP at the tips); box / stock = derived TCP z + pad_length/2 (TCP at the
    pad centre; the tip = the pads' distal end). The derived TCP (not a station F_T_TCP override) is the geometry."""
    info = sm.info
    fk = info.get("finger_kit") or {}
    if fk.get("flange_to_tip_m") is not None:
        return float(fk["flange_to_tip_m"])
    z = float(np.asarray(info["F_T_TCP_derived"], float)[2, 3])
    return z if info.get("tcp_at") == "tip" else z + float(info["pad_length_m"]) / 2


def design_fields(ctx) -> dict:
    """A9 design keys of one combo: finger mode, the swept key and its value, flange_to_tip, TCP definition."""
    info = ctx.sm.info
    mode = str(info.get("finger_mode", "box"))
    key = SWEEP_KEY[mode]
    d = {"finger_mode": mode, "swept_key": key,
         "swept_value": None if key is None else float(ctx.st.get(f"tooling.finger_adapter.{key}")),
         "flange_to_tip_m": flange_to_tip_m(ctx.sm), "tcp_at": info.get("tcp_at"),
         "pad_length_m": float(info["pad_length_m"])}
    if mode == "kit":
        d["adapter_length_m"] = d["swept_value"]
        fk = info.get("finger_kit") or {}
        d["finger_kit"] = {k: fk.get(k) for k in ("id", "sha", "variant")}
    return d


def combo_dir(robot, ext, side, mode: str = "box") -> str:
    """Box: ``<robot>_ext<e>_<py|my>`` (unchanged); kit: ``<robot>_len<L>_<side>``; stock: ``<robot>_stock_<side>``."""
    sd = f"{'p' if side == '+y' else 'm'}y"
    if mode == "kit":
        return f"{robot}_len{ext:g}_{sd}"
    if mode == "stock":
        return f"{robot}_stock_{sd}"
    return f"{robot}_ext{ext:g}_{sd}"


def _json(obj) -> str:
    return json.dumps(obj, indent=1, default=float)


def strip_times(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "wall_s"}


def design_row(res: dict, robot, ext, side) -> dict:
    s = res["summary"]
    dz = res["design"]
    return {"robot": robot, "extension_m": dz["extension_m"], "camera_side": dz["camera_side"],
            **{k: dz.get(k) for k in ("finger_mode", "swept_key", "swept_value", "flange_to_tip_m")},
            "n_feasible": s["n_feasible"],
            "n_feasible_dq_ok": s["n_feasible_dq_ok"], "peak_dq_ratio_median": s["peak_dq_ratio_median"],
            "n_demos": s["n_demos"], "worst_tooling_clearance_m": s["worst_tooling_clearance_m"],
            "worst_tooling_clearance_feasible_m": s["worst_tooling_clearance_feasible_m"],
            "worst_joint_margin_feasible_deg": s["worst_joint_margin_feasible_deg"],
            "worst_joint_margin_all_demos_deg": s["worst_joint_margin_all_demos_deg"],
            "clamp_fraction": s["clamp_fraction"], "time_above_vmax_fraction": s["time_above_vmax_fraction"],
            "tcp_time_above_vmax_fraction": s["tcp_time_above_vmax_fraction"],
            "n_demos_over_cone": s["n_demos_over_cone"], "cone_violation_fraction_mean": s["cone_violation_fraction_mean"],
            "cone_max_angle_deg": s["cone_max_angle_deg"], "cone_max_deg": s["cone_max_deg"],
            "n_demos_outside_box": s["n_demos_outside_box"], "gates": s["gates"],
            "slowdown_mean": s["slowdown_mean"], "slowdown_max": s["slowdown_max"],
            "placement": {**res["recommendation"]["grid_point"], "yaw_deg": res["recommendation"]["yaw_deg"]},
            "wall_s": res["wall_s"]}


def _wm(r) -> float:
    m = r["worst_joint_margin_feasible_deg"]
    return -1e9 if m is None else m


def _sv(r):
    """The row's swept length (A9: extension_m in box mode, adapter_length_m in kit, None in stock)."""
    return r["extension_m"] if r.get("finger_mode") is None else r["swept_value"]


def recommend_design(rows: list[dict]) -> dict:
    """Per robot: the best (length, side) by (n_feasible, worst tooling clearance, worst feasible-demo margin,
    -clamp fraction). Overall: the (length, side) with the best worst-robot score if some design suits every robot
    (``robot_agnostic``: every robot reaches >= 95 % of the demos), else the best design of the robot with the most
    feasible demos. ``feasible_fraction_by_robot`` is reported either way. The length is the swept key of the finger
    mode (``swept_key``; box: ``extension_m``, kit: ``adapter_length_m``, stock: none); the chosen design's
    ``flange_to_tip_m`` is reported (``flange_to_tip_m_by_robot`` per robot)."""
    def key(r):
        return (r["n_feasible"], r["worst_tooling_clearance_m"], _wm(r), -r["clamp_fraction"])

    robots = sorted({r["robot"] for r in rows})
    best = {rb: max((r for r in rows if r["robot"] == rb), key=key) for rb in robots}
    groups = {}
    for r in rows:
        groups.setdefault((_sv(r), r["camera_side"]), []).append(r)
    agn = max(groups, key=lambda g: (min(x["n_feasible"] for x in groups[g]),
                                     min(_wm(x) for x in groups[g]),
                                     -max(x["clamp_fraction"] for x in groups[g])))
    n_demos = rows[0]["n_demos"]
    agnostic = min(x["n_feasible"] for x in groups[agn]) >= 0.95 * n_demos
    lead = max(robots, key=lambda rb: key(best[rb]))
    ch = (agn if agnostic else (_sv(best[lead]), best[lead]["camera_side"]))
    key0 = rows[0].get("swept_key", "extension_m")
    fields = ("extension_m", "camera_side", "finger_mode", "swept_key", "swept_value", "flange_to_tip_m", "n_feasible", "n_feasible_dq_ok", "worst_joint_margin_feasible_deg",
              "worst_joint_margin_all_demos_deg", "worst_tooling_clearance_m", "clamp_fraction", "placement")
    frac = {x["robot"]: x["n_feasible"] / x["n_demos"] for x in groups[ch]} if ch in groups else {}
    chosen = groups.get(ch, [])
    f2t = {x["robot"]: x.get("flange_to_tip_m") for x in chosen}
    lead_row = next((x for x in chosen if x["robot"] == lead), chosen[0] if chosen else {})
    out = {"extension_m": ch[0] if key0 == "extension_m" else lead_row.get("extension_m"),
           "camera_side": ch[1], "robot_agnostic": agnostic, "robot": None if agnostic else lead,
           "robot_agnostic_rule": "every robot >= 95 % of the demos feasible with the same design",
           "feasible_fraction_by_robot": frac,
           "per_robot_best": {rb: {k: best[rb].get(k) for k in fields} for rb in robots}}
    if "swept_key" in rows[0]:
        out.update(finger_mode=rows[0]["finger_mode"], swept_key=key0, swept_value=ch[0],
                   flange_to_tip_m=lead_row.get("flange_to_tip_m"), flange_to_tip_m_by_robot=f2t)
        if key0 not in (None, "extension_m"):
            out[key0] = ch[0]
    return out


def summary_text(rows, rec) -> str:
    f = lambda x, k=1.0, fmt="{:.1f}": "-" if x is None else fmt.format(x * k)  # noqa: E731
    key0 = rows[0].get("swept_key", "extension_m") if rows else "extension_m"
    lab = {"extension_m": "ext_mm", "adapter_length_m": "len_mm"}.get(key0, "len_mm")
    hdr = (f"{'robot':6s} {lab:>6s} {'f2tip_mm':>8s} {'side':>4s} {'feasible':>9s} {'tool_clr_mm':>11s} {'clr_feas_mm':>11s} "
           f"{'marg_feas':>9s} {'marg_all':>8s} {'clamp':>6s} {'t>vmax':>6s} {'tcp_t>v':>7s} {'slowdown':>9s} "
           f"{'cone>':>5s} {'cone_frac':>9s} {'cone_maxdeg':>11s} {'box>':>4s} {'dq<=1':>5s}  placement (gx, gy, yaw)")
    lines = [hdr]
    for r in rows:
        p = r["placement"]
        v, ft = _sv(r), r.get("flange_to_tip_m")
        lines.append(f"{r['robot']:6s} {v * 1e3 if v is not None else float('nan'):6.0f} "
                     f"{ft * 1e3 if ft is not None else float('nan'):8.1f} {r['camera_side']:>4s} {r['n_feasible']:>4d}/{r['n_demos']:<4d} "
                     f"{r['worst_tooling_clearance_m'] * 1e3:11.1f} {f(r['worst_tooling_clearance_feasible_m'], 1e3):>11s} "
                     f"{f(r['worst_joint_margin_feasible_deg']):>9s} {r['worst_joint_margin_all_demos_deg']:8.1f} "
                     f"{r['clamp_fraction']:6.3f} {r['time_above_vmax_fraction']:6.3f} {r['tcp_time_above_vmax_fraction']:7.3f} "
                     f"{r['slowdown_mean']:5.2f}/{r['slowdown_max']:<4.2f} {r['n_demos_over_cone']:5d} "
                     f"{r['cone_violation_fraction_mean']:9.3f} {r['cone_max_angle_deg']:11.1f} {r['n_demos_outside_box']:4d} "
                     f"{r['n_feasible_dq_ok']:5d} ({p['gx']:.2f}, {p['gy']:.2f}, {p['yaw_deg']:.0f})")
    g = rows[0]["gates"] if rows else {}
    lines.append(f"columns: {lab} = tooling.finger_adapter.{key0} (finger mode {rows[0].get('finger_mode', 'box') if rows else '?'}"
                 f"{'; stock fingers have no length axis' if key0 is None else ''}), f2tip = flange to fingertip along "
                 "flange z (from the built model); tool_clr = worst tooling clearance over all demos, clr_feas / marg_feas = over feasible demos, "
                 "marg_all = worst joint margin over all demos (incl. infeasible); clamp = SafetyFilter clamp fraction and "
                 "t>vmax = time above v_max, both on the flange path; tcp_t>v = time above v_max on the TCP path "
                 "(dataset statistic); cone> / box> = demos with any frame over the bridge cone "
                 f"({rows[0]['cone_max_deg'] if rows else '?'} deg) / outside the box; cone_frac = mean fraction of "
                 f"frames over the cone; gates cone={g.get('cone')} box={g.get('box')}")
    k = rec.get("swept_key", "extension_m")
    ft = rec.get("flange_to_tip_m")
    what = (f"finger mode {rec.get('finger_mode', 'box')}, " +
            ("no length axis" if k is None else f"{k} = {rec.get('swept_value', rec['extension_m'])} m") +
            (f" (flange_to_tip {ft:.4f} m)" if ft is not None else ""))
    lines.append(f"recommended design: {what}, camera side {rec['camera_side']}; "
                 f"robot-agnostic (>= 95 % on every robot): {rec['robot_agnostic']}; feasible fraction by robot "
                 f"{json.dumps({k: round(v, 3) for k, v in rec['feasible_fraction_by_robot'].items()})}; best per robot "
                 f"{json.dumps(rec['per_robot_best'], default=float)}")
    return "\n".join(lines)


def sweep(args, log=print) -> dict:
    from franka_sim.model import STATION_ROOT

    ids = dict(kv.split("=", 1) for kv in args.ids.split(","))
    run = dict(kv.split("=", 1) for kv in args.run.split(",")) if args.run else {}
    run.setdefault("criteria", "block_raise")
    run.setdefault("sim", "stage1")
    robots = args.robots.split(",") if args.robots else [ids.get("robot", "panda")]
    sets = list(args.set)
    if getattr(args, "finger_mode", None):
        sets.append(f"tooling.finger_adapter.mode={args.finger_mode}")
    lengths = getattr(args, "lengths", None)
    if lengths and args.finger_extensions:
        raise SystemExit("give --lengths or its alias --finger-extensions, not both")
    lengths = lengths or args.finger_extensions
    mode = sweep_mode(ids, run, robots[0], sets, args.root)
    key = SWEEP_KEY[mode]
    if lengths and key is None:
        raise SystemExit("finger_adapter.mode=stock has no length axis: drop --lengths / --finger-extensions")
    exts = [float(x) for x in lengths.split(",")] if lengths else [None]
    sides = args.camera_sides.split(",") if args.camera_sides else [None]
    out = Path(args.out).resolve()
    try:
        from franka_deploy.paths import REPO_ROOT

        if out.is_relative_to(REPO_ROOT):
            raise SystemExit(f"refusing to write inside the repo: {out}")
    except ImportError:                                                  # pragma: no cover
        pass
    out.mkdir(parents=True, exist_ok=True)
    sel = None if args.demos in (None, "all") else [int(x) for x in args.demos.split(",")]
    demos = [prepare_demo(d) for d in load_demos(args.zarr, sel)]
    log(f"{len(demos)} demos; grasp from phases: {sum(d['grasp_from'] == 'phases' for d in demos)}; finger mode "
        f"{mode}, length axis {key and 'tooling.finger_adapter.' + key}")
    root = args.root or STATION_ROOT
    rows = []
    t0 = time.time()
    for robot, ext, side in itertools.product(robots, exts, sides):
        spec = station_spec(ids, run, robot, ext, side, sets, key=key)
        log(f"[{time.time() - t0:.0f}s] {robot} {key or 'stock'}={ext} side={side}")
        res = run_combo(spec, demos, root=root, jobs=args.jobs, quick=args.quick, log=log)
        dz = res["design"]
        if dz["finger_mode"] != mode:                                   # pragma: no cover (mode is per tooling)
            raise SystemExit(f"robot {robot}: finger mode {dz['finger_mode']} != {mode} (sweep is per mode)")
        d = out / combo_dir(robot, dz["extension_m"] if mode == "box" else dz["swept_value"], dz["camera_side"], mode)
        d.mkdir(exist_ok=True)
        (d / "stage1.json").write_text(_json(res))
        rows.append(design_row(res, robot, ext, side))
        log("  " + json.dumps({k: rows[-1][k] for k in ("flange_to_tip_m", "n_feasible", "worst_joint_margin_feasible_deg",
                                                         "worst_tooling_clearance_m", "clamp_fraction")}, default=float))
    rec = recommend_design(rows)
    summ = {"schema": "stage1_design/v1", "rows": rows, "recommended": rec, "wall_s": round(time.time() - t0, 1)}
    (out / "design_summary.json").write_text(_json(summ))
    text = summary_text(rows, rec)
    (out / "summary.txt").write_text(text + "\n")
    log(text)
    return summ


def sweep_mode(ids: dict, run: dict, robot: str, sets, root=None) -> str:
    """The finger mode of the station the sweep starts from (decides the length key before any model is built)."""
    from franka_sim.model import STATION_ROOT, finger_mode, load_station

    st = load_station({"station": {**ids, "robot": robot}, "run": dict(run)},
                      root=Path(root) if root else STATION_ROOT, sets=list(sets))
    return finger_mode(st.get("tooling.finger_adapter"))


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--zarr", required=True, help="dataset zarr zip (low-dim arrays) or an npz fixture")
    p.add_argument("--ids", required=True, help="robot=panda,tooling=...,sensor=...,scene=...")
    p.add_argument("--run", default="criteria=block_raise,sim=stage1", help="run groups, e.g. criteria=block_raise,sim=stage1")
    p.add_argument("--root", default=None, help="station config root")
    p.add_argument("--set", action="append", default=[], metavar="group.path=value")
    p.add_argument("--robots", help="comma list, e.g. panda,fr3")
    p.add_argument("--finger-mode", choices=sorted(SWEEP_KEY), default=None,
                   help="sets tooling.finger_adapter.mode (default: the tooling's own mode)")
    p.add_argument("--lengths", help="metres, e.g. 0,0.05,0.1: box -> finger_adapter.extension_m, kit -> "
                                     "finger_adapter.adapter_length_m; not allowed in stock mode")
    p.add_argument("--finger-extensions", help="alias of --lengths (kept for old scripts), metres, e.g. 0,0.05,0.1")
    p.add_argument("--camera-sides", help="+y,-y (tooling.tcp_x_along_hand); a single -y needs the = form: --camera-sides=-y")
    p.add_argument("--demos", default="all", help="all or indices 0,3,7")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="coarse 3x3 grid, one fine placement")
    p.add_argument("--out", required=True, help="output directory (outside the repo)")


def run(args: argparse.Namespace) -> int:
    sweep(args)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_sim stage1", description=__doc__.split("\n\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
