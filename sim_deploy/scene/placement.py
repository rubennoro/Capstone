"""Eraser placements: seeded random ones for benchmarks, and the "replay placement" for check G6.

    from sim_deploy.scene.placement import Scene
    sc = Scene.load()                                  # model.json + station + randomization.yaml
    pose = sc.random_pose(seed, n)                     # [x, y, z, qw, qx, qy, qz] in O (scene REP set_object pose)
    rp = sc.replay_placement("replay:<zarr>#<index>")  # dict with 'pose' and how it was derived

Replay placement (G6). The replay policy (copy's policy/replay_policy.py) makes the arm repeat the recorded motion
relative to wherever the TCP is when the rollout starts: O_T_TCP(t) = O_T_TCP_start . inv(P_s) . P_j. The eraser
must therefore sit where the recorded grasp happens, carried over the same way:

    O_T_grasp = O_T_TCP_start . inv(P_s) . P_g

P_s = the episode's first recorded TCP pose, P_g = the recorded TCP pose when the fingers start closing (row g, from
the width signal: the last row before the width falls below open - 15 % of the open-to-hold drop, with the drop
reaching hold = the episode's minimum width), O_T_TCP_start = the sim's TCP where the rollout starts (home "start":
the start pose of stations/start_pose.yaml, P2; home "deploy": model.json home.O_T_tcp). Assumptions (the data has no object pose):
  1. The eraser centre lies on the grasp TCP's approach axis, ``centre_behind_tip_m`` (default half the eraser's
     resting height) back from the fingertip point toward the hand: the fingertips at the eraser's bottom, i.e. at
     the table, as the recordings show (their frame has the table at z ~ 0 and the grasp tip at -1..+1 cm).
  2. The fingers close across the eraser's body y (its width when flat, its thickness on_long_edge): body y is laid
     along the grasp closing axis (TCP x) projected onto the table, body x (its length) along TCP y projected.
  3. The eraser rests flat on the table: x, y come from step 1, z is the table top + half its height. The difference
     between the step-1 height and the resting height is reported as ``height_error_m``: a large value means the
     recorded motion would grasp above (> 0) or below (< 0) the eraser from the sim's home (start the arm elsewhere,
     or accept that G6 checks the path but not the grasp).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sim_deploy import ROOT
from sim_deploy.model.build import eraser_box, eraser_qpos, load_station_file
from sim_deploy.scene.zarr_lowdim import load_episode, parse_replay_target

DEFAULT_MODEL = ROOT / "build" / "sim_rig" / "model.json"
DEFAULT_RANDOM = ROOT / "scene" / "randomization.yaml"
HOMES = ("start", "deploy")              # P2: the start pose (PC key p, stations/start_pose.yaml), the deployment home
HOME_ALIASES = {"trainstart": "start"}   # E7's training-like home is now the start pose (its q is the stand-in)
RANDOM_START = ROOT / "scene" / "randomization_start.yaml"


def canonical_home(home: str) -> str:
    home = HOME_ALIASES.get(home, home)
    if home not in HOMES:
        raise ValueError(f"home {home!r}: one of {HOMES} (or the alias trainstart = start)")
    return home


def grasp_rows(widths: np.ndarray) -> dict:
    """Rows of the first closing: start (fingers begin to close) and end (hold reached) from the width signal."""
    w = np.asarray(widths, float)
    n0 = max(1, min(30, len(w) // 10))
    w_open = float(np.median(w[:n0]))
    i_min = int(np.argmin(w))
    w_hold = float(w[i_min])
    drop = w_open - w_hold
    if drop < 0.005:
        raise ValueError(f"no grasp in this episode: width drops only {drop * 1e3:.1f} mm")
    below = np.flatnonzero(w[: i_min + 1] < w_open - 0.5 * drop)
    i_half = int(below[0])
    start = i_half
    while start > 0 and w[start] < w_open - 0.15 * drop:
        start -= 1
    end = i_half
    while end < len(w) - 1 and w[end] > w_hold + 0.1 * drop:
        end += 1
    return {"start": start, "half": i_half, "end": end, "w_open": w_open, "w_hold": w_hold}


@dataclass
class Scene:
    model: dict
    station: dict
    random: dict
    home: str = "deploy"
    start: dict | None = None            # home "start": q, O_T_tcp, cameras, heading (scene/start_pose.py)
    safety: dict | None = None           # the sim PC's effective safety block (control_sim.yaml over config.yaml)

    @classmethod
    def load(cls, model_json=DEFAULT_MODEL, random_yaml=None, home: str = "deploy", start_pose=None) -> "Scene":
        """home "deploy" (the copied config.yaml q_home_rad, model.json's key home; region randomization.yaml) or
        "start" (P2: the arm at stations/start_pose.yaml's q_start_rad, after the PC's key p; region
        randomization_start.yaml, relative to that pose). "trainstart" is an alias of "start"."""
        from sim_deploy.scene import start_pose as sp

        home = canonical_home(home)
        model = json.loads(Path(model_json).read_text())
        station = load_station_file(ROOT.parent / model["station_file"])
        st = None
        if home == "start":
            path = Path(start_pose) if start_pose is not None else sp.START_POSE_YAML
            q = sp.read_q(path)
            if q is None:
                raise sp.StartPoseError(f"{path}: q_start_rad is not set: python -m sim_deploy.scene start-pose "
                                        f"--q ... | --from-probe FILE | --from-git (or use home deploy)")
            f = sp.fk(q, model_json)
            num = sp.tcp_numbers(f["O_T_tcp"], f["table_top_z_m"])
            if num["heading_deg"] is None:
                raise sp.StartPoseError(f"{path}: the start pose points the fingertips straight down: no heading to "
                                        f"place the eraser ahead of")
            st = {"q": q, "file": str(path.resolve()), "O_T_tcp": f["O_T_tcp"], "O_T_cam_cv": f["O_T_cam_cv"],
                  "contacts": f["contacts"], **num}
            if random_yaml is None:
                random_yaml = RANDOM_START
        random_yaml = DEFAULT_RANDOM if random_yaml is None else random_yaml
        rnd = yaml.safe_load(Path(random_yaml).read_text())
        want = "sim_deploy.scene_random_start/v1" if home == "start" else "sim_deploy.scene_random/v1"
        if rnd.get("schema") != want:
            raise ValueError(f"{random_yaml}: schema {rnd.get('schema')!r}, home {home} needs {want!r}")
        rnd["_path"] = str(Path(random_yaml).resolve())
        safety = sp.effective_config(sp.CONTROL_SIM)["safety"]
        return cls(model, station, rnd, home, st, safety)

    @property
    def sc(self) -> dict:
        return self.station["scene"]

    @property
    def eraser_size(self) -> np.ndarray:
        return eraser_box(self.sc)

    def pose(self, x: float, y: float, yaw_deg: float) -> list:
        return eraser_qpos(self.sc, x, y, yaw_deg)

    def default_pose(self) -> list:
        dp = self.sc["eraser"]["default_pose"]
        return self.pose(*dp["xy_m"], dp["yaw_deg"])

    def random_pose(self, seed: int, rollout: int) -> list:
        """Rollout ``rollout`` of seed ``seed``: [x, y, z, qw, qx, qy, qz] in O (scene REP set_object)."""
        return self.random_placement(seed, rollout)["pose"]

    def random_placement(self, seed: int, rollout: int) -> dict:
        """The pose and how it was drawn. deploy: x, y, yaw uniform in randomization.yaml (O). start: ahead, left,
        yaw_rel uniform in randomization_start.yaml relative to the start pose, then clipped to the table + box."""
        rng = np.random.default_rng([int(seed), int(rollout)])
        r = self.random
        if self.home != "start":
            x = rng.uniform(*r["x_m"])
            y = rng.uniform(*r["y_m"])
            yaw = float(self.sc["eraser"]["default_pose"]["yaw_deg"]) + rng.uniform(*r["yaw_deg"])
            return {"pose": self.pose(x, y, yaw), "clipped": False}
        ahead, left, yaw_rel = rng.uniform(*r["ahead_m"]), rng.uniform(*r["left_m"]), rng.uniform(*r["yaw_rel_deg"])
        return self.start_relative(ahead, left, yaw_rel)

    def start_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(origin on the table, ahead, left, heading deg) of the start pose's eraser frame (scene/start_pose.py)."""
        T = np.asarray(self.start["O_T_tcp"], float)
        h = np.deg2rad(self.start["heading_deg"])
        ahead = np.array([np.cos(h), np.sin(h)])
        left = np.array([-ahead[1], ahead[0]])
        return T[:2, 3].copy(), ahead, left, float(self.start["heading_deg"])

    def start_relative(self, ahead_m: float, left_m: float, yaw_rel_deg: float, clip: bool = True) -> dict:
        o, f, lft, hdg = self.start_frame()
        xy = o + ahead_m * f + left_m * lft
        yaw = (hdg + yaw_rel_deg + 90.0) % 180.0 - 90.0      # the box is symmetric: [-90, 90)
        raw = self.pose(xy[0], xy[1], yaw)
        if not clip:
            return {"pose": raw, "clipped": False, "ahead_m": ahead_m, "left_m": left_m, "yaw_rel_deg": yaw_rel_deg}
        c = self.footprint(raw)[:, :2]
        half = np.max(np.abs(c - xy), axis=0)
        t = self.sc["table"]
        lo = np.maximum([t["x_m"][0], t["y_m"][0]], np.asarray(self.safety["workspace_min_m"], float)[:2]) + half
        hi = np.minimum([t["x_m"][1], t["y_m"][1]], np.asarray(self.safety["workspace_max_m"], float)[:2]) - half
        if np.any(lo > hi):
            raise ValueError(f"the eraser does not fit on the table inside the box (lo {lo}, hi {hi})")
        xy_c = np.clip(xy, lo, hi)
        moved = float(np.linalg.norm(xy_c - xy))
        return {"pose": self.pose(xy_c[0], xy_c[1], yaw), "clipped": moved > 1e-9, "clip_moved_m": moved,
                "ahead_m": ahead_m, "left_m": left_m, "yaw_rel_deg": yaw_rel_deg, "unclipped_xy": xy.tolist()}

    def home_tcp(self) -> np.ndarray:
        """The TCP where each rollout starts (c is pressed): the start pose's, or the deploy home's."""
        if self.home == "start":
            return np.array(self.start["O_T_tcp"], float)
        return np.array(self.model["home"]["O_T_tcp"], float)

    def home_q(self) -> list:
        return list(self.start["q"]) if self.home == "start" else list(self.model["home"]["q_arm_rad"])

    def replay_placement(self, target: str, centre_behind_tip_m: float | None = None) -> dict:
        if centre_behind_tip_m is None:                    # tips at the eraser's bottom = the table (as recorded)
            centre_behind_tip_m = float(self.eraser_size[2]) / 2
        path, index = parse_replay_target(target)
        ep = load_episode(path, index)
        P = ep["poses"]
        g = grasp_rows(ep["widths"])
        G = np.linalg.inv(P[0]) @ P[g["start"]]
        O_T_grasp = self.home_tcp() @ G
        z_t = O_T_grasp[:3, 2]
        c = O_T_grasp[:3, 3] - centre_behind_tip_m * z_t
        x_t = O_T_grasp[:3, 0]
        ang = float(np.degrees(np.arctan2(x_t[1], x_t[0]))) - 90.0      # local y (width) along the closing axis
        yaw = (ang + 90.0) % 180.0 - 90.0                                 # the box is symmetric: [-90, 90)
        pose = self.pose(c[0], c[1], yaw)
        rest_z = pose[2]
        tilt = float(np.degrees(np.arccos(np.clip(-z_t[2], -1, 1))))
        return {
            "target": target, "dataset": str(path), "index": index, "episode_name": ep["name"],
            "rows": ep["rows"], "grasp_rows": g, "grasp_t_s": g["start"] / 60.0,
            "G_rel_start": G.tolist(), "O_T_grasp": O_T_grasp.tolist(),
            "centre_behind_tip_m": centre_behind_tip_m, "pose": pose, "yaw_deg": yaw,
            "height_error_m": float(c[2] - rest_z), "approach_tilt_deg": tilt,
            "grasp_tip_above_table_m": float(O_T_grasp[2, 3] - float(self.sc["table"]["top_z_m"])),
            "on_table": self.on_table(pose), "in_workspace": self.in_workspace(O_T_grasp[:3, 3]),
            # the recording's own frame has z up with the table at z ~ 0 (umiproc): where the handheld really was
            "recorded_start_tip_height_m": float(P[0][2, 3]), "recorded_grasp_tip_height_m": float(P[g["start"]][2, 3]),
            "recorded_start_tilt_deg": float(np.degrees(np.arccos(np.clip(-P[0][2, 2], -1, 1)))),
            "recorded_grasp_tilt_deg": float(np.degrees(np.arccos(np.clip(-P[g["start"]][2, 2], -1, 1)))),
            "recorded_closing_axis_vertical_component": float(P[g["start"]][2, 0]),
        }

    def on_table(self, pose) -> bool:
        t = self.sc["table"]
        corners = self.footprint(pose)
        return bool(np.all((corners[:, 0] >= t["x_m"][0]) & (corners[:, 0] <= t["x_m"][1])
                           & (corners[:, 1] >= t["y_m"][0]) & (corners[:, 1] <= t["y_m"][1])))

    def footprint(self, pose) -> np.ndarray:
        """(8, 3) corners of the eraser box at ``pose`` in O."""
        from scipy.spatial.transform import Rotation

        R = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
        h = self.eraser_size / 2
        s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float)
        return (s * h) @ R.T + np.asarray(pose[:3])

    def in_workspace(self, p) -> bool:
        """p inside the sim PC's workspace box (control_sim.yaml's safety over config.yaml's, P2; was config.yaml's)."""
        lo = np.array(self.safety["workspace_min_m"], float)
        hi = np.array(self.safety["workspace_max_m"], float)
        return bool(np.all(np.asarray(p) >= lo) and np.all(np.asarray(p) <= hi))

    def in_view(self, pose, cam: str = "cam_imx219", margin_px: float = 0.0) -> bool:
        """All 8 eraser corners project inside the model camera's pinhole image at the home (no occlusion test)."""
        c = self.model["cameras"][cam]
        T = np.array(self.start["O_T_cam_cv"][cam] if self.home == "start" else c["O_T_cam_cv_home"], float)
        K = np.array(c["K_opencv"], float)
        w, h = c["resolution_px"]
        p = (self.footprint(pose) - T[:3, 3]) @ T[:3, :3]
        if np.any(p[:, 2] <= 0):
            return False
        uv = (p[:, :2] / p[:, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        return bool(np.all((uv >= -0.5 + margin_px) & (uv <= np.array([w, h]) - 0.5 - margin_px)))

    def check_region(self) -> dict:
        """Corners of the randomisation region (deploy: x, y, yaw extremes in O; start: ahead, left, yaw_rel extremes
        at the start pose, before clipping): on the table, in the box, in the IMX219 view (at the home / start)."""
        r = self.random
        rows = []
        if self.home == "start":
            for ah in r["ahead_m"]:
                for le in r["left_m"]:
                    for yr in r["yaw_rel_deg"]:
                        rp = self.start_relative(ah, le, yr, clip=False)
                        p = rp["pose"]
                        rows.append({"ahead": ah, "left": le, "yaw_rel": yr, "x": round(p[0], 4), "y": round(p[1], 4),
                                     **self._corner_checks(p)})
            med = r.get("median") or {}
            if med:
                p = self.start_relative(med["ahead_m"], med["left_m"], med["yaw_rel_deg"], clip=False)["pose"]
                rows.append({"median": True, "x": round(p[0], 4), "y": round(p[1], 4), **self._corner_checks(p)})
        else:
            base = float(self.sc["eraser"]["default_pose"]["yaw_deg"])
            for x in r["x_m"]:
                for y in r["y_m"]:
                    for yaw in r["yaw_deg"]:
                        p = self.pose(x, y, base + yaw)
                        rows.append({"x": x, "y": y, "yaw": base + yaw, **self._corner_checks(p)})
        return {"corners": rows, "ok": all(c["on_table"] and c["in_workspace"] and c["in_view_imx219"] for c in rows)}

    def _corner_checks(self, p) -> dict:
        fp = self.footprint(p)
        return {"on_table": self.on_table(p),
                "in_workspace": bool(self.in_workspace(p[:3]) and all(self.in_workspace([c[0], c[1], p[2]]) for c in fp)),
                "in_view_imx219": self.in_view(p, "cam_imx219"), "in_view_imx519": self.in_view(p, "cam_imx519")}
