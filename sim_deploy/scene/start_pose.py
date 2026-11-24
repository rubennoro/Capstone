"""The start pose (PLAN.md section 9, P2): the real PC's key `p` moves the arm from home to `homing.q_start_rad`, and
every eval episode starts there (`--home start`). This module owns ``stations/start_pose.yaml`` (the sim's
``q_start_rad``; run.sh merges it into the sim PC's station with P1's ``sim_pc/merge_station.py``), its checks in
the sim, and the eraser's place relative to it.

    python -m sim_deploy.scene start-pose --show                     the current one, with its checks
    python -m sim_deploy.scene start-pose --q "q1,...,q7"            write it from 7 joint angles (rad)
    python -m sim_deploy.scene start-pose --from-probe FILE          ... from a `pc_probe --start-pose` printout
    python -m sim_deploy.scene start-pose --from-git [REF]           ... from REF's deployment/control_deploy/station.yaml
    python -m sim_deploy.scene start-data [--dataset ZIP] [--write]  the eraser relative to the demos' start

Checks (a write is refused when one fails, unless --force): the q inside the Panda's joint limits (the real loader's
table) and the model's, no contact at the pose (the eraser aside), the TCP inside the sim PC's workspace box and its
tilt within ``safety.cone_max_deg`` (both as the real PC checks at the end of `p`: box without margin, cone without
margin), the move from home within ``homing.timeout_s`` (the PC refuses `p` otherwise), and a heading (the fingertips
not straight down: the eraser is placed ahead of them). Information only: the
distance from the demos' start (tilt, height above the table) and whether the eraser region fits at that pose.

Frame of the eraser region (``randomization_start.yaml``): origin = the start TCP projected on the table, ``ahead``
= the fingertip direction (TCP z) projected on the table, ``left`` = up x ahead; the eraser's yaw is its long axis
relative to ``ahead``. Measured on the demos by ``start-data``: per episode the eraser centre (the grasp TCP, half
the eraser height back along its approach axis, as the replay placement) relative to the episode's first pose.
"""

from __future__ import annotations

import copy
import datetime as dt
import functools
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml

from sim_deploy import REPO, ROOT, upstream

START_POSE_YAML = ROOT / "stations" / "start_pose.yaml"
RANDOM_START = ROOT / "scene" / "randomization_start.yaml"
CONTROL_SIM = ROOT / "stations" / "control_sim.yaml"
CONFIG = REPO / "deployment" / "control_deploy" / "franka_control" / "config.yaml"
OPERATOR_PY = REPO / "deployment" / "policy_deploy" / "policy_deploy" / "manager" / "operator.py"
DEFAULT_MODEL = ROOT / "build" / "sim_rig" / "model.json"
DEFAULT_DATASET = Path("~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip").expanduser()
DEFAULT_GIT_REF = upstream.ref()          # env SIM_DEPLOY_UPSTREAM_REF (sim_deploy/upstream.py)
GIT_STATION = "deployment/control_deploy/station.yaml"
# The stand-in until the arm is measured (PLAN.md section 9: E7's trainstart solution).
STAND_IN = [-0.6225485, 0.3138391, -0.1643961, -1.7825973, 0.5934631, 1.3978677, -1.6275796]
# The Panda's joint limits as franka_control's config loader checks q_start_rad (Franka's datasheet), rad.
PANDA_Q_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_Q_MAX = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
PEAK_SPEED_FACTOR = 1.875            # franka_control homing_motion.cpp: duration = max(min, 1.875 * max|dq| / v)
DEFAULT_AT_POSE_TOL = 0.05           # homing.at_pose_tol_rad of the real config.yaml (2026-11-20)
_BASE_WORDS = {(0, 1): "forward (away from the base)", (0, -1): "back (toward the base)",
               (1, 1): "to the robot's left", (1, -1): "to the robot's right"}   # pc_probe's words


class StartPoseError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- the yaml files


def read_q(path: Path = START_POSE_YAML) -> list[float] | None:
    """franka_control.homing.q_start_rad of start_pose.yaml: 7 floats, or None (no file, null)."""
    path = Path(path)
    if not path.exists():
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    q = ((doc.get("franka_control") or {}).get("homing") or {}).get("q_start_rad")
    if q is None:
        return None
    return _seven(q, f"{path}: franka_control.homing.q_start_rad")


def _seven(q, what: str) -> list[float]:
    if not isinstance(q, (list, tuple)) or len(q) != 7:
        raise StartPoseError(f"{what}: expected a list of 7 numbers, got {q!r}")
    try:
        v = [float(x) for x in q]
    except (TypeError, ValueError) as e:
        raise StartPoseError(f"{what}: expected a list of 7 numbers, got {q!r}") from e
    if not all(math.isfinite(x) for x in v):
        raise StartPoseError(f"{what}: not finite: {q!r}")
    return v


def parse_q_text(text: str) -> list[float]:
    """--q "q1,q2,...,q7" (commas and/or spaces, optional brackets)."""
    parts = [p for p in re.split(r"[\s,\[\]]+", text.strip()) if p]
    return _seven(parts, "--q")


PROBE_LINE = re.compile(r"q_start_rad:\s*\[([^\]]*)\]")


def parse_probe(text: str) -> list[float]:
    """The `q_start_rad: [...]` line `pc_probe --start-pose` prints (the last one when there are several)."""
    found = PROBE_LINE.findall(text)
    if not found:
        raise StartPoseError("no 'q_start_rad: [...]' line in the probe output (run pc_probe with --start-pose)")
    return _seven([p for p in re.split(r"[\s,]+", found[-1].strip()) if p], "the probe's q_start_rad line")


def from_git(ref: str | None = None, repo: Path | None = None) -> list[float]:
    """franka_control.homing.q_start_rad of REF:deployment/control_deploy/station.yaml (read only: git show; REF and
    the repo default to sim_deploy/upstream.py's setting)."""
    ref = ref or upstream.ref()
    repo = repo or upstream.repo()
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{GIT_STATION}"], capture_output=True, text=True,
                       env=env)
    if r.returncode != 0:
        raise StartPoseError(f"git show {ref}:{GIT_STATION} failed: {(r.stderr or r.stdout).strip()}")
    doc = yaml.safe_load(r.stdout) or {}
    q = ((doc.get("franka_control") or {}).get("homing") or {}).get("q_start_rad")
    if q is None:
        raise StartPoseError(f"{ref}:{GIT_STATION} has no franka_control.homing.q_start_rad yet (not set yet: the "
                             f"start pose is measured on the arm with pc_probe --start-pose, then pasted there)")
    return _seven(q, f"{ref}:{GIT_STATION} franka_control.homing.q_start_rad")


def compare_git(path: Path = START_POSE_YAML, ref: str | None = None, tol_rad: float = 1e-6) -> dict:
    """start_pose.yaml vs REF's station.yaml q_start_rad (read only: from_git's `git show`, in the upstream worktree
    of sim_deploy/upstream.py when it exists; REF defaults to its ref). Also returns "repo" and "ref" (the ones used). {"same": True | False | None, "text": one line}; None = could not compare. run.sh prints
    it at each start; nothing is written (stations/start_pose.yaml changes only through `start-pose --from-git`)."""
    ref = ref or upstream.ref()
    repo = upstream.repo()
    used = {"ref": ref, "repo": str(repo)}
    try:
        mine = read_q(path)
        theirs = from_git(ref, repo)
    except Exception as e:                                # noqa: BLE001 - bad YAML, no git: report, never raise
        return {"same": None, **used, "text": f"start pose: cannot compare with {ref} ({repo}): {type(e).__name__}: {e}"}
    if mine is None:
        return {"same": False, **used, "ours": None, "git": theirs,
                "text": f"start pose WARNING: {path} has no q_start_rad, {ref} has {theirs}"}
    d = max(abs(a - b) for a, b in zip(mine, theirs))
    if d <= tol_rad:
        return {"same": True, **used, "ours": mine, "git": theirs,
                "text": f"start pose = {ref}'s station.yaml homing.q_start_rad ({Path(path).name})"}
    return {"same": False, **used, "ours": mine, "git": theirs, "max_diff_rad": d,
            "text": (f"start pose WARNING: {Path(path).name} differs from {ref}'s station.yaml (max {d:.4g} rad): "
                     f"ours {[round(v, 7) for v in mine]}, git {[round(v, 7) for v in theirs]}; to adopt git's: "
                     f"python -m sim_deploy.scene start-pose --from-git")}


def write_start_pose(q: list[float], source: str, info: dict | None = None, path: Path = START_POSE_YAML) -> Path:
    now = dt.datetime.now().isoformat(timespec="seconds")
    lines = ["# sim_deploy: the sim's start pose (PLAN.md section 9, key p): franka_control.homing.q_start_rad, merged by",
             "# run.sh into the sim PC's station (sim_pc/merge_station.py; --home deploy uses null instead). Written by",
             "# `python -m sim_deploy.scene start-pose`; edit through it (it checks the pose in the sim).",
             f"# source: {source}",
             f"# date: {now}"]
    if info:
        lines.append(f"# in the sim: {summary_line(info)}")
    qs = ", ".join(f"{x:.7f}" for x in q)
    text = "\n".join(lines) + f"\nfranka_control:\n  homing:\n    q_start_rad: [{qs}]\n"
    path.write_text(text)
    return path


# ------------------------------------------------------------------------------------- config and the station


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def effective_config(station: Path = CONTROL_SIM, start_pose: Path | None = None) -> dict:
    """The copy's franka_control/config.yaml with the station's franka_control overrides (and start_pose.yaml's),
    as franka_control's loader merges them."""
    cfg = yaml.safe_load(CONFIG.read_text())
    st = yaml.safe_load(Path(station).read_text()) or {}
    cfg = _deep_merge(cfg, st.get("franka_control") or {})
    if start_pose is not None and Path(start_pose).exists():
        cfg = _deep_merge(cfg, (yaml.safe_load(Path(start_pose).read_text()) or {}).get("franka_control") or {})
    return cfg


def config_has_q_start() -> bool:
    """Whether the copy's franka_control knows homing.q_start_rad (P1's port of key p): its loader refuses unknown
    keys in a station."""
    return "q_start_rad" in ((yaml.safe_load(CONFIG.read_text()) or {}).get("homing") or {})


def laptop_has_home_key() -> bool:
    """Whether the copy's policy_deploy still has the laptop's `h` (final-deployment's start-pose change removes it:
    homing is then the PC's key only)."""
    try:
        return re.search(r'^\s*HOME\s*=\s*"h"', OPERATOR_PY.read_text(), re.M) is not None
    except OSError:
        return False


# ------------------------------------------------------------------------------------------------ FK and checks


@functools.lru_cache(maxsize=8)
def _model(scene_xml: str):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    m = mujoco.MjModel.from_xml_path(scene_xml)
    return m


def fk(q, model_json: Path = DEFAULT_MODEL) -> dict:
    """The arm at q in the sim model (fingers open, eraser at its default pose): TCP and camera poses in O, the
    contacts that do not involve the eraser, the model's joint-limit margin."""
    import mujoco

    model = json.loads(Path(model_json).read_text())
    m = _model(str(Path(model_json).with_name(model["scene_xml"])))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    d.qpos[:7] = np.asarray(q, float)
    mujoco.mj_forward(m, d)
    sid = m.site("tcp").id
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
    cams = {}
    for cam in ("cam_imx219", "cam_imx519"):
        c = m.camera(cam).id
        Tc = np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = d.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), d.cam_xpos[c]
        cams[cam] = Tc
    names = lambda g: m.geom(g).name or m.body(m.geom_bodyid[g]).name   # noqa: E731
    contacts = [(names(c.geom1), names(c.geom2), round(float(c.dist), 4)) for c in d.contact[:d.ncon]
                if "eraser" not in (m.geom(c.geom1).name, m.geom(c.geom2).name)]
    jr = m.jnt_range[:7]
    qa = np.asarray(q, float)
    return {"O_T_tcp": T, "O_T_cam_cv": cams, "contacts": contacts,
            "model_joint_margin_rad": float(np.min(np.minimum(qa - jr[:, 0], jr[:, 1] - qa))),
            "model_sha256": model["model_sha256"], "table_top_z_m": float(model["table"]["top_z_m"])}


def heading_words(x: float, y: float) -> str:
    """Which way a horizontal direction points, in pc_probe's words (one base axis within 25 deg, else between two)."""
    n = math.hypot(x, y)
    v = (x / n, y / n)
    big = 0 if abs(v[0]) >= abs(v[1]) else 1
    first = _BASE_WORDS[(big, 1 if v[big] > 0 else -1)]
    if abs(v[big]) > 0.906:
        return first
    return f"between {first} and {_BASE_WORDS[(1 - big, 1 if v[1 - big] > 0 else -1)]}"


def tcp_numbers(T: np.ndarray, table_top_z_m: float) -> dict:
    """Tilt of the fingertip direction (TCP z) from straight down, its heading on the table, height above it."""
    z = np.asarray(T)[:3, 2]
    tilt = float(np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0))))
    h = math.hypot(z[0], z[1])
    return {"tcp_O_m": [round(float(v), 4) for v in np.asarray(T)[:3, 3]],
            "fingertip_dir_O": [round(float(v), 3) for v in z], "tilt_deg": round(tilt, 2),
            "heading_deg": round(float(np.degrees(np.arctan2(z[1], z[0]))), 2) if h > 0.035 else None,
            "heading_words": f"pointing {heading_words(z[0], z[1])}" if h > 0.035 else "no horizontal direction",
            "height_above_table_m": round(float(np.asarray(T)[2, 3] - table_top_z_m), 4)}


def describe(q, model_json: Path = DEFAULT_MODEL, station: Path = CONTROL_SIM, region: Path = RANDOM_START) -> dict:
    """Everything start-pose --show prints, and whether the pose passes the checks."""
    q = [float(x) for x in q]
    f = fk(q, model_json)
    cfg = effective_config(station)
    s, hm = cfg["safety"], cfg["homing"]
    T = f["O_T_tcp"]
    num = tcp_numbers(T, f["table_top_z_m"])
    p = T[:3, 3]
    lo, hi = np.asarray(s["workspace_min_m"], float), np.asarray(s["workspace_max_m"], float)
    axis = np.asarray(s.get("cone_axis", [0, 0, -1]), float)
    axis /= np.linalg.norm(axis)
    cone_tilt = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ axis, -1.0, 1.0))))
    q_home = [float(x) for x in hm["q_home_rad"]]
    dur = max(float(hm["min_duration_s"]), PEAK_SPEED_FACTOR * float(np.max(np.abs(np.subtract(q, q_home))))
              / float(hm["max_joint_speed_rad_per_s"]))
    outside = [i + 1 for i in range(7) if not (PANDA_Q_MIN[i] <= q[i] <= PANDA_Q_MAX[i])]
    checks = {
        "panda_joint_limits": {"ok": not outside, "detail": f"joints outside: {outside}" if outside else "inside"},
        "model_joint_limits": {"ok": f["model_joint_margin_rad"] > 0,
                               "detail": f"margin {f['model_joint_margin_rad']:.3f} rad"},
        "box": {"ok": bool(np.all(p >= lo) and np.all(p <= hi)),
                "detail": f"TCP {np.round(p, 3).tolist()} in [{lo.tolist()}, {hi.tolist()}]"},
        "cone": {"ok": cone_tilt <= float(s["cone_max_deg"]),
                 "detail": f"{cone_tilt:.1f} deg from the cone axis {axis.round(3).tolist()} (max "
                           f"{float(s['cone_max_deg']):.1f}, no margin)"},
        "no_contact": {"ok": not f["contacts"], "detail": str(f["contacts"]) if f["contacts"] else "none"},
        "heading": {"ok": num["heading_deg"] is not None,
                    "detail": num["heading_words"] + ("" if num["heading_deg"] is not None else
                                                      ": the sim cannot place the eraser ahead of it")},
        "move_time": {"ok": dur < float(hm["timeout_s"]),
                      "detail": f"{dur:.1f} s from home (homing.timeout_s {float(hm['timeout_s']):.0f})"},
    }
    rnd = yaml.safe_load(Path(region).read_text()) if Path(region).exists() else {}
    demos = rnd.get("demos") or {}
    vs = {}
    for key, val, scale, unit in (("tilt_deg", num["tilt_deg"], 1.0, "deg"),
                                  ("tip_height_m", num["height_above_table_m"], 100.0, "cm")):
        dm = demos.get(key)
        if dm:
            vs[key] = {"sim": val, "demos_median": dm["median"], "demos_range": dm["range"],
                       "inside_range": dm["range"][0] <= val <= dm["range"][1],
                       "text": f"{val * scale:.1f} {unit} vs demos {dm['median'] * scale:.1f} "
                               f"({dm['range'][0] * scale:.0f}-{dm['range'][1] * scale:.0f}): "
                               f"{(val - dm['median']) * scale:+.1f} {unit}"}
    return {"q_start_rad": q, **num, "checks": checks, "ok": all(c["ok"] for c in checks.values()),
            "vs_demos": vs, "cone_tilt_deg": round(cone_tilt, 2), "move_time_s": round(dur, 2),
            "model_sha256": f["model_sha256"], "station": str(station),
            "O_T_tcp": T.tolist(), "O_T_cam_cv": {k: v.tolist() for k, v in f["O_T_cam_cv"].items()}}


def summary_line(info: dict) -> str:
    t = info["tcp_O_m"]
    return (f"TCP O ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}), {info['height_above_table_m'] * 100:.1f} cm above the "
            f"table, tilted {info['tilt_deg']:.1f} deg from straight down, {info['heading_words']}")


def format_info(info: dict, source: str = "") -> list[str]:
    q = ", ".join(f"{x:.4f}" for x in info["q_start_rad"])
    out = [f"q_start_rad: [{q}]" + (f"   ({source})" if source else ""),
           f"in the sim: {summary_line(info)}",
           f"fingertip direction (TCP z) -> O {info['fingertip_dir_O']}"]
    for name, c in info["checks"].items():
        out.append(f"  {'ok  ' if c['ok'] else 'FAIL'} {name:<18} {c['detail']}")
    for name, v in info["vs_demos"].items():
        out.append(f"  {'info' if v['inside_range'] else 'WARN'} vs demos {name:<9} {v['text']}")
    reg = info.get("region")
    if reg:
        out.append(f"  {'ok  ' if reg['ok'] else 'WARN'} eraser region      {reg['detail']}")
    out.append("checks: OK" if info["ok"] else "checks: FAIL")
    return out


def show(model=DEFAULT_MODEL, path: Path = START_POSE_YAML) -> dict:
    """--show as a dict (also GET /api/start_pose of the UI): the file's q, its source line, describe() and the region
    check at that pose; {"set": False, ...} when q_start_rad is null."""
    path = Path(path)
    from sim_deploy.scene.placement import Scene

    q = read_q(path)
    src, date, cont = "", "", False
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.startswith("# source:"):
                src, cont = ln[len("# source:"):].strip(), True
            elif cont and re.match(r"#\s{2,}\S", ln):       # an indented continuation of the source line
                src += " " + ln[1:].strip()
            else:
                cont = False
                if ln.startswith("# date:"):
                    date = ln[len("# date:"):].strip()
    if date:
        src += f" ({date})"
    if q is None:
        return {"set": False, "file": str(path), "source": src,
                "text": f"{path}: q_start_rad is not set (null): --home start cannot run"}
    info = describe(q, model)
    info.update(set=True, file=str(path), source=src)
    try:
        sc = Scene.load(model, home="start", start_pose=path)
        rep = sc.check_region()
        bad = [c for c in rep["corners"] if not (c["on_table"] and c["in_workspace"] and c["in_view_imx219"])]
        info["region"] = {"ok": rep["ok"], "detail": "every corner on the table, in the box, in the IMX219 view"
                          if rep["ok"] else f"{len(bad)} of {len(rep['corners'])} corners fail (scene check)"}
    except Exception as e:                                # noqa: BLE001
        info["region"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    return info


# --------------------------------------------------------------------------------- the demos (start-data)


def relative_to_start(dataset=DEFAULT_DATASET, centre_behind_tip_m: float = 0.006) -> dict:
    """Per episode, the eraser centre relative to the episode's first pose, in the start's heading frame."""
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode, n_episodes

    rows = []
    for i in range(n_episodes(dataset)):
        ep = load_episode(dataset, i)
        P, g = ep["poses"], grasp_rows(ep["widths"])
        P0, Pg = P[0], P[g["start"]]
        f = P0[:3, 2].copy()
        f[2] = 0.0
        f /= np.linalg.norm(f)
        lft = np.cross([0.0, 0.0, 1.0], f)
        c = Pg[:3, 3] - centre_behind_tip_m * Pg[:3, 2]
        d = c - P0[:3, 3]
        xg = Pg[:3, 0].copy()                                # closing axis: the eraser's width; its length across
        xg[2] = 0.0
        xg /= np.linalg.norm(xg)
        close = float(np.degrees(np.arctan2(xg @ lft, xg @ f)))
        yaw = (close - 90.0 + 90.0) % 180.0 - 90.0           # long axis vs ahead, in [-90, 90)
        rows.append({"episode": i, "ahead_m": float(d @ f), "left_m": float(d @ lft), "below_m": float(-d[2]),
                     "start_tip_height_m": float(P0[2, 3]),
                     "start_tilt_deg": float(np.degrees(np.arccos(np.clip(-P0[2, 2], -1, 1)))),
                     "yaw_rel_deg": yaw})
    stats = {}
    for k in ("ahead_m", "left_m", "below_m", "start_tip_height_m", "start_tilt_deg", "yaw_rel_deg"):
        v = np.array([r[k] for r in rows])
        stats[k] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                    "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}
    return {"dataset": str(dataset), "n": len(rows), "episodes": rows, "stats": stats}


def write_region(rel: dict, path: Path = RANDOM_START) -> Path:
    s = rel["stats"]
    r = lambda k, n=3: [round(s[k]["p10"], n), round(s[k]["p90"], n)]   # noqa: E731
    doc = {
        "schema": "sim_deploy.scene_random_start/v1", "id": "pick_eraser_start_v1",
        "ahead_m": r("ahead_m"), "left_m": r("left_m"), "yaw_rel_deg": r("yaw_rel_deg", 1),
        "median": {"ahead_m": round(s["ahead_m"]["median"], 3), "left_m": round(s["left_m"]["median"], 3),
                   "yaw_rel_deg": round(s["yaw_rel_deg"]["median"], 1)},
        "settle_s": 0.3,
        "demos": {"dataset": rel["dataset"], "episodes": rel["n"],
                  "tilt_deg": {"median": round(s["start_tilt_deg"]["median"], 1),
                               "range": [round(s["start_tilt_deg"]["min"], 1), round(s["start_tilt_deg"]["max"], 1)]},
                  "tip_height_m": {"median": round(s["start_tip_height_m"]["median"], 3),
                                   "range": [round(s["start_tip_height_m"]["min"], 3),
                                             round(s["start_tip_height_m"]["max"], 3)]},
                  "below_m": {"median": round(s["below_m"]["median"], 3),
                              "range": [round(s["below_m"]["min"], 3), round(s["below_m"]["max"], 3)]},
                  "ahead_m": {"median": round(s["ahead_m"]["median"], 3),
                              "range": [round(s["ahead_m"]["min"], 3), round(s["ahead_m"]["max"], 3)]},
                  "left_m": {"median": round(s["left_m"]["median"], 3),
                             "range": [round(s["left_m"]["min"], 3), round(s["left_m"]["max"], 3)]},
                  "yaw_rel_deg": {"median": round(s["yaw_rel_deg"]["median"], 1),
                                  "range": [round(s["yaw_rel_deg"]["min"], 1), round(s["yaw_rel_deg"]["max"], 1)]}},
    }
    hdr = f"""\
# The eraser relative to the start pose (P2, PLAN.md section 9; `--home start`). Written by
# `python -m sim_deploy.scene start-data --write` from {rel['dataset']}
# ({rel['n']} episodes); the region keys may be edited by hand (start-data --write overwrites them).
# Frame (scene/start_pose.py): origin = the start TCP (fingertip point) projected on the table; ahead = the fingertip
# direction (TCP z) projected on the table; left = up x ahead. yaw_rel_deg = the eraser's long axis relative to ahead
# (the fingers close across its 51 mm width). Per episode of the demos: the eraser centre (the grasp TCP, half the
# eraser height back along its approach axis) relative to the episode's first pose.
# Rollout n of an eval with --seed S draws ahead, left, yaw uniformly in these ranges (the demos' 10th-90th
# percentiles, around their median) from numpy.random.default_rng([S, n]); the centre is then clipped so the whole
# eraser stays on the table and inside the sim PC's workspace box (xy). `python -m sim_deploy.scene check` (home start)
# checks the region's corners at the current start pose (stations/start_pose.yaml).
# demos: the demos' start (tilt from straight down, fingertip height above the table, the eraser below the fingertips),
# for start-pose --show. pc_probe's text: "~48 deg (38-64), ~16 cm above the table (6-27), object ~16 cm ahead
# horizontally and ~17 cm below the fingertips".
"""
    path.write_text(hdr + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120))
    return path
