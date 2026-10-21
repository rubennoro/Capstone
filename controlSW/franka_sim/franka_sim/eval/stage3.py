"""Stage 3: the closed loop. N episodes x M eraser poses on the full stack, with a judge and a between-episode reset.

Why: Stages 1-2 teacher-force a demo. Stage 3 lets the policy act, so it measures what only a closed loop shows: plumbing,
safety (collisions, reflexes), timing (tracking, gripper lag, inference latency, receiver age, observation motion) and,
informationally, success. With rendered or stub images the success rate is not a real-world estimate (the report says so).

One launcher (``python -m franka_deploy run --profile P --no-tty``, a derived deploy.yaml in the run dir, ports shifted by the
test offset) serves every episode. The driver types into the session's ``commands.txt``:

    per pose p, episode e:  sim.reset(pose) -> settle -> ``c`` -> (Judge starts) -> the judge writes ``y``/``n`` -> episode_finalized
                            -> (reflex: bridge ``recover``) -> ``r`` -> wait until the arm is home
    end:                    ``q``; a dead sender (``sender_died``) or a failed launcher is INCOMPLETE with the reason.

The simulator is a ``SimHandle`` (``simlink``): ``StubSim`` (``--sim-stub SCENARIO``, no MuJoCo, scripted eraser) now,
``BridgeSim`` (A6) later. A reflex or a collision is a FAIL (a gate), not INCOMPLETE.

Per-episode metrics come from the episode dir: ``label.json`` / ``sim_truth.json`` (label, truth), ``controller_trace.npz``
(tracking, gripper lag), ``tap/call_*.npz`` (inference latency, observation motion), ``bridge.jsonl`` (reflexes),
``summary.json`` (stale-image counter); receiver age from the launcher's camera stats. Metric names = the stage3 gate bases.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

from franka_sim.eval import judge as jg
from franka_sim.eval import runner as rn
from franka_sim.eval import simlink
from franka_sim.eval import stage2b as s2b

DEFAULTS = {
    "n_episodes_per_pose": 3,
    "poses": {"center": "home", "offsets_m": [[0.0, 0.0], [0.03, 0.0], [-0.03, 0.02]], "yaw_deg": [0.0, 15.0, -15.0]},
    "reset_timeout_s": 3.0, "home_timeout_s": 15.0, "settle_s": 0.5, "start_retries_n": 3,
    "finger_geom_patterns": ["pad_", "tag_", "finger"],
    "ignore_contact_pairs": [["eraser", "table"]],
}
GRIPPER_ERR_START_M, GRIPPER_ERR_OK_M = 0.004, 0.002     # a lag interval opens above 4 mm of error and closes below 2 mm
EVENT = re.compile(r"^\[franka_task\] (\w+) (\{.*\})\s*$")


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def stage3_cfg(R: rn.Run) -> dict:
    """``DEFAULTS`` < the station's ``sim.stage3`` < the CLI (--episodes-per-pose, --poses)."""
    cfg = _merge(DEFAULTS, R.station.get("sim.stage3", {}) or {})
    a = R.args
    if getattr(a, "episodes_per_pose", None):
        cfg["n_episodes_per_pose"] = int(a.episodes_per_pose)
    if getattr(a, "poses", None):
        cfg["n_poses"] = int(a.poses)
    return cfg


def make_poses(cfg: dict, center_xy) -> list:
    """[(x, y, yaw_rad)]: the centre plus each offset, with yaw_deg[i]; ``n_poses`` keeps the first N."""
    p = cfg["poses"]
    offs, yaws = p["offsets_m"], p["yaw_deg"]
    n = max(len(offs), len(yaws))
    poses = [(float(center_xy[0]) + float(offs[i % len(offs)][0]), float(center_xy[1]) + float(offs[i % len(offs)][1]),
              math.radians(float(yaws[i % len(yaws)]))) for i in range(n)]
    return poses[: int(cfg["n_poses"])] if cfg.get("n_poses") else poses


def pose_center(R: rn.Run, cfg: dict) -> tuple:
    """(xy, note): ``stage1`` = recommendation.grasp_xyz_base of --stage1 DIR; ``home`` = robot.home_tcp.xyz_m."""
    home = R.station.get("robot.home_tcp.xyz_m")
    mode = str(cfg["poses"].get("center", "home"))
    if mode == "stage1":
        from franka_sim.eval import stage1 as s1w

        d = getattr(R.args, "stage1", None)
        p = s1w.stage1_json_in(Path(d)) if d else None
        if p is None:
            raise rn.StageIncomplete("sim.stage3.poses.center is 'stage1' but --stage1 DIR has no stage1.json")
        g = json.loads(p.read_text())["recommendation"]["grasp_xyz_base"]
        return (float(g[0]), float(g[1])), f"pose centre: Stage 1 grasp point {g[:2]}"
    return (float(home[0]), float(home[1])), f"pose centre: robot.home_tcp xy {list(home[:2])}"


# ------------------------------------------------------------------------------------------------ pure metrics
def gripper_lag_steps(t, meas, cmd, step_s: float, grasped_fn=None) -> list:
    """Action steps (``step_s`` = 1/agent_update_freq_hz) the gripper needs to follow a command. A lag interval opens when
    |commanded - measured| rises above 4 mm (after having been below 2 mm) and closes when it falls below 2 mm; its lag is
    floor(duration / step_s). Intervals that start while the sim says ``grasped`` are excluded (the object blocks the
    fingers); an interval that never closes before the end of the trace counts to the end."""
    t, meas, cmd = np.asarray(t, float), np.asarray(meas, float), np.asarray(cmd, float)
    err = np.abs(cmd - meas)
    out, open_i, armed = [], None, True
    for i in range(len(t)):
        if open_i is None:
            if armed and err[i] > GRIPPER_ERR_START_M:
                open_i, armed = i, False
            elif err[i] < GRIPPER_ERR_OK_M:
                armed = True
        elif err[i] < GRIPPER_ERR_OK_M:
            if not (grasped_fn and grasped_fn(float(t[open_i]))):
                out.append(int((t[i] - t[open_i]) // step_s))
            open_i, armed = None, True
    if open_i is not None and not (grasped_fn and grasped_fn(float(t[open_i]))):
        out.append(int((t[-1] - t[open_i]) // step_s))
    return out


def tracking_err_mm(safe_target, fk) -> np.ndarray:
    return np.linalg.norm(np.asarray(safe_target)[:, :3] - np.asarray(fk)[:, :3], axis=1) * 1000.0


def obs_motion_mm(t, fk, main_ts_last: float, state_ts_last: float) -> float:
    """||fk(main_ts[-1]) - fk(state_ts[-1])|| in mm, fk interpolated from the controller trace."""
    t, fk = np.asarray(t, float), np.asarray(fk, float)
    a = np.array([np.interp(main_ts_last, t, fk[:, j]) for j in range(3)])
    b = np.array([np.interp(state_ts_last, t, fk[:, j]) for j in range(3)])
    return float(np.linalg.norm(a - b) * 1000.0)


def pct(x, q):
    x = np.asarray(x, float)
    return float(np.percentile(x, q)) if x.size else None


def episode_metrics(ep_dir: Path, grasped_fn, step_s: float) -> dict:
    """Arrays/numbers from one episode dir (missing files give empty results, never an exception)."""
    ep_dir = Path(ep_dir)
    out = {"tracking_mm": np.array([]), "gripper_lag": [], "infer_s": [], "obs_motion_mm": [], "reflexes": 0, "stale_image": 0,
           "label": None, "n_calls": 0, "holds": 0}
    tp = ep_dir / "controller_trace.npz"
    tr = None
    for _ in range(60):             # the controller writes the trace after the task's finalize message (may be half-written)
        if tp.exists():
            try:
                tr = np.load(tp)
                tr["t"]
                break
            except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
                tr = None
        time.sleep(0.1)
    if tr is not None:
        ref = tr["safe_target"] if "safe_target" in tr.files else tr["commanded"]
        out["tracking_mm"] = tracking_err_mm(ref, tr["fk"])
        out["gripper_lag"] = gripper_lag_steps(tr["t"], tr["width_measured"], tr["width_commanded"], step_s, grasped_fn)
    for f in sorted((ep_dir / "tap").glob("call_*.npz")):
        c = np.load(f, allow_pickle=False)
        out["n_calls"] += 1
        if "info_inference_s" in c.files:
            out["infer_s"].append(float(c["info_inference_s"]))
        if tr is not None and "info_main_ts" in c.files and "info_state_ts" in c.files:
            out["obs_motion_mm"].append(obs_motion_mm(tr["t"], tr["fk"], float(c["info_main_ts"][-1]), float(c["info_state_ts"][-1])))
    bj = ep_dir / "bridge.jsonl"
    if bj.exists():
        for ln in bj.read_text().splitlines():
            try:
                out["reflexes"] += json.loads(ln).get("kind") == "reflex"
            except ValueError:
                pass
    sj = ep_dir / "summary.json"
    if sj.exists():
        sm = json.loads(sj.read_text())
        out["stale_image"] = int((sm.get("task_counters") or {}).get("stale_image", 0))
        out["holds"] = int((sm.get("task_counters") or {}).get("holds", 0))
    lj = ep_dir / "label.json"
    if lj.exists():
        out["label"] = json.loads(lj.read_text()).get("label")
    return out


def aggregate(eps: list, R: rn.Run, sim_source: str, n_gripper_events: int | None = None) -> tuple[dict, list]:
    """Metrics (stage3 gate bases) and one row per episode from the driver's episode records."""
    from franka_deploy.eval.metrics import wilson

    labels = [e["label"] for e in eps]
    scored = [x for x in labels if x in ("y", "n")]
    k = sum(x == "y" for x in scored)
    p, lo, hi = wilson(k, len(scored))
    trk = np.concatenate([e["m"]["tracking_mm"] for e in eps] or [np.array([])])
    lag = [x for e in eps for x in e["m"]["gripper_lag"]]
    inf = [x for e in eps for x in e["m"]["infer_s"]]
    om = [x for e in eps for x in e["m"]["obs_motion_mm"]]
    m = {
        "n_episodes": len(eps), "n_poses": len({tuple(e["pose"]) for e in eps}), "labels": "".join(str(x)[:1] for x in labels),
        "success_rate": p if scored else None, "success_k": k, "success_n": len(scored),
        "success_ci95_lo": lo if scored else None, "success_ci95_hi": hi if scored else None,
        "sim_source": sim_source, "image_source": R.image_source,
        "collisions": int(sum(len(e["collisions"]) for e in eps)),
        "reflex": int(sum(max(e["m"]["reflexes"], int(bool(e["truth"].get("reflex")))) for e in eps)),
        "tracking_err_mm_p95": pct(trk, 95) if trk.size else None,
        "gripper_width_lag_steps_p95": pct(lag, 95) if lag else (0.0 if eps else None),
        "n_gripper_lag_events": len(lag),
        "inference_latency_s_p95": pct(inf, 95), "obs_motion_mm_p95": pct(om, 95),
        "stale_image_errors": int(sum(e["m"]["stale_image"] for e in eps)), "holds": int(sum(e["m"]["holds"] for e in eps)),
        "slip_exceeds_tol_episodes": int(sum(bool(e["truth"].get("slip_exceeds_tol")) for e in eps)),
        "max_slip_m": max([e["truth"].get("slip_m", 0.0) for e in eps] or [0.0]),
        "knocked_over_episodes": int(sum(bool(e["truth"].get("knocked_over")) for e in eps)),
    }
    rows = [{"pose": e["pose_i"], "ep": e["ep_i"], "label": e["label"], "reason": e["truth"].get("reason"),
             "max_lift_mm": round(1000 * e["truth"].get("max_lift_m", 0.0), 1), "slip_mm": round(1000 * e["truth"].get("slip_m", 0.0), 2),
             "collisions": len(e["collisions"]), "reflexes": e["m"]["reflexes"], "calls": e["m"]["n_calls"],
             "track_p95_mm": round(pct(e["m"]["tracking_mm"], 95) or 0.0, 2), "wall_s": e["wall_s"]} for e in eps]
    return m, rows


# ------------------------------------------------------------------------------------------------ driver
class LogTail:
    """Incremental reader of ``logs/task.log``: yields ``(kind, kwargs)`` for ``[franka_task] <kind> {json}`` lines."""

    def __init__(self, path: Path):
        self.path, self.pos, self.buf = Path(path), 0, ""

    def poll(self) -> list:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)
        self.buf += data.decode(errors="replace")
        *lines, self.buf = self.buf.split("\n")
        out = []
        for ln in lines:
            m = EVENT.match(ln.strip())
            if m:
                try:
                    out.append((m.group(1), json.loads(m.group(2))))
                except ValueError:
                    pass
        return out


class Driver:
    """Holds one launcher run and the helpers the episode loop needs. All waits are bounded; none can hang."""

    def __init__(self, R: rn.Run, d: Path, cfg: dict, port_offset: int, budget_s: float):
        self.R, self.d, self.cfg, self.o, self.budget_s = R, d, cfg, int(port_offset), float(budget_s)
        self.proc = None
        self.session: Path | None = None
        self.tail: LogTail | None = None
        self.bridge = None
        self.home_xyz = None
        self.home_max = 0.05
        self.deployed: dict = {}
        self._log_pos = 0

    # ---- launcher
    def derive(self, scenario: str | None) -> Path:
        R = self.R
        raw = yaml.safe_load(Path(R.cfg["config_path"]).read_text())
        prof = copy.deepcopy(raw["profiles"][R.profile])
        prof["station_set"] = list(R.sets)
        if R.policy.get("kind") in ("bp_server", "command"):
            prof["user_site_children"] = sorted(set(prof.get("user_site_children") or []) | {"inference"})
        prof["work_root"] = str(self.d / "work")
        prof["run"] = {**prof.get("run", {}), **R.run_ids}
        if scenario == "reflex":
            prof.setdefault("bridge", {})["allow_inject"] = True
        s2b.apply_port_offset(R, prof, self.o)
        prof.setdefault("sim_state_endpoint", f"tcp://127.0.0.1:{5570 + self.o}")
        raw["profiles"][R.profile] = prof
        p = self.d / "deploy_stage3.yaml"
        p.write_text(yaml.safe_dump(raw))
        self.deployed = prof
        return p

    def launch(self, dep: Path):
        cmd = [self.R.python_re, "-u", "-m", "franka_deploy", "run", "--config", str(dep), "--profile", self.R.profile, "--no-tty",
               "--script", ",".join(x for x in (os.environ.get("EVAL_3_INJECT"), f"{self.budget_s:.0f}:q") if x)]   # INJECT: fault tests
        self.lg = open(self.d / "launcher.log", "w")
        self.proc = subprocess.Popen(cmd, stdout=self.lg, stderr=subprocess.STDOUT, env=rn.child_env(), cwd=str(self.d),
                                     start_new_session=True)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def check_sender(self):
        """A dead sim sender (launcher event ``sender_died``) aborts the episode loop: the receiver goes stale and every later
        episode would only measure that."""
        p = self.d / "launcher.log"
        if not p.exists():
            return
        with open(p, errors="replace") as f:
            f.seek(self._log_pos)
            new = f.read()
            self._log_pos = f.tell()
        if "sender_died" in new:
            line = next(ln for ln in new.splitlines() if "sender_died" in ln)
            raise rn.StageIncomplete(f"sender_died {line.split('sender_died', 1)[1].strip()} (episode aborted)")

    def fail_if_dead(self, what: str):
        if not self.alive():
            raise rn.StageIncomplete(f"launcher exited (rc {self.proc.returncode}) while {what}; see {self.d / 'launcher.log'}")

    def wait_ready(self, timeout_s: float):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.fail_if_dead("waiting for the task")
            ss = sorted((self.d / "work").glob("session_*"))
            if ss and (ss[-1] / "task_ready.json").exists():
                self.session = ss[-1]
                break
            time.sleep(0.2)
        else:
            raise rn.StageIncomplete(f"no task_ready.json within {timeout_s:.0f} s; see {self.d / 'launcher.log'}")
        self.tail = LogTail(self.session / "logs" / "task.log")
        dc = yaml.safe_load((self.session / "config" / "deploy_resolved.yaml").read_text())
        from franka_deploy.bridge.link import BridgeClient

        self.bridge = BridgeClient(str(dc["bridge"]["endpoint"]).replace("0.0.0.0", "127.0.0.1"), 1.0)
        home = dc.get("_station_home") or {}
        self.home_xyz = (np.array(home["home_O_T_EE_xyz_wxyz"][:3], float) if home.get("home_O_T_EE_xyz_wxyz") else None)
        self.home_max = float((dc.get("home") or {}).get("max_dist_m", 0.05))
        self.deploy_resolved = dc

    def key(self, k: str):
        with open(self.session / "commands.txt", "a") as f:
            f.write(k + "\n")

    def wait_event(self, kinds, timeout_s: float):
        """First ``(kind, kw)`` among ``kinds`` from the task log, or None on timeout."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            for kind, kw in self.tail.poll():
                if kind in kinds:
                    return kind, kw
            if not self.alive():
                self.tail.poll()
                raise rn.StageIncomplete(f"launcher exited (rc {self.proc.returncode}) while waiting for {sorted(kinds)}")
            self.check_sender()
            time.sleep(0.05)
        return None

    def bridge_state(self) -> dict | None:
        try:
            return self.bridge.request("get_state", None, timeout_s=1.0)
        except Exception:       # noqa: BLE001
            return None

    def recover_if_reflex(self) -> bool:
        st = self.bridge_state()
        if st and (st.get("error") or st.get("bridge_state") == "reflex"):
            try:
                self.bridge.request("recover", None, timeout_s=2.0)     # operator recovery: recover (-> idle), then start (-> running)
                self.bridge.request("start", None, timeout_s=2.0)
            except Exception as e:      # noqa: BLE001
                raise rn.StageIncomplete(f"bridge recover/start failed after a reflex: {e!r}") from e
            return True
        return False

    def home_dist(self) -> float | None:
        st = self.bridge_state()
        if st is None or self.home_xyz is None:
            return None
        return float(np.linalg.norm(np.asarray(st["O_T_EE"], float)[:3, 3] - self.home_xyz))

    def go_home(self):
        """``r`` and wait until the bridge's O_T_EE is within home.max_dist_m of the station home."""
        self.key("r")
        t0 = time.monotonic()
        tmo = float(self.cfg["home_timeout_s"])
        if self.home_xyz is None:
            time.sleep(min(tmo, 3.0))
            return
        time.sleep(0.3)
        while time.monotonic() - t0 < tmo:
            self.fail_if_dead("going home")
            self.check_sender()
            d = self.home_dist()
            if d is not None and d <= self.home_max:
                return
            time.sleep(0.1)
        raise rn.StageIncomplete(f"the arm did not reach home within {tmo:.0f} s after r (last distance {self.home_dist()})")

    def finish(self, timeout_s: float = 40.0) -> int | None:
        """``q`` and wait for the launcher; SIGINT, then SIGKILL, if it does not leave."""
        if self.session is not None and self.alive():
            self.key("q")
        try:
            return self.proc.wait(timeout_s) if self.proc else None
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                return self.proc.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                return self.proc.wait()
        finally:
            try:
                self.lg.close()
            except Exception:       # noqa: BLE001
                pass


def run_episode(dr: Driver, sim, judge_cfg: dict, success: dict, pose: tuple, scenario: str | None, step_s: float) -> dict:
    """One episode, per the module docstring. Raises ``StageIncomplete`` when the loop cannot continue."""
    cfg = dr.cfg
    t_wall = time.monotonic()
    try:
        sim.reset(pose)
    except simlink.SimResetRefused:
        dr.go_home()
        try:
            sim.reset(pose)
        except simlink.SimResetRefused as e:
            raise rn.StageIncomplete(f"sim_reset refused twice: {e}") from e
    time.sleep(float(cfg["settle_s"]))
    reason = None
    for _ in range(int(cfg["start_retries_n"])):
        dr.key("c")
        ev = dr.wait_event(("episode_start", "refused_start"), 15.0)
        if ev is None:
            raise rn.StageIncomplete("the task did not answer c (no episode_start / refused_start within 15 s)")
        if ev[0] == "episode_start":
            break
        reason = ev[1].get("reason")
        dr.recover_if_reflex()
        dr.go_home()
    else:
        raise rn.StageIncomplete(f"episode start refused {cfg['start_retries_n']} times: {reason}")
    t0 = time.monotonic()
    ep_dir = Path(ev[1]["episode_dir"])
    if hasattr(sim, "on_episode_start"):
        sim.on_episode_start(t0)
    judge = jg.Judge(sim.state_endpoint, dr.session / "commands.txt", success, judge_cfg["finger_geom_patterns"],
                     judge_cfg["ignore_contact_pairs"], t_start=t0)
    judge.start()
    timers = []
    for rel, action in getattr(sim, "driver_actions", []):
        if action == "inject_reflex":
            def _inject():          # its own REQ socket: the main thread's dr.bridge is not shared across threads
                from franka_deploy.bridge.link import BridgeClient

                bc = None
                try:
                    bc = BridgeClient(str(dr.deploy_resolved["bridge"]["endpoint"]).replace("0.0.0.0", "127.0.0.1"), 1.0)
                    bc.request("inject_reflex", "cartesian_reflex (stage3 stub)", timeout_s=1.0)
                except Exception:       # noqa: BLE001
                    pass
                finally:
                    if bc is not None:
                        bc.close()
            tm = threading.Timer(float(rel), _inject)
            tm.daemon = True
            tm.start()
            timers.append(tm)
    try:
        t_limit = float(success["timeout_s"]) + 40.0
        fin, forced = None, False
        while fin is None and time.monotonic() - t0 < t_limit:
            fin = dr.wait_event(("episode_finalized",), 0.5)
            if judge.error:
                raise rn.StageIncomplete(f"judge failed: {judge.error}")
            if fin is None and not forced and time.monotonic() - t0 > float(success["timeout_s"]) + 10.0:
                dr.key("s")         # the judge should have decided by now: end the episode unlabeled rather than hang
                forced = True
        if fin is None:
            raise rn.StageIncomplete(f"episode not finalized within {t_limit:.0f} s of its start")
    finally:
        for tm in timers:
            tm.cancel()
        judge.stop()
        if hasattr(sim, "on_episode_end"):
            sim.on_episode_end()
    truth = {**judge.tracker.result(), "label_written": judge.label, "forced_stop": forced}
    (ep_dir / "sim_truth.json").write_text(json.dumps({"pose_xy_yaw": list(pose), "scenario": scenario, "truth": truth,
                                                        "collisions": judge.collisions, "n_states": judge.n_states}, indent=1))
    m = episode_metrics(ep_dir, judge.grasped_at, step_s)
    rec = {"ep_dir": str(ep_dir), "pose": pose, "label": m["label"], "truth": truth, "collisions": judge.collisions, "m": m,
           "wall_s": round(time.monotonic() - t_wall, 1)}
    # after the episode: a reflex needs the bridge recovered before the arm can go home
    if m["reflexes"] or truth.get("reflex") or dr.recover_if_reflex():
        dr.recover_if_reflex()
    dr.go_home()
    return rec


def run(R: rn.Run, d: Path, results: dict) -> dict:
    cfg = stage3_cfg(R)
    scenario = getattr(R.args, "sim_stub", None)
    po = getattr(R.args, "port_offset", None)
    po = int(os.environ.get("FD_TEST_PORT_OFFSET", 0)) if po is None else int(po)
    center, centre_note = pose_center(R, cfg)
    poses = make_poses(cfg, center)
    n_per = int(cfg["n_episodes_per_pose"])
    success = R.station.get("criteria.success")
    step_s = 1.0 / float(R.cfg.get("task", {}).get("agent_hz", 20))
    backend = ((R.cfg.get("bridge") or {}).get("override") or {}).get("backend", {}).get("type")
    if scenario is None and backend != "mujoco":
        raise rn.StageIncomplete("Stage 3 needs --sim-stub SCENARIO or a MuJoCo profile (backend.type mujoco: A5/A6/A7)")
    notes = [centre_note]
    budget = len(poses) * n_per * (float(success["timeout_s"]) + float(cfg["home_timeout_s"]) + 40.0) + 300.0
    dr = Driver(R, d, cfg, po, budget)
    sim = None
    eps: list = []
    reason = None
    sender_died = None
    rc = None
    try:
        dep = dr.derive(scenario)
        state_ep = dr.deployed["sim_state_endpoint"]
        if scenario:
            sim = simlink.StubSim(state_ep, scenario)
        dr.launch(dep)
        dr.wait_ready(float(os.environ.get("EVAL_3_READY_S", 240)))
        if not scenario:
            sim = simlink.BridgeSim(dr.deploy_resolved["bridge"]["endpoint"], state_ep, float(cfg["reset_timeout_s"]))
        dr.go_home()
        for pi, pose in enumerate(poses):
            for ei in range(n_per):
                try:
                    rec = run_episode(dr, sim, cfg, success, pose, scenario, step_s)
                except rn.StageIncomplete as e:
                    reason = f"pose {pi} episode {ei}: {e}"
                    break
                rec.update(pose_i=pi, ep_i=ei)
                eps.append(rec)
            if reason:
                break
    except rn.StageIncomplete as e:
        reason = str(e)
    finally:
        rc = dr.finish()
        if sim is not None:
            sim.close()
    summ_p = (dr.session / "launcher_summary.json") if dr.session else None
    summ = json.loads(summ_p.read_text()) if summ_p and summ_p.exists() else {}
    if summ.get("sender_died"):
        sender_died = summ["sender_died"]
        reason = reason or f"sender_died {sender_died} (episode aborted)"
    elif rc not in (0, None) and not reason:
        reason = f"launcher failed rc {rc}: {summ.get('exit_reason')}; see {d / 'launcher.log'}"
    if not eps and not reason:
        reason = "no episode ran"
    sim_source = getattr(sim, "sim_source", "mujoco" if not scenario else "stub")
    for e in eps:
        e["pose"] = list(e["pose"])
    m, rows = aggregate(eps, R, sim_source)
    if eps:
        cs = (summ.get("camera_stats") or {}).get("streams", {}).get("main", {})
        m["receiver_age_s_p95"] = (cs.get("age_s") or {}).get("p95")
    else:
        m["receiver_age_s_p95"] = None
    m["scenario"] = scenario
    m["sender_died"] = bool(sender_died)
    if reason:
        m["incomplete_reason"] = reason
    # wall-clock dependent numbers are timing (reproducibility checks skip them)
    m["timing"] = sorted(k for k in m if k in ("tracking_err_mm_p95", "gripper_width_lag_steps_p95", "inference_latency_s_p95",
                                               "receiver_age_s_p95", "obs_motion_mm_p95", "holds", "stale_image_errors"))
    if sim_source != "mujoco":
        notes.append(f"sim_source {sim_source}: scripted stub eraser (scenario {scenario}), NOT a physics simulation; success and "
                     "collisions only exercise the judge and the driver")
    notes.append(f"{R.image_source} images: success is informational (Stage 3 measures plumbing, safety and timing)")
    if m["n_gripper_lag_events"] == 0 and eps:
        notes.append("no gripper command change left the 2 mm band: gripper_width_lag_steps_p95 is 0 by default (vacuous for this policy)")
    notes.append(f"run dir {d}: launcher.log, work/session_*/task/episodes/*/*/*/episode_*/sim_truth.json")
    return {"metrics": m, "rows": rows, "notes": notes}
