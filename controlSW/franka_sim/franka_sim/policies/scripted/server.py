"""Scripted pick-and-place policy server (the five robotmq topics of ``controlSW/docs/adding_a_policy.md``).

    python -m franka_sim.policies.scripted.server --endpoint tcp://0.0.0.0:18765 --ready-file F --work-dir W

Structure copied from ``franka_deploy/mocks/fake_policy_server.py`` (``--no-prompt-rules`` behaviour: use_prompting
false, a prompt is answered "OK" and ignored). The policy ignores the observation: it reads the simulator's state PUB
(I-SIMSTATE, 100 Hz) in a background thread and keeps the newest state. Each ``policy_inference``:
``planner.plan`` (on the measured state) -> ``trajectory.step_chunk`` (one rest-to-rest min-jerk move towards the
goal, K = 16 at 50 ms) -> poses relative to the reference pose (step 0 = identity), rot6d = first two rows, width in
training units.

Reference pose: the task (anchor "target") applies the relative chunk to its COMMANDED pose, which is the last executed
action of the previous chunk (pause mode). Relative to the measured ``tcp_T_base`` (the first version) the tracking
lag at each call was added to the command (up to ~10 mm: the fingers were driven 6.5 mm below the grasp goal into the
table -> cartesian reflex). So the reference is this server's own estimate of that commanded pose (sample exec_h - 1
of its previous chunk). The measured pose is used (closing the loop: the arm has settled on the task's real anchor
by then, so a residual offset is corrected) at the first call after a reset, once the previous chunk already ended
at the same goal, and whenever the estimate is more than 2 cm from it (the task's own anchor fallback). Before any sim
state arrives it replies an identity hold chunk (open width). ``policy_reset`` clears the planner memory (phase back to ABOVE).

Config: ``W/../config/deploy_resolved.yaml`` (``sim_state_endpoint``) and ``station_resolved.yaml`` (``groups.policy.
params``, ``groups.scene`` place / table / grasp_tcp_height_m, ``groups.tooling`` width_map / finger_adapter,
``groups.criteria`` limits / task.open_gap_min_m). Missing files or keys fall back to the defaults (``planner.Params``,
tcp://127.0.0.1:5570), so a hand-started server works too. Phase changes are printed to stdout (the inference log).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import robotmq
import yaml

from franka_sim.policies.scripted import planner as pl
from franka_sim.policies.scripted import trajectory as tj

DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5570"
CONFIG = {"workspace": {"proprio_length": 1, "img_obs_horizon": 2, "obs_down_sample_steps": 3,
                        "ultrawide_down_sample_steps": 1},
          "use_prompting": False, "policy_name": "scripted_pick_place", "run_name": "scripted", "epoch": 0,
          "date_str": "x", "time_str": "y"}
TOPICS = ("policy_config", "policy_reset", "policy_prompt", "policy_inference", "export_recorded_data")
ANCHOR_MAX_DEV_M = 0.02           # the task's own anchor rule (franka_task anchor_max_dev_m): beyond it it anchors on FK
SAME_GOAL_M, SAME_GOAL_RAD = 0.0005, 0.0035   # a goal within 0.5 mm / 0.2 deg of the previous one is the same goal
IDENTITY_R6 = np.array([1, 0, 0, 0, 1, 0], np.float32)


def _get(d, dotted, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class _Station:
    """The ``.get(dotted, default)`` that ``franka_sim.scene.grasp_tcp_height_m`` reads, over the resolved groups."""

    def __init__(self, groups: dict):
        self.groups = groups

    def get(self, dotted, default=None):
        return _get(self.groups, dotted, default)


def grasp_tcp_height(groups: dict) -> tuple[float, str]:
    """(TCP height above the table top, how) via the sim's own rule ``franka_sim.scene.grasp_tcp_height_m``. The model
    info it reads (``tcp_at``) follows from the finger mode: kit -> fingertip TCP; box -> pad centre (pad length =
    the station's pad_size_m, exactly what model.py uses); stock -> approximated with the station pad size."""
    from franka_sim.scene import grasp_tcp_height_m

    mode = _get(groups, "tooling.finger_adapter.mode", "box")
    info = {"tcp_at": "tip"} if mode == "kit" else {}
    h = float(grasp_tcp_height_m(SimpleNamespace(info=info), _Station(groups)))
    return h, f"scene.grasp_tcp_height_m(mode={mode})"


def load_params(work_dir: str | None) -> tuple[pl.Params, str, dict]:
    """(Params, sim_state_endpoint, info) from ``work_dir/../config``; defaults where absent."""
    cfg_dir = Path(work_dir).resolve().parent / "config" if work_dir else None
    dep, st = {}, {}
    if cfg_dir is not None:
        for name, tgt in (("deploy_resolved.yaml", dep), ("station_resolved.yaml", st)):
            f = cfg_dir / name
            if f.is_file():
                try:
                    tgt.update(yaml.safe_load(f.read_text()) or {})
                except Exception as e:          # noqa: BLE001
                    print(f"[scripted] cannot read {f}: {e}", flush=True)
    g = st.get("groups") or {}
    kw = dict(_get(g, "policy.params", {}) or {})
    info = {"config_dir": str(cfg_dir) if cfg_dir else None, "station": bool(g), "deploy": bool(dep)}
    if g:
        h, how = grasp_tcp_height(g)
        top = float(_get(g, "scene.table.top_z_base_m", 0.0))
        kw["grasp_z_m"] = top + h
        info["grasp_z"] = f"table top {top} + {how} {h}"
        if _get(g, "scene.place.xy_m") is not None:
            kw["place_xy_m"] = tuple(float(x) for x in _get(g, "scene.place.xy_m"))
        wm = _get(g, "tooling.width_map", {}) or {}
        if "a" in wm:
            kw["width_a"] = float(wm["a"])
        if "b" in wm:
            kw["width_b"] = float(wm["b"])
        for src, dst in (("criteria.limits.v_max_m_s", "v_max_m_s"), ("criteria.limits.w_max_rad_s", "w_max_rad_s"),
                         ("criteria.task.open_gap_min_m", "open_gap_min_m")):
            v = _get(g, src)
            if v is not None:
                kw[dst] = float(v)
        if _get(g, "policy.contract.exec_h") is not None:
            kw["exec_h"] = int(_get(g, "policy.contract.exec_h"))
    ep = str(dep.get("sim_state_endpoint") or DEFAULT_STATE_ENDPOINT).replace("0.0.0.0", "127.0.0.1")
    return pl.Params.from_dict(kw), ep, info


class StateSub:
    """Background SUB on the sim state PUB (as ``franka_sim.eval.judge``); keeps only the newest state."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.state, self.t_recv, self.n = None, None, 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name="simstate-sub", daemon=True)
        self._th.start()

    def _run(self):
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.CONFLATE, 1)          # one part per message: keep only the newest
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(self.endpoint)
        try:
            while not self._stop.is_set():
                if not sock.poll(50, zmq.POLLIN):
                    continue
                s = robotmq.deserialize(sock.recv())
                if isinstance(s, dict) and "tcp_T_base" in s and "eraser_T_base" in s:
                    self.state, self.t_recv = s, time.monotonic()
                    self.n += 1
        finally:
            sock.close()

    def stop(self):
        self._stop.set()


class ScriptedPolicyServer:
    def __init__(self, endpoint: str, ready_file: str | None, params: pl.Params, state_endpoint: str | None,
                 k: int = tj.K_STEPS):
        self.endpoint, self.ready_file, self.p, self.k = endpoint, ready_file, params, int(k)
        self.mem = pl.Memory()
        self.sub = StateSub(state_endpoint) if state_endpoint else None
        self._stop = threading.Event()
        self._last_phase = None
        self.n_inf = 0
        self.A_est = None               # the task's anchor (its commanded pose) as this server last commanded it
        self.at_goal = False            # the previous chunk ended exactly at its goal G_prev
        self.G_prev = None
        self.n_fallback = 0

    def hold_chunk(self, width: float) -> dict:
        return {"action0_eef_pos": np.zeros((self.k, 3), np.float32),
                "action0_eef_rot_axis_angle": np.tile(IDENTITY_R6, (self.k, 1)),
                "action0_gripper_width": np.full((self.k, 1), width, np.float32)}

    def infer(self, s: dict | None) -> dict:
        self.n_inf += 1
        if s is None:
            return self.hold_chunk(self.p.width_open)
        T = np.asarray(s["tcp_T_base"], float)
        phase, G, w = pl.plan(s, self.p, self.mem)
        if phase != self._last_phase:
            E = np.asarray(s["eraser_T_base"], float)
            print(f"[scripted] phase {self._last_phase} -> {phase} inf={self.n_inf} sim_t={s.get('sim_t')} "
                  f"tcp={np.round(T[:3, 3], 4).tolist()} goal={np.round(G[:3, 3], 4).tolist()} "
                  f"eraser={np.round(E[:3, 3], 4).tolist()} gap={s.get('hand_gap_m')} "
                  f"hand_grasped={s.get('hand_grasped')} width={w:.4f}", flush=True)
            self._last_phase = phase
        ref = T
        same_goal = (self.G_prev is not None and float(np.linalg.norm(self.G_prev[:3, 3] - G[:3, 3])) <= SAME_GOAL_M
                     and tj.rot_angle(self.G_prev[:3, :3], G[:3, :3]) <= SAME_GOAL_RAD)   # live goals jitter slightly
        if self.A_est is not None and not (same_goal and self.at_goal):
            dev = float(np.linalg.norm(self.A_est[:3, 3] - T[:3, 3]))
            if dev <= ANCHOR_MAX_DEV_M:
                ref = self.A_est
            else:
                self.n_fallback += 1
                print(f"[scripted] anchor estimate {1000 * dev:.1f} mm from the measured TCP: relative to measured "
                      f"(fallback {self.n_fallback})", flush=True)
        Ts, self.at_goal = tj.step_chunk(ref, G, self.p.v_max_m_s, self.p.w_max_rad_s, self.p.speed_frac_ratio,
                                         self.p.min_segment_s, self.k, exec_h=self.p.exec_h)
        pos, r6 = tj.to_relative(ref, Ts)
        self.A_est, self.G_prev = Ts[self.p.exec_h - 1].copy(), G.copy()
        return {"action0_eef_pos": pos.astype(np.float32), "action0_eef_rot_axis_angle": r6.astype(np.float32),
                "action0_gripper_width": np.full((self.k, 1), w, np.float32)}

    def reply(self, topic, data):
        if topic == "policy_config":
            return CONFIG
        if topic == "policy_reset":
            self.mem.reset()
            self.A_est, self.G_prev, self.at_goal = None, None, False
            self._last_phase = None
            print("[scripted] reset -> ABOVE", flush=True)
            return "OK"
        if topic == "policy_prompt":
            return "OK"
        if topic == "policy_inference":
            try:
                return self.infer(self.sub.state if self.sub is not None else None)
            except Exception as e:          # noqa: BLE001
                import traceback

                print(f"[scripted] inference error: {e!r}\n{traceback.format_exc()}", flush=True)
                return f"ERROR: scripted policy: {e!r}"
        return ""

    def run(self):
        server = robotmq.RMQServer("policy_server", self.endpoint, robotmq.RMQLogLevel.WARNING)
        for t in TOPICS:
            server.add_topic(t, message_remaining_time_s=60)
        if self.ready_file:
            tmp = self.ready_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"endpoint": self.endpoint, "pid": os.getpid(), "ready_monotonic": time.monotonic(),
                           "kind": "scripted_pick_place"}, f)
            os.replace(tmp, self.ready_file)
        while not self._stop.is_set():
            data, topic = server.wait_for_request(timeout_s=0.05)
            if not topic:
                continue
            server.reply_request(topic=topic, data=robotmq.serialize(self.reply(topic, data)))

    def stop(self):
        self._stop.set()
        if self.sub is not None:
            self.sub.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", default="tcp://0.0.0.0:18765")
    ap.add_argument("--ready-file", default=None, help="JSON written atomically once serving")
    ap.add_argument("--work-dir", default=None, help="<session>/server; the config is read from ../config")
    ap.add_argument("--state-endpoint", default=None, help="override the sim state PUB endpoint")
    a = ap.parse_args(argv)
    if a.work_dir:
        Path(a.work_dir).mkdir(parents=True, exist_ok=True)
    params, ep, info = load_params(a.work_dir)
    ep = a.state_endpoint or ep
    print(f"[scripted] params {params} state_endpoint {ep} info {info} width open {params.width_open:.4f} "
          f"closed {params.width_closed:.4f}", flush=True)
    s = ScriptedPolicyServer(a.endpoint, a.ready_file, params, ep)
    signal.signal(signal.SIGTERM, lambda *_: s.stop())
    try:
        s.run()
    except KeyboardInterrupt:
        pass
    finally:
        s.stop()


if __name__ == "__main__":
    main()
