"""MuJoCoBackend: the simulated Franka behind franka_deploy's bridge ``Backend`` interface (A4).

Why: the unchanged franka_bridge, FrankaCartesian controller and task drive the sim exactly like the real arm. A
wall-clock thread steps SimCore (simcore.py) at ``rate_hz`` (1 kHz): each wake it runs the control steps that are due
(``floor((now - t0) rate) - k_done``, at most ``max_catchup_steps``), resyncs when more than 50 ms behind (counted in
``rt_resyncs``), then publishes a cached state snapshot, so ``get_state`` never touches MuJoCo.

bridge.yaml (``backend``; the launcher's sim profiles set ``type: mujoco``):
  type: mujoco
  panda_py: {...}            # the real arm's impedance/damping/nullspace/filter/collision: used by the sim too
  mujoco:                    # all optional
    station: null            # a franka_deploy.station PROFILES name, {station: {...}, run: {...}}, {robot: .., ...}
                             # ids, or a path to a launcher station_resolved.yaml; null -> profile sim-iphone
    robot: null              # panda | fr3: overrides the station's robot id (not for station_resolved.yaml)
    rate_hz: 1000
    q_init: null             # 7 joint angles; null -> IK of home_O_T_EE near the station home_q
    max_catchup_steps: 10
    physics: {}              # run/sim physics overrides (payload_error_kg, reflex, ...)
    viewer: false            # mujoco.viewer passive window (needs a display)
    hand: true               # Franka Hand on hand_cmd / hand_state (franka_sim.hand_sim, A5); false = A4 PD hold
    state_endpoint: tcp://127.0.0.1:5570   # I-SIMSTATE PUB (A6; the launcher sets it); null / "" = no PUB
    state_pub_hz: 100        # sim-state build + PUB rate (wall clock, built in the loop wake that is due); 0 = off
    sim_log_hz: 25           # bridge.jsonl "kind: sim" lines (written from sim_tick when --log-jsonl is set)
    impedance / damping_ratio / nullspace_stiffness / filter_coeff / collision: overrides of panda_py

Sim topics (A6, I-SIMSTATE): ``sim_state`` returns the newest sim-state dict; ``sim_reset {eraser_xy_yaw?: [x, y, yaw],
object_pose?: [x,y,z,qw,qx,qy,qz], hand_gap_m?}`` ({} = the home key pose) moves the eraser (and the fingers), never the
arm; refused while grasping or when the new pose penetrates anything but the table. The bridge gates it behind
--allow-sim-reset / --allow-inject. The sim-state dict is built in the 1 kHz loop under _core_lock and sent by its own
PUB thread (one part, robotmq.serialize).

Usage: backend = MuJoCoBackend.from_config(cfg["backend"], home_O_T_EE)   (via franka_deploy make_backend)
"""

from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from franka_deploy.bridge.backends import Backend, cycle_hist

RESYNC_S = 0.05
STATE_ENDPOINT = "tcp://127.0.0.1:5570"
STATE_PUB_HZ = 100.0
SIM_LOG_HZ = 25.0
_UNSET = object()


def load_station_spec(spec=None, robot: str | None = None):
    """A franka_deploy Station from the backend.mujoco ``station`` value (see the module docstring)."""
    import yaml

    from franka_deploy.station import PROFILES, Station, load_station

    if spec is None:
        spec = "sim-iphone"
    if isinstance(spec, str) and spec in PROFILES:
        spec = {k: dict(v) for k, v in PROFILES[spec].items()}
    elif isinstance(spec, (str, Path)):
        p = Path(spec)
        if not p.is_file():
            raise FileNotFoundError(f"backend.mujoco.station: no profile or file {spec!r} (profiles: {sorted(PROFILES)})")
        doc = yaml.safe_load(p.read_text())
        if "groups" in doc and "ids" in doc:          # launcher station_resolved.yaml (Station.to_yaml)
            st = Station(groups=doc["groups"], ids=doc["ids"], provenance=doc.get("provenance", {}),
                         critical=tuple(doc.get("critical", ())), overrides=tuple(doc.get("overrides", ())))
            if robot and st.ids.get("robot") != robot:
                raise ValueError(f"backend.mujoco.robot {robot!r} contradicts {p} (robot {st.ids.get('robot')!r})")
            return st
        spec = doc
    spec = dict(spec)
    if "station" not in spec and "run" not in spec:
        spec = {"station": spec}
    spec = {k: dict(v or {}) for k, v in spec.items() if k in ("station", "run")}
    if robot:
        spec.setdefault("station", {})["robot"] = robot
    return load_station(spec)


def _warn_tcp_mismatch(station, sm) -> None:
    """Loud warning when the controller's F_T_TCP (the station config it receives) is not the model's pad TCP."""
    from franka_sim.model import T_err, orthonormalize

    st_F = station.get("tooling.F_T_TCP", None)
    if st_F is None:
        return
    e_t, e_r = T_err(np.asarray(sm.F_T_TCP, float), orthonormalize(st_F))
    sm.info["controller_F_T_TCP_deviation"] = {"mm": e_t * 1e3, "deg": e_r}
    if e_t > 1e-3 or e_r > 0.1:
        print(f"[mujoco] WARNING: the controller F_T_TCP differs from the sim's actual TCP (model.F_T_TCP) by "
              f"{e_t * 1e3:.1f} mm / {e_r:.1f} deg: grasps will miss. Use franka_sim.model.derived_tcp(station).",
              flush=True)


class MuJoCoBackend(Backend):
    extra_topics: tuple = ("hand_cmd", "hand_state", "sim_reset", "sim_state")   # A5 I-HAND + A6 I-SIMSTATE

    def __init__(self, sm, core_cfg: dict, q_init, rate_hz: float = 1000.0, max_catchup_steps: int = 10,
                 viewer: bool = False, tooling: dict | None = None, hand: bool = True,
                 state_endpoint: str | None = None, state_pub_hz: float = STATE_PUB_HZ, sim_log_hz: float = SIM_LOG_HZ):
        from franka_sim.hand_sim import HandSim
        from franka_sim.simcore import SimCore

        self.sm = sm
        self.core = SimCore(sm, core_cfg, q_init)
        # Franka Hand: libfranka move/grasp semantics on the finger tendon (A5); without it SimCore's PD hold stays
        self.hand = HandSim(self.core, tooling) if hand else None
        if self.hand is not None:
            self.core.hand_ctrl = self.hand.ctrl
        self.rate = float(rate_hz)
        self.steps_per_tick = max(1, int(round(1.0 / (self.rate * self.core.control_dt))))
        self.max_catchup = int(max_catchup_steps)
        self._core_lock = threading.Lock()       # SimCore is not thread-safe: commands and stepping take this
        self._lock = threading.Lock()            # snapshot + statistics
        self._pending = (0, None)               # (seq, T) latest set_target; one tuple assignment (atomic under the GIL)
        self._applied_seq = 0
        self._run = True
        self._periods: list[float] = []
        self._cycles: list[float] = []
        self._lag_max = 0.0
        self._spw_max = 0
        self.rt_resyncs = 0                  # cumulative (campaign.py reads max - min)
        self.rt_resyncs_win = 0              # per loop_stats window
        self._win = (time.monotonic(), 0.0)      # (wall, sim_t) at the last loop_stats
        self._viewer = None
        if viewer:
            try:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(sm.model, self.core.d)
            except Exception as e:  # pragma: no cover - needs a display
                print(f"[mujoco] viewer disabled: {e}", flush=True)
        if sys.getswitchinterval() > 0.001:      # the 1 kHz thread must get the GIL back within ~1 ms
            sys.setswitchinterval(0.001)
        # -- A6: scene runtime, sim-state dict, PUB
        from franka_sim.scene import ContactProbe, Scene

        # state_pub_hz 0 = no sim state at all (no build, PUB, sim_* data; the A6 loop-cost baseline)
        self.scene = Scene(sm) if "object" in sm.names.get("bodies", {}) and float(state_pub_hz) > 0 else None
        self.probe = ContactProbe(self.scene) if self.scene is not None else None
        self.core.contact_probe = self.probe
        self.state_period = 1.0 / float(state_pub_hz) if self.scene is not None else 0.0   # wall-clock cadence
        self._next_state = 0.0
        self.sim_log_period = 1.0 / float(sim_log_hz) if sim_log_hz else 0.0
        self._next_sim_log = 0.0
        self._fext_max = 0.0
        self._sim_state = None
        self.n_state_pub = 0
        self._pub_evt = threading.Event()
        self._pub = None
        self.state_endpoint = None
        if state_endpoint and self.scene is not None:
            import zmq

            ep = str(state_endpoint).replace("localhost", "127.0.0.1")
            sock = zmq.Context.instance().socket(zmq.PUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 1000)
            try:
                sock.bind(ep)
            except zmq.ZMQError as e:
                sock.close(linger=0)
                raise RuntimeError(f"MuJoCoBackend: cannot bind the sim-state PUB on {ep}: {e}") from e
            self._pub, self.state_endpoint = sock, ep
        self._snapshot()
        if self.scene is not None:
            with self._core_lock:
                self._build_sim_state(self.scene.contacts(self.core.d))   # SimCore ran mj_forward
        self._pub_th = None
        if self._pub is not None:
            self._pub_th = threading.Thread(target=self._pub_loop, name="mujoco-simstate-pub", daemon=True)
            self._pub_th.start()
        self._th = threading.Thread(target=self._loop, name="mujoco-1khz", daemon=True)
        self._th.start()

    # ------------------------------------------------------------------------------------------- construction

    @classmethod
    def from_config(cls, backend_cfg: dict, home_O_T_EE) -> "MuJoCoBackend":
        from franka_sim.kinematics import Kin
        from franka_sim.model import build_model, sim_config
        from franka_sim.simcore import make_core_cfg

        mj = dict(backend_cfg.get("mujoco") or {})
        station = load_station_spec(mj.get("station"), mj.get("robot"))
        physics = {**sim_config(station, None), **dict(mj.get("physics") or {})}
        sm = build_model(station, physics, with_scene=bool(mj.get("with_scene", True)))
        core_cfg = make_core_cfg(station, backend_cfg.get("panda_py"), mj, physics)
        q_init = mj.get("q_init")
        if q_init is None:
            kin = Kin(sm, "ee")
            q0 = np.asarray(station.get("robot.home_q_rad"), float)
            q_init, info = kin.ik(np.asarray(home_O_T_EE, float), q0, q_rest=q0)
            if not info["ok"]:
                raise ValueError(f"MuJoCoBackend: home_O_T_EE is not reachable from home_q (IK err "
                                 f"{info['err_p'] * 1e3:.2f} mm / {np.degrees(info['err_r']):.2f} deg); set "
                                 f"backend.mujoco.q_init")
        _warn_tcp_mismatch(station, sm)
        print(f"[mujoco] robot={sm.robot} station={dict(station.ids)} rate_hz={mj.get('rate_hz', 1000)} "
              f"q_init={np.round(np.asarray(q_init, float), 4).tolist()}", flush=True)
        ep = mj.get("state_endpoint", _UNSET)
        return cls(sm, core_cfg, q_init, rate_hz=float(mj.get("rate_hz", 1000.0)),
                   max_catchup_steps=int(mj.get("max_catchup_steps", 10)), viewer=bool(mj.get("viewer", False)),
                   tooling=station.get("tooling", None), hand=bool(mj.get("hand", True)),
                   state_endpoint=STATE_ENDPOINT if ep is _UNSET else (ep or None),
                   state_pub_hz=float(mj.get("state_pub_hz", STATE_PUB_HZ)),
                   sim_log_hz=float(mj.get("sim_log_hz", SIM_LOG_HZ)))

    # ---------------------------------------------------------------------------------------------- the loop

    def _snapshot(self) -> None:
        # also called from command threads without _core_lock: a read torn by a concurrent step lasts <= 1 ms
        s = self.core.state()
        s.pop("tau_ext", None)                  # get_state keys = FakeBackend's + sim_t
        s["t"] = time.monotonic()
        hs = self.hand.state(s["t"]) if self.hand is not None else None
        with self._lock:
            self._state = s
            self._hand_state = hs

    def _loop(self) -> None:
        dt = 1.0 / self.rate
        t0 = time.monotonic()
        k_done = 0
        t_prev = t0
        next_view = 0.0
        while self._run:
            now = time.monotonic()
            behind = (now - t0) * self.rate - k_done
            lag_seen = behind * dt                     # recorded before a resync zeroes it (the stall is the point)
            if behind * dt > RESYNC_S:                 # far behind: drop the backlog instead of fast-forwarding
                t0 = now - k_done * dt
                behind = 0.0
                with self._lock:
                    self.rt_resyncs += 1
                    self.rt_resyncs_win += 1
            due = int(math.floor(behind))
            n = min(due, self.max_catchup)
            c0 = time.perf_counter()
            if n > 0:
                # A6: a sim state is due on a wall-clock schedule (the rate holds even when the sim lags real time)
                sim_due = self.scene is not None and now >= self._next_state
                if sim_due:
                    self._next_state = max(self._next_state + self.state_period, now - 3.0 * self.state_period)
                with self._core_lock:
                    seq, T = self._pending
                    if seq != self._applied_seq:
                        self._applied_seq = seq
                        self.core.set_target(T)
                    if sim_due:
                        self.probe.arm(n * self.steps_per_tick)      # capture the contacts of the last step
                    self.core.step(n * self.steps_per_tick)
                    if sim_due:
                        self._build_sim_state(self.probe.contacts)
                k_done += n
                self._snapshot()
                if sim_due and self._pub is not None:
                    self._pub_evt.set()
                if self._viewer is not None and now >= next_view:   # pragma: no cover - needs a display
                    next_view = now + 1 / 30
                    with self._core_lock:
                        self._viewer.sync()
            c1 = time.perf_counter()
            with self._lock:
                if n > 0:
                    self._periods.append(now - t_prev)
                    self._cycles.append(c1 - c0)
                    t_prev = now
                    if len(self._periods) > 20000:
                        del self._periods[:10000]
                        del self._cycles[:10000]
                self._lag_max = max(self._lag_max, lag_seen)
                self._spw_max = max(self._spw_max, n)
            nxt = t0 + (k_done + 1) * dt
            time.sleep(max(0.0, nxt - time.monotonic()))

    # ------------------------------------------------------------------------------------------- Backend API

    def start(self):
        with self._core_lock:
            self._applied_seq = self._pending[0]    # targets sent before start are stale
            self.core.start()
        self._snapshot()

    def set_target(self, T):
        self._pending = (self._pending[0] + 1, np.array(T, float))   # applied by the loop before its next step

    def get_state(self) -> dict:
        with self._lock:
            s = self._state
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}

    def move_joints(self, q, speed=0.2):
        """Blocking joint move, like PandaPyBackend: stop the controller, move, restart the impedance controller."""
        with self._core_lock:
            self.core.stop()
            T = self.core.begin_joint_move(q, speed)
        self._snapshot()                               # publish "moving" before waiting on the cached mode
        deadline = time.monotonic() + T + 5.0
        while True:
            time.sleep(0.002)
            st = self.get_state()
            if st["mode"] != "moving":
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"joint move did not finish within {T + 5.0:.1f} s")
        if st["error"]:
            raise RuntimeError(f"joint move aborted: {st['error']}")
        self.start()

    def stop(self):
        with self._core_lock:
            self.core.stop()
        self._snapshot()

    def recover(self):
        with self._core_lock:
            self.core.recover()
        self._snapshot()

    def inject_reflex(self, msg="cartesian_reflex (injected)"):
        with self._core_lock:
            self.core.inject_reflex(msg)
        self._snapshot()

    def handle_extra(self, topic, data):
        """I-HAND topics, served from the bridge's REP thread without _core_lock (HandSim has its own lock; the
        command is applied at the next control step and the state is the loop's last snapshot). A6: sim_state (no
        _core_lock) and sim_reset (under _core_lock)."""
        if topic == "sim_state":
            return self.sim_state()
        if topic == "sim_reset":
            return self.sim_reset(data)
        if self.hand is not None:
            if topic == "hand_cmd":
                return self.hand.command(data)
            if topic == "hand_state":
                with self._lock:
                    return dict(self._hand_state)
        raise ValueError(f"unknown request {topic!r}")

    # ------------------------------------------------------------------------------------- A6 sim state

    def _build_sim_state(self, contacts) -> None:
        """Under _core_lock: the I-SIMSTATE dict of the current data, stored under _lock."""
        core = self.core
        s = self.scene.sim_state(core, contacts, time.monotonic(), self.hand)
        f = float(np.max(np.abs(core.O_F_ext[:3])))
        with self._lock:
            self._sim_state = s
            self._fext_max = max(self._fext_max, f)

    def sim_state(self) -> dict | None:
        """The newest sim-state dict (a shallow copy; the arrays are never written after the build)."""
        with self._lock:
            s = self._sim_state
        return dict(s) if s is not None else None

    def _pub_loop(self) -> None:
        from robotmq import serialize

        try:
            while self._run:
                if not self._pub_evt.wait(0.1):
                    continue
                self._pub_evt.clear()
                with self._lock:
                    s = self._sim_state
                if s is None:
                    continue
                try:
                    self._pub.send(serialize(s), copy=False)
                    self.n_state_pub += 1
                except Exception as e:      # noqa: BLE001 - never kill the sim over a PUB error
                    print(f"[mujoco] sim-state PUB error: {e!r}", flush=True)
        finally:
            self._pub.close(linger=0)

    def sim_reset(self, data) -> dict:
        """Move the eraser (and optionally the fingers) under _core_lock; never the arm. Raises RuntimeError (the
        bridge replies ok: false) while grasping or when the pose penetrates anything but the table."""
        import mujoco

        if self.scene is None:
            raise RuntimeError("sim_reset: the model has no scene (backend.mujoco.with_scene false)")
        data = {} if data is None else data
        if not isinstance(data, dict):
            raise ValueError(f"sim_reset expects a dict, got {type(data).__name__}")
        sc, core = self.scene, self.core
        quat, z = None, None
        if data.get("object_pose") is not None:
            p = [float(v) for v in data["object_pose"]]
            if len(p) != 7:
                raise ValueError(f"sim_reset object_pose needs 7 values [x,y,z,qw,qx,qy,qz], got {len(p)}")
            x, y, z, quat = p[0], p[1], p[2], p[3:]
            yaw = None
        elif data.get("eraser_xy_yaw") is not None:
            x, y, yaw = (float(v) for v in data["eraser_xy_yaw"])
        else:
            (x, y), yaw = sc.home_xy, 0.0
        gap = data.get("hand_gap_m")
        if gap is not None:
            gap = float(gap)
            gmax = float(self.sm.limits.get("gap_max_m", 0.08))
            if not 0.0 <= gap <= gmax:
                raise ValueError(f"sim_reset hand_gap_m {gap} outside [0, {gmax}]")
        vals = [x, y] + ([yaw] if yaw is not None else [z, *quat]) + ([gap] if gap is not None else [])
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"sim_reset: non-finite value in {data}")
        with self._core_lock:
            ss = self._sim_state
            mode = self.hand.mode if self.hand is not None else None
            if (ss is not None and ss["grasped"]) or mode in ("grasping", "holding"):
                raise RuntimeError(f"sim_reset refused: grasping (physical grasp {bool(ss and ss['grasped'])}, "
                                   f"hand mode {mode})")
            d = core.d
            q0, v0 = d.qpos.copy(), d.qvel.copy()
            sc.set_eraser(d, x, y, yaw or 0.0, z=z, quat=quat)
            if gap is not None:
                sc.set_gap(d, gap)
            mujoco.mj_forward(self.sm.model, d)
            pen = sc.penetrations(d)
            if pen:
                d.qpos[:], d.qvel[:] = q0, v0
                mujoco.mj_forward(self.sm.model, d)
                raise RuntimeError(f"sim_reset refused: the eraser would penetrate {', '.join(pen)}")
            if gap is not None and self.hand is not None:
                self.hand.reset(gap, core)
            core._update_kin()
            self._build_sim_state(sc.contacts(d))         # valid right after mj_forward
            s = self._sim_state
        self._snapshot()
        if self._pub is not None:
            self._pub_evt.set()
        return {"ok": True, "eraser_xy_yaw": sc.eraser_xy_yaw(d), "hand_gap_m": gap, "t_mono": float(s["t_mono"])}

    def sim_tick(self, bridge) -> None:
        """Called by FrankaBridge.tick (200 Hz): a bridge.jsonl ``kind: sim`` line at sim_log_hz, written directly to
        bridge._jsonl (not bridge.log: no print, no bridge.events growth)."""
        f = getattr(bridge, "_jsonl", None)
        if f is None or self.scene is None or not self.sim_log_period:
            return
        now = time.monotonic()
        if now < self._next_sim_log:
            return
        self._next_sim_log = max(self._next_sim_log + self.sim_log_period, now - self.sim_log_period / 2)
        from franka_sim.scene import sim_line

        with self._lock:
            s = self._sim_state
            fmax, self._fext_max = self._fext_max, 0.0
        if s is None:
            return
        import json

        f.write(json.dumps(sim_line(s, fmax)) + "\n")

    def loop_stats(self) -> dict:
        now = time.monotonic()
        with self._lock:
            p, c = np.array(self._periods), np.array(self._cycles)
            self._periods, self._cycles = [], []
            lag, spw, rs, rsw = self._lag_max, self._spw_max, self.rt_resyncs, self.rt_resyncs_win
            self._lag_max, self._spw_max, self.rt_resyncs_win = 0.0, 0, 0
            sim_t = float(self._state["sim_t"])
        w0, s0 = self._win
        self._win = (now, sim_t)
        out = {"sim_lag_ms_max": lag * 1e3, "steps_per_wake_max": int(spw), "rt_resyncs": int(rs),
               "rt_resyncs_window": int(rsw),
               "realtime_factor": (sim_t - s0) / (now - w0) if now > w0 else float("nan")}
        if p.size:
            out.update({"n": int(p.size), "p50_ms": float(np.percentile(p, 50) * 1e3),
                        "p99_ms": float(np.percentile(p, 99) * 1e3), "p999_ms": float(np.percentile(p, 99.9) * 1e3),
                        "max_ms": float(p.max() * 1e3)})
            out.update(cycle_hist(c))
        return out

    def close(self):
        self._run = False
        self._th.join(2.0)
        if self._pub_th is not None:
            self._pub_evt.set()
            self._pub_th.join(2.0)
        if self._viewer is not None:  # pragma: no cover - needs a display
            self._viewer.close()
            self._viewer = None
