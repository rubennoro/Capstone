"""HandSim: libfranka Gripper move/grasp/stop/homing semantics on the MuJoCo Franka Hand finger tendon (A5).

Why: the controller's FrankaHandDriver talks I-HAND (hand_cmd / hand_state) to the bridge in sim and to hand_server
on hardware with identical code; this class is the sim end. It is installed as ``SimCore.hand_ctrl`` (A4 hook) and
returns the ``motor_hand`` tendon force every 1 ms control step (clipped to +-limits["hand_force_max_n"]).

Geometry: gap (libfranka width) = finger_joint1 + finger_joint2 = 2 x the ``split`` tendon length L. A tendon force T
splits 0.5 / 0.5 onto the two finger joints, so a squeeze of F N per finger is T = -2F (verify-on-hardware:
docs/panda_py_checklist.md §4b, libfranka's ``force`` may be the total, not per finger).

Semantics (one pending command, latest wins, applied at the next control step):
  move(width, speed)   reference ramps at ``speed`` (gap rate, trapezoid with accel_m_s2) to width; PD on L with
                       velocity feed-forward, force-capped. |gap - width| < 1 mm after the ramp -> move_ok;
                       |ref - gap| > 2 mm for > 0.2 s -> move_failed, hold where it is.
  grasp(w, v, F, ei, eo) close at v toward max(0, w - ei); stall (|gap_vel| < 2 mm/s for 50 ms with |ref - gap| > 1 mm)
                       -> mode holding, tendon force -2F; grasp_ok if w - ei <= gap <= w + eo else grasp_failed (still
                       holding: libfranka keeps the force); ``grasped`` is re-evaluated every step while holding.
                       Reaching the target without a stall -> grasp_failed, position hold.
  stop                 position hold at the current gap -> stopped.   homing: move to gap_max_m -> homing_ok.
hand_state = {gap_m, gap_vel_m_s, busy, mode, grasped, last_result, seq_done, t}.
"""

from __future__ import annotations

import math
import threading

import numpy as np

import mujoco

from franka_sim.simcore import HAND_KD, HAND_KP

KINDS = ("move", "grasp", "stop", "homing")
DONE_TOL_M = 0.001
BLOCK_ERR_M, BLOCK_S = 0.002, 0.2
STALL_VEL_M_S, STALL_S, STALL_ERR_M = 0.002, 0.05, 0.001
SPEED_MAX_M_S = 0.1
FORCE_MAX_N = 70.0
ACCEL_M_S2 = 2.0             # reference ramp acceleration (gap): 0.1 m/s in 50 ms


class HandSim:
    def __init__(self, core, tooling: dict | None = None, kp: float = HAND_KP, kd: float = HAND_KD,
                 accel_m_s2: float = ACCEL_M_S2):
        tooling = tooling or {}
        wm = tooling.get("width_map") or {}
        hand = tooling.get("hand") or {}
        self.gap_min = float(wm.get("gap_min_m", 0.0))
        self.gap_max = float(wm.get("gap_max_m", core.sm.limits.get("gap_max_m", 0.08)))
        self.force_max = min(float(hand.get("force_max_n", FORCE_MAX_N)), FORCE_MAX_N)
        self.f_lim = float(core.sm.limits["hand_force_max_n"])
        self.kp, self.kd, self.accel = float(kp), float(kd), float(accel_m_s2)
        self.dt = float(core.control_dt)
        self._ten = core._tendon
        self._lock = threading.Lock()
        self._pending = None
        g = self.gap(core)
        self.mode, self.last_result, self.seq_done, self.grasped = "idle", "", 0, False
        self._cmd = None                # active command dict
        self._ref, self._ref_v = g, 0.0     # gap reference and its rate
        self._target = g
        self._hold = g
        self._t_err = 0.0               # time with a large tracking error (move) / stall time (grasp)
        self._F = 0.0
        self.m_ten = self._tendon_inertia(core)          # acceleration feed-forward (kg on the tendon length)
        self._ref_a = 0.0
        self._state = self._make_state(core)

    def _tendon_inertia(self, core) -> float:
        """Effective inertia seen by the tendon force: 1 / (J M^-1 J^T), J = d(L)/dq (fingers + armature)."""
        m, d = core.m, core.d
        J = np.array(d.ten_J[self._ten], float).reshape(1, m.nv)
        x = np.zeros((1, m.nv))
        mujoco.mj_solveM(m, d, x, J)
        return float(1.0 / (J @ x.T)[0, 0])

    # ------------------------------------------------------------------------------------------ accessors
    def gap(self, core) -> float:
        return 2.0 * float(core.d.ten_length[self._ten])

    def gap_vel(self, core) -> float:
        return 2.0 * float(core.d.ten_velocity[self._ten])

    # -------------------------------------------------------------------------------- command (any thread)
    def command(self, msg: dict) -> dict:
        seq = int((msg or {}).get("seq", 0)) if isinstance(msg, dict) else 0
        if not isinstance(msg, dict) or msg.get("kind") not in KINDS:
            return {"accepted": False, "seq": seq, "error": f"kind must be one of {KINDS}"}
        c = {"kind": msg["kind"], "seq": seq}
        try:
            c["width_m"] = float(np.clip(float(msg.get("width_m") or 0.0), self.gap_min, self.gap_max))
            c["speed_m_s"] = float(np.clip(float(msg.get("speed_m_s") or SPEED_MAX_M_S), 1e-4, SPEED_MAX_M_S))
            c["force_n"] = float(np.clip(float(msg.get("force_n") or 0.0), 0.0, self.force_max))
            c["eps_inner_m"] = max(0.0, float(msg.get("eps_inner_m") or 0.0))
            c["eps_outer_m"] = max(0.0, float(msg.get("eps_outer_m") or 0.0))
        except (TypeError, ValueError) as e:
            return {"accepted": False, "seq": seq, "error": str(e)}
        if not all(math.isfinite(c[k]) for k in ("width_m", "speed_m_s", "force_n", "eps_inner_m", "eps_outer_m")):
            return {"accepted": False, "seq": seq, "error": "non-finite value"}
        with self._lock:
            self._pending = c
        return {"accepted": True, "seq": seq}

    def state(self, t_mono: float) -> dict:
        with self._lock:
            s = dict(self._state)
        s["t"] = float(t_mono)
        return s

    def reset(self, gap_m: float, core=None) -> None:
        """A6 sim_reset: drop any command and hold at gap_m (mode idle, last_result "reset"). The caller has put the finger
        joints at gap_m / 2 (under the core lock)."""
        g = float(gap_m)
        with self._lock:
            self._pending = None
            self._cmd = None
            self.mode, self.last_result, self.grasped = "idle", "reset", False
            self._t_err, self._F = 0.0, 0.0
            self._target = self._hold = self._ref = g
            self._ref_v = self._ref_a = 0.0
            self._state = self._make_state(core) if core is not None else {**self._state, "gap_m": g, "gap_vel_m_s": 0.0,
                                                                              "busy": False, "mode": "idle", "grasped": False,
                                                                              "last_result": "reset"}

    # ------------------------------------------------------------------------------ control step (A4 hook)
    def _finish(self, result: str, mode: str, hold: float) -> None:
        self.last_result, self.mode = result, mode
        self.seq_done = self._cmd["seq"] if self._cmd else self.seq_done
        self._cmd = None
        self._hold = hold
        self._ref, self._ref_v = hold, 0.0

    def _start(self, c: dict, g: float) -> None:
        k = c["kind"]
        self.grasped = False
        self._t_err = 0.0
        if k == "stop":
            self._cmd = c
            self._finish("stopped", "stopped", g)
            return
        self._cmd = c
        if self.mode not in ("moving", "grasping", "homing"):
            self._ref, self._ref_v = g, 0.0        # ramp from where the fingers are
        if k == "move":
            self._target, self.mode = c["width_m"], "moving"
        elif k == "homing":
            self._target, self.mode = self.gap_max, "homing"
            c["speed_m_s"] = SPEED_MAX_M_S
        else:
            self._target, self.mode = max(self.gap_min, c["width_m"] - c["eps_inner_m"]), "grasping"

    def _ramp(self, speed: float) -> None:
        """Trapezoid reference toward self._target at |rate| <= speed, |accel| <= self.accel."""
        d = self._target - self._ref
        self._ref_a = 0.0
        if abs(d) < 1e-12 and abs(self._ref_v) < 1e-9:
            self._ref_v = 0.0
            return
        a, dt = self.accel, self.dt
        v_stop = math.sqrt(2 * a * abs(d))                 # max speed that can still stop at the target
        v_des = math.copysign(min(speed, v_stop), d)
        dv = float(np.clip(v_des - self._ref_v, -a * dt, a * dt))
        self._ref_v += dv
        self._ref_a = dv / dt
        step = self._ref_v * dt
        if abs(step) >= abs(d) or (d > 0) != (step > 0) and abs(d) < 1e-5:
            self._ref, self._ref_v, self._ref_a = self._target, 0.0, 0.0
        else:
            self._ref += step

    def _pd(self, g: float, gv: float, ref: float, ref_v: float, ref_a: float = 0.0) -> float:
        # tendon-length PD (L = gap/2) with velocity and acceleration feed-forward
        return self.kp * (ref - g) / 2.0 + self.kd * (ref_v - gv) / 2.0 + self.m_ten * ref_a / 2.0

    def ctrl(self, core) -> float:
        with self._lock:
            c, self._pending = self._pending, None
        g, gv = self.gap(core), self.gap_vel(core)
        if c is not None:
            self._start(c, g)
        m = self.mode
        if m in ("moving", "homing"):
            self._ramp(self._cmd["speed_m_s"])
            F = self._pd(g, gv, self._ref, self._ref_v, self._ref_a)
            err = abs(self._ref - g)
            self._t_err = self._t_err + self.dt if err > BLOCK_ERR_M else 0.0
            if self._t_err > BLOCK_S:
                self._finish("move_failed" if m == "moving" else "error:homing blocked", "idle" if m == "moving"
                             else "error", g)
                F = self._pd(g, gv, g, 0.0)
            elif self._ref == self._target and abs(g - self._target) < DONE_TOL_M and abs(gv) < 0.01:
                self._finish("move_ok" if m == "moving" else "homing_ok", "idle", self._target)
        elif m == "grasping":
            cmd = self._cmd
            self._ramp(cmd["speed_m_s"])
            F = self._pd(g, gv, self._ref, self._ref_v, self._ref_a)
            stalled = abs(gv) < STALL_VEL_M_S and abs(self._ref - g) > STALL_ERR_M
            self._t_err = self._t_err + self.dt if stalled else 0.0
            if self._t_err >= STALL_S:
                self.mode = "holding"
                ok = cmd["width_m"] - cmd["eps_inner_m"] <= g <= cmd["width_m"] + cmd["eps_outer_m"]
                self.last_result = "grasp_ok" if ok else "grasp_failed"
                self.seq_done = cmd["seq"]
                F = -2.0 * cmd["force_n"]
            elif self._ref == self._target and abs(g - self._target) < DONE_TOL_M:
                self._finish("grasp_failed", "idle", self._target)
        elif m == "holding":
            cmd = self._cmd
            F = -2.0 * cmd["force_n"] - self.kd * gv / 2.0
        else:                                              # idle / stopped / error: position hold
            F = self._pd(g, gv, self._hold, 0.0)
        if self.mode == "holding":
            cmd = self._cmd
            self.grasped = cmd["width_m"] - cmd["eps_inner_m"] <= g <= cmd["width_m"] + cmd["eps_outer_m"]
        else:
            self.grasped = False
        self._F = float(np.clip(F, -self.f_lim, self.f_lim))
        st = self._make_state(core, g, gv)
        with self._lock:
            self._state = st
        return self._F

    def _make_state(self, core, g=None, gv=None) -> dict:
        g = self.gap(core) if g is None else g
        gv = self.gap_vel(core) if gv is None else gv
        return {"gap_m": float(g), "gap_vel_m_s": float(gv), "busy": self.mode in ("moving", "grasping", "homing"),
                "mode": self.mode, "grasped": bool(self.grasped), "last_result": self.last_result,
                "seq_done": int(self.seq_done), "t": 0.0}
