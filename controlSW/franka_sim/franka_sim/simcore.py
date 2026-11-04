"""SimCore: a deterministic, steppable simulated Franka arm (no threads) that reproduces panda-py's CartesianImpedance
controller with libfranka-style torque limits and reflexes on the A1 MuJoCo model.

Why: the unchanged bridge, controller and task must drive a simulated arm exactly like the real one (A4). The
MuJoCoBackend (backend.py) wraps this class in a 1 kHz wall-clock thread; tests and ``characterize`` step it directly.

Control law (one control step = 1 ms = ``n_substeps`` mj_steps; panda-py source was not available offline, so this is
the documented law of franka_example_controllers' cartesian_impedance_example_controller, which panda-py's
CartesianImpedance follows -- marked verify-on-hardware in docs/panda_py_checklist.md):
  1. target filter  p_d <- fc p_t + (1 - fc) p_d, quaternion slerp(q_d, q_t, fc)
  2. errors         e_p = p - p_d; flip q_d if q.q_d < 0; e_q = q^-1 q_d; e_r = -R vec(e_q)
  3. task torque    tau_task = J^T (-K e - D J dq), D = 2 zeta sqrt(K) (elementwise), J = 6x7 Jacobian of site `ee`
  4. null space     tau_ns = (I - J^T pinv(J^T)) (k_ns (q_ns - q) - 2 sqrt(k_ns) dq), q_ns = q at start()
  5. tau_d = tau_task + tau_ns + Coriolis (qfrc_bias minus gravity)
  6. |tau_d - tau_prev| <= 1 Nm per ms, then saturate to +-tau_max
  7. applied: ctrl = tau_d + gravity(q) (+ J_com^T dm g for physics.payload_error_kg = configured - true payload)
Modes: idle / stopped (joint PD hold at the q captured on entry), running (impedance), reflex (joint PD hold; targets
ignored until recover()), moving (quintic joint trajectory from begin_joint_move; start() refused).
Reflexes (running/moving only; libfranka has no reflexes without an active controller or motion; the first hit wins):
joint_position_limits_violation (station datasheet limits intersected with the MJCF range), joint_velocity_violation, self_collision_avoidance_violation, joint_reflex
(|tau_ext_i| > tau thr), cartesian_reflex (|O_F_ext_j| > F thr); optional (physics.reflex): cartesian_speed,
torque_discontinuity. tau_ext = arm-DOF qfrc_constraint + qfrc_applied + xfrc_applied on the arm; O_F_ext =
pinv(J^T) tau_ext.

Usage: core = SimCore(sm, make_core_cfg(station), q_init); core.start(); core.set_target(T); core.step(1000)
"""

from __future__ import annotations

import math

import numpy as np

import mujoco

from franka_sim.model import SimModel, sim_config

LIBFRANKA_TAU_THR = (20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0)
LIBFRANKA_F_THR = (20.0, 20.0, 20.0, 25.0, 25.0, 25.0)
JOINT_KP = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)       # franka joint_impedance_example gains
JOINT_KD = (50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0)
DTAU_MAX_PER_MS = 1.0                                              # libfranka saturateTorqueRate (Nm per 1 ms)
CONTROL_DT = 0.001
HAND_KP, HAND_KD = 2000.0, 20.0                                    # finger hold until A5 replaces it (N/m, N s/m)
REFLEX_DEFAULTS = {"contact": True, "velocity": True, "joint_limit": True, "self_collision": True,
                   "cartesian_speed": False, "torque_discontinuity": False}


def make_core_cfg(station, panda_py: dict | None = None, overrides: dict | None = None,
                  physics: dict | None = None) -> dict:
    """SimCore config: robot impedance/limits from the station, then the real arm's backend.panda_py dict (impedance
    6x6, damping_ratio, nullspace_stiffness, filter_coeff, collision), then ``overrides`` (backend.mujoco)."""
    robot = station.get("robot")
    imp = robot["impedance"]
    t, r = float(imp["translational_n_m"]), float(imp["rotational_nm_rad"])
    cfg = {"impedance": [t, t, t, r, r, r], "damping_ratio": float(imp["damping_ratio"]),
           "nullspace_stiffness": float(imp["nullspace_stiffness"]), "filter_coeff": float(imp["filter_coeff"]),
           "collision": imp.get("collision_thresholds"),
           "dq_max_rad_s": list(robot["joints"]["dq_max_rad_s"]),
           "v_max_m_s": float(robot["cartesian"]["v_max_m_s"]),
           "physics": dict(physics) if physics is not None else sim_config(station, None)}
    for src in (panda_py or {}, overrides or {}):
        for k in ("impedance", "damping_ratio", "nullspace_stiffness", "filter_coeff", "collision"):
            if k in src and (src[k] is not None or k == "collision"):
                cfg[k] = src[k]
        if "physics" in src:
            cfg["physics"] = {**cfg["physics"], **dict(src["physics"])}
        for k in ("dq_max_rad_s", "v_max_m_s", "joint_kp", "joint_kd", "pinv_damping"):
            if k in src:
                cfg[k] = src[k]
    return cfg


def _stiffness6(imp) -> np.ndarray:
    a = np.asarray(imp, float)
    if a.shape == (6, 6):
        return np.diag(a).copy()
    if a.shape == (6,):
        return a.copy()
    if a.shape == (2,):
        return np.r_[[a[0]] * 3, [a[1]] * 3]
    raise ValueError(f"impedance must be 6x6, 6 or [translational, rotational], got shape {a.shape}")


def collision_thresholds(coll) -> tuple[np.ndarray, np.ndarray]:
    """(tau 7, F 6) reflex thresholds from backend.panda_py.collision / robot.impedance.collision_thresholds.
    Accepted: None (libfranka defaults); {tau_nm|torque, force_n|force}; [tau7, F6]; the setCollisionBehavior
    argument list [lower_tau7, upper_tau7, lower_F6, upper_F6] (the upper = reflex thresholds are used)."""
    if coll is None:
        return np.array(LIBFRANKA_TAU_THR), np.array(LIBFRANKA_F_THR)
    if isinstance(coll, dict):
        tau = coll.get("tau_nm", coll.get("torque", LIBFRANKA_TAU_THR))
        f = coll.get("force_n", coll.get("force", LIBFRANKA_F_THR))
        return np.broadcast_to(np.asarray(tau, float), (7,)).copy(), np.broadcast_to(np.asarray(f, float), (6,)).copy()
    c = list(coll)
    if len(c) == 2:
        return np.broadcast_to(np.asarray(c[0], float), (7,)).copy(), np.broadcast_to(np.asarray(c[1], float), (6,)).copy()
    if len(c) == 4:
        return np.asarray(c[1], float).reshape(7), np.asarray(c[3], float).reshape(6)
    raise ValueError(f"unsupported collision thresholds {coll!r}")


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(min(1.0, d))
        s = math.sin(th)
        q = (math.sin((1 - a) * th) / s) * q0 + (math.sin(a * th) / s) * q1
    return q / np.linalg.norm(q)


def _mat2quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q


class SimCore:
    def __init__(self, sm: SimModel, cfg: dict, q_init=None):
        self.sm, self.cfg = sm, dict(cfg)
        m = self.m = sm.model
        self.d = mujoco.MjData(m)
        nm = sm.names
        mujoco.mj_resetDataKeyframe(m, self.d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, nm["key_home"]))
        self._jid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm["joints"][f"j{i}"]) for i in range(1, 8)])
        self._qadr = m.jnt_qposadr[self._jid].copy()
        self._dadr = m.jnt_dofadr[self._jid].copy()
        self._jbody = m.jnt_bodyid[self._jid].copy()
        self._act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm["actuators"][f"j{i}"])
                              for i in range(1, 8)])
        self._act_hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm["actuators"]["hand"])
        self._tendon = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, nm["tendon"])
        self._site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm["sites"]["ee"])
        self._hand_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm["bodies"]["hand"])
        self._tool_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm["bodies"]["tool"])
        # arm subtree bodies (link1 and below): where external forces count as tau_ext
        root = int(self._jbody[0])
        sub = [root]
        for b in range(root + 1, m.nbody):
            if m.body_parentid[b] in sub:
                sub.append(b)
        self._arm_bodies = np.array(sub)
        self._g = np.array(m.opt.gravity, float)

        # -- config
        phys = dict(cfg.get("physics") or {})
        self.n_substeps = int(phys.get("n_substeps", 1))
        self.control_dt = float(m.opt.timestep) * self.n_substeps
        if abs(self.control_dt - CONTROL_DT) > 1e-9:
            raise ValueError(f"timestep_s x n_substeps must be {CONTROL_DT} s (got {m.opt.timestep} x {self.n_substeps})")
        self.payload_error_kg = float(phys.get("payload_error_kg", 0.0))
        self.reflex_on = {**REFLEX_DEFAULTS, **dict(phys.get("reflex") or {})}
        self.K = _stiffness6(cfg.get("impedance", [800.0] * 3 + [40.0] * 3))
        self.D = 2.0 * float(cfg.get("damping_ratio", 1.0)) * np.sqrt(self.K)
        self.k_ns = float(cfg.get("nullspace_stiffness", 0.5))
        self.fc = float(cfg.get("filter_coeff", 1.0))
        self.pinv_damping = float(cfg.get("pinv_damping", 0.0))
        self.tau_thr, self.f_thr = collision_thresholds(cfg.get("collision"))
        self.tau_max = np.asarray(sm.limits["tau_max_nm"], float)
        self.dq_max = np.asarray(cfg.get("dq_max_rad_s", [2.175] * 4 + [2.61] * 3), float)
        self.v_max = float(cfg.get("v_max_m_s", 1.7))
        self.kp = np.asarray(cfg.get("joint_kp", JOINT_KP), float)
        self.kd = np.asarray(cfg.get("joint_kd", JOINT_KD), float)
        rng = sm.limits["mjcf_range_rad"]              # station datasheet limits within the MJCF range (as Kin)
        self.q_lo = np.maximum(np.asarray(sm.limits["q_min_rad"], float), rng[:, 0])
        self.q_hi = np.minimum(np.asarray(sm.limits["q_max_rad"], float), rng[:, 1])
        self._self_pairs = self._contact_pairs()

        # -- initial state
        if q_init is not None:
            self.d.qpos[self._qadr] = np.asarray(q_init, float)
        self.d.qvel[:] = 0
        mujoco.mj_forward(m, self.d)
        self._hand_L = float(self.d.ten_length[self._tendon])     # finger hold target (A5 replaces the hold)
        self.hand_ctrl = None                                     # optional callable(core) -> tendon force (A5)
        self.contact_probe = None                                 # optional callable(core) after mj_step2 (A6 scene.ContactProbe)
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._J, self._chol = np.zeros((6, 7)), np.zeros((6, 6))
        self._lam_eye = (self.pinv_damping ** 2 if self.pinv_damping > 0 else 1e-10) * np.eye(6)
        self._e = np.zeros(6)
        self._gz_only = bool(self._g[0] == 0 and self._g[1] == 0)
        self._qfrc = np.zeros(m.nv)
        self.tau_prev = np.zeros(7)
        self.tau_ext = np.zeros(7)
        self.O_F_ext = np.zeros(6)
        self.dtau_max_seen = 0.0
        self.mode, self.error = "idle", None
        self.n_steps = 0
        self._q_hold = self.q.copy()
        self._move = None
        self._q_ns = self.q.copy()
        self._update_kin()
        self._p_t, self._q_t = self._p.copy(), self._quat.copy()
        self._p_d, self._q_d = self._p.copy(), self._quat.copy()

    # ------------------------------------------------------------------------------------------- state access

    @property
    def q(self) -> np.ndarray:
        return self.d.qpos[self._qadr].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.d.qvel[self._dadr].copy()

    def _contact_pairs(self) -> set:
        """Kinematics' self + tooling geom pairs. MuJoCo never generates contacts for some of them (e.g. housing/cradle
        vs link 6: the tool body is welded to link 7, whose parent is link 6, so filterparent drops the pair); those
        are checked by signed distance every control step instead (``self._dist_pairs``)."""
        from franka_sim.kinematics import Kin

        m = self.m
        kin = Kin(self.sm, "ee")
        pairs = {(min(a, b), max(a, b)) for grp in ("self", "tooling") for a, b in kin._pairs[grp]}
        filt_parent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        excl = {int(x) for x in m.exclude_signature[:m.nexclude]}

        def generated(g1: int, g2: int) -> bool:
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                return False
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
            if w1 == w2:
                return False
            pw1, pw2 = int(m.body_weldid[m.body_parentid[w1]]), int(m.body_weldid[m.body_parentid[w2]])
            if filt_parent and w1 and w2 and (w1 == pw2 or w2 == pw1):
                return False
            return ((min(b1, b2) << 16) + max(b1, b2)) not in excl

        self._dist_pairs = [(a, b) for a, b in sorted(pairs) if not generated(a, b)]
        self._fromto = np.zeros(6)
        return pairs

    def _update_kin(self) -> None:
        """Everything derived from the current (q, dq) after mj_step1 / mj_forward: EE pose, J and the Cholesky factor
        of J J^T (+ pinv_damping^2 I), so that pinv(J^T) x = (J J^T)^-1 J x costs one mju_cholSolve per vector."""
        d = self.d
        self._p = d.site_xpos[self._site].copy()
        self._R = d.site_xmat[self._site].reshape(3, 3).copy()
        self._quat = _mat2quat(self._R)
        mujoco.mj_jacSite(self.m, d, self._jp, self._jr, self._site)
        J = self._J
        J[:3] = self._jp[:, self._dadr]
        J[3:] = self._jr[:, self._dadr]
        self._chol[:] = J @ J.T
        self._chol += self._lam_eye
        mujoco.mju_cholFactor(self._chol, 0.0)

    def _pinvJt(self, x: np.ndarray) -> np.ndarray:
        """pinv(J^T) x for a 7-vector x (6-vector result)."""
        out = np.zeros(6)
        mujoco.mju_cholSolve(out, self._chol, self._J @ x)
        return out

    def state(self) -> dict:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self._R, self._p
        return {"O_T_EE": T, "q": self.q, "dq": self.dq, "O_F_ext": self.O_F_ext.copy(), "tau_ext": self.tau_ext.copy(),
                "mode": self.mode, "error": self.error, "sim_t": float(self.d.time), "success_rate": 1.0}

    # ------------------------------------------------------------------------------------------------ commands

    def set_target(self, O_T_EE) -> None:
        if self.mode != "running" or self.error:
            return
        T = np.asarray(O_T_EE, float)
        self._p_t = T[:3, 3].copy()
        self._q_t = _mat2quat(T[:3, :3])

    def start(self) -> None:
        if self.error:
            raise RuntimeError(f"cannot start: reflex active ({self.error}); recover first")
        if self.mode == "moving":
            raise RuntimeError("cannot start: joint move in progress")
        self._p_t, self._q_t = self._p.copy(), self._quat.copy()
        self._p_d, self._q_d = self._p.copy(), self._quat.copy()
        self._q_ns = self.q
        self.mode = "running"

    def _hold(self, mode: str) -> None:
        self.mode = mode
        self._q_hold = self.q
        self._move = None

    def stop(self) -> None:
        if self.mode != "reflex":
            self._hold("stopped")

    def recover(self) -> None:
        self.error = None
        self._hold("idle")

    def inject_reflex(self, msg="cartesian_reflex (injected)") -> None:
        self._reflex(str(msg))

    def _reflex(self, msg: str) -> None:
        self.error = msg
        self._hold("reflex")

    def begin_joint_move(self, q_goal, speed: float = 0.2) -> float:
        """Quintic joint trajectory to q_goal; duration so that the peak joint speed is speed x dq_max. Returns it (s)."""
        if self.error:
            raise RuntimeError(f"cannot move: reflex active ({self.error}); recover first")
        q0, q1 = self.q, np.asarray(q_goal, float).reshape(7)
        speed = float(np.clip(speed, 1e-3, 1.0))
        T = max(0.05, float(np.max(1.875 * np.abs(q1 - q0) / (speed * self.dq_max))))
        self._move = {"q0": q0, "q1": q1, "T": T, "t0": float(self.d.time)}
        self.mode = "moving"
        return T

    # ------------------------------------------------------------------------------------------------ stepping

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self._control_step()

    def _gravity(self, extra_mass: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(gravity bias on the arm DOFs = the part of qfrc_bias due to gravity at zero velocity, torque that holds an
        extra point mass at the payload COM against gravity)."""
        d = self.d
        anchors, axes = d.xanchor[self._jid], d.xaxis[self._jid]
        com = d.subtree_com[self._jbody]
        mass = self.m.body_subtreemass[self._jbody]
        gb = -self._torque_of(com - anchors, mass, axes)
        pay = np.zeros(7)
        if extra_mass:
            mh, mt = self.m.body_subtreemass[self._hand_b], self.m.body_subtreemass[self._tool_b]
            c = (mh * d.subtree_com[self._hand_b] + mt * d.subtree_com[self._tool_b]) / (mh + mt)
            pay = -self._torque_of(c[None, :] - anchors, np.full(7, extra_mass), axes)
        return gb, pay

    def _torque_of(self, r: np.ndarray, mass: np.ndarray, axes: np.ndarray) -> np.ndarray:
        """Joint torques a_i . (r_i x m_i g) of gravity on point masses at lever arms r_i (7x3)."""
        if self._gz_only:                      # g = (0, 0, gz): r x F = (r_y F_z, -r_x F_z, 0)
            fz = mass * self._g[2]
            return (axes[:, 0] * r[:, 1] - axes[:, 1] * r[:, 0]) * fz
        return np.einsum("ij,ij->i", np.cross(r, mass[:, None] * self._g), axes)

    def _control_law(self, q, dq, coriolis) -> np.ndarray:
        if self.mode == "running":
            fc = self.fc
            if fc >= 1.0:
                self._p_d, self._q_d = self._p_t.copy(), self._q_t.copy()
            else:
                self._p_d = fc * self._p_t + (1 - fc) * self._p_d
                self._q_d = _slerp(self._q_d, self._q_t, fc)
            qd = self._q_d if float(self._quat @ self._q_d) >= 0 else -self._q_d
            qinv = np.zeros(4)
            mujoco.mju_negQuat(qinv, self._quat)
            eq = np.zeros(4)
            mujoco.mju_mulQuat(eq, qinv, qd)
            e = self._e
            e[:3] = self._p - self._p_d
            e[3:] = -self._R @ eq[1:]
            J = self._J
            tau_task = J.T @ (-self.K * e - self.D * (J @ dq))
            x = self.k_ns * (self._q_ns - q) - 2.0 * math.sqrt(self.k_ns) * dq
            tau_ns = x - J.T @ self._pinvJt(x)                       # (I - J^T pinv(J^T)) x
            return tau_task + tau_ns + coriolis
        if self.mode == "moving":
            mv = self._move
            s = min(1.0, (float(self.d.time) - mv["t0"]) / mv["T"])
            pos = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            vel = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / mv["T"]
            dqd = mv["q1"] - mv["q0"]
            qr, dqr = mv["q0"] + pos * dqd, vel * dqd
            if s >= 1.0 and (np.max(np.abs(q - mv["q1"])) < 2e-4 or float(self.d.time) - mv["t0"] > mv["T"] + 2.0):
                self.mode, self._q_hold, self._move = "idle", mv["q1"].copy(), None
            return self.kp * (qr - q) + self.kd * (dqr - dq) + coriolis
        return self.kp * (self._q_hold - q) - self.kd * dq + coriolis

    def _control_step(self) -> None:
        m, d = self.m, self.d
        q, dq = d.qpos[self._qadr], d.qvel[self._dadr]
        bias = d.qfrc_bias[self._dadr]
        gb, pay = self._gravity(self.payload_error_kg)
        tau_raw = self._control_law(q, dq, bias - gb)
        dlim = DTAU_MAX_PER_MS * self.control_dt / 1e-3
        jump = tau_raw - self.tau_prev
        tau = np.clip(self.tau_prev + np.clip(jump, -dlim, dlim), -self.tau_max, self.tau_max)
        self.dtau_max_seen = max(self.dtau_max_seen, float(np.max(np.abs(tau - self.tau_prev))))
        self.tau_prev = tau
        d.ctrl[self._act] = tau + gb + pay
        if self.hand_ctrl is not None:
            d.ctrl[self._act_hand] = float(self.hand_ctrl(self))
        else:
            f = HAND_KP * (self._hand_L - d.ten_length[self._tendon]) - HAND_KD * d.ten_velocity[self._tendon]
            lim = self.sm.limits["hand_force_max_n"]
            d.ctrl[self._act_hand] = float(np.clip(f, -lim, lim))
        mujoco.mj_step2(m, d)
        for _ in range(self.n_substeps - 1):
            mujoco.mj_step1(m, d)
            mujoco.mj_step2(m, d)
        if self.contact_probe is not None:          # d.contact and d.efc_force still match here (A6 contact forces)
            self.contact_probe(self)
        tau_ext = d.qfrc_constraint[self._dadr] + d.qfrc_applied[self._dadr]
        mujoco.mj_step1(m, d)                       # kinematics, collisions and bias forces of the new state
        xf = d.xfrc_applied[self._arm_bodies]
        if np.any(xf):
            self._qfrc[:] = 0
            for b, w in zip(self._arm_bodies, xf):
                if np.any(w):
                    mujoco.mj_applyFT(m, d, w[:3].copy(), w[3:].copy(), d.xipos[b].copy(), int(b), self._qfrc)
            tau_ext = tau_ext + self._qfrc[self._dadr]
        self.tau_ext = tau_ext
        self._update_kin()
        self.O_F_ext = self._pinvJt(tau_ext) if np.any(tau_ext) else np.zeros(6)
        self.n_steps += 1
        if self.mode in ("running", "moving") and not self.error:
            msg = self._check_reflex(np.abs(jump).max() > dlim)
            if msg:
                self._reflex(msg)

    def _gname(self, g: int) -> str:
        return self.m.geom(g).name or f"{self.m.body(int(self.m.geom_bodyid[g])).name}:geom{g}"

    def _check_reflex(self, tau_jump: bool) -> str | None:
        on = self.reflex_on
        q, dq = self.d.qpos[self._qadr], self.d.qvel[self._dadr]
        if on["joint_limit"] and ((q < self.q_lo).any() or (q > self.q_hi).any()):
            bad = np.flatnonzero((q < self.q_lo) | (q > self.q_hi))
            if bad.size:
                i = int(bad[0])
                return f"joint_position_limits_violation: q{i + 1} {q[i]:.4f} outside [{self.q_lo[i]:.4f}, {self.q_hi[i]:.4f}]"
        if on["velocity"] and (np.abs(dq) > self.dq_max).any():
            bad = np.flatnonzero(np.abs(dq) > self.dq_max)
            if bad.size:
                i = int(bad[0])
                return f"joint_velocity_violation: |dq{i + 1}| {abs(dq[i]):.3f} > {self.dq_max[i]:.3f} rad/s"
        if on["self_collision"] and on["contact"] and self.d.ncon:
            con = self.d.contact
            for k in np.flatnonzero(con.dist < 0):
                g1, g2 = (int(x) for x in con.geom[k])
                if (min(g1, g2), max(g1, g2)) in self._self_pairs:
                    return (f"self_collision_avoidance_violation: {self._gname(g1)} / {self._gname(g2)}")
        if on["self_collision"] and on["contact"]:
            for g1, g2 in self._dist_pairs:            # pairs MuJoCo does not collide (see _contact_pairs)
                if mujoco.mj_geomDistance(self.m, self.d, g1, g2, 0.0, self._fromto) < 0:
                    return (f"self_collision_avoidance_violation: {self._gname(g1)} / {self._gname(g2)} "
                            f"(distance check)")
        if on["contact"] and ((np.abs(self.tau_ext) > self.tau_thr).any() or (np.abs(self.O_F_ext) > self.f_thr).any()):
            bad = np.flatnonzero(np.abs(self.tau_ext) > self.tau_thr)
            if bad.size:
                i = int(bad[0])
                return f"joint_reflex: |tau_ext{i + 1}| {abs(self.tau_ext[i]):.1f} > {self.tau_thr[i]:.1f} Nm"
            bad = np.flatnonzero(np.abs(self.O_F_ext) > self.f_thr)
            if bad.size:
                i = int(bad[0])
                return f"cartesian_reflex: |O_F_ext{i}| {abs(self.O_F_ext[i]):.1f} > {self.f_thr[i]:.1f}"
        if on["cartesian_speed"]:
            v = float(np.linalg.norm(self._J[:3] @ dq))
            if v > self.v_max:
                return f"cartesian_velocity_violation: {v:.3f} > {self.v_max:.3f} m/s"
        if on["torque_discontinuity"] and tau_jump:
            return "controller_torque_discontinuity: commanded torque step > 1 Nm/ms"
        return None
