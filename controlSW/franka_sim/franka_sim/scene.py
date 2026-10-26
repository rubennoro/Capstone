"""Scene runtime helpers on (model, data) (A6): the eraser pose get/set, named contacts with normal forces, the physical
grasp test, the eraser tilt and the I-SIMSTATE dict. The scene itself (table, eraser free joint, grip solref/solimp) is
built by ``model.build_model``; this module only reads and moves it.

Contact names (I-SIMSTATE): ``m.geom(g).name``, or ``f"{body_name}_g{g}"`` for an unnamed geom (the menagerie arm and
hand geoms are unnamed), so B9's substring patterns match: ``link6`` / ``fr3_link6``, ``left_finger`` / ``right_finger``
(``finger``), ``pad_l`` / ``pad_r`` (``pad_``), ``table``, ``eraser``, ``housing``, ``cradle_*``.

Contact forces are only consistent with ``d.contact`` right after ``mj_step2`` (``mj_step1`` re-runs collision and
rebuilds the constraint arrays) or after ``mj_forward``: ``ContactProbe`` is installed as ``SimCore.contact_probe`` and
captures the list at the end of the control step it is armed for.

Usage:
    sc = Scene(sm)                         # needs a model built with_scene=True
    sc.eraser_T(d); sc.set_eraser(d, x, y, yaw); sc.contacts(d)   # (after mj_forward / inside a probe)
"""

from __future__ import annotations

import math

import numpy as np

import mujoco

GRASP_FORCE_MIN_N = 0.5          # physical grasp: both pads touch the eraser with at least this normal force
CONTACT_MIN_N = 0.1              # contacts below this are dropped
CONTACTS_MAX = 32
REST_LIFT_M = 0.0005             # sim_reset drops the eraser from 0.5 mm above the table
PENETRATION_MAX_M = 0.001        # sim_reset refuses a pose that penetrates anything but the table by more
PAD_TABLE_CLEARANCE_M = 0.005    # grasp_tcp_height_m: the pad bottoms stay this far above the table top (A6 review
                                 # finding 4: 3 mm left a 0.07 mm margin after the descent undershoot)


def geom_name(m, g: int) -> str:
    return m.geom(g).name or f"{m.body(int(m.geom_bodyid[g])).name}_g{g}"


def quat_to_R(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def R_to_quat(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def tilt_deg(T) -> float:
    """Angle between the eraser z axis and the base z axis (the judge's rule)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(T[2][2])))))


def grasp_tcp_height_m(sm, station) -> float:
    """TCP height above the table top for the scripted grasp: the station's ``scene.grasp_tcp_height_m``, raised so
    the pads (the TCP is at the pad centre, pads are ``pad_size_m[2]`` long) clear the table by PAD_TABLE_CLEARANCE_M. With the
    station's 0.01 m and 30 mm pads the station value would put the pads 5 mm into the table."""
    h = float(station.get("scene.grasp_tcp_height_m", 0.01))
    pl = float(station.get("tooling.finger_adapter.pad_size_m")[2])
    return max(h, pl / 2 + PAD_TABLE_CLEARANCE_M)


class Scene:
    def __init__(self, sm):
        m = self.m = sm.model
        nm = sm.names
        if "object" not in nm.get("bodies", {}):
            raise ValueError("Scene needs a model built with_scene=True (no eraser body)")
        oid = lambda kind, n: mujoco.mj_name2id(m, kind, n)  # noqa: E731
        self.body = oid(mujoco.mjtObj.mjOBJ_BODY, nm["bodies"]["object"])
        j = oid(mujoco.mjtObj.mjOBJ_JOINT, nm["joints"]["object"])
        self.qa, self.va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.g_eraser = oid(mujoco.mjtObj.mjOBJ_GEOM, nm["geoms"]["object"])
        self.g_table = oid(mujoco.mjtObj.mjOBJ_GEOM, nm["geoms"]["table"])
        self.tcp_site = oid(mujoco.mjtObj.mjOBJ_SITE, nm["sites"]["tcp"])
        self.pads = (oid(mujoco.mjtObj.mjOBJ_GEOM, nm["geoms"]["pad_l"]), oid(mujoco.mjtObj.mjOBJ_GEOM, nm["geoms"]["pad_r"]))
        self.fingers = [int(m.jnt_qposadr[oid(mujoco.mjtObj.mjOBJ_JOINT, f)]) for f in nm["fingers"].values()]
        self.finger_dofs = [int(m.jnt_dofadr[oid(mujoco.mjtObj.mjOBJ_JOINT, f)]) for f in nm["fingers"].values()]
        self.size = 2.0 * np.array(m.geom_size[self.g_eraser], float)          # full box size
        key = oid(mujoco.mjtObj.mjOBJ_KEY, nm["key_home"])
        kq = np.array(m.key_qpos[key][self.qa:self.qa + 7], float)
        self.home_xy = (float(kq[0]), float(kq[1]))
        self.z_rest = float(kq[2])                       # top_z + h/2 (model.py puts the key eraser on the table)
        self.top_z = self.z_rest - self.size[2] / 2
        self.names = [geom_name(m, g) for g in range(m.ngeom)]
        self._f = np.zeros(6)

    # ------------------------------------------------------------------------------------------ eraser pose
    def eraser_pose7(self, d) -> np.ndarray:
        return np.r_[d.xpos[self.body], d.xquat[self.body]].astype(float)

    def eraser_T(self, d) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = d.xmat[self.body].reshape(3, 3)
        T[:3, 3] = d.xpos[self.body]
        return T

    def eraser_xy_yaw(self, d) -> list:
        R = d.xmat[self.body].reshape(3, 3)
        return [float(d.xpos[self.body, 0]), float(d.xpos[self.body, 1]), float(math.atan2(R[1, 0], R[0, 0]))]

    def set_eraser(self, d, x: float, y: float, yaw: float, z: float | None = None, quat=None) -> None:
        """Free-joint qpos/qvel only; the caller runs mj_forward."""
        d.qpos[self.qa:self.qa + 3] = [x, y, self.z_rest + REST_LIFT_M if z is None else z]
        d.qpos[self.qa + 3:self.qa + 7] = yaw_quat(yaw) if quat is None else np.asarray(quat, float) / np.linalg.norm(quat)
        d.qvel[self.va:self.va + 6] = 0.0

    def set_gap(self, d, gap_m: float) -> None:
        for a, v in zip(self.fingers, self.finger_dofs):
            d.qpos[a] = gap_m / 2
            d.qvel[v] = 0.0

    # ------------------------------------------------------------------------------------------ contacts
    def contacts(self, d, min_force_n: float = CONTACT_MIN_N, max_n: int = CONTACTS_MAX) -> list:
        """[{a, b, force_n}]: |normal force| per geom pair (duplicates summed, < min_force_n dropped, largest first).
        Valid only where d.efc_force matches d.contact (after mj_step2 / mj_forward)."""
        m = self.m
        acc: dict = {}
        f = self._f
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            mujoco.mj_contactForce(m, d, i, f)
            key = (g1, g2) if g1 < g2 else (g2, g1)
            acc[key] = acc.get(key, 0.0) + abs(float(f[0]))
        out = [{"a": self.names[a], "b": self.names[b], "force_n": round(v, 4)} for (a, b), v in acc.items() if v >= min_force_n]
        out.sort(key=lambda c: -c["force_n"])
        return out[:max_n]

    def grasped(self, contacts) -> bool:
        """Physical grasp: pad_l AND pad_r touch the eraser with >= 0.5 N each."""
        e = self.names[self.g_eraser]
        pl, pr = self.names[self.pads[0]], self.names[self.pads[1]]
        f = {pl: 0.0, pr: 0.0}
        for c in contacts:
            for p in (pl, pr):
                if {c["a"], c["b"]} == {p, e}:
                    f[p] += c["force_n"]
        return f[pl] >= GRASP_FORCE_MIN_N and f[pr] >= GRASP_FORCE_MIN_N

    def penetrations(self, d, max_m: float = PENETRATION_MAX_M) -> list:
        """Geoms other than the table that the eraser penetrates by more than max_m (d.contact after mj_forward)."""
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if self.g_eraser not in (g1, g2) or c.dist >= -max_m:
                continue
            other = g2 if g1 == self.g_eraser else g1
            if other != self.g_table:
                out.add(f"{self.names[other]} ({-c.dist * 1e3:.1f} mm)")
        return sorted(out)

    # ------------------------------------------------------------------------------------------ I-SIMSTATE
    def sim_state(self, core, contacts, t_mono: float, hand=None) -> dict:
        """The I-SIMSTATE dict (keys simlink/judge read, plus hand_mode, hand_grasped, sim_t, and tcp_z: the TCP site z
        in the base frame, so in-hand slip = the drop of eraser z - tcp_z while grasped, A7)."""
        d = core.d
        hs = hand._state if hand is not None else None
        gap = float(hs["gap_m"]) if hs is not None else 2.0 * float(d.ten_length[core._tendon])
        return {"t_mono": float(t_mono), "qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "eraser_T_base": self.eraser_T(d),
                "hand_gap_m": gap, "grasped": bool(self.grasped(contacts)), "contacts": contacts,
                "reflex": core.error, "tau_ext": np.asarray(core.tau_ext, float).copy(),
                "hand_mode": hs["mode"] if hs is not None else None,
                "hand_grasped": bool(hs["grasped"]) if hs is not None else None, "sim_t": float(d.time),
                "tcp_z": float(d.site_xpos[self.tcp_site][2])}


class ContactProbe:
    """``SimCore.contact_probe``: ``arm(n)`` makes it capture ``Scene.contacts`` after the mj_step2 of the n-th next
    control step (1 = the next one); the list lands in ``self.contacts`` and ``self.seq`` increments."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self._left = 0
        self.contacts: list = []
        self.seq = 0

    def arm(self, n_steps: int = 1) -> None:
        self._left = max(1, int(n_steps))

    def __call__(self, core) -> None:
        if self._left <= 0:
            return
        self._left -= 1
        if self._left == 0:
            self.contacts = self.scene.contacts(core.d)
            self.seq += 1


def sim_line(state: dict, F_ext_max: float = 0.0) -> dict:
    """The bridge.jsonl ``kind: sim`` line of one I-SIMSTATE dict (plain JSON types)."""
    T = np.asarray(state["eraser_T_base"], float)
    q = R_to_quat(T[:3, :3])
    return {"t": float(state["t_mono"]), "kind": "sim", "obj": [round(float(v), 6) for v in (*T[:3, 3], *q)],
            "gap": round(float(state["hand_gap_m"]), 6), "grasped": bool(state["grasped"]),
            "hand_mode": str(state.get("hand_mode")), "contacts": [dict(c) for c in state["contacts"]],
            "F_ext_max": round(float(F_ext_max), 4), "reflex": None if state["reflex"] is None else str(state["reflex"]),
            **({"tcp_z": round(float(state["tcp_z"]), 6)} if state.get("tcp_z") is not None else {})}
