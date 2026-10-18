"""Fast kinematics on the A1 model: FK, Jacobian, damped-least-squares IK, collision/distance queries, trajectories.

Why: Stage 1 (A3) and the Stage 2a evaluator (B8) must decide, thousands of times, whether a recorded TCP trajectory is
reachable by a given arm at a given base placement, with what joint margin and without self/table/tooling collisions.
That needs IK at about a millisecond per warm-started frame, so ``Kin`` keeps its own ``MjData`` (never a shared one)
and uses only ``mj_kinematics`` + ``mj_comPos`` (+ ``mj_geomDistance`` for contact queries).

World = robot base frame O (z up, base at the origin). ``frame`` is a model site: ``flange`` (libfranka F, link 7 +
0.107 z), ``ee`` (F.F_T_NE.NE_T_EE) or ``tcp`` (training TCP). FK was verified against the Franka modified-DH (Craig)
table on both Panda and FR3 with no frame offset (agreement ~1e-15 m): the MJCF joint axes, link frames and the
``flange`` site coincide with the DH ones, and the arm base sits at the world origin.

Collision groups (``Kin.contacts`` / ``min_distance``), pairs filtered to non-adjacent, non-excluded, collidable geoms:
  self     arm/hand geoms vs each other (parent-child bodies and the hand's own fingers/pads excluded)
  table    arm links 1-7, hand, finger pads and tooling vs the table (link 0 stands on it: excluded)
  tooling  housing/cradle vs links 0-6, link 6 included although link 7 is its child (link 7 and the hand are rigid
           with the tooling, so they are skipped); only the self group applies the parent-child adjacency rule
  object   (opt-in) non-pad geoms of arm/hand/tooling vs the eraser
Excluded by design: finger pads vs the eraser (that is the grasp) and the eraser vs the table.

Usage: kin = Kin(sm, "ee"); T = kin.fk(q); q, info = kin.ik(T_goal, q0); out = check_traj(sm, T_seq, q_seed)
"""

from __future__ import annotations

import numpy as np

import mujoco

DEFAULT_GROUPS = ("self", "table", "tooling")


def rot_err(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """World-frame rotation vector w with ``R_des = exp(w) R_cur`` (axis-angle, |w| in [0, pi])."""
    dR = R_des @ R_cur.T
    v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) * 0.5
    s = np.linalg.norm(v)
    c = (np.trace(dR) - 1.0) * 0.5
    if s < 1e-9:
        if c > 0:
            return v
        # angle ~ pi: axis from the symmetric part
        A = (dR + dR.T) * 0.5 - c * np.eye(3)
        k = int(np.argmax(np.diag(A)))
        ax = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        return np.pi * ax / max(np.linalg.norm(ax), 1e-12)
    return v / s * np.arctan2(s, c)


class Kin:
    def __init__(self, sm, frame: str = "flange"):
        self.sm = sm
        self.m = sm.model
        self.d = mujoco.MjData(self.m)
        m, nm = self.m, sm.names
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, nm["key_home"])
        mujoco.mj_resetDataKeyframe(m, self.d, key)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm["joints"][f"j{i}"]) for i in range(1, 8)]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self._dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm["sites"].get(frame, frame))
        if self.site < 0:
            raise KeyError(f"no site '{frame}' (have {sorted(nm['sites'])})")
        self.frame = frame
        lim = sm.limits
        self.q_lo = np.maximum(lim["q_min_rad"], lim["mjcf_range_rad"][:, 0])
        self.q_hi = np.minimum(lim["q_max_rad"], lim["mjcf_range_rad"][:, 1])
        self.q_home = self.d.qpos[self._qadr].copy()
        self._rng = np.random.default_rng(0)
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self._groups, self._pairs = self._build_groups()

    # ------------------------------------------------------------------------------------------ FK / Jacobian

    def _set(self, q) -> None:
        self.d.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def fk(self, q) -> np.ndarray:
        """4x4 world pose of the frame site."""
        self._set(q)
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.site].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.site]
        return T

    def _jac_now(self) -> np.ndarray:
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self._dadr], self._jr[:, self._dadr]])

    def jac(self, q) -> np.ndarray:
        """6x7 geometric Jacobian [v; w] of the frame site, expressed in the world frame."""
        self._set(q)
        return self._jac_now()

    def manipulability(self, q) -> float:
        """Yoshikawa measure sqrt(det(J J^T))."""
        J = self.jac(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def joint_margin(self, q) -> np.ndarray:
        """Per-joint distance (rad) to the nearest limit (station limits intersected with the MJCF range)."""
        q = np.asarray(q, float)
        return np.minimum(q - self.q_lo, self.q_hi - q)

    # ------------------------------------------------------------------------------------------------- IK

    def ik(self, T, q0, *, q_rest=None, tol_p: float = 1e-4, tol_r: float = 1e-3, iters: int = 100,
           damping: float = 1e-2, limit_margin: float = 0.02, w_null: float = 0.1,
           restarts: int = 24) -> tuple[np.ndarray, dict]:
        """Damped-least-squares (Levenberg-Marquardt) IK to the world pose ``T`` of the frame site, warm-started at q0.

        Step ``dq = J^T (J J^T + damping I)^-1 e`` plus a posture pull ``w_null * (I - J+ J) (q_rest - q)`` (capped
        at the task step size so it cannot stop convergence); the damping adapts (x5 on a rejected step, x0.3 on
        success, never below ``1e-3 * damping``). ``q`` stays inside ``[lo + limit_margin, hi - limit_margin]``;
        joints pressing on a bound against the error are frozen. If a run stalls (joint limit / local minimum), up
        to ``restarts`` further runs start from random in-limit configurations (deterministic rng); a warm start that
        converges never pays for this.
        Returns ``(q, info)``, info = ok, err_p (m), err_r (rad), iters (last run), margin_min (rad), restarts_used.
        Never raises on unreachable/singular targets: ``ok`` is False and q is the best effort.
        """
        kw = dict(q_rest=q_rest, tol_p=tol_p, tol_r=tol_r, iters=iters, damping=damping, limit_margin=limit_margin,
                  w_null=w_null)
        score = lambda i: i["err_p"] / tol_p + i["err_r"] / tol_r  # noqa: E731
        q, info = self._ik_once(T, q0, **kw)
        used = 0
        while not info["ok"] and used < restarts:
            used += 1
            if score(info) < 100:      # slow convergence near a singularity rather than a local minimum: keep going
                q, info = self._ik_once(T, q, **{**kw, "iters": 2 * iters})
                if info["ok"]:
                    break
            q2, info2 = self._ik_once(T, self._rng.uniform(self.q_lo + limit_margin, self.q_hi - limit_margin), **kw)
            if info2["ok"] or score(info2) < score(info):
                q, info = q2, info2
        return q, {**info, "restarts_used": used}

    def _ik_once(self, T, q0, *, q_rest, tol_p, tol_r, iters, damping, limit_margin, w_null):
        T = np.asarray(T, float)
        lo, hi = self.q_lo + limit_margin, self.q_hi - limit_margin
        rest = self.q_home if q_rest is None else np.asarray(q_rest, float)
        q = np.clip(np.asarray(q0, float), lo, hi)
        Rd, pd = T[:3, :3], T[:3, 3]
        eye6, eye7 = np.eye(6), np.eye(7)
        lam_min, lam = 1e-3 * damping, damping

        def err(qq):
            self._set(qq)
            return np.r_[pd - self.d.site_xpos[self.site], rot_err(self.d.site_xmat[self.site].reshape(3, 3), Rd)]

        e = err(q)
        n = 0
        while n < iters:
            ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if not (np.isfinite(ep) and np.isfinite(er)):
                return q, {"ok": False, "err_p": np.inf, "err_r": np.inf, "iters": n, "margin_min": -np.inf}
            if ep <= tol_p and er <= tol_r:
                break
            n += 1
            J = self._jac_now()                                        # kinematics are current for q
            g = J.T @ e                                                # direction the task error pushes each joint
            free = ~(((q <= lo + 1e-12) & (g < 0)) | ((q >= hi - 1e-12) & (g > 0)))
            Jf = J * free
            cost = e @ e
            while n <= iters:                                          # LM: raise damping until the step helps
                Jp = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam * eye6, eye6)      # damped pseudo-inverse (7x6)
                dq = Jp @ e
                if w_null > 0:                                         # posture pull, never larger than the task step
                    dn = w_null * (eye7 - Jp @ Jf) @ (rest - q) * free
                    nn = np.linalg.norm(dn)
                    if nn > 0:
                        dq = dq + dn * min(1.0, np.linalg.norm(dq) / nn)
                big = np.abs(dq).max()
                if big > 0.3:
                    dq *= 0.3 / big
                qn = np.clip(q + dq, lo, hi)
                en = err(qn)
                if en @ en < cost:
                    q, e, lam = qn, en, max(lam * 0.3, lam_min)
                    break
                lam *= 5.0
                n += 1
                if lam > 1e4:                                          # stalled (limit/singularity): give up
                    n = iters + 1
            else:
                self._set(q)
                break
        ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        ok = bool(ep <= tol_p and er <= tol_r)
        return q, {"ok": ok, "err_p": ep, "err_r": er, "iters": n, "margin_min": float(self.joint_margin(q).min())}

    def ik_multistart(self, T, seeds, **kw) -> tuple[np.ndarray, dict]:
        """IK from each seed; the best is the ``ok`` solution with the largest minimum joint margin (else the lowest
        error). ``info`` is that run's info plus ``seed`` (index) and ``n_ok``."""
        kw.setdefault("restarts", 0)
        runs = [(*self.ik(T, s, **kw), i) for i, s in enumerate(np.atleast_2d(np.asarray(seeds, float)))]
        oks = [r for r in runs if r[1]["ok"]]
        if oks:
            q, info, i = max(oks, key=lambda r: r[1]["margin_min"])
        else:
            tp, tr = kw.get("tol_p", 1e-4), kw.get("tol_r", 1e-3)
            q, info, i = min(runs, key=lambda r: r[1]["err_p"] / tp + r[1]["err_r"] / tr)
        return q, {**info, "seed": i, "n_ok": len(oks)}

    # ---------------------------------------------------------------------------------- collisions / distances

    def _build_groups(self):
        m, nm = self.m, self.sm.names
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        geom = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        collidable = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])  # noqa: E731

        def subtree(b: int) -> set[int]:
            out = {b}
            for i in range(m.nbody):
                if i not in out and m.body_parentid[i] in out and i > b:
                    out.add(i)
            return out

        def geoms_of(bodies) -> list[int]:
            return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and collidable(g)]

        link7 = body(nm["bodies"]["link7"])
        chain = [link7]
        while m.body_parentid[chain[-1]] > 0:
            chain.append(int(m.body_parentid[chain[-1]]))
        chain = chain[::-1]                                  # [link0 (base), link1, ..., link7]
        links = {f"link{i}": geoms_of({b}) for i, b in enumerate(chain)}
        hand_b = subtree(body(nm["bodies"]["hand"]))
        tool_g = [geom(g) for g in nm["geoms"]["tooling"]]
        g = {**links, "hand": geoms_of(hand_b), "pads": [geom("pad_l"), geom("pad_r")], "tooling": tool_g,
             "arm": [x for i in range(1, 8) for x in links[f"link{i}"]]}
        has_scene = "table" in nm["geoms"]
        g["table"] = [geom(nm["geoms"]["table"])] if has_scene else []
        g["eraser"] = [geom(nm["geoms"]["object"])] if has_scene else []
        excl = {int(s) for s in m.exclude_signature[:m.nexclude]}

        def rigid_root(b: int) -> int:        # bodies without joints move with their parent (attachment, tool, ...)
            while b > 1 and m.body_jntnum[b] == 0 and m.body_parentid[b] > 0:
                b = int(m.body_parentid[b])
            return b

        def skip(a: int, b: int, adjacent: bool = False) -> bool:
            """Same rigid body or an MJCF exclude; with ``adjacent`` also parent-child rigid bodies (self group only:
            tooling/table/explicit queries must see e.g. housing vs link6, although link7 is link6's child)."""
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            ra, rb = rigid_root(ba), rigid_root(bb)
            if ra == rb or (adjacent and (m.body_parentid[ra] == rb or m.body_parentid[rb] == ra)):
                return True
            lo_, hi_ = min(ba, bb), max(ba, bb)
            return (lo_ << 16) + hi_ in excl

        self._skip = skip

        def pairs(A, B, *, same=False):
            out = []
            for i, a in enumerate(A):
                for b in (B[i + 1:] if same else B):
                    if a != b and not skip(a, b, adjacent=same):
                        out.append((a, b))
            return out

        arm_hand = g["arm"] + g["hand"]
        self_pairs = [(a, b) for a, b in pairs(arm_hand, arm_hand, same=True)
                      if not (m.geom_bodyid[a] in hand_b and m.geom_bodyid[b] in hand_b)]
        movers = arm_hand + g["tooling"]
        pr = {"self": self_pairs,
              "table": pairs(movers, g["table"]),
              "tooling": [p for i in range(0, 7) for p in pairs(g["tooling"], links[f"link{i}"])],
              "object": pairs([x for x in movers if x not in g["pads"]], g["eraser"])}
        return g, pr

    @property
    def groups(self) -> dict[str, list[int]]:
        """Named geom-id groups: link0-7, arm (links 1-7), hand (+ fingers + pads), pads, tooling, table, eraser."""
        return self._groups

    def _gname(self, g: int) -> str:
        return self.m.geom(g).name or f"{self.m.body(int(self.m.geom_bodyid[g])).name}:geom{g}"

    def _dist(self, a: int, b: int, dmax: float, ft=np.zeros(6)) -> float:
        return mujoco.mj_geomDistance(self.m, self.d, a, b, dmax, ft)

    def contacts(self, q, *, margin: float = 0.0, groups=DEFAULT_GROUPS) -> list[dict]:
        """Geom pairs closer than ``margin`` (m; negative distance = penetration) in the given collision groups.
        Each item: {a, b (geom names), group, dist}. Sorted by distance."""
        self._set(q)
        out = []
        dmax = max(margin, 0.0) + 1e-3
        for grp in groups:
            for a, b in self._pairs[grp]:
                dist = self._dist(a, b, dmax)
                if dist < margin:
                    out.append({"a": self._gname(a), "b": self._gname(b), "group": grp, "dist": float(dist)})
        return sorted(out, key=lambda c: c["dist"])

    def n_contacts(self, q, *, margin: float = 0.0, groups=DEFAULT_GROUPS) -> int:
        """Fast count version of :meth:`contacts` (no dicts); same semantics."""
        self._set(q)
        dmax = max(margin, 0.0) + 1e-3
        return sum(1 for grp in groups for a, b in self._pairs[grp] if self._dist(a, b, dmax) < margin)

    def min_distance(self, q, pairs, *, dmax: float = 1.0) -> float:
        """Smallest signed distance (m) between named geom groups (see :attr:`groups`), e.g.
        ``[("tooling", "link5"), ("tooling", "link6")]``. Entries may also be single geom names. ``dmax`` caps the
        search (returned when everything is farther). Same-body, adjacent and excluded geom pairs are skipped."""
        self._set(q)
        best = dmax

        def ids(x):
            if x in self._groups:
                return self._groups[x]
            gi = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, x)
            if gi < 0:
                raise KeyError(f"unknown geom group or geom '{x}' (groups: {sorted(self._groups)})")
            return [gi]

        for ga, gb in pairs:
            for a in ids(ga):
                for b in ids(gb):
                    if a != b and not self._skip(a, b):
                        best = min(best, self._dist(a, b, dmax))
        return float(best)


def check_traj(sm, T_seq, q_seed, frame: str = "ee", *, tol_p: float = 1e-3, tol_r: float = np.deg2rad(0.5),
               collision: bool = True) -> dict:
    """Track a pose sequence (N, 4, 4; world = base frame) with warm-started IK.

    Each frame starts from the previous reachable solution (``q_seed`` for the first); a failed frame is retried
    from a few extra seeds (previous, ``q_seed``, home, two fixed random). Returns arrays over frames:
    reachable (bool), q (N,7; best effort when unreachable), err_p, err_r, joint_margin_rad (N,7), joint_margin_deg
    (N,7), collisions (count of self/table/tooling contacts at q; best effort too), manip.
    """
    kin = Kin(sm, frame)
    T_seq = np.asarray(T_seq, float).reshape(-1, 4, 4)
    N = len(T_seq)
    rng = np.random.default_rng(0)
    extra = rng.uniform(kin.q_lo, kin.q_hi, size=(2, 7))
    out = {"reachable": np.zeros(N, bool), "q": np.zeros((N, 7)), "err_p": np.zeros(N), "err_r": np.zeros(N),
           "joint_margin_rad": np.zeros((N, 7)), "collisions": np.zeros(N, int), "manip": np.zeros(N)}
    q_prev = np.asarray(q_seed, float)
    for i, T in enumerate(T_seq):
        q, info = kin.ik(T, q_prev, tol_p=tol_p, tol_r=tol_r, restarts=0)
        if not info["ok"]:
            seeds = np.vstack([q_prev, q_seed, kin.q_home, extra])
            q, info = kin.ik_multistart(T, seeds, tol_p=tol_p, tol_r=tol_r)
        out["reachable"][i] = info["ok"]
        out["q"][i], out["err_p"][i], out["err_r"][i] = q, info["err_p"], info["err_r"]
        out["joint_margin_rad"][i] = kin.joint_margin(q)
        out["manip"][i] = kin.manipulability(q)
        if collision:
            out["collisions"][i] = kin.n_contacts(q)
        if info["ok"]:
            q_prev = q
    out["joint_margin_deg"] = np.rad2deg(out["joint_margin_rad"])
    return out
