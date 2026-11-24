"""Geometry of the IMX519 (main) view, handheld vs Franka mount, for camera study B step corr_main (CPU only).

    from sim_deploy.corrections import geometry_main as G
    geo = G.MainGeometry.load()                     # rig file, model scene.xml (no rendering), finger kit
    cams = geo.cameras()                            # {name: tcp_T_cam (4x4, OpenCV axes)}
    pts = geo.points(O_T_tcp, width_m, eraser)      # {class: (n, 3) points in the TCP frame}
    uv640, uv224, ok = geo.project(cams["handheld"], pts["tags"])

Used by ``fit_main.py`` (run that). Conventions:

* TCP frame = training's fingertip TCP (x across the fingers, y away from the cameras, z toward the fingertips); in
  the sim it is the model's site ``tcp`` (flange x frames.yaml F_T_TCP), the same physical point (E3: kit tip
  hand z 235.65 mm, model.json F_T_TCP_vs_kit 0.002 mm).
* Cameras: OpenCV axes. ``handheld_nominal`` = training's rig chain (handheld_check.camera_poses: T_cam_tcp for the
  IMX219, IMX519 via both T_imu_cam); ``handheld`` = the same + E7's fix, both cameras rigidly translated by
  (-4.0, +2.5, +8.8) mm in IMX219 axes (DESIGN.md E7 open item 4); ``handheld_e2rel`` = the fixed IMX219 with E2's CAD
  IMX219->IMX519 relation instead of training's T_imu_cam (DESIGN.md E5 finding 3); ``mount`` = the model's
  ``cam_imx519`` (E2's CAD on the Franka mount); ``mount_trrel`` = the model's ``cam_imx219`` with training's
  IMX219->IMX519 relation (cross-check).
* Main camera model: the copy's ``physical_model(rig, 'main')`` (PinholeRadtan, K 492.5 @ 640x480, radtan -0.02, 0.01)
  -> 640x480 pixels (OpenCV centres) -> ``letterbox_point`` -> 224x224 (MainPrep).
* Points (TCP frame): fingers = front-facing face centres (to the handheld IMX519) of the kit's soft fingers and
  holders (iphumi_umi_soft_v1, both sides slid so the tags are E7's tag distance width + 44.29 mm apart, i.e.
  gap = width), voxel-thinned to 4 mm; tags = the 16 mm marker corners + centre (E7's real sticker positions); the
  mount's fingers/tags are the same points plus the model's per-side offset (the kit's 0.28 mm asymmetry, measured on
  the compiled pads); table = a 40 px grid of the handheld 640 image cast onto the table plane (z 0 of the package's
  tag frame = O z 0); eraser = top-face grid + 4 top corners of the ERASER_M (127 x 30 x 12 mm) box centred half its height
  behind the grasp tip on the grasp TCP's approach axis, resting on the table, its width along the closing axis
  (scene/placement.py replay rule, applied to the recorded grasp pose itself). Table/eraser points hidden behind the
  fingers (inside a finger's projected hull in either view) are dropped; no other occlusion test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sim_deploy.sim_pi.render as _R  # noqa: F401  (EGL vendor before mujoco; no GL context is made here)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_deploy import eraser as eraser_mod  # noqa: E402
from sim_deploy.sim_pi import _deploy as D  # noqa: E402
from sim_deploy.sim_pi import handheld_check as H  # noqa: E402

REPO = D.REPO
SCENE_XML = REPO / "sim_deploy" / "build" / "sim_rig" / "scene.xml"
E7_DT_M = np.array([-4.0, 2.5, 8.8]) * 1e-3          # E7's camera fix, IMX219 axes
IN_RES = (640, 480)
ERASER_M = np.array(eraser_mod.size_m(0.127, 0.012))   # 127 x sim width (sim_deploy/eraser.py, 30 mm) x 12 mm
CLASSES = ("fingers", "tags", "table", "eraser")


def _T(R=None, t=None) -> np.ndarray:
    return H._T(R, t)


def _inv(T: np.ndarray) -> np.ndarray:
    return H._inv(T)


def _apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra.T @ Rb
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def _voxel(P: np.ndarray, size: float) -> np.ndarray:
    if len(P) == 0:
        return P
    k = np.floor(P / size).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return P[np.sort(idx)]


@dataclass
class MainGeometry:
    rig: object
    prep: object
    model_main: object
    kit: object
    mount: dict           # {"tcp_T_imx519", "tcp_T_imx219", "side_offset": {+1/-1: (3,)}}

    @classmethod
    def load(cls, scene_xml: Path = SCENE_XML) -> "MainGeometry":
        from franka_sim import finger_kit as FK
        rig = D.load_live_rig(D.DEFAULT_RIG_FILE)
        prep = D.ImagePrep(rig)
        kit = FK.load_kit(H.KIT)
        mount = mount_cameras(scene_xml, kit)
        return cls(rig, prep, D.physical_model(rig, "main"), kit, mount)

    # -------------------------------------------------------------------------------------------- cameras
    def cameras(self) -> dict[str, np.ndarray]:
        """tcp_T_cam (OpenCV) of the IMX519 for every variant (see the module docstring)."""
        nom = H.camera_poses(self.rig)                       # assembly frame
        Ai = _inv(_T(H.TCP_AXES, H.TCP_POS))
        uw_n, main_n = Ai @ nom["ultrawide"], Ai @ nom["main"]
        uw_T_main_tr = _inv(uw_n) @ main_n                   # training's IMX219 -> IMX519 (T_imu_cam chain)
        from sim_deploy.rig.add_rig import hand_T_cam, load_rig
        rd = load_rig(H.RIG_DIR)
        uw_T_main_e2 = _inv(hand_T_cam(rd, "cam_imx219")) @ hand_T_cam(rd, "cam_imx519")
        uw_f = uw_n @ _T(None, E7_DT_M)
        return {
            "handheld": uw_f @ uw_T_main_tr,
            "handheld_nominal": main_n,
            "handheld_e2rel": uw_f @ uw_T_main_e2,
            "mount": self.mount["tcp_T_imx519"],
            "mount_trrel": self.mount["tcp_T_imx219"] @ uw_T_main_tr,
            "_uw_handheld": uw_f, "_uw_handheld_nominal": uw_n, "_uw_mount": self.mount["tcp_T_imx219"],
            "_uw_T_main_training": uw_T_main_tr, "_uw_T_main_e2": uw_T_main_e2,
        }

    # -------------------------------------------------------------------------------------------- projection
    def project(self, tcp_T_cam: np.ndarray, P_tcp: np.ndarray):
        """(n, 3) TCP-frame points -> (uv640 (n, 2), uv224 (n, 2), ok (n,)) through the copy's main model."""
        Pc = _apply(_inv(tcp_T_cam), P_tcp)
        uv, ok = self.model_main.project(Pc)
        ok = ok & (uv[:, 0] >= -0.5) & (uv[:, 0] <= IN_RES[0] - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= IN_RES[1] - 0.5)
        uv224 = D.letterbox_point(np.nan_to_num(uv), IN_RES)
        return uv, uv224, ok

    # -------------------------------------------------------------------------------------------- points
    def finger_points(self, width_m: float, cull_cam: np.ndarray) -> dict:
        """{'fingers': (n,3), 'tags': (m,3), 'finger_side': (n,), 'tag_side': (m,), 'hull_parts': [...]} in the TCP
        frame (handheld placement: gap = width, E7 real sticker positions)."""
        Ai = _inv(_T(H.TCP_AXES, H.TCP_POS))
        dist = H.E7_TAG_DIST(width_m)
        cam_pos_asm = _apply(_inv(Ai), cull_cam[:3, 3])[0]
        fingers, fside, tags, tside, hulls = [], [], [], [], []
        for _, side in self.kit.sides.items():
            t0 = side.tags[0]
            native = t0.centre[0] + np.sign(t0.centre[0]) * H.TAG_OUTWARD_M
            dx = -np.sign(native) * (abs(native) - dist / 2)
            sgn = float(np.sign(native))
            for pn in side.parts:
                part = self.kit.parts[pn]
                v = part.vertices + [dx, 0.0, 0.0]
                f = part.faces
                a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
                n = np.cross(b - a, c - a)
                cen = (a + b + c) / 3
                front = np.einsum("ij,ij->i", n, cam_pos_asm - cen) > 0
                P = _voxel(cen[front], 0.004)
                fingers.append(_apply(Ai, P))
                fside.append(np.full(len(P), sgn))
                hulls.append((sgn, _apply(Ai, v)))
            for t in side.tags:
                cs = H.tag_corners_3d(t, dist, 0.016)
                P = np.vstack([cs, cs.mean(0)])
                tags.append(_apply(Ai, P))
                tside.append(np.full(len(P), sgn))
        return {"fingers": np.vstack(fingers), "finger_side": np.concatenate(fside), "tags": np.vstack(tags),
                "tag_side": np.concatenate(tside), "hulls": hulls}

    def mount_points(self, P: np.ndarray, side: np.ndarray) -> np.ndarray:
        """The same finger/tag points on the mount: + the model's per-side offset (TCP frame)."""
        off = np.array([self.mount["side_offset"][float(s)] for s in side]).reshape(-1, 3)
        return P + off

    def table_points(self, O_T_tcp: np.ndarray, cam: np.ndarray, step_px: int = 40, max_range_m: float = 1.5):
        """TCP-frame points on the table plane (O z 0) under a grid of ``cam``'s 640 image."""
        us = np.arange(step_px / 2, IN_RES[0], step_px)
        vs = np.arange(step_px / 2, IN_RES[1], step_px)
        uv = np.array([[u, v] for v in vs for u in us])
        rays, ok = self.model_main.unproject(uv)
        O_T_cam = O_T_tcp @ cam
        d = rays @ O_T_cam[:3, :3].T
        o = O_T_cam[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -o[2] / d[:, 2]
        good = ok & (s > 0) & (s * np.linalg.norm(d, axis=1) < max_range_m)
        P_O = o + s[good, None] * d[good]
        return _apply(_inv(O_T_tcp), P_O)

    @staticmethod
    def eraser_pose(O_T_grasp: np.ndarray) -> np.ndarray:
        """O_T_eraser (z up, x = its length, y = its width along the closing axis), resting on the table."""
        z_t = O_T_grasp[:3, 2]
        c = O_T_grasp[:3, 3] - ERASER_M[2] / 2 * z_t
        x_t = O_T_grasp[:3, 0]
        yw = np.array([x_t[0], x_t[1], 0.0])
        yw /= np.linalg.norm(yw)
        zw = np.array([0.0, 0.0, 1.0])
        xw = np.cross(yw, zw)
        return _T(np.column_stack([xw, yw, zw]), [c[0], c[1], ERASER_M[2] / 2])

    @staticmethod
    def eraser_points(O_T_e: np.ndarray, O_T_tcp: np.ndarray) -> np.ndarray:
        h = ERASER_M / 2
        top = [[x, y, h[2]] for x in np.linspace(-h[0], h[0], 9) for y in np.linspace(-h[1], h[1], 3)]
        return _apply(_inv(O_T_tcp) @ O_T_e, np.array(top))


def mount_cameras(scene_xml: Path, kit) -> dict:
    """The model's cam_imx519 / cam_imx219 in its tcp site frame (OpenCV axes), and the per-side offset of the
    model's fingers vs the handheld placement (compiled pad vertices, nearest neighbours, TCP frame)."""
    from scipy.spatial import cKDTree
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    gap = 0.06
    for j in ("finger_joint1", "finger_joint2"):
        d.qpos[m.joint(j).qposadr[0]] = gap / 2
    mujoco.mj_forward(m, d)

    def pose(xpos, xmat, cv=False):
        T = np.eye(4)
        T[:3, :3] = xmat.reshape(3, 3) @ (np.diag([1.0, -1.0, -1.0]) if cv else np.eye(3))
        T[:3, 3] = xpos
        return T

    s = m.site("tcp").id
    Oti = _inv(pose(d.site_xpos[s], d.site_xmat[s]))
    cams = {c: Oti @ pose(d.cam_xpos[m.camera(c).id], d.cam_xmat[m.camera(c).id], cv=True)
            for c in ("cam_imx519", "cam_imx219")}
    # handheld pads at the same gap (gap = width)
    Ai = _inv(_T(H.TCP_AXES, H.TCP_POS))
    dist = H.E7_TAG_DIST(gap)
    offs = {}
    for _, side in kit.sides.items():
        t0 = side.tags[0]
        native = t0.centre[0] + np.sign(t0.centre[0]) * H.TAG_OUTWARD_M
        dx = -np.sign(native) * (abs(native) - dist / 2)
        vh = _apply(Ai, kit.parts[side.contact_part].vertices + [dx, 0, 0])
        sgn = float(np.sign(native))
        gname = "pad_l" if sgn > 0 else "pad_r"
        g = m.geom(gname).id
        if not np.sign(_apply(Oti, d.geom_xpos[g])[0, 0]) == sgn:
            gname = "pad_r" if gname == "pad_l" else "pad_l"
            g = m.geom(gname).id
        mid = m.geom_dataid[g]
        w = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        wt = _apply(Oti @ pose(d.geom_xpos[g], d.geom_xmat[g]), w)
        _, i = cKDTree(vh).query(wt)
        off = np.median(wt - vh[i], axis=0)
        resid, _ = cKDTree(wt).query(vh + off)
        offs[sgn] = off
        offs[f"{sgn:+.0f}_resid_max_mm"] = float(resid.max() * 1e3)
    return {"tcp_T_imx519": cams["cam_imx519"], "tcp_T_imx219": cams["cam_imx219"], "side_offset": offs}


def hull_mask(uv224_parts: list[np.ndarray], uv: np.ndarray) -> np.ndarray:
    """True where uv (n, 2) lies inside any of the convex hulls of the projected parts."""
    inside = np.zeros(len(uv), bool)
    for q in uv224_parts:
        q = q[np.isfinite(q).all(1)]
        if len(q) < 3:
            continue
        hull = cv2.convexHull(q.astype(np.float32))
        for k, p in enumerate(uv):
            if not inside[k] and cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0:
                inside[k] = True
    return inside
