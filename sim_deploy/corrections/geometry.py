"""Camera study B (sim only): handheld vs Franka-mount camera geometry relative to the fingertip TCP, the 3D point
sets the policy's cameras see, and their projection into the policy's 224x224 images. Pure math on CPU: MuJoCo is
used only to read the sim model's kinematics (``mj_kinematics``, no rendering, no GL).

    ~/envs/sim_deploy/bin/python -m sim_deploy.corrections.geometry      # prints the camera transforms (smoke)

API (all frames OpenCV for cameras, metres, 4x4 ``T_a_b`` maps points in b to a):

* ``camera_poses_tcp(kind) -> {"ultrawide": T_tcp_cam, "main": T_tcp_cam}``; ``kind``:
  ``"handheld"`` = training's ``T_cam_tcp`` (rig file) + E7's camera fix (-4.0, +2.5, +8.8) mm in IMX219 axes, the
  IMX519 rigidly with it (``T_imu_cam`` chain, as ``sim_pi/handheld_check.HandheldScene``);
  ``"handheld_nominal"`` = training's ``T_cam_tcp`` as it is; ``"mount"`` = the sim model (``build/sim_rig/scene.xml``:
  cameras ``cam_imx219`` / ``cam_imx519``, ``R_cv = R_mj diag(1,-1,-1)``, relative to site ``tcp``).
* ``finger_points(kind, width_m) -> {"tags": (8,3), "fingertips": (N,3), "holders": (M,3), "tips": (2,3)}`` in the TCP
  frame at fingertip gap = width (D8 revised). ``kind`` "handheld*" = the kit ``iphumi_umi_soft_v1`` with E7's real
  sticker positions (HandheldScene); "mount" = the sim model's fingers (finger joints at gap = width) and its tag
  geoms. The same asm-frame vertex subsets on both, so rows correspond one to one (tags: ids 0 then 1, ArUco corner
  order top-left, top-right, bottom-right, bottom-left, marker top = toward the fingertips).
* ``load_refs()`` -> the 15 reference frames of ``sim_evals/_camera_study/B/layout.json`` (TCP pose in the tag/table
  frame, width, grasp pose); ``scene_points(ref)`` -> table grid (near/mid/far) + eraser in the TCP frame;
  ``point_sets(ref, kind)`` = finger points + scene points: ``{class: (N,3)}``.
* ``project(role, T_tcp_cam, P_tcp, rig, R_phys_virt=None, K_virt=None) -> (uv224 (N,2), valid (N,))``: ultrawide =
  physical fisheye coverage check, then the copy's ``UltrawidePrep`` virtual pinhole (``d_phys = R_phys_virt d_virt``,
  ``K_virt`` = [fx, fy, cx, cy], defaults = the rig's) and ``letterbox_point``; main = the rig's pinhole-radtan and
  the letterbox. ``valid`` = in front, inside the physical image (and the model's range), inside the virtual image.
  ``project_native`` gives the physical 640x480 pixels (for B5's IMX519 crop/zoom).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim_deploy import eraser as eraser_mod
from sim_deploy.sim_pi import _deploy as D

REPO = D.REPO
MODEL_XML = REPO / "sim_deploy" / "build" / "sim_rig" / "scene.xml"
LAYOUT = REPO / "sim_evals" / "_camera_study" / "B" / "layout.json"
KIT_HANDHELD = REPO / "controlSW" / "franka_sim" / "franka_sim" / "assets" / "finger_kits" / "iphumi_umi_soft_v1"
DEFAULT_RIG_FILE = D.DEFAULT_RIG_FILE

# Training's TCP in the kit's assembly frame (umi-processing rig_geometry_from_cad.py; = handheld_check TCP_POS/AXES)
TCP_POS = np.array([0.0, 0.20135, -0.0022])
TCP_AXES = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)
E7_CAMERA_DT_M = np.array([-0.0040, 0.0025, 0.0088])      # DESIGN.md E7 open item 4 (IMX219 axes)
TAG_OUTWARD_M = 0.001855                                   # E7: real stickers outward of the CAD outline, per side
D8_TAG_OFFSET_M = 0.040577                                 # CAD tag-centre distance minus fingertip gap (kit)
MARKER_M = 0.016                                           # ArUco marker side (inside the 20 mm sticker)
HANDHELD_TAG_LIFT_M = 0.0002                               # HandheldScene's tag plane above the holder surface
ERASER_M = tuple(eraser_mod.size_m(0.127, 0.012))   # 127 x sim width (sim_deploy/eraser.py, 30 mm) x 12 mm
CLASSES = ("tags", "fingertips", "holders", "table_near", "table_mid", "table_far", "eraser")
FINGER_CLASSES = ("tags", "fingertips", "holders")
TABLE_CLASSES = ("table_near", "table_mid", "table_far")
ROLES = ("ultrawide", "main")


def T_(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv(T: np.ndarray) -> np.ndarray:
    return T_(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def apply(T: np.ndarray, P) -> np.ndarray:
    P = np.asarray(P, float).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def pose_diff(T_a: np.ndarray, T_b: np.ndarray) -> dict:
    """Pose b relative to a, both in one frame: translation (mm, in that frame and in a's axes), rotation."""
    dT = inv(T_a) @ T_b
    rv = Rotation.from_matrix(dT[:3, :3]).as_rotvec()
    return {"dt_frame_mm": ((T_b[:3, 3] - T_a[:3, 3]) * 1e3).tolist(), "dt_in_a_axes_mm": (dT[:3, 3] * 1e3).tolist(),
            "dist_mm": float(np.linalg.norm(dT[:3, 3]) * 1e3), "rot_deg": float(np.degrees(np.linalg.norm(rv))),
            "rotvec_deg_in_a_axes": np.degrees(rv).tolist(),
            "euler_xyz_deg_in_a_axes": Rotation.from_matrix(dT[:3, :3]).as_euler("xyz", degrees=True).tolist()}


@lru_cache(maxsize=4)
def load_rig(path: str | None = None):
    """The copy's ``load_live_rig`` (default: the 640 rig the laptop uses)."""
    return D.load_live_rig(Path(path) if path else DEFAULT_RIG_FILE)


# ----------------------------------------------------------------------------------------------- the sim model

class _Model:
    """scene.xml compiled on CPU (read-only) + the spec's local geom placements (meshes in the kit's asm frame)."""

    def __init__(self, xml: Path = MODEL_XML):
        import mujoco
        self.mj = mujoco
        self.spec = mujoco.MjSpec.from_file(str(xml))
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        self.q_finger = [self.m.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)              # camera poses are not part of mj_kinematics

    def geom_local(self, name: str) -> np.ndarray:
        """body_T_geom of a geom as written in scene.xml (for mesh geoms: body_T_meshfile)."""
        g = self.spec.geom(name)
        R = np.zeros(9)
        self.mj.mju_quat2Mat(R, np.asarray(g.quat, float))
        return T_(R.reshape(3, 3), np.asarray(g.pos, float))

    def body_world(self, name: str) -> np.ndarray:
        b = self.d.body(name)
        return T_(b.xmat.reshape(3, 3).copy(), b.xpos.copy())

    def site_world(self, name: str) -> np.ndarray:
        s = self.d.site(name)
        return T_(s.xmat.reshape(3, 3).copy(), s.xpos.copy())

    def cam_world_cv(self, name: str) -> np.ndarray:
        c = self.d.cam(name)
        return T_(c.xmat.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), c.xpos.copy())

    def set_fingers(self, q1: float, q2: float) -> None:
        self.d.qpos[self.q_finger[0]] = q1
        self.d.qpos[self.q_finger[1]] = q2
        self.mj.mj_kinematics(self.m, self.d)
        self.mj.mj_camlight(self.m, self.d)


@lru_cache(maxsize=1)
def model() -> _Model:
    return _Model()


@lru_cache(maxsize=1)
def kit():
    from franka_sim import finger_kit as FK
    return FK.load_kit(KIT_HANDHELD)


# ----------------------------------------------------------------------------------------------- camera poses

def T_tcp_asm() -> np.ndarray:
    """Kit assembly frame -> training TCP frame (handheld)."""
    return inv(T_(TCP_AXES, TCP_POS))


def camera_poses_tcp(kind: str = "handheld", rig=None) -> dict[str, np.ndarray]:
    """{"ultrawide", "main"}: camera pose (OpenCV axes) in the fingertip TCP frame, T_tcp_cam."""
    if kind in ("handheld", "handheld_nominal"):
        rig = rig or load_rig()
        uw = inv(np.asarray(rig.T_cam_tcp, float))
        uw_T_main = inv(np.asarray(rig.cameras["ultrawide"].T_imu_cam, float)) @ np.asarray(
            rig.cameras["main"].T_imu_cam, float)
        if kind == "handheld":
            uw = uw @ T_(None, E7_CAMERA_DT_M)
        return {"ultrawide": uw, "main": uw @ uw_T_main}
    if kind == "mount":
        M = model()
        O_tcp = M.site_world("tcp")
        return {"ultrawide": inv(O_tcp) @ M.cam_world_cv("cam_imx219"),
                "main": inv(O_tcp) @ M.cam_world_cv("cam_imx519")}
    raise ValueError(f"camera kind {kind!r}: handheld | handheld_nominal | mount")


def e2_formula_T_cam_tcp(role: str, z_tip_mm: float) -> np.ndarray:
    """DESIGN.md E2 'Camera -> fingertip point': T_cam_tcp of the mount from the hand-frame fingertip height."""
    if role == "ultrawide":
        t = (8.38, 139.41 - 0.2588 * z_tip_mm, 9.59 + 0.9659 * z_tip_mm)
        R = Rotation.from_euler("x", 15.0, degrees=True).as_matrix()
    else:
        t = (8.18, 158.54 - 0.5 * z_tip_mm, 55.82 + 0.8660 * z_tip_mm)
        R = Rotation.from_euler("x", 30.0, degrees=True).as_matrix()
    return T_(R, np.array(t) * 1e-3)


# ----------------------------------------------------------------------------------------------- finger points

def voxel_subset(P: np.ndarray, voxel_m: float) -> np.ndarray:
    """Indices of one vertex per voxel (the one nearest the voxel's centroid): a deterministic thinning."""
    key = np.floor(P / voxel_m).astype(np.int64)
    _, inv_idx = np.unique(key, axis=0, return_inverse=True)
    inv_idx = inv_idx.ravel()
    out = []
    for k in range(inv_idx.max() + 1):
        idx = np.flatnonzero(inv_idx == k)
        c = P[idx].mean(0)
        out.append(idx[np.argmin(np.linalg.norm(P[idx] - c, axis=1))])
    return np.sort(np.array(out))


@lru_cache(maxsize=1)
def _asm_subsets() -> dict:
    """Per asm part name: the vertex subset used (fingers 5 mm voxels, holders 8 mm), in asm coords."""
    K = kit()
    out = {}
    for name, part in K.parts.items():
        v = np.asarray(part.vertices, float)
        out[name] = v[voxel_subset(v, 0.005 if part.role == "contact" else 0.008)]
    return out


def _kit_tag_corners_asm(tag, centre_x: float, lift_m: float) -> np.ndarray:
    """ArUco corner order (TL, TR, BR, BL; marker top = tag up) of the 16 mm marker, asm coords."""
    n, up = np.asarray(tag.normal, float), np.asarray(tag.up, float)
    x = np.cross(up, n)
    c = np.asarray(tag.centre, float).copy()
    c[0] = centre_x
    c = c + lift_m * n
    h = MARKER_M / 2
    return np.array([c - h * x + h * up, c + h * x + h * up, c + h * x - h * up, c - h * x - h * up])


def _handheld_fingers(width_m: float) -> dict:
    """The kit in the asm frame with E7's sliding rule (HandheldScene.set_tag_distance at tag distance = width +
    44.29 mm), -> TCP frame."""
    K = kit()
    sub = _asm_subsets()
    dist = width_m + D8_TAG_OFFSET_M + 2 * TAG_OUTWARD_M
    T = T_tcp_asm()
    pts = {"fingertips": [], "holders": [], "tips": [], "tags": {}}
    for side_name in ("left", "right"):
        side = K.sides[side_name]
        tag = side.tags[0]
        native = float(tag.centre[0]) + np.sign(tag.centre[0]) * TAG_OUTWARD_M
        dx = -np.sign(native) * (abs(native) - dist / 2)
        sh = np.array([dx, 0.0, 0.0])
        for pn in side.parts:
            cls = "fingertips" if K.parts[pn].role == "contact" else "holders"
            pts[cls].append(apply(T, sub[pn] + sh))
        pts["tips"].append(apply(T, np.asarray(side.tip, float) + sh))
        pts["tags"][tag.id] = apply(T, _kit_tag_corners_asm(tag, native + dx, HANDHELD_TAG_LIFT_M))
    return {"tags": np.concatenate([pts["tags"][0], pts["tags"][1]]), "fingertips": np.concatenate(pts["fingertips"]),
            "holders": np.concatenate(pts["holders"]), "tips": np.concatenate(pts["tips"])}


# model body -> (kit side its meshes come from, geom names of contact / holder / adapter / tag)
_MODEL_SIDES = {"left_finger": ("right", "pad_l", "finger_holder_l", "finger_holder_l_1", "tag_l"),
                "right_finger": ("left", "pad_r", "finger_holder_r", "finger_holder_r_1", "tag_r")}


@lru_cache(maxsize=1)
def model_gap_fn():
    """Fingertip gap (m, between the two kit tip points along TCP x) as a function of the finger joints, and its
    inverse for symmetric joints. Returns (gap(q1, q2), q_of_gap(g)) - the gap is linear in q1 + q2."""
    def gap(q1, q2):
        P = _model_fingers_q(q1, q2)["tips"]
        return float(abs(P[1, 0] - P[0, 0]))
    g0, g1 = gap(0.0, 0.0), gap(0.02, 0.02)
    slope = (g1 - g0) / 0.04
    return gap, (lambda g: max(0.0, (g - g0) / slope / 2)), {"gap_at_q0_mm": g0 * 1e3, "dgap_d(q1+q2)": slope}


def _model_fingers_q(q1: float, q2: float) -> dict:
    M = model()
    M.set_fingers(q1, q2)
    T_tcp_O = inv(M.site_world("tcp"))
    K = kit()
    sub = _asm_subsets()
    out = {"fingertips": [], "holders": [], "tips": [], "tags": {}}
    for body in ("right_finger", "left_finger"):        # kit side left (tag 0, TCP -x) first, as the handheld
        kit_side, pad, holder, _adapter, tag_geom = _MODEL_SIDES[body]
        side = K.sides[kit_side]
        T_b = T_tcp_O @ M.body_world(body)
        T_pad = T_b @ M.geom_local(pad)                 # pad and holder share one placement (checked below)
        T_hold = T_b @ M.geom_local(holder)
        for pn in side.parts:
            if K.parts[pn].role == "contact":
                out["fingertips"].append(apply(T_pad, sub[pn]))
            else:
                out["holders"].append(apply(T_hold, sub[pn]))
        out["tips"].append(apply(T_pad, np.asarray(side.tip, float)))
        # the model's tag geom: textured slab, geom x = normal (0 .. 0.4 mm), y/z in the plane, 20 mm sticker
        T_tag = T_b @ M.geom_local(tag_geom)
        cam = camera_poses_tcp("mount")["ultrawide"][:3, 3]
        faces = [apply(T_tag, [[x, 0, 0]])[0] for x in (0.0, 0.0004)]
        xo = (0.0, 0.0004)[int(np.argmin([np.linalg.norm(f - cam) for f in faces]))]
        h = MARKER_M / 2
        corners = apply(T_tag, [[xo, sy * h, sz * h] for sy in (-1, 1) for sz in (-1, 1)])
        out["tags"][side.tags[0].id] = corners
    return {"tags": out["tags"], "fingertips": np.concatenate(out["fingertips"]),
            "holders": np.concatenate(out["holders"]), "tips": np.concatenate(out["tips"])}


def _order_like(corners: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Reorder 4 corners to the nearest of ``ref``'s 4 (the kit's ArUco order)."""
    from scipy.optimize import linear_sum_assignment
    C = np.linalg.norm(ref[:, None, :] - corners[None, :, :], axis=2)
    r, c = linear_sum_assignment(C)
    return corners[c[np.argsort(r)]]


def finger_points(kind: str, width_m: float) -> dict[str, np.ndarray]:
    """Tags (8 corners: id 0 then id 1), fingertip pads, holders and the two tip points, TCP frame, gap = width."""
    hand = _handheld_fingers(width_m)
    if kind.startswith("handheld"):
        return hand
    if kind != "mount":
        raise ValueError(kind)
    _, q_of_gap, _ = model_gap_fn()
    q = q_of_gap(width_m)
    mo = _model_fingers_q(q, q)
    tags = np.concatenate([_order_like(mo["tags"][i], hand["tags"][4 * i:4 * i + 4]) for i in (0, 1)])
    return {"tags": tags, "fingertips": mo["fingertips"], "holders": mo["holders"], "tips": mo["tips"]}


# ----------------------------------------------------------------------------------------------- references, scene

@lru_cache(maxsize=1)
def layout() -> dict:
    return json.loads(LAYOUT.read_text())


@lru_cache(maxsize=8)
def _episode(index: int) -> dict:
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode
    ep = load_episode(layout()["dataset"], index)
    ep["grasp"] = grasp_rows(ep["widths"])
    return ep


def load_refs() -> list[dict]:
    """The 15 reference frames: name ep<index>_<label>, T_tag_tcp (table/tag frame), width, grasp pose."""
    out = []
    for sel in layout()["selection"]:
        ep = _episode(int(sel["index"]))
        g = ep["grasp"]
        for row, lab in zip(sel["rows_in_episode"], sel["labels"]):
            out.append({"name": f"ep{int(sel['index']):02d}_{lab}", "index": int(sel["index"]), "row": int(row),
                        "label": lab, "T_tag_tcp": ep["poses"][row], "width_m": float(ep["widths"][row]),
                        "grasp_row": int(g["end"]), "T_tag_tcp_grasp": ep["poses"][int(g["end"])],
                        "episode_name": ep["name"]})
    return out


def eraser_box_tag(ref: dict, table_z_m: float = 0.0) -> np.ndarray:
    """Eraser (ERASER_M: 127 x 30 x 12 mm) flat on the table (tag z 0) at the grasp TCP xy, long axis perpendicular to the
    closing axis (the TCP x axis at the grasp, in the table plane): 8 corners + a 5x3 grid on the top face."""
    T = ref["T_tag_tcp_grasp"]
    c = np.array([T[0, 3], T[1, 3], table_z_m])
    a = T[:3, 0].copy()
    a[2] = 0.0
    a /= np.linalg.norm(a)                       # closing axis (short side, ERASER_M[1])
    b = np.array([-a[1], a[0], 0.0])             # long axis (127 mm)
    L, W, H = ERASER_M
    pts = [c + sl * L / 2 * b + sw * W / 2 * a + np.array([0, 0, z]) for z in (0.0, H) for sl in (-1, 1)
           for sw in (-1, 1)]
    pts += [c + sl * L / 2 * b + sw * W / 2 * a + np.array([0, 0, H]) for sl in np.linspace(-0.5, 0.5, 5)
            for sw in (-0.5, 0.0, 0.5)]
    return np.array(pts)


def table_grid_tag(ref: dict, T_tcp_cam: np.ndarray | None = None, rig=None, step_px: int = 32,
                   max_dist_m: float = 1.0, table_z_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Points of the table plane (tag z 0) the (handheld) ultrawide sees: virtual pixels on a grid (step_px in the
    640x480 virtual image) cast to the plane, kept within max_dist_m of the camera. Returns (P_tag, dist_m)."""
    rig = rig or load_rig()
    T_tcp_cam = camera_poses_tcp("handheld", rig)["ultrawide"] if T_tcp_cam is None else T_tcp_cam
    virt = _virtual(rig)
    u, v = np.meshgrid(np.arange(step_px / 2, 640, step_px), np.arange(step_px / 2, 480, step_px))
    rays, _ = virt.unproject(np.stack([u.ravel(), v.ravel()], 1))
    R_pv = np.asarray(rig.virtual_ultrawide.R_phys_virt, float)
    T_tag_cam = ref["T_tag_tcp"] @ T_tcp_cam
    d = (rays @ R_pv.T) @ T_tag_cam[:3, :3].T
    o = T_tag_cam[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (table_z_m - o[2]) / d[:, 2]
    ok = np.isfinite(t) & (t > 0) & (t <= max_dist_m)
    P = o + t[ok, None] * d[ok]
    return P, t[ok]


def scene_points(ref: dict, rig=None, table_z_m: float = 0.0) -> dict[str, np.ndarray]:
    """Table near/mid/far (terciles of the camera distance of the grid points) + eraser, in the TCP frame. The
    table is the plane tag z = table_z_m (layout.json: 0)."""
    P, dist = table_grid_tag(ref, rig=rig, table_z_m=table_z_m)
    q1, q2 = np.quantile(dist, [1 / 3, 2 / 3])
    T = inv(ref["T_tag_tcp"])
    out = {"table_near": apply(T, P[dist <= q1]), "table_mid": apply(T, P[(dist > q1) & (dist <= q2)]),
           "table_far": apply(T, P[dist > q2]), "eraser": apply(T, eraser_box_tag(ref, table_z_m))}
    out["_table_dist_edges_m"] = np.array([dist.min(), q1, q2, dist.max()])
    return out


def point_sets(ref: dict, kind: str, rig=None, table_z_m: float = 0.0) -> dict[str, np.ndarray]:
    """{class: (N,3)} in the TCP frame: finger points of ``kind`` at the reference's width + the scene points."""
    f = finger_points(kind, ref["width_m"])
    s = scene_points(ref, rig, table_z_m)
    return {"tags": f["tags"], "fingertips": np.concatenate([f["tips"], f["fingertips"]]), "holders": f["holders"],
            **{k: v for k, v in s.items() if not k.startswith("_")}}


# ----------------------------------------------------------------------------------------------- projection

def _virtual(rig, K_virt=None):
    from policy_deploy.frames._dataproc.geometry.camera_model import VirtualPinhole
    v = rig.virtual_ultrawide
    w, h = v.resolution
    fx, fy, cx, cy = v.intrinsics if K_virt is None else K_virt
    return VirtualPinhole(int(w), int(h), float(fx), float(fy), float(cx), float(cy))


def project_native(role: str, T_tcp_cam: np.ndarray, P_tcp, rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical-camera pixels (rig resolution) of TCP-frame points: (uv, valid, P_cam); valid = the model's range
    and inside the physical image."""
    phys = D.physical_model(rig, role)
    P_cam = apply(inv(T_tcp_cam), P_tcp)
    uv, ok = phys.project(P_cam)
    return uv, ok & phys.in_image(uv), P_cam


def project(role: str, T_tcp_cam: np.ndarray, P_tcp, rig, R_phys_virt=None, K_virt=None
            ) -> tuple[np.ndarray, np.ndarray]:
    """TCP-frame points -> the policy's 224x224 image of ``role``. Ultrawide: through the virtual pinhole with
    ``R_phys_virt`` (3x3, d_phys = R d_virt) and ``K_virt`` ([fx, fy, cx, cy]); None = the rig's values."""
    uv_phys, ok_phys, P_cam = project_native(role, T_tcp_cam, P_tcp, rig)
    if role == "main":
        res = tuple(int(x) for x in rig.cameras["main"].resolution)
        return D.letterbox_point(uv_phys, res), ok_phys
    R = np.asarray(rig.virtual_ultrawide.R_phys_virt if R_phys_virt is None else R_phys_virt, float)
    virt = _virtual(rig, K_virt)
    uv_v, ok_v = virt.project(P_cam @ R)                 # d_virt = R^T d_phys (row vectors: P R)
    return D.letterbox_point(uv_v, (virt.width, virt.height)), ok_phys & ok_v


def coverage_mask_224(rig, R_phys_virt=None, K_virt=None) -> np.ndarray:
    """(224, 224) fraction of each 224 pixel's virtual footprint that the physical fisheye covers (1 = fully seen,
    0 = black). The letterbox rows (outside the 640x480 virtual image) count as 0."""
    import cv2
    from policy_deploy.frames._dataproc.geometry.camera_model import coverage_mask
    R = np.asarray(rig.virtual_ultrawide.R_phys_virt if R_phys_virt is None else R_phys_virt, float)
    m = coverage_mask(D.physical_model(rig, "ultrawide"), _virtual(rig, K_virt), R).astype(np.uint8) * 255
    from policy_deploy.frames._dataproc.imageprep import letterbox
    rgb = np.repeat(m[:, :, None], 3, axis=2)
    return letterbox(rgb)[:, :, 0].astype(float) / 255.0


def _smoke() -> None:
    rig = load_rig()
    np.set_printoptions(precision=2, suppress=True)
    for kind in ("handheld", "handheld_nominal", "mount"):
        for role, T in camera_poses_tcp(kind, rig).items():
            print(f"{kind:17s} {role:9s} t_tcp_cam {T[:3, 3] * 1e3} mm, T_cam_tcp t {inv(T)[:3, 3] * 1e3} mm")
    print("mount vs handheld (ultrawide):", pose_diff(camera_poses_tcp("handheld")["ultrawide"],
                                                       camera_poses_tcp("mount")["ultrawide"]))
    print("model gap:", model_gap_fn()[2])


if __name__ == "__main__":
    _smoke()
