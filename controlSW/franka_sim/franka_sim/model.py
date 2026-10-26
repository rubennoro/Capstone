"""MuJoCo model of the evaluation station: Panda/FR3 arm + Franka Hand + printed finger adapters + phone/rig housing
and cradle + the two cameras + table + eraser, built with MjSpec from a B1 ``Station``.

Why: every other sim component (kinematics, Stage 1, the backend, rendering) needs one model whose frames match the
training data. The key derived results live here: the training TCP in the flange frame (``F_T_TCP``), the camera poses
(placed from the measured ``C_T_TCP``) and the clearance between the phone/cradle and the arm (a hardware design input).

Frames (INTERFACES.md section 3): ``flange`` = libfranka F (link 7 + 0.107 z, no rotation), ``ee`` = F.F_T_NE.NE_T_EE,
``tcp`` = training TCP (z approach, x along the finger closing axis, y = z x x), ``C`` = OpenCV camera frame. MuJoCo
cameras look along -z with y up, so ``R_mj = R_cv . diag(1, -1, -1)``. Both arms use one path: the no-hand arm MJCF
plus ``hand.xml``'s ``hand`` body attached at ``attachment_site`` (with a correction frame where the site lacks the
stock -45 deg hand rotation, i.e. on FR3).

Usage: python -m franka_sim model --profile sim-iphone [--robot fr3] [--out /tmp/frames.json] [--clearance]
       python -m franka_sim model --ids robot=panda,tooling=franka_hand_iphumi_rig1,sensor=rig1_wifi,scene=...
       sm = build_model(load_station(spec)); frames = export_frames(sm); rep = clearance_report(sm)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

_DEPLOY_ROOT = Path(__file__).resolve().parents[2] / "franka_deploy"   # controlSW/franka_deploy (station loader)
if str(_DEPLOY_ROOT) not in sys.path:
    sys.path.append(str(_DEPLOY_ROOT))

from franka_deploy.station import PROFILES, STATION_ROOT, Station, load_station, station_from_deploy_cfg  # noqa: E402

CONTROLSW = STATION_ROOT.parent
HAND_XML = Path(__file__).resolve().parent / "assets" / "franka_emika_panda" / "hand.xml"
HAND_PREFIX = "hand/"
# Stock panda.xml: hand body in link7 at z 0.107 with quat (0.9238795, 0, 0, -0.3826834) = -45 deg about z (7 digits;
# reproduced exactly so that Panda and FR3 hands are bit-identical to the stock model).
FLANGE_Z_M = 0.107
HAND_IN_LINK7 = (0.0, 0.0, FLANGE_Z_M, 0.9238795, 0.0, 0.0, -0.3826834)
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])            # R_mj = R_cv @ CV_TO_MJ (and back: R_cv = R_mj @ CV_TO_MJ)
UW_PLACEHOLDER = np.array([[1, 0, 0, 0], [0, 1, 0, 0.012], [0, 0, 1, 0], [0, 0, 0, 1]], float)  # UW 12 mm below main
TAG_SIZE_M = 0.016
# Tag centre along the approach axis relative to the pad centre (= the TCP height), TCP-relative so it does not depend on
# ``finger_adapter.extension_m`` (the camera is fixed to the TCP through C_T_TCP, and so are the tags). The real tags sit
# ~98 mm from the UW camera along its optical axis (iPhUMI gripper_calibration artag_z_distance_from_ultrawide 0.098079),
# the TCP ~211 mm: camera-check measured the sim tags 104 mm too far in UW z when they were at the pad centre; a pure
# shift of the tag along TCP z explains the whole (0, +23, -104) mm UW-frame offset. Override: finger_adapter.tag_z_from_pad_m.
TAG_Z_FROM_PAD_M = -0.108
# Lateral position: camera-check on 5 firstreal demos (200 frames) found the real tag centres separated by (zarr width + 44 mm)
# = (finger gap + 87 mm) while the sim had gap + 24 mm: a constant sim-minus-real of -62.3 mm (p50 -62.25, p95 -62.2, ~0.1 mm
# spread over widths 45-84 mm), i.e. the real tags sit 31.2 mm further out per side than b/2 + TAG/2. Override:
# finger_adapter.tag_y_extra_m.
TAG_Y_EXTRA_M = 0.0312
# Inner-face separation of the printed pads at finger gap 0 (pad separation = gap + PAD_OFFSET_M). Placeholder (CAD
# pending): it was tied to width_map.b until the A5 review showed b is the TRAINING width at gap 0 (~0.0431, tag
# separation), not a geometry offset. Override with tooling.finger_adapter.pad_offset_m.
PAD_OFFSET_M = 0.008
MIN_CLEARANCE_M = 0.010
STOCK_PAD_CLASS = re.compile(r"fingertip_pad_collision_\d")


@dataclass
class SimModel:
    model: "mujoco.MjModel"
    spec: "mujoco.MjSpec"
    names: dict
    F_T_TCP: np.ndarray
    EE_T_TCP: np.ndarray
    limits: dict
    robot: str = ""
    station_fingerprint: str = ""
    info: dict = field(default_factory=dict)  # derived numbers for reports (payload, F_T_TCP deviation, intrinsics)


# --------------------------------------------------------------------------------------------------- small math


def Rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def orthonormalize(T) -> np.ndarray:
    """Nearest proper rotation (SVD polar factor) with the translation kept. YAML matrices carry 3-4 digit rotations."""
    T = np.array(T, float)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return make_T(R, T[:3, 3])


def pose_to_T(p) -> np.ndarray:
    """4x4 from a nested 4x4 list or a 7-vector [x, y, z, qw, qx, qy, qz]."""
    a = np.array(p, float)
    if a.shape == (4, 4):
        return orthonormalize(a)
    if a.shape == (7,):
        import mujoco

        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, a[3:] / np.linalg.norm(a[3:]))
        return make_T(R.reshape(3, 3), a[:3])
    raise ValueError(f"pose must be 4x4 or [x,y,z,qw,qx,qy,qz], got shape {a.shape}")


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q


def T_err(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(translation error m, rotation error deg) between two 4x4 poses."""
    dR = A[:3, :3].T @ B[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(ang)


def xform_of(d, kind: str, idx: int) -> np.ndarray:
    """World 4x4 of a body/site/geom/cam from MjData (MuJoCo axes)."""
    pos, mat = {"body": (d.xpos, d.xmat), "site": (d.site_xpos, d.site_xmat), "geom": (d.geom_xpos, d.geom_xmat),
                "cam": (d.cam_xpos, d.cam_xmat)}[kind]
    return make_T(mat[idx].reshape(3, 3), pos[idx])


# ---------------------------------------------------------------------------------------------------- config


def resolve_mjcf(rel: str) -> Path:
    """robot.model.mjcf is relative to controlSW/ (STATION_ROOT.parent)."""
    p = Path(rel)
    p = p if p.is_absolute() else CONTROLSW / p
    if not p.is_file():
        raise FileNotFoundError(f"robot.model.mjcf not found: {p} (paths are relative to {CONTROLSW})")
    return p


def sim_config(station: Station, sim: dict | None) -> dict:
    """The run/sim physics settings: the explicit dict, else the station's sim group if it is a physics file,
    else controlSW/configs/run/sim/physics.yaml."""
    if sim is not None:
        return dict(sim)
    if station.ids.get("sim") and "timestep_s" in station.groups.get("sim", {}):
        return dict(station.groups["sim"])
    return dict(load_station({"run": {"sim": "physics"}}).groups["sim"])


# ----------------------------------------------------------------------------------------------------- build


def _arm_joints(spec) -> dict:
    out = {}
    for j in spec.joints:
        m = re.search(r"(?:^|_)joint([1-7])$", j.name)
        if m and not j.name.startswith(HAND_PREFIX):
            out[f"j{m.group(1)}"] = j.name
    if sorted(out) != [f"j{i}" for i in range(1, 8)]:
        raise ValueError(f"arm MJCF must have 7 joints named *joint1..7, found {sorted(out.values())}")
    return out


def _body_in(m, d, child: str, parent: str) -> np.ndarray:
    import mujoco

    a = xform_of(d, "body", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, parent))
    b = xform_of(d, "body", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, child))
    return inv_T(a) @ b


def _site_frames(spec, site_name: str, j7_body: str) -> tuple[str, np.ndarray, np.ndarray]:
    """(site parent body, parentbody_T_site, j7body_T_site) from a compile of the arm spec alone. The site must be on
    the joint-7 body or on a joint-free descendant of it (panda_nohand.xml: body 'attachment')."""
    import mujoco

    m = spec.compile()
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"flange_site '{site_name}' not in the arm MJCF")
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pb = int(m.site_bodyid[sid])
    b = pb
    while m.body(b).name != j7_body:
        if m.body_jntnum[b] or b == 0:
            raise ValueError(f"flange_site '{site_name}' is not rigidly attached to {j7_body}")
        b = int(m.body_parentid[b])
    W_site = xform_of(d, "site", sid)
    return (m.body(pb).name, inv_T(xform_of(d, "body", pb)) @ W_site,
            inv_T(xform_of(d, "body", b)) @ W_site)


def _finger_tip_z(m, body: str) -> float:
    """Max z (finger body frame) over the finger's mesh geoms: the stock fingertip."""
    import mujoco

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    zmax = -np.inf
    for g in np.flatnonzero(m.geom_bodyid == bid):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        zmax = max(zmax, float((v @ R.reshape(3, 3).T + m.geom_pos[g])[:, 2].max()))
    return zmax


def _finger_mesh_inner_y(m, body: str) -> float:
    """Min y (finger body frame, i.e. the closing side) over the finger's collision mesh geoms."""
    import mujoco

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    ymin = np.inf
    for g in np.flatnonzero(m.geom_bodyid == bid):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        ymin = min(ymin, float((v @ R.reshape(3, 3).T + m.geom_pos[g])[:, 1].min()))
    return ymin


def grip_contact(simc: dict) -> dict:
    """solref/solimp of the finger pads and the eraser (A5). MuJoCo's defaults (solref 0.02 s, solimp 0.9/0.95) let a
    15 g box under a 2 x 30 N squeeze sink ~2 mm into each pad and creep out of the grasp within 2 s; a stiff contact
    (time constant 2 x timestep, the documented minimum; solimp 0.95/0.99) holds it with 30.0 N per pad and < 0.2 mm
    drop. Override with run/sim keys grip_solref / grip_solimp."""
    ts = float(simc.get("timestep_s", 0.002))
    solref = [float(x) for x in (simc.get("grip_solref") or [2.0 * ts, 1.0])]
    solimp = [float(x) for x in (simc.get("grip_solimp") or [0.95, 0.99, 0.001, 0.5, 2.0])]
    if len(solref) != 2:
        raise ValueError(f"run/sim grip_solref must have 2 values (timeconst, dampratio), got {solref}")
    if len(solimp) == 3:                       # MjSpec needs all 5: pad with MuJoCo's defaults (width, power)
        solimp += [0.5, 2.0]
    if len(solimp) != 5:
        raise ValueError(f"run/sim grip_solimp must have 3 or 5 values (dmin, dmax, width[, midpoint, power]), "
                         f"got {solimp}")
    return {"solref": solref, "solimp": solimp}


def _box(body, name: str, size_full, T: np.ndarray, rgba, *, collide: bool, mass: float | None = None,
         friction=None, condim: int | None = None, group: int = 0, solref=None, solimp=None):
    import mujoco

    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = np.asarray(size_full, float) / 2
    g.pos = T[:3, 3]
    g.quat = mat_to_quat(T[:3, :3])
    g.rgba = rgba
    g.group = group
    if not collide:
        g.contype = g.conaffinity = 0
        g.mass = 0.0
    if mass is not None:
        g.mass = float(mass)
    if friction is not None:
        f = np.atleast_1d(np.asarray(friction, float))
        g.friction = np.r_[f, [0.005, 0.0001][len(f) - 1:]][:3] if len(f) < 3 else f[:3]
    if condim is not None:
        g.condim = condim
    if solref is not None:
        g.solref = [float(x) for x in solref]
    if solimp is not None:
        g.solimp = [float(x) for x in solimp]
    return g


def _set_cam_intrinsics(cam, K, width: int, height: int) -> None:
    """MjSpec camera intrinsics (mujoco 3.3.0). Verified by rendering (WORKLOG): MuJoCo's principal_pixel is the
    offset from the image centre with the x sign reversed relative to OpenCV: (W/2 - cx, cy - H/2). MuJoCo measures
    from the continuous image centre W/2 while OpenCV puts pixel centres at integer coordinates, so the OpenCV
    principal point (cx, cy) is (cx + 0.5, cy + 0.5) in MuJoCo's convention (A1 review finding 1; A7 test_render
    checks the projection to <= 0.15 px)."""
    K = np.asarray(K, float)
    cam.resolution = [int(width), int(height)]
    cam.sensor_size = [width * 1e-6, height * 1e-6]       # any non-zero size switches MuJoCo to intrinsic mode
    cam.focal_pixel = [K[0, 0], K[1, 1]]
    cam.principal_pixel = [width / 2 - (K[0, 2] + 0.5), (K[1, 2] + 0.5) - height / 2]


def build_model(station: Station, sim: dict | None = None, with_scene: bool = True) -> SimModel:
    import mujoco

    simc = sim_config(station, sim)
    robot = station.get("robot")
    tool = station.get("tooling")
    sensor = station.get("sensor")
    info: dict = {}

    arm = mujoco.MjSpec.from_file(str(resolve_mjcf(robot["model"]["mjcf"])))
    names: dict = {"joints": _arm_joints(arm)}
    j7_body = arm.joint(names["joints"]["j7"]).parent.name
    site_body, pb_T_site, l7_T_site = _site_frames(arm, robot["model"]["flange_site"], j7_body)

    # -- 1. hand at attachment_site, so that hand-in-link7 = (0, 0, 0.107), -45 deg about z (stock panda.xml)
    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML))
    hand_body = hand_spec.body("hand")
    T_hand_local = pose_to_T(np.r_[hand_body.pos, hand_body.quat])
    want = pose_to_T(HAND_IN_LINK7)
    corr = inv_T(l7_T_site) @ want @ inv_T(T_hand_local)        # extra frame between the site and the hand body
    site = arm.site(robot["model"]["flange_site"])
    if np.allclose(corr, np.eye(4), atol=1e-6):
        site.attach_body(hand_body, HAND_PREFIX, "")
        info["hand_attach"] = "attachment_site (already carries the hand rotation)"
    else:
        fr = arm.body(site_body).add_frame()
        T_fr = pb_T_site @ corr
        fr.pos, fr.quat = T_fr[:3, 3], mat_to_quat(T_fr[:3, :3])
        fr.attach_body(hand_body, HAND_PREFIX, "")
        info["hand_attach"] = f"frame on {site_body} (attachment_site + correction yaw " \
                              f"{np.degrees(np.arctan2(corr[1, 0], corr[0, 0])):.1f} deg)"
    hp = HAND_PREFIX
    names.update(fingers={"finger_l": hp + "finger_joint1", "finger_r": hp + "finger_joint2"}, tendon=hp + "split",
                 bodies={"hand": hp + "hand", "finger_l": hp + "left_finger", "finger_r": hp + "right_finger",
                         "link5": arm.joint(names["joints"]["j5"]).parent.name,
                         "link6": arm.joint(names["joints"]["j6"]).parent.name, "link7": j7_body, "tool": "tool"})

    # -- 2. actuators: torque motors on the arm joints and on the finger tendon; friction/damping scaling
    for a in list(arm.actuators):
        a.delete()
    tau = robot["joints"]["tau_max_nm"]
    fl_scale = float(simc.get("joint_frictionloss_scale", 0.0))
    d_scale = float(simc.get("damping_scale", 1.0))
    acts = {}
    for i in range(1, 8):
        jn = names["joints"][f"j{i}"]
        j = arm.joint(jn)
        j.frictionloss = j.frictionloss * fl_scale
        j.damping = j.damping * d_scale
        a = arm.add_actuator()
        a.name, a.target, a.trntype = f"motor_j{i}", jn, mujoco.mjtTrn.mjTRN_JOINT
        a.gainprm[0], a.ctrllimited, a.ctrlrange = 1.0, 1, [-tau[i - 1], tau[i - 1]]
        acts[f"j{i}"] = a.name
    f_hand = 2.0 * float(tool["hand"]["force_max_n"])
    a = arm.add_actuator()
    a.name, a.target, a.trntype = "motor_hand", names["tendon"], mujoco.mjtTrn.mjTRN_TENDON
    a.gainprm[0], a.ctrllimited, a.ctrlrange = 1.0, 1, [-f_hand, f_hand]
    acts["hand"] = a.name
    names["actuators"] = acts

    # -- first compile: hand pose and finger geometry
    m0 = arm.compile()
    d0 = mujoco.MjData(m0)
    mujoco.mj_kinematics(m0, d0)
    F_in_l7 = make_T(None, [0, 0, FLANGE_Z_M])
    F_T_hand = inv_T(F_in_l7) @ _body_in(m0, d0, names["bodies"]["hand"], j7_body)
    hand_T_fl = _body_in(m0, d0, names["bodies"]["finger_l"], names["bodies"]["hand"])   # at q_finger = 0
    tip_z = _finger_tip_z(m0, names["bodies"]["finger_l"])
    mesh_inner_y = _finger_mesh_inner_y(m0, names["bodies"]["finger_l"])

    # -- 3. printed finger adapters (finger frame: y = closing side outward, z = approach)
    fa, wm = tool["finger_adapter"], tool["width_map"]
    grip_sol = grip_contact(simc)
    b = float(fa.get("pad_offset_m", PAD_OFFSET_M))          # pad offset (geometry), NOT width_map.b (training units)
    info["pad_offset_m"] = b
    info["tag_y_extra_m"] = float(fa.get("tag_y_extra_m", TAG_Y_EXTRA_M))
    info["tag_z_from_pad_m"] = float(fa.get("tag_z_from_pad_m", TAG_Z_FROM_PAD_M))
    info["pad_offset_source"] = ("tooling.finger_adapter.pad_offset_m" if "pad_offset_m" in fa
                                 else "model.PAD_OFFSET_M placeholder (CAD pending)")
    px, pt, pl = (float(v) for v in fa["pad_size_m"])
    pad_c = np.array([0.0, b / 2 + pt / 2, tip_z + float(fa["extension_m"]) - pl / 2])
    info["stock_finger_mesh_collision_off"] = bool(mesh_inner_y < b / 2)
    info["tcp_x_along_hand"] = str(tool.get("tcp_x_along_hand", "+y"))
    sgn = {"+y": 1.0, "-y": -1.0}[info["tcp_x_along_hand"]]
    # TCP in the hand frame: x = sgn * hand y, z = hand z, y = z x x; origin = midpoint of the pad centres
    x_h = np.array([0, sgn, 0.0])
    z_h = np.array([0, 0, 1.0])
    hand_T_TCP = make_T(np.column_stack([x_h, np.cross(z_h, x_h), z_h]),
                        [0, 0, hand_T_fl[2, 3] + pad_c[2]])
    cam_dir_h = -hand_T_TCP[:3, 1]                         # TCP -y (camera side) in the hand frame
    for side, fb in (("l", names["bodies"]["finger_l"]), ("r", names["bodies"]["finger_r"])):
        body = arm.body(fb)
        for g in body.geoms:
            cls = g.classname.name if g.classname is not None else ""
            is_coll = g.contype != 0 or g.conaffinity != 0
            if STOCK_PAD_CLASS.search(cls or "") or (info["stock_finger_mesh_collision_off"] and is_coll
                                                     and g.type == mujoco.mjtGeom.mjGEOM_MESH):
                g.contype = g.conaffinity = 0
        _box(body, f"pad_{side}", [px, pt, pl], make_T(None, pad_c), [0.15, 0.15, 0.15, 1], collide=True,
             friction=fa["friction"], condim=4, **grip_sol)
        # tag patch on the camera-facing pad face (see report: the literal "outer face" is invisible to the camera)
        hT = _body_in(m0, d0, fb, names["bodies"]["hand"])
        dir_f = hT[:3, :3].T @ cam_dir_h                   # camera direction in this finger's frame (+-x)
        face = np.sign(dir_f[0]) * px / 2
        tag_z = pad_c[2] + float(fa.get("tag_z_from_pad_m", TAG_Z_FROM_PAD_M))
        tag_c = np.array([face + np.sign(dir_f[0]) * 0.0005, b / 2 + TAG_SIZE_M / 2 + float(fa.get("tag_y_extra_m", TAG_Y_EXTRA_M)), tag_z])
        _box(body, f"tag_{side}", [0.001, TAG_SIZE_M, TAG_SIZE_M], make_T(None, tag_c), [1, 1, 1, 1], collide=False)
        tag_in = tag_c + [np.sign(dir_f[0]) * 0.0006, 0, 0]
        _box(body, f"tag_{side}_ink", [0.001, 0.6 * TAG_SIZE_M, 0.6 * TAG_SIZE_M], make_T(None, tag_in),
             [0, 0, 0, 1], collide=False)
    names["geoms"] = {"pad_l": "pad_l", "pad_r": "pad_r", "tag_l": "tag_l", "tag_r": "tag_r"}

    # -- 4. TCP
    F_T_TCP_derived = F_T_hand @ hand_T_TCP
    F_T_TCP = F_T_TCP_derived
    st_F = tool.get("F_T_TCP")
    st_prov = station.prov("tooling.F_T_TCP").status if st_F is not None else None
    info["station_F_T_TCP_status"] = st_prov
    if st_F is not None:
        e_t, e_r = T_err(F_T_TCP_derived, orthonormalize(st_F))
        info["station_F_T_TCP_deviation"] = {"mm": e_t * 1e3, "deg": e_r}
        if st_prov not in ("placeholder", "hw") and simc.get("use_station_F_T_TCP", False):
            F_T_TCP = orthonormalize(st_F)
            info["F_T_TCP_source"] = "station"
    info.setdefault("F_T_TCP_source", "derived")
    F_T_EE = np.array(robot["F_T_NE"], float) @ np.array(robot["NE_T_EE"], float)
    EE_T_TCP = inv_T(F_T_EE) @ F_T_TCP

    # tool body = flange frame, rigid with link 7: sites, cameras, housing, cradle
    tb = arm.body(j7_body).add_body()
    tb.name = "tool"
    tb.pos, tb.quat = F_in_l7[:3, 3], [1, 0, 0, 0]
    for nm, T in (("flange", np.eye(4)), ("ee", F_T_EE), ("tcp", F_T_TCP)):
        s = tb.add_site()
        s.name, s.pos, s.quat, s.size = nm, T[:3, 3], mat_to_quat(T[:3, :3]), [0.004, 0, 0]
        s.group = 4
    names["sites"] = {"flange": "flange", "ee": "ee", "tcp": "tcp"}

    # -- 5. cameras
    C_T_TCP_raw = np.array(tool["C_T_TCP"], float)
    C_T_TCP = orthonormalize(C_T_TCP_raw)
    info["C_T_TCP_orthonormalize_deg"] = T_err(make_T(C_T_TCP_raw[:3, :3] / np.linalg.norm(C_T_TCP_raw[:3, 0])),
                                               C_T_TCP)[1]
    F_T_C = F_T_TCP @ inv_T(C_T_TCP)
    uw_raw = tool.get("C_main_T_C_uw")
    C_T_uw = UW_PLACEHOLDER if uw_raw is None else pose_to_T(uw_raw)
    info["C_main_T_C_uw_source"] = "placeholder (12 mm along +y, parallel)" if uw_raw is None else "station"
    F_T_Cuw = F_T_C @ C_T_uw
    for nm, T, stream in (("cam_main", F_T_C, "main"), ("cam_uw", F_T_Cuw, "ultrawide")):
        st = sensor["streams"][stream]
        c = tb.add_camera()
        c.name, c.pos, c.quat = nm, T[:3, 3], mat_to_quat(T[:3, :3] @ CV_TO_MJ)
        _set_cam_intrinsics(c, st["K"], st["width_px"], st["height_px"])
    info["intrinsics_method"] = "MjSpec camera resolution/focal_pixel/principal_pixel/sensor_size (mujoco 3.3.0)"
    names["cams"] = {"cam_main": "cam_main", "cam_uw": "cam_uw"}
    W = max(int(s["width_px"]) for s in sensor["streams"].values())
    H = max(int(s["height_px"]) for s in sensor["streams"].values())
    arm.visual.global_.offwidth = max(arm.visual.global_.offwidth, W)
    arm.visual.global_.offheight = max(arm.visual.global_.offheight, H)

    # -- 6. housing (main lens point at the cam_main origin; housing frame = C axes) and cradle boxes
    hs = tool["housing"]
    F_T_body = F_T_C @ make_T(None, -np.asarray(hs["main_lens_in_body_m"], float))
    tool_geoms = ["housing"]
    masses = {"housing": float(hs.get("mass_kg", 0.0))}
    _box(tb, "housing", hs["size_m"], F_T_body, [0.1, 0.1, 0.12, 1], collide=True, mass=masses["housing"])
    for cb in tool.get("cradle_boxes") or []:
        gname = f"cradle_{cb['name']}"
        _box(tb, gname, cb["size_m"], F_T_body @ pose_to_T(cb["body_T_box"]), [0.55, 0.55, 0.6, 1], collide=True,
             mass=float(cb.get("mass_kg", 0.0)))
        tool_geoms.append(gname)
        masses[gname] = float(cb.get("mass_kg", 0.0))
    names["geoms"]["tooling"] = tool_geoms
    ex = arm.add_exclude()
    ex.bodyname1, ex.bodyname2 = "tool", names["bodies"]["hand"]      # the mount is rigid with the hand

    # -- 7. scene and options
    arm.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    arm.option.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
                       "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[str(simc.get("cone", "elliptic"))]
    arm.option.impratio = float(simc.get("impratio", 10.0))
    arm.option.timestep = float(simc.get("timestep_s", 0.002))
    eraser_qpos = []
    if with_scene:
        sc = station.get("scene")
        tbl = sc["table"]
        ts = np.asarray(tbl["size_m"], float)
        top = float(tbl["top_z_base_m"])
        base_T_table = sc.get("robot_base_T_table")
        tb_T = pose_to_T(base_T_table) if base_T_table is not None else make_T(None, [ts[0] / 2 - 0.2, 0, top])
        tbody = arm.worldbody.add_body()
        tbody.name = "table"
        tbody.pos, tbody.quat = tb_T[:3, 3], mat_to_quat(tb_T[:3, :3])  # body frame: centre of the table top
        _box(tbody, "table", ts, make_T(None, [0, 0, -ts[2] / 2]), [0.6, 0.5, 0.4, 1], collide=True,
             friction=tbl["friction"])
        bd = arm.worldbody.add_geom()          # visual-only floor so that the renders have no black void past the table
        bd.name, bd.type, bd.size = "backdrop", mujoco.mjtGeom.mjGEOM_PLANE, [5.0, 5.0, 0.1]
        bd.pos = [float(tb_T[0, 3]), float(tb_T[1, 3]), float(tb_T[2, 3]) - 0.70]
        bd.contype = bd.conaffinity = 0
        bd.rgba = [0.36, 0.36, 0.38, 1]
        base_body = arm.worldbody.bodies[0].name
        for nm in {base_body, *[b.name for b in arm.worldbody.bodies[0].bodies]}:   # robot base stands on the table
            exb = arm.add_exclude()
            exb.bodyname1, exb.bodyname2 = "table", nm
        er = sc["eraser"]
        es = np.asarray(er["size_m"], float)
        eb = arm.worldbody.add_body()
        eb.name = "eraser"
        jn = eb.add_freejoint()
        jn.name = "eraser_free"
        _box(eb, "eraser", es, np.eye(4), [0.9, 0.3, 0.3, 1], collide=True, mass=float(er["mass_kg"]),
             friction=er["friction"], condim=4, **grip_sol)
        home_xy = np.asarray(robot["home_tcp"]["xyz_m"], float)[:2]
        eraser_qpos = [home_xy[0], home_xy[1], top + es[2] / 2, 1, 0, 0, 0]
        names["bodies"].update(table="table", object="eraser")
        names["joints"]["object"] = "eraser_free"
        names["geoms"].update(table="table", object="eraser")
        light = arm.worldbody.add_light()
        light.pos, light.dir, light.directional = [0.4, 0, 1.5], [0, 0, -1], True

    # -- keyframe "home": station home q, hand at home width, eraser at its default pose
    for k in list(arm.keys):
        k.delete()
    # home_width_m is a TRAINING width (run_task.py arx5_home_gripper_pos_m): the key gap is WidthMap.gap of it
    gap0 = float(np.clip((float(tool["hand"]["home_width_m"]) - float(wm["b"])) / float(wm["a"]),
                         float(wm["gap_min_m"]), float(wm["gap_max_m"]))) / 2
    key = arm.add_key()
    key.name = "home"
    key.qpos = list(robot["home_q_rad"]) + [gap0, gap0] + eraser_qpos
    key.ctrl = [0.0] * 8
    names["key_home"] = "home"

    model = arm.compile()

    # payload: housing + cradle (+ adapters, massless here) vs payload.mass_kg - hand
    hb = model.body(names["bodies"]["hand"]).id
    hand_mass = float(sum(model.body_mass[i] for i in range(model.nbody)
                          if i == hb or model.body_parentid[i] == hb))
    tooling_mass = float(sum(masses.values()))
    pay = float(tool["payload"]["mass_kg"])
    info["payload"] = {"payload_mass_kg": pay, "hand_mass_kg": hand_mass, "tooling_mass_kg": tooling_mass,
                       "tooling_masses_kg": masses, "expected_tooling_kg": pay - hand_mass,
                       "difference_kg": tooling_mass - (pay - hand_mass)}

    limits = {
        "q_min_rad": np.array(robot["joints"]["q_min_rad"], float),
        "q_max_rad": np.array(robot["joints"]["q_max_rad"], float),
        "tau_max_nm": np.array(tau, float),
        "mjcf_range_rad": np.array([model.jnt_range[model.joint(names["joints"][f"j{i}"]).id] for i in range(1, 8)]),
        "hand_force_max_n": f_hand,
        "pad_offset_m": float(info["pad_offset_m"]),
        "gap_max_m": float(wm["gap_max_m"]),
    }
    return SimModel(model=model, spec=arm, names=names, F_T_TCP=F_T_TCP, EE_T_TCP=EE_T_TCP, limits=limits,
                    robot=str(robot["id"]), station_fingerprint=station.fingerprint(), info=info)


# -------------------------------------------------------------------------------------------- frames / reports


def _id(m, kind, name: str) -> int:
    import mujoco

    i = mujoco.mj_name2id(m, {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE,
                              "geom": mujoco.mjtObj.mjOBJ_GEOM, "cam": mujoco.mjtObj.mjOBJ_CAMERA,
                              "joint": mujoco.mjtObj.mjOBJ_JOINT, "key": mujoco.mjtObj.mjOBJ_KEY}[kind], name)
    if i < 0:
        raise KeyError(f"no {kind} '{name}' in the model")
    return i


def forward_at(sm: SimModel, q=None, gap_m: float | None = None):
    """MjData after mj_forward at the home key, with the arm at q (7) and the finger joints at gap_m/2 if given."""
    import mujoco

    m = sm.model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, _id(m, "key", "home"))
    if q is not None:
        for i, v in enumerate(np.asarray(q, float)[:7], 1):
            d.qpos[m.jnt_qposadr[_id(m, "joint", sm.names["joints"][f"j{i}"])]] = v
    if gap_m is not None:
        for f in sm.names["fingers"].values():
            d.qpos[m.jnt_qposadr[_id(m, "joint", f)]] = gap_m / 2
    mujoco.mj_forward(m, d)
    return d


def cam_T_cv(m, d, name: str) -> np.ndarray:
    """World pose of a MuJoCo camera in OpenCV axes."""
    T = xform_of(d, "cam", _id(m, "cam", name))
    T[:3, :3] = T[:3, :3] @ CV_TO_MJ
    return T


def export_frames(sm: SimModel, q=None) -> dict:
    """Frames every other component relies on (4x4 lists, OpenCV axes for cameras), measured from the model."""
    m = sm.model
    d = forward_at(sm, q)
    W_F = xform_of(d, "site", _id(m, "site", "flange"))
    W_TCP = xform_of(d, "site", _id(m, "site", "tcp"))
    F_inv = inv_T(W_F)
    out = {
        "robot": sm.robot, "station_fingerprint": sm.station_fingerprint,
        "F_T_TCP": (F_inv @ W_TCP).tolist(), "EE_T_TCP": sm.EE_T_TCP.tolist(),
        "F_T_ee": (F_inv @ xform_of(d, "site", _id(m, "site", "ee"))).tolist(),
        "F_T_cam_main": (F_inv @ cam_T_cv(m, d, "cam_main")).tolist(),
        "F_T_cam_uw": (F_inv @ cam_T_cv(m, d, "cam_uw")).tolist(),
        "F_T_hand": (F_inv @ xform_of(d, "body", _id(m, "body", sm.names["bodies"]["hand"]))).tolist(),
        "camera_axes": "OpenCV (x right, y down, z optical)",
    }
    TCP_inv = inv_T(W_TCP)
    for side in ("l", "r"):
        key = {"l": "TCP_T_tag_left", "r": "TCP_T_tag_right"}[side]
        out[key] = (TCP_inv @ xform_of(d, "geom", _id(m, "geom", f"tag_{side}"))).tolist()
    T = np.array(out["F_T_TCP"])
    rpy = _rpy_deg(T[:3, :3])
    out["F_T_TCP_summary"] = {"xyz_mm": (T[:3, 3] * 1e3).round(3).tolist(), "rpy_deg": rpy,
                              "source": sm.info.get("F_T_TCP_source")}
    out["payload"] = sm.info.get("payload")
    out["station_F_T_TCP_deviation"] = sm.info.get("station_F_T_TCP_deviation")
    out["station_F_T_TCP_status"] = sm.info.get("station_F_T_TCP_status")
    out["intrinsics_method"] = sm.info.get("intrinsics_method")
    out["hand_attach"] = sm.info.get("hand_attach")
    return out


def _rpy_deg(R: np.ndarray) -> list:
    """Extrinsic xyz (roll, pitch, yaw) in degrees."""
    from scipy.spatial.transform import Rotation

    return np.round(Rotation.from_matrix(R).as_euler("xyz", degrees=True), 3).tolist()


def _collision_geoms(m, body_ids) -> list[int]:
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in body_ids
            and (m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0)]


def clearance_report(sm: SimModel, q6_steps: int = 13, q7_steps: int = 13, q_base=None) -> dict:
    """Minimum distance between the housing/cradle and links 5, 6 (swept over q6 x q7 within the station limits),
    and the static distances to link 7 and the hand (rigid with the tooling). Distances < 0 are penetrations."""
    import mujoco

    m = sm.model
    tool = [_id(m, "geom", g) for g in sm.names["geoms"]["tooling"]]
    groups = {k: _collision_geoms(m, {_id(m, "body", sm.names["bodies"][k])})
              for k in ("link5", "link6", "link7")}
    hb = _id(m, "body", sm.names["bodies"]["hand"])
    groups["hand"] = _collision_geoms(m, {hb, *np.flatnonzero(m.body_parentid == hb).tolist()})
    qb = np.array(q_base if q_base is not None else forward_at(sm).qpos[:7], float)  # arm joints come first
    lo, hi = sm.limits["q_min_rad"], sm.limits["q_max_rad"]
    rng = sm.limits["mjcf_range_rad"]
    lo, hi = np.maximum(lo, rng[:, 0]), np.minimum(hi, rng[:, 1])
    d = forward_at(sm)
    adr = [m.jnt_qposadr[_id(m, "joint", sm.names["joints"][f"j{i}"])] for i in range(1, 8)]
    fromto = np.zeros(6)
    per_link: dict = {k: {"min_m": np.inf} for k in groups}
    n_clear = 0

    def moving_min(q) -> float:
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        return min(mujoco.mj_geomDistance(m, d, tg, g, 0.5, fromto)
                   for k in ("link5", "link6") for tg in tool for g in groups[k])

    home_m = moving_min(qb)
    for q6 in np.linspace(lo[5], hi[5], q6_steps):
        for q7 in np.linspace(lo[6], hi[6], q7_steps):
            q = qb.copy()
            q[5], q[6] = q6, q7
            n_clear += moving_min(q) >= MIN_CLEARANCE_M
            for link, gs in groups.items():
                for tg in tool:
                    for g in gs:
                        dist = mujoco.mj_geomDistance(m, d, tg, g, 0.5, fromto)
                        if dist < per_link[link]["min_m"]:
                            per_link[link] = {"min_m": float(dist), "tool_geom": m.geom(tg).name,
                                              "link_geom": m.geom(g).name or f"geom{g}",
                                              "q": np.round(q, 4).tolist()}
    worst_link = min(per_link, key=lambda k: per_link[k]["min_m"])
    moving = min(("link5", "link6"), key=lambda k: per_link[k]["min_m"])
    return {"robot": sm.robot, "tcp_x_along_hand": sm.info.get("tcp_x_along_hand"), "per_link": per_link,
            "min_m": per_link[worst_link]["min_m"], "min_link": worst_link,
            "min_moving_m": per_link[moving]["min_m"], "min_moving_link": moving, "q_at_min": per_link[moving]["q"],
            "min_clearance_m": MIN_CLEARANCE_M, "ok": bool(per_link[moving]["min_m"] >= MIN_CLEARANCE_M),
            "at_q_base_m": float(home_m), "q_base": np.round(qb, 4).tolist(),
            "clear_fraction": n_clear / (q6_steps * q7_steps), "grid": [q6_steps, q7_steps],
            "q6_range_rad": [float(lo[5]), float(hi[5])], "q7_range_rad": [float(lo[6]), float(hi[6])]}


# ---------------------------------------------------------------------------------------------------------- CLI


def station_from_args(args) -> Station:
    if args.ids:
        ids = dict(kv.split("=", 1) for kv in args.ids.split(","))
        spec = {"station": ids}
    elif args.profile and not args.deploy_config:
        spec = {k: dict(v) for k, v in PROFILES[args.profile].items()}
    elif args.deploy_config:
        import yaml

        cfg = yaml.safe_load(Path(args.deploy_config).read_text())
        if args.profile:
            cfg = {**cfg, **(cfg.get("profiles", {}).get(args.profile) or {})}
        st = station_from_deploy_cfg(cfg)
        if st is None:
            raise SystemExit(f"{args.deploy_config} has no 'station' key")
        spec = {"station": dict(st.ids), "run": {}}
        spec["station"] = {g: st.ids[g] for g in st.ids if g in ("robot", "tooling", "sensor", "scene", "host")}
        spec["run"] = {g: st.ids[g] for g in st.ids if g not in spec["station"]}
    else:
        raise SystemExit("give --profile, --deploy-config or --ids")
    if args.robot:
        spec.setdefault("station", {})["robot"] = args.robot
    return load_station(spec, root=Path(args.root), sets=args.set)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--deploy-config", help="deploy.yaml with a station key")
    p.add_argument("--profile", choices=sorted(PROFILES), help="design profile (franka_deploy.station.PROFILES)")
    p.add_argument("--ids", help="robot=panda,tooling=...,sensor=...,scene=...")
    p.add_argument("--robot", choices=["panda", "fr3"], help="override the robot id")
    p.add_argument("--root", default=str(STATION_ROOT), help="station config root")
    p.add_argument("--set", action="append", default=[], metavar="group.path=value")
    p.add_argument("--no-scene", action="store_true")
    p.add_argument("--out", help="write frames (+ clearance) JSON here (outside the repo)")
    p.add_argument("--clearance", action="store_true", help="also run the q6 x q7 clearance sweep")
    p.add_argument("--steps", type=int, default=13, help="clearance grid steps per joint")


def _fmt_T(T) -> str:
    T = np.asarray(T)
    return f"xyz_mm={np.round(T[:3, 3] * 1e3, 2).tolist()} rpy_deg={_rpy_deg(T[:3, :3])}"


def run(args: argparse.Namespace) -> int:
    st = station_from_args(args)
    sm = build_model(st, with_scene=not args.no_scene)
    fr = export_frames(sm)
    print(f"robot={sm.robot} tooling={st.ids.get('tooling')} sensor={st.ids.get('sensor')} "
          f"tcp_x_along_hand={st.get('tooling.tcp_x_along_hand', '+y')} fingerprint={sm.station_fingerprint}")
    print(f"  hand attach: {fr['hand_attach']}")
    for k in ("F_T_TCP", "EE_T_TCP", "F_T_cam_main", "F_T_cam_uw", "TCP_T_tag_left", "TCP_T_tag_right"):
        print(f"  {k:16s} {_fmt_T(fr[k])}")
    print(f"  F_T_TCP source={fr['F_T_TCP_summary']['source']}; station F_T_TCP ({fr['station_F_T_TCP_status']}) "
          f"deviation={fr['station_F_T_TCP_deviation']}")
    pay = fr["payload"]
    print(f"  payload: tooling {pay['tooling_mass_kg']:.3f} kg vs payload-hand {pay['expected_tooling_kg']:.3f} kg "
          f"(diff {pay['difference_kg']:+.3f} kg)")
    print(f"  intrinsics: {fr['intrinsics_method']}")
    if args.clearance:
        rep = clearance_report(sm, args.steps, args.steps)
        fr["clearance"] = rep
        print(f"  clearance (q6 x q7 {rep['grid']}): moving min {rep['min_moving_m'] * 1e3:.1f} mm "
              f"({rep['min_moving_link']}) at q={rep['q_at_min']} -> {'OK' if rep['ok'] else 'BELOW'} "
              f"{rep['min_clearance_m'] * 1e3:.0f} mm")
        print(f"    at q_base {rep['at_q_base_m'] * 1e3:.1f} mm; grid fraction >= "
              f"{rep['min_clearance_m'] * 1e3:.0f} mm: {rep['clear_fraction']:.2f}")
        for k, v in rep["per_link"].items():
            print(f"    {k:6s} min {v['min_m'] * 1e3:7.1f} mm  {v.get('tool_geom')} <-> {v.get('link_geom')}")
    if args.out:
        out = Path(args.out).resolve()
        from franka_deploy.paths import REPO_ROOT

        if out.is_relative_to(REPO_ROOT):
            print(f"refusing to write inside the repo: {out}", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fr, indent=1, default=float))
        print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_sim model", description=__doc__.split("\n\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
