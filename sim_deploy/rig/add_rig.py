"""Put the CAD rig (rig.yaml + assets/, written by extract.py) on the Franka Hand of a MuJoCo MjSpec.

    from sim_deploy.rig.add_rig import add_rig, load_rig, cam_T_tip
    info = add_rig(spec, "hand/hand", "sim_deploy/rig")     # before spec.compile()

What it adds (DESIGN.md section 2):
  - body ``rig`` (child of the hand body, frame = hand frame, no joint) with the rig's total mass, COM and full inertia
    as an explicit inertial (rig.yaml ``mass``);
  - visual mesh geoms ``rig_<part>`` (group 2, no contacts) and collision geoms ``rig_<part>_col`` (convex hulls of
    mount, plate, cover, Pi 5; group 5, contype = conaffinity = 1) on ``rig``. Bodies without joints share the weld
    of link 7, so MuJoCo never collides them with the hand, link 7 or (filterparent) link 6 and the fingers; they do
    collide with links 0-5, the table and objects;
  - cameras ``cam_imx519`` and ``cam_imx219`` on the hand body at the lens optical centres (MuJoCo axes: looks along -z,
    +y up; R_mj = R_cv @ diag(1, -1, -1)). Intrinsics: ``intrinsics={name: (K 3x3, width, height)}`` (MuJoCo
    focal/principal pixel, same +0.5 px convention as franka_sim.model._set_cam_intrinsics), else a ``fovy_deg``.

Needs only mujoco, numpy and pyyaml (runs in ~/envs/env_sim).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

SCHEMA = "sim_deploy.rig/v1"
CAMERAS = ("cam_imx519", "cam_imx219")
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
VISUAL_GROUP = 2
COLLISION_GROUP = 5


def load_rig(rig_dir) -> dict:
    rig_dir = Path(rig_dir)
    doc = yaml.safe_load((rig_dir / "rig.yaml").read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{rig_dir / 'rig.yaml'}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def _quat(R: np.ndarray) -> np.ndarray:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q


def hand_T_cam(rig: dict, name: str) -> np.ndarray:
    """4x4 pose of a camera in the hand frame, OpenCV axes (x right, y down, z optical)."""
    c = rig["cameras"][name]
    T = np.eye(4)
    T[:3, :3] = np.array(c["hand_R_cam_cv"], float)
    T[:3, 3] = c["pos_hand_m"]
    return T


def cam_T_tip(rig: dict, name: str, z_tip_m: float) -> np.ndarray:
    """Camera (OpenCV) -> fingertip point (0, 0, z_tip) in the hand frame with training TCP axes (hand_R_tcp)."""
    T_tip = np.eye(4)
    T_tip[:3, :3] = np.array(rig["frames"]["hand_R_tcp"], float)
    T_tip[2, 3] = z_tip_m
    T = hand_T_cam(rig, name)
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti @ T_tip


def _set_intrinsics(cam, K, width: int, height: int) -> None:
    K = np.asarray(K, float)
    cam.resolution = [int(width), int(height)]
    cam.sensor_size = [width * 1e-6, height * 1e-6]
    cam.focal_pixel = [K[0, 0], K[1, 1]]
    cam.principal_pixel = [width / 2 - (K[0, 2] + 0.5), (K[1, 2] + 0.5) - height / 2]


def add_rig(spec, hand_body: str, rig_dir, *, intrinsics: dict | None = None, fovy_deg: float = 90.0,
            prefix: str = "rig", collide: bool = True) -> dict:
    """Add the rig's meshes, collision hulls, inertial body and the two cameras to ``spec`` on ``hand_body``.

    Returns {"body", "visual_geoms", "collision_geoms", "cameras", "mesh_files", "mass_kg", "com_hand_m",
    "rig_yaml_sha256"} (mesh files are absolute paths: a builder that writes scene.xml copies them)."""
    import hashlib

    import mujoco

    rig_dir = Path(rig_dir).resolve()
    rig = load_rig(rig_dir)
    hand = spec.body(hand_body)
    if hand is None:
        raise KeyError(f"no body {hand_body!r} in the spec")

    body = hand.add_body()
    body.name = prefix
    body.pos, body.quat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    ms = rig["mass"]
    I = np.array(ms["inertia_com_hand_kgm2"], float)
    body.mass = float(ms["total_kg"])
    body.ipos = list(ms["com_hand_m"])
    body.iquat = [1.0, 0.0, 0.0, 0.0]
    body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
    body.explicitinertial = True

    files, vis, col = [], [], []
    for part, m in rig["meshes"].items():
        mesh = spec.add_mesh()
        mesh.name = f"{prefix}_{part}"
        mesh.file = str(rig_dir / m["file"])
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL     # not used (explicit inertial), never fails
        files.append(mesh.file)
        g = body.add_geom()
        g.name = f"{prefix}_{part}"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = mesh.name
        g.group = VISUAL_GROUP
        g.contype = g.conaffinity = 0
        g.rgba = list(rig["visual_rgba"][part])
        vis.append(g.name)
        if "collision_file" in m:
            cm = spec.add_mesh()
            cm.name = f"{prefix}_{part}_col"
            cm.file = str(rig_dir / m["collision_file"])
            files.append(cm.file)
            cg = body.add_geom()
            cg.name = f"{prefix}_{part}_col"
            cg.type = mujoco.mjtGeom.mjGEOM_MESH
            cg.meshname = cm.name
            cg.group = COLLISION_GROUP
            cg.contype = cg.conaffinity = 1 if collide else 0
            cg.rgba = [0.9, 0.2, 0.2, 0.3]
            col.append(cg.name)

    cams = {}
    for name in CAMERAS:
        T = hand_T_cam(rig, name)
        c = hand.add_camera()
        c.name = name
        c.pos = T[:3, 3]
        c.quat = _quat(T[:3, :3] @ CV_TO_MJ)
        if intrinsics and name in intrinsics:
            K, w, h = intrinsics[name]
            _set_intrinsics(c, K, w, h)
        else:
            c.fovy = float(fovy_deg)
        cams[name] = name

    return {"body": prefix, "visual_geoms": vis, "collision_geoms": col, "cameras": cams, "mesh_files": files,
            "mass_kg": float(ms["total_kg"]), "com_hand_m": list(ms["com_hand_m"]),
            "rig_yaml_sha256": hashlib.sha256((rig_dir / "rig.yaml").read_bytes()).hexdigest()}
