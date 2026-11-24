"""Camera study B3 / W2 "sets": the same scene from the HANDHELD camera pose and from the Franka MOUNT (PLAN.md 10, 11).

    unset PYTHONPATH; cd ~/Capstone-main-sim
    ~/envs/sim_deploy/bin/python -m sim_deploy.camera_study.render_sets [--station sim_deploy/stations/<id>.yaml]
        [--out sim_evals/_pipeline/W2/sets] [--look-id ID] [--refit] [--only ep14_first ...]
        [--motion-blur-s EXPOSURE_S [--motion-blur-k K]]

``--look-id`` defaults to the station's ``id`` (a station may reuse another build through ``sim_pi.model``; its sets
still get their own folder). ``--out`` defaults to ``sim_evals/_pipeline/W2/sets``. The station's optional
``sim_pi.camera_look`` (a file, relative to the station) is loaded by ``sim_deploy.look.camera_look.load_camera_look``
and applied to every 640x480 frame before JPEG; the station settings are otherwise read as the sim Pi reads them.

For each of the 15 references of ``sim_evals/_camera_study/B/layout.json`` (5 PICK_TEST1 episodes x rows first /
approach_1of3 / approach_2of3) one arm configuration per data row is shared by every camera placement, so only the
camera-to-finger geometry differs between the sets:

* **scene**: the recorded TCP at rows T-6, T-3, T (package low-dim, tag frame) -> O by ``layout.json T_O_tag``; the
  arm solved by IK for the model's ``tcp`` site (``trainstart.solve_ik``, seeded from ``start_pose.yaml`` q_start and
  then row to row); fingertip gap = the recorded width at that row (D8: gap = width; model gap = finger_joint1 +
  finger_joint2, set past the joints' 0.04 m range when the width is wider than 80 mm, as the handheld opens to 90 mm);
  the eraser flat on the table where it lay in that episode: at the hold TCP (``placement.grasp_rows`` end) with its
  width (the model's: sim_deploy/eraser.py, 30 mm) along the closing axis, then refined per episode by image evidence (``fit_placement``: x, y, yaw, shared
  by the episode's 3 references, least squares of the eraser blob's centroid / size / second moments in the real
  224 images through the copy's camera models). The real eraser images as ~102 x 50 mm, not the model's 127 x 30 mm (51 before 2026-11-21),
  so the chosen fit (``xy_yaw_size``) frees the box length and width for the fit only (the model's eraser is
  unchanged): its centre lands on the real eraser's centre; the fits with the model's size (``xy_yaw``, and
  ``xy_yaw_dz`` with a table-height offset) are reported too. Stored in ``<out>/placement.json`` (reused by every
  look; ``--refit`` recomputes it).
* **handheld**: both cameras at training's pose relative to ``tcp`` (rig file ``T_cam_tcp``, IMX519 via both
  ``T_imu_cam``, as ``handheld_check.camera_poses``) with E7's fix (IMX219 + IMX519 translated by (-4.0, +2.5, +8.8) mm
  in IMX219 axes); the rig parts that ride with the cameras (cover, Pi 5, LED, button, camera and IMU boards) moved
  rigidly with the IMX219 (E2's CAD relation, as handheld_check); arm links, Franka hand, Franka mount + plate and the
  printed finger adapters removed from the render model (pads, holders and tags stay). The "nominal" handheld pose
  (without E7's fix) is evaluated by projection only.
* **mount**: the model's own ``cam_imx219`` / ``cam_imx519`` at the same arm configuration (only each camera's own
  board hidden, as the sim Pi does).
* **mount_start** (``first`` references only): the arm at q_start, the eraser placed at the same position / yaw
  relative to the fingertips' table projection (ahead / right of the TCP heading) as in that reference at row T.
* Finger gap: every MOUNT pose uses the ROBOT's fingertip gap, ``robot_gap(width)`` = the deployment gripper map
  (training 0.051 <-> robot 0.030, 0.090 <-> 0.040, clipped to 0..0.040), as on the real arm; handheld keeps the
  recorded gap. ``mount_recgap`` / ``mount_start_recgap`` are the same mount scenes with the recorded gap (the gap
  effect separately). The obs.npz low-dim is the reference's (recorded width) in every pose.
* **robot_trials** (``robot_trials.py``, run after the sets unless ``--no-robot-trials``): the arm at q_start, gap
  0.040, the eraser fitted to the real arm's mount images of each trial's static start.

Rendering = the sim Pi path: ``model_file.load_render_model`` (its private compile; the handheld variant runs the same
steps on an edited spec), ``CameraChain(role, rig)`` with the camera's own board hidden, ``GLRenderer``, no motion blur
(unless ``--motion-blur-s``, below),
the station's ``sim_pi:`` settings (shadows, keep_model_lights, jpeg_quality, optional ``camera_look`` applied to each
640x480 frame before JPEG), JPEG encode + decode, then the copy's ``ImagePrep(rig_pi_v3_640.yaml)``. To stay under
200 MB of VRAM, views taller than ``STRIP_H`` (640) rows are rendered as horizontal strips (exact sub-frusta of the
same pinhole) into the same atlas and the offscreen buffer is sized to (widest view, 640): checked against one render
per view: 640x480 frames max |diff| 1 (IMX519) / 14 on 26 of 307 200 pixels (IMX219), 224 images max 2, coverage
mask identical; VRAM 139-157 MiB instead of 233-252 MiB.

Outputs, ``<out>/``: ``<look>/<pose>/ep<ii>_<label>/{main_0,main_1,uw_0,uw_1}.png`` (224, oldest first: main rows
T-3, T; IMX219 rows T-6, T), ``obs.npz`` (exactly the C4 Observation keys: the images above as ``main`` /
``ultrawide`` (2, 224, 224, 3) uint8, the low-dim ``eef_pos`` / ``eef_rot6d`` / ``eef_rot6d_wrt_start`` /
``gripper_width_m`` copied unchanged from the B1 reference ``sim_evals/_camera_study/B/reference/<key>/obs.npz``),
``jpg640_*.jpg`` (the exact JPEG bytes), ``meta.json``; ``index.json`` (one row per look / pose / reference, with the
model and camera_look sha256; rewritten atomically when a look completes); ``geometry.json`` (per look and set: ArUco
corners, eraser blobs, fingertips; handheld / mount / real compared, by detection in the images and by pure
projection; IK and contacts); ``placement.json``; ``contact_<look>.png`` (real | handheld | mount | mount_start).
Real reference images come from the package (``sets_real_export.py`` in the torch env); B's cache
``sim_evals/_camera_study/B/sets/_real/`` is used read-only when complete, else they are exported to ``<out>/_real/``.

Motion blur (opt-in, ``--motion-blur-s EXPOSURE_S``, ``--motion-blur-k K`` sub-frames, default 9; off = the output
above, unchanged): every frame of the per-row poses (handheld, mount, mount_recgap; mount_start* and robot_trials are
static and stay sharp) is the mean of K sub-frames at times row + (k / (K-1) - 1/2) * EXPOSURE_S * 60 rows (60 Hz
rows, window centred on the frame's row). Per sub-frame the recorded TCP is interpolated between the two neighbouring
rows (position linear, rotation slerp), the arm solved by IK (seeded from the row's IK solution), fingertip gap and
eraser as at the row; so the camera (and the fingers / rig parts riding with it) moves, the table and eraser stay put.
The K 640x480 frames are averaged in linear light (decode gamma 2.2, mean, re-encode) before the camera_look and JPEG.
The eraser coverage mask and the stored camera poses are those of the row itself. Recorded in meta.json and the
index rows as ``motion_blur``.

Eraser blobs: real images by colour (lime core: HSV hue 33-90, then a soft alpha by unmixing the core colour against
the surrounding background, so blur does not bias area or centroid); sim images by an exact coverage mask: a second
render of the same views with the eraser emissive white and everything else black (no texture, no skybox, no JPEG),
the same remap / INTER_AREA / ImagePrep path. The colour method applied to the sim images is reported as its check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import sim_deploy.sim_pi.render as R_  # noqa: F401  (EGL + NVIDIA before mujoco)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from sim_deploy.sim_pi import _deploy as D  # noqa: E402
from sim_deploy.sim_pi import handheld_check as HC  # noqa: E402
from sim_deploy.sim_pi.cameras import CameraChain  # noqa: E402
from sim_deploy.sim_pi.model_file import CAMERAS, OWN_BOARD, RenderModel, load_render_model, read_doc  # noqa: E402
from sim_deploy.sim_pi.render import GLRenderer, Lighting, View, finish_model, prepare_spec  # noqa: E402
from sim_deploy.sim_pi import settings as SPS  # noqa: E402
from sim_deploy.scene.placement import grasp_rows  # noqa: E402
from sim_deploy.scene.trainstart import solve_ik  # noqa: E402
from sim_deploy.scene.zarr_lowdim import load_episode  # noqa: E402

cv2.setNumThreads(4)

ROW_HZ = 60.0                                    # low-dim rows per second (the timing of the motion-blur window)
BLUR_GAMMA = 2.2                                 # sub-frames are averaged in linear light

REPO = D.REPO
B_DIR = REPO / "sim_evals" / "_camera_study" / "B"
LAYOUT = B_DIR / "layout.json"
OUT = REPO / "sim_evals" / "_pipeline" / "W2" / "sets"          # default --out
REAL_CACHE = B_DIR / "sets" / "_real"                          # B's export (read-only; checked complete)
REAL_EXPORT = REPO / "sim_deploy" / "camera_study" / "sets_real_export.py"
DEFAULT_STATION = REPO / "sim_deploy" / "stations" / "sim_rig.yaml"
TORCH_PY = Path("~/envs/sim_policy_deploy_torch/bin/python").expanduser()
OBS_KEYS = ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m", "main", "ultrawide")   # C4 Observation
LOWDIM_KEYS = OBS_KEYS[:4]
KIT_SIM = REPO / "sim_deploy" / "adapter" / "kit_franka_v1"
E7_DT_M = (-0.0040, 0.0025, 0.0088)              # E7's camera fix, IMX219 axes (DESIGN.md E7 open item 4)
ROLES = ("main", "ultrawide")
SHORT = {"main": "main", "ultrawide": "uw"}
ARM_BODIES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")
HANDHELD_REMOVE = ("rig_mount", "rig_plate", "finger_holder_l_1", "finger_holder_r_1")   # not on the handheld
RIDE_WITH_CAMERAS = ("rig_cover", "rig_pi5", "rig_led_strip", "rig_button_board", "rig_imx519_board",
                     "rig_imx219_board", "rig_imu_board")
MARKER_M = 0.016
#: The robot's fingertip gap for a training width (deployment policy/gripper_map.py, the W2 coordinator's constants):
#: one line through (training 0.051, robot 0.030) and (training 0.090, robot 0.040), clipped to the robot's 0..0.040.
GRIPPER_MAP = {"training_closed_m": 0.051, "training_open_m": 0.090, "robot_closed_m": 0.030, "robot_open_m": 0.040}
#: pose name -> (scene variant, gap rule). Mount poses use the ROBOT's gap (the arm's fingers), handheld the recorded one.
POSES = {"handheld": ("per_row", "gap"), "mount": ("per_row", "gap_robot"), "mount_recgap": ("per_row", "gap"),
         "mount_start": ("start", "gap_robot"), "mount_start_recgap": ("start", "gap")}
POSE_NOTES = {
    "handheld": "training camera pose (rig T_cam_tcp + T_imu_cam + E7 fix) at the recorded TCP; recorded fingertip gap",
    "mount": "model mount cameras (E2) at the recorded TCP (IK); ROBOT fingertip gap = gripper_map(recorded width)",
    "mount_recgap": "as mount but the recorded (handheld) fingertip gap: separates the gap effect",
    "mount_start": "model mount cameras, arm at q_start, eraser at the reference's fingertip-relative position; ROBOT gap",
    "mount_start_recgap": "as mount_start with the recorded (handheld) fingertip gap",
    "robot_trials": "arm at q_start, gap 0.040, eraser fitted to the real arm's mount images (robot_trials.py)"}


def robot_gap(width_m: float) -> float:
    g = GRIPPER_MAP
    s = (g["robot_open_m"] - g["robot_closed_m"]) / (g["training_open_m"] - g["training_closed_m"])
    return float(np.clip(g["robot_closed_m"] + (float(width_m) - g["training_closed_m"]) * s, 0.0, g["robot_open_m"]))


STRIP_H = 640                    # Renderer: offscreen buffer height cap (VRAM <= 200 MB); views split into strips


# ------------------------------------------------------------------------------------------------ small math
def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    return _T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def _quat_wxyz(R: np.ndarray) -> list:
    q = Rotation.from_matrix(R).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _mat_from_wxyz(q) -> np.ndarray:
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _r(x, n=4):
    return np.round(np.asarray(x, float), n).tolist()


def _wrap90(a: float) -> float:
    return (a + 90.0) % 180.0 - 90.0


# ------------------------------------------------------------------------------------------------ references
def load_layout() -> dict:
    return json.loads(LAYOUT.read_text())


def references(layout: dict) -> list[dict]:
    out = []
    for sel in layout["selection"]:
        for T, label in zip(sel["rows_in_episode"], sel["labels"]):
            out.append({"key": f"ep{int(sel['index']):02d}_{label}", "index": int(sel["index"]), "T": int(T),
                        "label": label})
    return out


def load_real(layout: dict, out_dir: Path | None = None) -> dict:
    """key -> npz dict of the real images: B's cache (read-only) when it holds every reference, else exported once
    (torch env) into ``<out_dir>/_real``."""
    refs = references(layout)
    cache = REAL_CACHE
    if not all((cache / f"{r['key']}.npz").is_file() for r in refs):
        cache = Path(out_dir if out_dir is not None else OUT) / "_real"
        if not all((cache / f"{r['key']}.npz").is_file() for r in refs):
            cmd = [str(TORCH_PY), str(REAL_EXPORT), str(LAYOUT), str(cache)]
            print("[sets] exporting the real references:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    out = {}
    for r in refs:
        z = np.load(cache / f"{r['key']}.npz")
        out[r["key"]] = {k: z[k] for k in z.files}
        out[r["key"]]["_cache"] = str(cache)
    return out


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_station_settings(station: Path):
    """The sim Pi's settings (``sim_pi.settings.load`` semantics: defaults, then the station's ``sim_pi:`` block, paths
    relative to the station, model default ``build/<id>/scene.xml``), tolerating the optional ``camera_look`` key
    whether or not ``SimPiSettings`` knows it. Returns (settings, camera_look path or None, station doc)."""
    import yaml
    station = Path(station).expanduser().resolve()
    doc = yaml.safe_load(station.read_text()) or {}
    block = dict(doc.get("sim_pi") or {})
    look = block.pop("camera_look", None)
    look_path = None
    if look:
        p = Path(str(look)).expanduser()
        look_path = p if p.is_absolute() else (station.parent / p).resolve()
    s = SPS.SimPiSettings()
    unknown = sorted(set(block) - set(s.fields))
    if unknown:
        raise ValueError(f"{station}: sim_pi.{unknown[0]}: unknown key (known: {', '.join(s.fields)}, camera_look)")
    for k, v in block.items():
        setattr(s, k, SPS._coerce(k, v, station.parent))
    if s.model is None and doc.get("id"):
        s.model = REPO / "sim_deploy" / "build" / str(doc["id"]) / "scene.xml"
    if s.model is None:
        raise ValueError(f"{station}: no model (no sim_pi.model and no id)")
    if "camera_look" in s.fields:
        setattr(s, "camera_look", look_path)
    return s, look_path, doc


def _write_json_atomic(path: Path, doc) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def b1_reference_check(real: dict) -> dict:
    """B1's reference PNGs (when present) against the package images used here (max abs difference)."""
    res = {}
    for key, R in real.items():
        d = B_DIR / "reference" / key
        row = {"obs_npz": (d / "obs.npz").is_file()}
        for name, img in (("main_1.png", R["main"][1]), ("uw_1.png", R["uw"][1]), ("main_0.png", R["main"][0]),
                          ("uw_0.png", R["uw"][0])):
            p = d / name
            if p.is_file():
                b = cv2.imread(str(p))[:, :, ::-1]
                row[name] = int(np.abs(b.astype(int) - img.astype(int)).max()) if b.shape == img.shape else "shape"
        res[key] = row
    return res


# ------------------------------------------------------------------------------------------------ kinematics
class Kin:
    """Plain (non-render) compile of the scene for IK, FK, contacts and the kit/tag/eraser geometry."""

    def __init__(self, scene_xml: Path, doc: dict):
        from franka_sim import finger_kit as FK
        self.m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.d = mujoco.MjData(self.m)
        self.spec = mujoco.MjSpec.from_file(str(scene_xml))
        self.doc = doc
        lay = doc["qpos_layout"]
        self.arm = slice(*lay["arm"])
        self.fing = slice(*lay["fingers"])
        self.er = slice(*lay["eraser"])
        self.box = np.asarray(doc["eraser"]["box_m"], float)
        self.table_z = float(doc["table"]["top_z_m"])
        self.jr = self.m.jnt_range[:7].copy()
        self.kit = FK.load_kit(KIT_SIM)
        # model finger body -> (kit side, body_T_kit): the pad geom places the kit's STL (kit coordinates)
        self.sides = {}
        for body, pad in (("left_finger", "pad_l"), ("right_finger", "pad_r")):
            g = self.spec.geom(pad)
            side = "right" if g.meshname.endswith("_right") else "left"
            self.sides[body] = (side, _T(_mat_from_wxyz(g.quat), np.asarray(g.pos, float)))
        self.tcp_id = self.m.site("tcp").id
        self.eraser_geom = self.m.geom("eraser").id

    def set(self, q_arm, gap: float, eraser_pose) -> None:
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[self.arm] = q_arm
        d.qpos[self.fing] = gap / 2.0
        d.qpos[self.er] = eraser_pose
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)

    def body_T(self, name: str) -> np.ndarray:
        b = self.m.body(name).id
        return _T(self.d.xmat[b].reshape(3, 3), self.d.xpos[b])

    def tcp_T(self) -> np.ndarray:
        return _T(self.d.site_xmat[self.tcp_id].reshape(3, 3), self.d.site_xpos[self.tcp_id])

    def cam_T(self, role: str) -> np.ndarray:
        c = self.m.camera(CAMERAS[role]).id
        return _T(self.d.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), self.d.cam_xpos[c])

    def kit_T(self, body: str) -> np.ndarray:
        return self.body_T(body) @ self.sides[body][1]

    def tips_world(self) -> dict:
        out = {}
        for body, (side, _) in self.sides.items():
            p = self.kit_T(body) @ np.r_[self.kit.sides[side].tip, 1.0]
            out[body] = p[:3]
        return out

    def tag_corners_world(self) -> dict[int, np.ndarray]:
        """ArUco corner order (marker TL, TR, BR, BL; marker top = the tag's up) of the 16 mm markers, world."""
        out = {}
        for body, (side, _) in self.sides.items():
            T = self.kit_T(body)
            for t in self.kit.sides[side].tags:
                n, up = np.asarray(t.normal, float), np.asarray(t.up, float)
                x = np.cross(up, n)
                c = np.asarray(t.centre, float)
                h = MARKER_M / 2
                P = np.array([c - h * x + h * up, c + h * x + h * up, c + h * x - h * up, c - h * x - h * up])
                out[int(t.id)] = (np.c_[P, np.ones(4)] @ T.T)[:, :3]
        return out

    def eraser_points(self, pose, n_edge: int = 24, box=None) -> np.ndarray:
        """Points along the 12 edges of the eraser box at ``pose`` (world); ``box`` (L, W, T m) overrides the model's
        size (resting on the same table: the centre height follows T)."""
        R = _mat_from_wxyz(pose[3:7])
        if box is None:
            h = self.box / 2
        else:
            h = np.asarray(box, float) / 2
            pose = list(pose)
            pose[2] = self.table_z + h[2]
        C = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * h
        pts = []
        t = np.linspace(0, 1, n_edge)[:, None]
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(C[i] != C[j]) == 1:
                    pts.append(C[i] + t * (C[j] - C[i]))
        P = np.concatenate(pts)
        return P @ R.T + np.asarray(pose[:3], float)

    def eraser_pose(self, x: float, y: float, yaw_deg: float) -> list:
        a = np.deg2rad(yaw_deg) / 2
        return [float(x), float(y), self.table_z + float(self.box[2]) / 2, float(np.cos(a)), 0.0, 0.0,
                float(np.sin(a))]

    def contacts(self) -> list:
        mujoco.mj_forward(self.m, self.d)
        out = []
        for c in self.d.contact[:self.d.ncon]:
            n1 = self.m.geom(c.geom1).name or self.m.body(self.m.geom_bodyid[c.geom1]).name
            n2 = self.m.geom(c.geom2).name or self.m.body(self.m.geom_bodyid[c.geom2]).name
            if {n1, n2} == {"eraser", "table"}:
                continue
            out.append([n1, n2, round(float(c.dist), 4)])
        return out

    def margins(self, q) -> float:
        q = np.asarray(q, float)
        return float(np.min(np.minimum(q - self.jr[:, 0], self.jr[:, 1] - q)))


def handheld_tcp_T_cam(rig, dt_m=E7_DT_M) -> dict:
    """tcp -> camera (OpenCV) of training's handheld cameras (handheld_check.camera_poses), shifted by dt (IMX219
    axes, both cameras rigidly)."""
    nom = HC.camera_poses(rig)
    A_uw = nom["ultrawide"] @ _T(None, np.asarray(dt_m, float))
    A = {"ultrawide": A_uw, "main": A_uw @ _inv(nom["ultrawide"]) @ nom["main"]}
    A_tcp = _T(HC.TCP_AXES, HC.TCP_POS)
    return {r: _inv(A_tcp) @ A[r] for r in ROLES}


# ------------------------------------------------------------------------------------------------ projection
class Projector:
    """Pure projection through the copy's camera models into the 224 images (independent of rendering)."""

    def __init__(self, rig, prep):
        self.prep = prep
        self.main = D.physical_model(rig, "main")
        self.main_res = tuple(int(v) for v in rig.cameras["main"].resolution)
        self.R_phys_virt = np.asarray(prep.ultrawide.R_phys_virt, float)   # identity in rig_pi_v3_640

    def __call__(self, role: str, O_T_cam: np.ndarray, P: np.ndarray) -> np.ndarray:
        pc = (np.c_[P, np.ones(len(P))] @ _inv(O_T_cam).T)[:, :3]
        if role == "ultrawide":
            uv = self.prep.ultrawide.virt.project(pc @ self.R_phys_virt)[0]       # virtual axes: R^T p
            return self.prep.ultrawide.virtual_to_out(uv)
        uv = self.main.project(pc)[0]
        return D.letterbox_point(uv, self.main_res)


def polygon_blob(uv: np.ndarray) -> dict | None:
    """Exact area / centroid / second moments (float64 shoelace) of the convex hull of projected points (224 px)."""
    from scipy.spatial import ConvexHull
    uv = np.asarray(uv, float)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) < 3:
        return None
    try:
        h = uv[ConvexHull(uv).vertices]
    except Exception:  # noqa: BLE001  (degenerate)
        return None
    x, y = h[:, 0], h[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    if abs(A) <= 1e-9:
        return None
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    exx = ((x * x + x * x1 + x1 * x1) * cr).sum() / (12 * A)
    eyy = ((y * y + y * y1 + y1 * y1) * cr).sum() / (12 * A)
    exy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / (24 * A)
    C = np.array([[exx - cx * cx, exy - cx * cy], [exy - cx * cy, eyy - cy * cy]])
    return {"area": float(abs(A)), "c": np.array([cx, cy]), "C": C}


def alpha_blob(alpha: np.ndarray) -> dict | None:
    a = float(alpha.sum())
    if a < 1.0:
        return None
    h, w = alpha.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    c = np.array([(alpha * u).sum() / a, (alpha * v).sum() / a])
    du, dv = u - c[0], v - c[1]
    C = np.array([[(alpha * du * du).sum(), (alpha * du * dv).sum()],
                  [(alpha * du * dv).sum(), (alpha * dv * dv).sum()]]) / a
    return {"area": a, "c": c, "C": C}


def _sqrtm2(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T


def blob_resid(sim: dict, real: dict) -> np.ndarray:
    """(dcx, dcy, d sqrt(area), d sqrtm(C) 11/22/12) in px."""
    S, Rr = _sqrtm2(sim["C"]), _sqrtm2(real["C"])
    return np.r_[sim["c"] - real["c"], np.sqrt(sim["area"]) - np.sqrt(real["area"]),
                 S[0, 0] - Rr[0, 0], S[1, 1] - Rr[1, 1], S[0, 1] - Rr[0, 1]]


def blob_json(b: dict | None) -> dict | None:
    if b is None:
        return None
    w, V = np.linalg.eigh(b["C"])
    ang = float(np.degrees(np.arctan2(V[1, 1], V[0, 1])))
    return {"area_px": round(b["area"], 2), "centre_px": _r(b["c"], 2), "axes_sd_px": _r(np.sqrt(np.maximum(w, 0))[::-1], 2),
            "major_axis_deg": round(_wrap90(ang), 1)}


def blob_compare(a: dict | None, b: dict | None) -> dict | None:
    """a relative to b: centre dx/dy, size ratio sqrt(area_a / area_b)."""
    if a is None or b is None:
        return None
    return {"centre_dxdy_px": _r(a["c"] - b["c"], 2), "size_ratio": round(float(np.sqrt(a["area"] / b["area"])), 4)}


# ------------------------------------------------------------------------------------------------ colour blobs
def colour_alpha(rgb: np.ndarray, core: np.ndarray | None = None) -> np.ndarray | None:
    """Soft eraser coverage of a colour image: a lime/yellow core (or the given one), then alpha by unmixing the core's
    colour against the background ring around it, within 2 px of the core."""
    if core is None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
        core = ((H >= 33) & (H <= 90) & (S >= 70) & (V >= 60)).astype(np.uint8)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    else:
        core = core.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 4:
        return None
    core = (lab == k).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    inner = cv2.erode(core, k3)
    if inner.sum() < 3:
        inner = core
    win = cv2.dilate(core, k3, iterations=2)
    ring = cv2.dilate(win, k3, iterations=3) & (1 - win)
    f = rgb.astype(np.float64)
    e = np.median(f[inner > 0], axis=0)
    bg = np.median(f[ring > 0], axis=0)
    dv = e - bg
    nn = float(dv @ dv)
    if nn < 100.0:
        return core.astype(np.float64)
    al = np.clip(((f - bg) @ dv) / nn, 0.0, 1.0)
    al[win == 0] = 0.0
    al[core > 0] = np.maximum(al[core > 0], 0.0)
    return al


# ------------------------------------------------------------------------------------------------ render models
def load_handheld_render_model(scene_xml: Path, lighting: Lighting, W: int, H: int, keep_model_lights: bool,
                               tcp_T_cam: dict) -> tuple[RenderModel, dict]:
    """load_render_model's private compile on an edited spec: the two cameras at the handheld pose relative to the
    tcp site, the rig parts that ride with them moved rigidly with the IMX219 (E2's CAD relation), the arm, hand,
    Franka mount + plate and printed adapters removed. qpos layout unchanged (checked)."""
    from sim_deploy.rig.add_rig import hand_T_cam, load_rig
    scene_xml = Path(scene_xml)
    doc, sha = read_doc(scene_xml)
    ref = mujoco.MjModel.from_xml_path(str(scene_xml))
    rd = mujoco.MjData(ref)
    mujoco.mj_kinematics(ref, rd)
    hb, ts = ref.body("hand").id, ref.site("tcp").id
    W_hand = _T(rd.xmat[hb].reshape(3, 3), rd.xpos[hb])
    W_tcp = _T(rd.site_xmat[ts].reshape(3, 3), rd.site_xpos[ts])
    hand_T_tcp = _inv(W_hand) @ W_tcp
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    hand_T_cam_hh = {r: hand_T_tcp @ tcp_T_cam[r] for r in ROLES}
    # the model's cameras (E2) vs E2's rig.yaml: must agree (the rig parts are placed through E2's IMX219)
    rdoc = load_rig(REPO / "sim_deploy" / "rig")
    E2_uw = hand_T_cam(rdoc, "cam_imx219")
    c = spec.camera(CAMERAS["ultrawide"])
    model_uw = _T(_mat_from_wxyz(c.quat) @ np.diag([1.0, -1.0, -1.0]), np.asarray(c.pos, float))
    e2_check_mm = float(np.linalg.norm(model_uw[:3, 3] - E2_uw[:3, 3]) * 1e3)
    M = hand_T_cam_hh["ultrawide"] @ _inv(E2_uw)            # hand frame: E2's rig parts -> handheld placement
    for role in ROLES:
        cam = spec.camera(CAMERAS[role])
        Tc = hand_T_cam_hh[role]
        cam.pos = Tc[:3, 3].tolist()
        cam.quat = _quat_wxyz(Tc[:3, :3] @ np.diag([1.0, -1.0, -1.0]))
    rig_body = spec.body("rig")
    hand_T_rig = _T(_mat_from_wxyz(rig_body.quat), np.asarray(rig_body.pos, float))
    for gname in RIDE_WITH_CAMERAS:
        g = spec.geom(gname)
        Tg = _inv(hand_T_rig) @ M @ hand_T_rig @ _T(_mat_from_wxyz(g.quat), np.asarray(g.pos, float))
        g.pos = Tg[:3, 3].tolist()
        g.quat = _quat_wxyz(Tg[:3, :3])
    removed = []
    for b in ARM_BODIES:
        for g in list(spec.body(b).geoms):
            if int(g.group) in (0, 1, 2, 3):
                removed.append(g.name or f"{b}/geom")
                g.delete()
    for gname in HANDHELD_REMOVE:
        spec.geom(gname).delete()
        removed.append(gname)
    prepare_spec(spec, lighting, W, H, keep_model_lights=keep_model_lights)
    m = spec.compile()
    finish_model(m, lighting)
    if m.nq != ref.nq or not np.array_equal(m.jnt_qposadr, ref.jnt_qposadr):
        raise RuntimeError("handheld render model's qpos layout differs from the file's")
    cam_id = {r: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n) for r, n in CAMERAS.items()}
    own = {r: (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)) for r, g in OWN_BOARD.items()}
    own = {r: (g if g >= 0 else None) for r, g in own.items()}
    d = mujoco.MjData(m)
    rm = RenderModel(scene_xml, doc, sha, m, d, cam_id, own, np.asarray(doc["home"]["qpos"], float))
    rm.set_qpos(rm.home_qpos)
    info = {"removed_geoms": removed, "moved_rig_geoms": list(RIDE_WITH_CAMERAS),
            "model_cam_imx219_vs_E2_rig_yaml_mm": round(e2_check_mm, 4),
            "rig_parts_shift_mm": _r(M[:3, 3] * 1e3, 2),
            "rig_parts_rotation_deg": round(float(np.degrees(np.linalg.norm(Rotation.from_matrix(M[:3, :3]).as_rotvec()))), 3)}
    return rm, info


class Renderer:
    """One render model + one GL context: the sim Pi's frame path, plus an exact eraser coverage pass."""

    def __init__(self, rm: RenderModel, chains: dict, settings, prep, look=None, strip_h: int | None = STRIP_H):
        """strip_h: views taller than this are rendered in horizontal strips (exact sub-frusta of the same pinhole)
        and the offscreen buffer is sized to (widest view, strip_h): the 4x MSAA buffer dominates VRAM (1621x1217:
        ~100 MB). None: the sim Pi's way (one render per view, buffer as prepare_spec sized it)."""
        self.rm, self.chains, self.s, self.prep, self.look = rm, chains, settings, prep, look
        for r, c in chains.items():
            c.set_hide([rm.own_board[r]])
        W = max(c.max_view[0] for c in chains.values())
        H = max(c.max_view[1] for c in chains.values())
        self.strip_h = int(strip_h) if strip_h else None
        if self.strip_h:
            H = min(H, self.strip_h)
            rm.model.vis.global_.offwidth = W            # MjrContext allocates the buffer from these
            rm.model.vis.global_.offheight = H
        self.gl = GLRenderer(rm.model, W, H)
        self.gl.set_shadows(settings.shadows)
        self.eraser_gid = rm.model.geom("eraser").id

    def close(self):
        self.gl.close()

    def _chain_frame(self, role: str, Rw: np.ndarray, pw: np.ndarray) -> np.ndarray:
        """CameraChain.frame, with views taller than strip_h rendered as strips into the same atlas rows."""
        chain = self.chains[role]
        if not self.strip_h:
            return chain.frame(self.gl, Rw, pw)
        atlas = chain.new_buffer()
        for v, y0 in zip(chain.views, chain._offsets):
            for a in range(0, v.h, self.strip_h):
                b = min(v.h, a + self.strip_h)
                sv = View(v.name, v.w, b - a, v.f, v.cx, v.cy - a, R_cam_view=v.R_cam_view, hide=v.hide)
                # atlas rows hold the view bottom row first: view rows [a, b) -> atlas rows y0 + h - b .. y0 + h - a
                atlas[y0 + v.h - b:y0 + v.h - a, :v.w] = self.gl.render_raw(sv, Rw, pw)
        return chain.stream_from(atlas)

    def frame(self, role: str, qpos: np.ndarray, rng, sub_qpos: list | None = None
              ) -> tuple[np.ndarray, bytes, np.ndarray, np.ndarray]:
        """(224 image, JPEG bytes, 640 decoded, 224 eraser coverage alpha). sub_qpos: motion blur, the frame is the
        mean of one render per sub-frame qpos in linear light (gamma BLUR_GAMMA); qpos (the row) gives the coverage
        mask and the camera pose left in rm."""
        rm, gl = self.rm, self.gl
        acc = None
        if sub_qpos:
            for sq in sub_qpos:
                rm.set_qpos(sq)
                gl.update(rm.data)
                Rs, ps = rm.camera_pose(role)
                lin = (self._chain_frame(role, Rs, ps).astype(np.float32) / 255.0) ** BLUR_GAMMA
                acc = lin if acc is None else acc + lin
        rm.set_qpos(qpos)
        gl.update(rm.data)
        Rw, pw = rm.camera_pose(role)
        if acc is not None:
            rgb = np.clip(np.rint(255.0 * (acc / len(sub_qpos)) ** (1.0 / BLUR_GAMMA)), 0, 255).astype(np.uint8)
        else:
            rgb = self._chain_frame(role, Rw, pw)
        if self.look is not None:
            rgb = np.ascontiguousarray(self.look.apply(role, rgb, rng), dtype=np.uint8)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(self.s.jpeg_quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        dec = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = self.prep.prep_main(dec) if role == "main" else self.prep.prep_ultrawide(dec)
        # coverage pass: same views, eraser emissive white, everything else black, no textures / skybox
        scn = gl.scn
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            is_er = g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == self.eraser_gid
            a = float(g.rgba[3])
            g.rgba[:] = [1.0, 1.0, 1.0, a] if is_er else [0.0, 0.0, 0.0, a]
            g.emission = 1.0 if is_er else 0.0
            g.specular = 0.0
            g.reflectance = 0.0
            g.matid = -1
        sky = scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]
        shadow = scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        try:
            mask = self._chain_frame(role, Rw, pw)
        finally:
            scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = sky
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
        m224 = self.prep.prep_main(mask) if role == "main" else self.prep.prep_ultrawide(mask)
        alpha = m224[..., 1].astype(np.float64) / 255.0
        return img, jpg.tobytes(), dec, alpha


# ------------------------------------------------------------------------------------------------ scene per reference
class Study:
    def __init__(self, station: Path, out: Path | None = None, look_id: str | None = None,
                 motion_blur_s: float | None = None, motion_blur_k: int = 9):
        self.station = Path(station).resolve()
        self.blur = None
        if motion_blur_s:
            if motion_blur_k < 2:
                raise ValueError("--motion-blur-k must be >= 2")
            self.blur = {"exposure_s": float(motion_blur_s), "k": int(motion_blur_k), "row_hz": ROW_HZ,
                         "gamma": BLUR_GAMMA, "window": "centred on the frame's row",
                         "subframes": "recorded TCP interpolated between neighbouring rows (lerp + slerp), IK per "
                                      "sub-frame seeded from the row's IK; gap and eraser as at the row",
                         "poses": ["handheld", "mount", "mount_recgap"]}
        self.s, look_path, self.station_doc = load_station_settings(self.station)
        self.look_id = str(look_id or self.station_doc.get("id") or self.s.model.parent.name)
        self.out = Path(out).resolve() if out is not None else OUT
        self.layout = load_layout()
        self.T_O_tag = np.asarray(self.layout["T_O_tag"], float)
        self.q_start = np.asarray(self.layout["q_start_rad"]["franka_control"]["homing"]["q_start_rad"], float)
        doc, _ = read_doc(self.s.model)
        self.doc = doc
        self.kin = Kin(self.s.model, doc)
        self.rig = D.load_live_rig(self.s.rig_file)
        self.prep = D.ImagePrep(self.rig)
        self.proj = Projector(self.rig, self.prep)
        self.tcp_T_cam = {"E7": handheld_tcp_T_cam(self.rig, E7_DT_M), "nominal": handheld_tcp_T_cam(self.rig, (0, 0, 0))}
        self.refs = references(self.layout)
        self.real = load_real(self.layout, self.out)
        self.episodes = {}
        for sel in self.layout["selection"]:
            ep = load_episode(self.layout["dataset"], int(sel["index"]))
            self.episodes[int(sel["index"])] = ep
        self.cam_look = None
        self.look_sha256 = None
        if look_path:
            from sim_deploy.look.camera_look import load_camera_look
            self.cam_look = load_camera_look(look_path)
            self.look_sha256 = sha256_file(look_path)
        self.look_path = str(look_path) if look_path else None
        self.real_blobs = {k: {(role, j): self._real_blob(k, role, j) for role in ROLES for j in (0, 1)}
                           for k in self.real}

    # --- rows and poses
    def rows(self, ref: dict) -> dict:
        R = self.real[ref["key"]]
        return {"main": [int(v) for v in R["main_rows"]], "ultrawide": [int(v) for v in R["uw_rows"]]}

    def O_T_tcp(self, index: int, row: int, dz: float = 0.0) -> np.ndarray:
        T = self.T_O_tag @ self.episodes[index]["poses"][row]
        T = T.copy()
        T[2, 3] -= dz
        return T

    def width(self, index: int, row: int) -> float:
        return float(self.episodes[index]["widths"][row])

    def _real_blob(self, key, role, j):
        img = self.real[key]["main" if role == "main" else "uw"][j]
        al = colour_alpha(img)
        return alpha_blob(al) if al is not None else None

    # --- eraser placement
    def base_placement(self, index: int) -> dict:
        ep = self.episodes[index]
        g = grasp_rows(ep["widths"])
        Th = self.T_O_tag @ ep["poses"][g["end"]]
        x_t = Th[:3, 0]
        yaw = _wrap90(float(np.degrees(np.arctan2(x_t[1], x_t[0]))) - 90.0)   # width (body y) along the closing axis
        Ts = self.T_O_tag @ ep["poses"][g["start"]]
        return {"grasp_rows": {k: (int(v) if isinstance(v, (int, np.integer)) else round(float(v), 5)) for k, v in g.items()},
                "hold_tcp_O": _r(Th[:3, 3], 5), "start_tcp_O": _r(Ts[:3, 3], 5),
                "x": float(Th[0, 3]), "y": float(Th[1, 3]), "yaw_deg": yaw,
                "hold_tip_height_m": round(float(Th[2, 3]), 5),
                "hold_tilt_deg": round(float(np.degrees(np.arccos(np.clip(-Th[2, 2], -1, 1)))), 2)}

    def _proj_blobs(self, index: int, pose, dz: float, cams="E7", box=None) -> dict:
        """Projected eraser hull blobs for every image of every reference of the episode (handheld camera)."""
        P = self.kin.eraser_points(pose, box=box)
        out = {}
        for ref in self.refs:
            if ref["index"] != index:
                continue
            rows = self.rows(ref)
            for role in ROLES:
                for j, row in enumerate(rows[role]):
                    O_cam = self.O_T_tcp(index, row, dz) @ self.tcp_T_cam[cams][role]
                    out[(ref["key"], role, j)] = polygon_blob(self.proj(role, O_cam, P))
        return out

    def _fit_resid(self, index, base, p, with_dz, box=None):
        """p = (dx mm, dy mm, dyaw deg[, dz mm]); box: eraser (L, W, T) m for the projection (default: the model's)."""
        dz = p[3] * 1e-3 if with_dz else 0.0
        pose = self.kin.eraser_pose(base["x"] + p[0] * 1e-3, base["y"] + p[1] * 1e-3, base["yaw_deg"] + p[2])
        pb = self._proj_blobs(index, pose, dz, box=box)
        res = []
        for (key, role, j), b in pb.items():
            rb = self.real_blobs[key][(role, j)]
            if b is None or rb is None:
                continue
            res.append(blob_resid(b, rb))
        return np.concatenate(res) if res else np.zeros(1)

    def fit_placement(self) -> dict:
        from scipy.optimize import least_squares
        out = {"method": "least squares (soft_l1, 1.5 px) of the eraser blob (centroid, sqrt area, sqrtm of the "
                         "covariance) in the real 224 images (IMX519 T-3, T; IMX219 T-6, T; all 3 references of the "
                         "episode) vs the eraser box hull projected through the copy's camera models from the "
                         "handheld (E7) camera at the recorded TCP; parameters dx, dy (O, mm), dyaw (deg) [, table "
                         "height offset dz (mm): table + eraser raised by dz relative to the recorded TCPs]",
               "episodes": {}}
        out["size_note"] = ("xy_yaw_size frees the eraser box length and width for the FIT only (thickness = the "
                            "model's): the real eraser images as ~100 x 50 mm, the model's is "
                            f"{self.kin.box[0] * 1e3:.0f} x {self.kin.box[1] * 1e3:.0f} mm, and with the model's size "
                            "the blob size / moments pull the centre off (and xy_yaw_dz explains the size by a 3 cm "
                            "lower table). The sim keeps the model's eraser; its centre is put on the fitted centre.")
        T_mm = float(self.kin.box[2]) * 1e3
        for index in sorted({r["index"] for r in self.refs}):
            base = self.base_placement(index)
            e0 = self._fit_resid(index, base, np.zeros(4), False).reshape(-1, 6)
            fits = {}
            for name, n, with_dz, sized in (("xy_yaw", 3, False, False), ("xy_yaw_dz", 4, True, False),
                                            ("xy_yaw_size", 5, False, True)):
                def fun(p, with_dz=with_dz, sized=sized):
                    box = np.array([p[3], p[4], T_mm]) * 1e-3 if sized else None
                    return self._fit_resid(index, base, p[:4] if with_dz else p[:3], with_dz, box=box)
                x0 = np.zeros(n)
                if sized:
                    x0[3:5] = self.kin.box[:2] * 1e3
                xs = np.array([10.0, 10.0, 2.0, 5.0] if with_dz else [10.0, 10.0, 2.0, 10.0, 5.0])[:n]
                f = least_squares(fun, x0, loss="soft_l1", f_scale=1.5, x_scale=xs, diff_step=1e-4)
                e = fun(f.x).reshape(-1, 6)
                fits[name] = {"dx_mm": round(float(f.x[0]), 2), "dy_mm": round(float(f.x[1]), 2),
                              "dyaw_deg": round(float(f.x[2]), 2),
                              "dz_mm": round(float(f.x[3]), 2) if with_dz else 0.0,
                              "resid": _resid_summary(e), "p": f.x[:4].tolist() if with_dz else f.x[:3].tolist()}
                if sized:
                    fits[name]["box_LW_mm"] = _r(f.x[3:5], 1)
                    # the model's eraser at that centre: residuals with the model's size (what the renders show)
                    fits[name]["resid_model_size"] = _resid_summary(self._fit_resid(index, base, f.x[:3], False))
            chosen = "xy_yaw_size"
            p = np.r_[fits[chosen]["p"], [0.0] * (4 - len(fits[chosen]["p"]))]
            pose = self.kin.eraser_pose(base["x"] + p[0] * 1e-3, base["y"] + p[1] * 1e-3, base["yaw_deg"] + p[2])
            out["episodes"][str(index)] = {"base": base, "resid_before": _resid_summary(e0), "fits": fits,
                                           "chosen": chosen, "dz_m": float(p[3]) * 1e-3, "eraser_pose_O": _r(pose, 6),
                                           "rule": "xy_yaw_size (centre fitted with a free box length / width; table "
                                                   "height as recorded, dz 0)"}
        return out


def _resid_summary(e: np.ndarray) -> dict:
    e = np.asarray(e, float).reshape(-1, 6)
    return {"n_images": int(len(e)), "rms_px": round(float(np.sqrt(np.mean(e ** 2))), 3),
            "centre_mean_abs_px": round(float(np.mean(np.linalg.norm(e[:, :2], axis=1))), 3),
            "centre_max_px": round(float(np.max(np.linalg.norm(e[:, :2], axis=1))), 3),
            "sqrt_area_mean_px": round(float(np.mean(e[:, 2])), 3)}


# ------------------------------------------------------------------------------------------------ main flow
def plan_scenes(st: Study, placement: dict) -> dict:
    """Per reference: q per row (IK), gap, eraser pose; the mount_start variant for first rows."""
    kin = st.kin
    scenes = {}
    for ref in st.refs:
        index, key = ref["index"], ref["key"]
        P = placement["episodes"][str(index)]
        pose = list(P["eraser_pose_O"])
        dz = float(P["dz_m"])
        rows = st.rows(ref)
        all_rows = sorted(set(rows["main"]) | set(rows["ultrawide"]))
        q = st.q_start.copy()
        per_row = {}
        for row in all_rows:
            Tg = st.O_T_tcp(index, row, dz)
            q, pe, ae = solve_ik(kin.m, kin.d, "tcp", Tg, q)
            gap = st.width(index, row)
            kin.set(q, gap, pose)
            per_row[row] = {"q": q.copy(), "gap": gap, "gap_robot": robot_gap(gap), "O_T_tcp_target": Tg, "ik_pos_err_mm": pe * 1e3,
                            "ik_ang_err_deg": ae, "joint_margin_rad": kin.margins(q),
                            "dist_to_q_start_rad": float(np.linalg.norm(q - st.q_start)),
                            "contacts": kin.contacts(), "O_T_tcp": kin.tcp_T(),
                            "finger_q_beyond_range": bool(gap / 2 > kin.m.jnt_range[7, 1] + 1e-9)}
        sc = {"ref": ref, "rows": rows, "per_row": per_row, "eraser_pose": pose, "dz_m": dz}
        if ref["label"] == "first":
            T = ref["T"]
            Tr = per_row[T]["O_T_tcp"]
            kin.set(st.q_start, per_row[T]["gap"], pose)
            Ts = kin.tcp_T()
            rel = _table_rel(Tr, pose)
            pose_s = _from_table_rel(Ts, rel, kin)
            per_row_s = {}
            for row in all_rows:
                gap = st.width(index, row)
                kin.set(st.q_start, gap, pose_s)
                per_row_s[row] = {"q": st.q_start.copy(), "gap": gap, "gap_robot": robot_gap(gap), "O_T_tcp": kin.tcp_T(),
                                  "joint_margin_rad": kin.margins(st.q_start), "dist_to_q_start_rad": 0.0,
                                  "contacts": kin.contacts(),
                                  "finger_q_beyond_range": bool(gap / 2 > kin.m.jnt_range[7, 1] + 1e-9)}
            sc["start"] = {"per_row": per_row_s, "eraser_pose": pose_s, "table_rel": rel}
        scenes[key] = sc
    return scenes


def _heading_frame(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = T[:3, 2].copy()
    z[2] = 0.0
    if np.linalg.norm(z) < 1e-6:
        z = T[:3, 1].copy()
        z[2] = 0.0
    ahead = z[:2] / np.linalg.norm(z[:2])
    right = np.array([ahead[1], -ahead[0]])
    return T[:2, 3].copy(), ahead, right


def _table_rel(T_tcp: np.ndarray, pose) -> dict:
    o, a, r = _heading_frame(T_tcp)
    d = np.asarray(pose[:2]) - o
    yaw = float(np.degrees(2 * np.arctan2(pose[6], pose[3])))
    hdg = float(np.degrees(np.arctan2(a[1], a[0])))
    return {"ahead_m": float(d @ a), "right_m": float(d @ r), "yaw_rel_deg": _wrap90(yaw - hdg)}


def _from_table_rel(T_tcp: np.ndarray, rel: dict, kin: Kin) -> list:
    o, a, r = _heading_frame(T_tcp)
    xy = o + rel["ahead_m"] * a + rel["right_m"] * r
    hdg = float(np.degrees(np.arctan2(a[1], a[0])))
    return kin.eraser_pose(xy[0], xy[1], _wrap90(hdg + rel["yaw_rel_deg"]))


def qpos_for(kin: Kin, q, gap, pose) -> np.ndarray:
    kin.set(q, gap, pose)
    return kin.d.qpos.copy()


def _interp_T(Ta: np.ndarray, Tb: np.ndarray, f: float) -> np.ndarray:
    from scipy.spatial.transform import Slerp
    T = np.eye(4)
    T[:3, 3] = (1 - f) * Ta[:3, 3] + f * Tb[:3, 3]
    T[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ta[:3, :3], Tb[:3, :3]])))([f]).as_matrix()[0]
    return T


def blur_subframes(st: Study, sc: dict, row: int) -> dict:
    """Motion blur sub-frames of one row (cached in the scene, shared by every per-row pose): {'q': [K arm q],
    't_rows': offsets, 'ik_pos_err_mm_max', 'ik_ang_err_deg_max'}."""
    cache = sc.setdefault("_blur", {})
    if row in cache:
        return cache[row]
    b = st.blur
    index, dz = sc["ref"]["index"], sc["dz_m"]
    n = len(st.episodes[index]["poses"])
    half = 0.5 * b["exposure_s"] * b["row_hz"]
    offs = np.linspace(-half, half, b["k"])
    q0 = sc["per_row"][row]["q"]
    qs, pes, aes = [], [], []
    for o in offs:
        t = float(np.clip(row + o, 0, n - 1))
        r0 = int(np.floor(t))
        r1 = min(r0 + 1, n - 1)
        Tg = _interp_T(st.O_T_tcp(index, r0, dz), st.O_T_tcp(index, r1, dz), t - r0)
        q, pe, ae = solve_ik(st.kin.m, st.kin.d, "tcp", Tg, q0)
        qs.append(q.copy())
        pes.append(pe * 1e3)
        aes.append(ae)
    cache[row] = {"q": qs, "t_rows": [round(float(o), 4) for o in offs],
                  "ik_pos_err_mm_max": round(float(max(pes)), 4), "ik_ang_err_deg_max": round(float(max(aes)), 5)}
    return cache[row]


def render_all(st: Study, scenes: dict, only: set | None) -> dict:
    """Renders every set of this look; returns {(pose, key): set record}."""
    chains = {r: CameraChain(r, st.rig, density=st.s.density) for r in ROLES}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    lighting = Lighting(shadows=st.s.shadows)
    records = {}
    for pose_kind in ("handheld", "mount"):
        t0 = time.perf_counter()
        if pose_kind == "handheld":
            rm, info = load_handheld_render_model(st.s.model, lighting, W, H, st.s.keep_model_lights,
                                                  st.tcp_T_cam["E7"])
        else:
            rm, info = load_render_model(st.s.model, lighting, W, H, keep_model_lights=st.s.keep_model_lights), {}
        rend = Renderer(rm, chains, st.s, st.prep, st.cam_look)
        vram = _vram()
        try:
            variants = [n for n in POSES if (n == "handheld") == (pose_kind == "handheld")]
            for name in variants:
                scene_kind, gap_key = POSES[name]
                for key, sc in scenes.items():
                    if only and key not in only:
                        continue
                    if scene_kind == "start":
                        if "start" not in sc:
                            continue
                        per_row, pose = sc["start"]["per_row"], sc["start"]["eraser_pose"]
                    else:
                        per_row, pose = sc["per_row"], sc["eraser_pose"]
                    records[(name, key)] = _render_set(st, rend, name, key, sc, per_row, pose, info, gap_key=gap_key)
            vram_end = _vram()
        finally:
            rend.close()
        print(f"[sets] {st.look_id}: {pose_kind} done in {time.perf_counter() - t0:.1f} s (VRAM {vram} after the "
              f"context, {vram_end} at the end)", flush=True)
        records.setdefault("_vram", {})[pose_kind] = {"after_context": vram, "end": vram_end}
    return records


def _vram() -> str | None:
    """This process's GPU memory from nvidia-smi's process table (EGL rendering shows as a "G" process, which
    ``--query-compute-apps`` does not list)."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return None
    for ln in out.splitlines():
        p = ln.strip("| ").split()
        if len(p) >= 6 and p[3] == str(os.getpid()):
            return p[-1]
    return "not listed"


def _rel(p) -> str:
    p = Path(p)
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def _render_set(st: Study, rend: Renderer, pose_kind: str, key: str, sc: dict, per_row: dict, pose, info,
                gap_key: str = "gap") -> dict:
    ref = sc["ref"]
    folder = st.out / st.look_id / pose_kind / key
    folder.mkdir(parents=True, exist_ok=True)
    kin = st.kin
    images, alphas, cams = {}, {}, {}
    blur = st.blur if (st.blur and POSES[pose_kind][0] == "per_row") else None
    blur_rows = {}
    for role in ROLES:
        for j, row in enumerate(sc["rows"][role]):
            pr = per_row[row]
            qpos = qpos_for(kin, pr["q"], pr[gap_key], pose)
            rng = np.random.default_rng([ref["index"], ref["T"], row, ROLES.index(role)])
            sub = None
            if blur:
                bs = blur_subframes(st, sc, row)
                sub = [qpos_for(kin, q, pr[gap_key], pose) for q in bs["q"]]
                blur_rows[str(row)] = {k: v for k, v in bs.items() if k != "q"}
            img, jpg, _, alpha = rend.frame(role, qpos, rng, sub_qpos=sub)
            nm = f"{SHORT[role]}_{j}"
            cv2.imwrite(str(folder / f"{nm}.png"), img[:, :, ::-1])
            (folder / f"jpg640_{nm}.jpg").write_bytes(jpg)
            images[(role, j)] = img
            alphas[(role, j)] = alpha
            Rw, pw = rend.rm.camera_pose(role)
            O_cam = _T(Rw, pw)
            cams[(role, j)] = O_cam
    # obs.npz: exactly the C4 Observation keys; low-dim unchanged from the B1 reference, images from this render
    lowdim_path = B_DIR / "reference" / key / "obs.npz"
    z = np.load(lowdim_path)
    obs = {k: z[k] for k in LOWDIM_KEYS}
    obs["main"] = np.stack([images[("main", 0)], images[("main", 1)]]).astype(np.uint8)
    obs["ultrawide"] = np.stack([images[("ultrawide", 0)], images[("ultrawide", 1)]]).astype(np.uint8)
    for k in OBS_KEYS:
        if obs[k].dtype != z[k].dtype or obs[k].shape != z[k].shape:
            raise RuntimeError(f"{key} obs.npz {k}: {obs[k].dtype} {obs[k].shape} != reference {z[k].dtype} {z[k].shape}")
    np.savez_compressed(folder / "obs.npz", **{k: obs[k] for k in OBS_KEYS})
    meta = {
        "schema": "camera_study.B.sets/v2", "reference": key, "episode_index": ref["index"],
        "episode_name": str(st.real[key]["name"]), "T": ref["T"], "label": ref["label"],
        "look": st.look_id, "station": str(st.station.relative_to(REPO)) if st.station.is_relative_to(REPO) else str(st.station),
        "model": _rel(st.s.model), "model_sha256": st.doc["model_sha256"],
        "camera_look": st.look_path, "camera_look_sha256": st.look_sha256,
        "tool_sha256": sha256_file(__file__), "strip_h": rend.strip_h,
        "pose": pose_kind,
        "obs_npz": {"file": "obs.npz", "keys": list(OBS_KEYS),
                    "lowdim_from": str(lowdim_path.relative_to(REPO)), "lowdim_sha256": sha256_file(lowdim_path),
                    "images": "main = [main_0, main_1], ultrawide = [uw_0, uw_1] (oldest first)"},
        "camera": ("training T_cam_tcp + T_imu_cam (rig_pi_v3_640.yaml) + E7 fix (-4.0, +2.5, +8.8) mm IMX219 axes"
                   if pose_kind == "handheld" else "model cam_imx219 / cam_imx519 (E2 CAD)"
                   + (", arm at q_start" if POSES[pose_kind][0] == "start" else "")),
        "pose_note": POSE_NOTES[pose_kind],
        "finger_gap": {"rule": gap_key, "gripper_map": GRIPPER_MAP if gap_key == "gap_robot" else None,
                       "note": "gap_robot = gripper_map(recorded width), clipped to 0..0.040; gap = recorded width; "
                               "the obs.npz low-dim keeps the recorded width either way"},
        "render": {"jpeg_quality": int(st.s.jpeg_quality), "shadows": bool(st.s.shadows),
                   "keep_model_lights": bool(st.s.keep_model_lights), "density": float(st.s.density),
                   "camera_look": st.look_path, "camera_look_sha256": st.look_sha256,
                   "rig_file": str(Path(st.s.rig_file).relative_to(REPO)),
                   "handheld_model_edits": info or None},
        "lowdim": str((B_DIR / "reference" / key / "obs.npz").relative_to(REPO)),
        "images": {f"{SHORT[r]}_{j}": {"row": int(sc["rows"][r][j]), "file": f"{SHORT[r]}_{j}.png",
                                        "jpeg": f"jpg640_{SHORT[r]}_{j}.jpg"} for r in ROLES for j in (0, 1)},
        "rows": {}, "eraser_pose_O": _r(pose, 6), "table_dz_m": sc["dz_m"],
        "cameras_O_T_cam_cv": {f"{SHORT[r]}_{j}": _r(T, 6) for (r, j), T in cams.items()},
        "cameras_tcp_T_cam_cv": {},
    }
    if st.blur:
        meta["motion_blur"] = dict(blur, rows=blur_rows) if blur else {"applied": False, "note": "static pose"}
    if pose_kind == "mount_start":
        meta["eraser_table_rel"] = {k: round(v, 5) for k, v in sc["start"]["table_rel"].items()}
    for row, pr in per_row.items():
        r = {"q_arm_rad": _r(pr["q"], 7), "finger_qpos_m": round(pr[gap_key] / 2, 6), "gap_m": round(pr[gap_key], 6),
             "recorded_width_m": round(pr["gap"], 6),
             "finger_q_beyond_range": pr["finger_q_beyond_range"], "O_T_tcp": _r(pr["O_T_tcp"], 6),
             "joint_margin_rad": round(pr["joint_margin_rad"], 4),
             "dist_to_q_start_rad": round(pr["dist_to_q_start_rad"], 4), "contacts": pr["contacts"]}
        if "ik_pos_err_mm" in pr:
            r["ik_pos_err_mm"] = round(pr["ik_pos_err_mm"], 4)
            r["ik_ang_err_deg"] = round(pr["ik_ang_err_deg"], 5)
        meta["rows"][str(row)] = r
    for (r, j), T in cams.items():
        row = sc["rows"][r][j]
        meta["cameras_tcp_T_cam_cv"][f"{SHORT[r]}_{j}"] = _r(_inv(per_row[row]["O_T_tcp"]) @ T, 6)
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"folder": folder, "images": images, "alphas": alphas, "cams": cams, "per_row": per_row, "pose": pose,
            "sc": sc, "gap_key": gap_key}


# ------------------------------------------------------------------------------------------------ geometry
def tags_compare(a: dict, b: dict) -> dict:
    out = {}
    for tid in (0, 1):
        if tid in a and tid in b:
            dv, k = HC.match_corners(a[tid], b[tid])
            e = np.linalg.norm(dv, axis=1)
            out[str(tid)] = {"corner_mean_px": round(float(e.mean()), 3), "corner_max_px": round(float(e.max()), 3),
                             "centre_dxdy_px": _r(a[tid].mean(0) - b[tid].mean(0), 3),
                             "size_ratio": round(HC._side(a[tid]) / HC._side(b[tid]), 4)}
        else:
            out[str(tid)] = None
    return out


def _tags_json(t: dict) -> dict:
    return {str(k): _r(v, 3) for k, v in t.items()}


def geometry(st: Study, records: dict) -> dict:
    det = HC.TagDetector()
    real_tags = {k: {(role, j): det(st.real[k]["main" if role == "main" else "uw"][j]) for role in ROLES
                     for j in (0, 1)} for k in st.real}
    kin = st.kin
    sets, summary = {}, {}
    for (pose_kind, key), rec in records.items():
        if pose_kind == "_vram":
            continue
        sc = rec["sc"]
        rows = sc["rows"]
        per = {}
        for role in ROLES:
            for j in (0, 1):
                row = rows[role][j]
                pr = rec["per_row"][row]
                kin.set(pr["q"], pr[rec.get("gap_key", "gap")], rec["pose"])
                Ocam = rec["cams"][(role, j)]
                img = rec["images"][(role, j)]
                alpha = rec["alphas"][(role, j)]
                sim_tags = det(img)
                proj_tags = {tid: st.proj(role, Ocam, P) for tid, P in kin.tag_corners_world().items()}
                tips = kin.tips_world()
                tips_px = {b: _r(st.proj(role, Ocam, p[None])[0], 2) for b, p in tips.items()}
                er_mask = alpha_blob(alpha)
                core = alpha > 0.5
                ca = colour_alpha(img, core if core.any() else None)
                er_colour = alpha_blob(ca) if ca is not None else None
                er_proj = polygon_blob(st.proj(role, Ocam, kin.eraser_points(rec["pose"])))
                rb = st.real_blobs[key][(role, j)]
                rt = real_tags[key][(role, j)]
                nm = f"{SHORT[role]}_{j}"
                per[nm] = {
                    "row": int(row),
                    "tags_detected": _tags_json(sim_tags), "tags_projected": _tags_json(proj_tags),
                    "tags_real_detected": _tags_json(rt),
                    "tags_vs_real": tags_compare(sim_tags, rt) if rt else None,
                    "tags_detected_vs_projected": tags_compare(sim_tags, proj_tags),
                    "tags_projected_vs_real": tags_compare(proj_tags, rt) if rt else None,
                    "fingertips_projected_px": tips_px,
                    "eraser_mask": blob_json(er_mask), "eraser_colour_method": blob_json(er_colour),
                    "eraser_projected": blob_json(er_proj), "eraser_real": blob_json(rb),
                    "eraser_mask_vs_real": blob_compare(er_mask, rb),
                    "eraser_projected_vs_real": blob_compare(er_proj, rb),
                    "eraser_colour_vs_mask": blob_compare(er_colour, er_mask),
                    "eraser_projected_vs_mask": blob_compare(er_proj, er_mask),
                    "_raw": {"tags": sim_tags, "proj_tags": proj_tags, "tips": tips_px, "er": er_mask, "er_proj": er_proj},
                }
        sets[f"{pose_kind}/{key}"] = per
    # pairwise: mount vs handheld (same reference, same image)
    for key, mpose in ((k, mp) for k in st.real for mp in ("mount", "mount_recgap")):
        h, m = sets.get(f"handheld/{key}"), sets.get(f"{mpose}/{key}")
        if not h or not m:
            continue
        for nm in h:
            a, b = m[nm]["_raw"], h[nm]["_raw"]
            m[nm]["vs_handheld"] = {
                "tags_detected": tags_compare(a["tags"], b["tags"]),
                "tags_projected": tags_compare(a["proj_tags"], b["proj_tags"]),
                "eraser_mask": blob_compare(a["er"], b["er"]), "eraser_projected": blob_compare(a["er_proj"], b["er_proj"]),
                "fingertips_dxdy_px": {bd: _r(np.subtract(a["tips"][bd], b["tips"][bd]), 2) for bd in a["tips"]}}
    # nominal handheld (no E7 fix): projection only, at the same arm configuration as the handheld sets
    nominal = {}
    for key in st.real:
        rec = records.get(("handheld", key))
        if rec is None:
            continue
        sc = rec["sc"]
        out = {}
        for role in ROLES:
            for j in (0, 1):
                row = sc["rows"][role][j]
                pr = rec["per_row"][row]
                kin.set(pr["q"], pr["gap"], rec["pose"])
                Tt = kin.tcp_T()
                Onom = Tt @ st.tcp_T_cam["nominal"][role]
                OE7 = Tt @ st.tcp_T_cam["E7"][role]
                pt_n = {tid: st.proj(role, Onom, P) for tid, P in kin.tag_corners_world().items()}
                pt_e = {tid: st.proj(role, OE7, P) for tid, P in kin.tag_corners_world().items()}
                rt = real_tags[key][(role, j)]
                en = polygon_blob(st.proj(role, Onom, kin.eraser_points(rec["pose"])))
                ee = polygon_blob(st.proj(role, OE7, kin.eraser_points(rec["pose"])))
                out[f"{SHORT[role]}_{j}"] = {
                    "tags_projected_vs_real": tags_compare(pt_n, rt) if rt else None,
                    "tags_projected_vs_E7": tags_compare(pt_n, pt_e),
                    "eraser_projected_vs_real": blob_compare(en, st.real_blobs[key][(role, j)]),
                    "eraser_projected_vs_E7": blob_compare(en, ee)}
        nominal[key] = out
    for v in sets.values():
        for nm in v.values():
            nm.pop("_raw", None)
    summary = summarise(sets, nominal)
    return {"sets": sets, "handheld_nominal_projection": nominal, "summary": summary}


def _agg(vals: list) -> dict | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.array(vals, float)
    if a.ndim == 1:
        return {"n": len(a), "mean": round(float(a.mean()), 3), "max": round(float(np.abs(a).max()), 3)}
    return {"n": len(a), "mean": _r(a.mean(0), 3), "max_abs": _r(np.abs(a).max(0), 3)}


def summarise(sets: dict, nominal: dict) -> dict:
    out = {}

    def tagstats(items):
        cm = [t["corner_mean_px"] for it in items if it for t in it.values() if t]
        cx = [t["corner_max_px"] for it in items if it for t in it.values() if t]
        dd = [t["centre_dxdy_px"] for it in items if it for t in it.values() if t]
        sr = [t["size_ratio"] for it in items if it for t in it.values() if t]
        if not cm:
            return None
        cd = np.linalg.norm(np.asarray(dd, float), axis=1)
        return {"n_tags": len(cm), "corner_mean_px": round(float(np.mean(cm)), 3), "corner_max_px": round(float(np.max(cx)), 3),
                "corner_mean_pct": round(float(np.mean(cm)) / 224 * 100, 2),
                "corner_max_pct": round(float(np.max(cx)) / 224 * 100, 2),
                "centre_dxdy_mean_px": _r(np.mean(dd, 0), 3), "centre_dist_mean_px": round(float(cd.mean()), 3),
                "centre_dist_max_px": round(float(cd.max()), 3), "size_ratio_mean": round(float(np.mean(sr)), 4),
                "size_ratio_min_max": _r([np.min(sr), np.max(sr)], 4)}

    def blobstats(items):
        items = [b for b in items if b]
        if not items:
            return None
        dd = np.array([b["centre_dxdy_px"] for b in items])
        sr = np.array([b["size_ratio"] for b in items])
        return {"n": len(items), "centre_dxdy_mean_px": _r(dd.mean(0), 3),
                "centre_dist_mean_px": round(float(np.linalg.norm(dd, axis=1).mean()), 3),
                "centre_dist_max_px": round(float(np.linalg.norm(dd, axis=1).max()), 3),
                "centre_dist_mean_pct": round(float(np.linalg.norm(dd, axis=1).mean()) / 224 * 100, 2),
                "centre_dist_max_pct": round(float(np.linalg.norm(dd, axis=1).max()) / 224 * 100, 2),
                "size_ratio_mean": round(float(sr.mean()), 4), "size_ratio_min_max": _r([sr.min(), sr.max()], 4)}

    for pose in POSES:
        s = {k: v for k, v in sets.items() if k.startswith(pose + "/")}
        if not s:
            continue
        o = {}
        for cam in ("uw", "main"):
            imgs = [v[nm] for v in s.values() for nm in v if nm.startswith(cam)]
            o[cam] = {"tags_detected_vs_real": tagstats([i["tags_vs_real"] for i in imgs]),
                      "tags_projected_vs_real": tagstats([i["tags_projected_vs_real"] for i in imgs]),
                      "tags_detected_vs_projected": tagstats([i["tags_detected_vs_projected"] for i in imgs]),
                      "eraser_mask_vs_real": blobstats([i["eraser_mask_vs_real"] for i in imgs]),
                      "eraser_projected_vs_real": blobstats([i["eraser_projected_vs_real"] for i in imgs]),
                      "eraser_colour_vs_mask": blobstats([i["eraser_colour_vs_mask"] for i in imgs]),
                      "eraser_projected_vs_mask": blobstats([i["eraser_projected_vs_mask"] for i in imgs])}
            if pose in ("mount", "mount_recgap"):
                vh = [i.get("vs_handheld") for i in imgs if i.get("vs_handheld")]
                o[cam]["vs_handheld"] = {
                    "tags_detected": tagstats([v["tags_detected"] for v in vh]),
                    "tags_projected": tagstats([v["tags_projected"] for v in vh]),
                    "eraser_mask": blobstats([v["eraser_mask"] for v in vh]),
                    "eraser_projected": blobstats([v["eraser_projected"] for v in vh]),
                    "fingertips_dxdy_mean_px": _agg([d for v in vh for d in v["fingertips_dxdy_px"].values()]),
                    "fingertips_dist_px": _agg([float(np.linalg.norm(d)) for v in vh
                                                for d in v["fingertips_dxdy_px"].values()])}
        out[pose] = o
    nom = {}
    for cam in ("uw", "main"):
        imgs = [v[nm] for v in nominal.values() for nm in v if nm.startswith(cam)]
        nom[cam] = {"tags_projected_vs_real": tagstats([i["tags_projected_vs_real"] for i in imgs]),
                    "tags_projected_vs_E7": tagstats([i["tags_projected_vs_E7"] for i in imgs]),
                    "eraser_projected_vs_real": blobstats([i["eraser_projected_vs_real"] for i in imgs]),
                    "eraser_projected_vs_E7": blobstats([i["eraser_projected_vs_E7"] for i in imgs])}
    out["handheld_nominal_projection"] = nom
    return out


def kit_check(st: Study) -> dict:
    """The model's finger / holder / tag geometry relative to tcp vs handheld_check's kit geometry (HandheldScene's
    convention: assembly frame, training TCP, sides slid so the tag distance = width + 44.29 mm)."""
    from franka_sim import finger_kit as FK
    kin = st.kin
    hk = FK.load_kit(HC.KIT)
    A_tcp = _T(HC.TCP_AXES, HC.TCP_POS)
    rows = []
    for w in (0.0, 0.052, 0.080, 0.088):
        kin.set(st.q_start, w, kin.eraser_pose(0.55, 0.0, 0.0))
        T_tcp = kin.tcp_T()
        model = {tid: (np.c_[P, np.ones(4)] @ _inv(T_tcp).T)[:, :3] for tid, P in kin.tag_corners_world().items()}
        mt = {b: (_inv(T_tcp) @ np.r_[p, 1])[:3] for b, p in kin.tips_world().items()}
        dist = HC.E7_TAG_DIST(w)
        hh, htips = {}, {}
        for side in hk.sides.values():
            for t in side.tags:
                native = float(t.centre[0]) + np.sign(t.centre[0]) * HC.TAG_OUTWARD_M
                dx = -np.sign(native) * (abs(native) - dist / 2)
                n, up = np.asarray(t.normal, float), np.asarray(t.up, float)
                x = np.cross(up, n)
                c = np.asarray(t.centre, float) + [np.sign(t.centre[0]) * HC.TAG_OUTWARD_M + dx, 0, 0]
                h = MARKER_M / 2
                P = np.array([c - h * x + h * up, c + h * x + h * up, c + h * x - h * up, c - h * x - h * up])
                hh[int(t.id)] = (np.c_[P, np.ones(4)] @ _inv(A_tcp).T)[:, :3]
                htips[int(t.id)] = (_inv(A_tcp) @ np.r_[np.asarray(side.tip, float) + [dx, 0, 0], 1])[:3]
        tag_mm = {tid: float(np.abs(model[tid] - hh[tid]).max() * 1e3) for tid in (0, 1)}
        tag_vec = {tid: _r((model[tid] - hh[tid]).mean(0) * 1e3, 3) for tid in (0, 1)}
        # tips: model body -> tag id on that body (kit side)
        tip_mm = {}
        for b, (side, _) in kin.sides.items():
            tid = int(kin.kit.sides[side].tags[0].id)
            tip_mm[tid] = _r((mt[b] - htips[tid]) * 1e3, 3)
        px = {}
        for cam in ("E7",):
            for role in ROLES:
                Oc = st.tcp_T_cam[cam][role]               # tcp frame as world here
                e = []
                for tid in (0, 1):
                    a = st.proj(role, Oc, model[tid])
                    b = st.proj(role, Oc, hh[tid])
                    if np.isfinite(a).all() and np.isfinite(b).all():
                        e.append(np.linalg.norm(a - b, axis=1).max())
                px[f"{cam}_{SHORT[role]}_max_px"] = round(float(max(e)), 3) if e else None
        rows.append({"width_m": w, "tag_corner_max_mm": {str(k): round(v, 3) for k, v in tag_mm.items()},
                     "tag_mean_offset_model_minus_handheld_mm": {str(k): v for k, v in tag_vec.items()},
                     "fingertip_offset_model_minus_handheld_mm": {str(k): v for k, v in tip_mm.items()},
                     "model_gap_mm": round(float(np.linalg.norm(mt["left_finger"] - mt["right_finger"]) * 1e3), 3),
                     "handheld_gap_mm": round(float(np.linalg.norm(htips[0] - htips[1]) * 1e3), 3),
                     "projected": px})
    # parts: the model's pad / holder meshes are the handheld kit's (same vertices, kit coordinates)
    same = {}
    for pn in ("finger_left", "finger_right", "finger_holder_left", "finger_holder_right"):
        a, b = kin.kit.parts[pn].vertices, hk.parts[pn].vertices
        same[pn] = bool(a.shape == b.shape and np.abs(a - b).max() < 1e-9)
    # the model's tag geoms (textured slabs) vs the kit tag centres (body frame)
    tag_geom = {}
    kin.set(st.q_start, 0.06, kin.eraser_pose(0.55, 0.0, 0.0))
    for gname, body in (("tag_l", "left_finger"), ("tag_r", "right_finger")):
        gid = kin.m.geom(gname).id
        side = kin.sides[body][0]
        t = kin.kit.sides[side].tags[0]
        c = (kin.kit_T(body) @ np.r_[t.centre, 1])[:3]
        tag_geom[gname] = {"aruco_id": int(t.id), "geom_centre_minus_kit_tag_centre_mm": _r((kin.d.geom_xpos[gid] - c) * 1e3, 3)}
    return {"widths": rows, "pad_holder_meshes_identical": same, "tag_geoms": tag_geom,
            "note": "model: kit_franka_v1 meshes placed by the pad geoms, sides at finger_joint = gap / 2 (gap = sum "
                    "of the two joints, 0 at q 0); handheld: iphumi_umi_soft_v1 + tag_outward 1.855 mm, sides slid "
                    "symmetrically to tag distance = width + 44.29 mm (HandheldScene.set_tag_distance)"}


# ------------------------------------------------------------------------------------------------ IK / contacts
def scene_checks(scenes: dict) -> dict:
    """IK residuals, joint margins and contacts (other than eraser-table) of every planned configuration."""
    rows, worst = {}, {"ik_pos_err_mm": 0.0, "ik_ang_err_deg": 0.0, "joint_margin_rad": 9.0}
    in_contact = []
    for key, sc in scenes.items():
        for variant, per_row in (("mount", sc["per_row"]), ("mount_start", sc.get("start", {}).get("per_row"))):
            if not per_row:
                continue
            for row, pr in per_row.items():
                r = {"joint_margin_rad": round(pr["joint_margin_rad"], 4), "contacts": pr["contacts"],
                     "finger_q_beyond_range": pr["finger_q_beyond_range"]}
                if "ik_pos_err_mm" in pr:
                    r["ik_pos_err_mm"] = round(pr["ik_pos_err_mm"], 4)
                    r["ik_ang_err_deg"] = round(pr["ik_ang_err_deg"], 5)
                    worst["ik_pos_err_mm"] = max(worst["ik_pos_err_mm"], r["ik_pos_err_mm"])
                    worst["ik_ang_err_deg"] = max(worst["ik_ang_err_deg"], r["ik_ang_err_deg"])
                worst["joint_margin_rad"] = min(worst["joint_margin_rad"], r["joint_margin_rad"])
                # pad-eraser pairs have a 3.5 mm soft margin: a "contact" there with dist > 0 is not a touch
                touch = [c for c in pr["contacts"] if c[2] <= 0.0]
                if touch:
                    in_contact.append([variant, key, int(row), touch])
                rows[f"{variant}/{key}/{row}"] = r
    return {"worst": worst, "penetrating_contacts": in_contact, "rows": rows}


# ------------------------------------------------------------------------------------------------ contact sheets
def contact_sheet(st: Study, look: str | None = None) -> Path:
    """One sheet per look: a row per reference, columns real | handheld | mount | mount_start, each main_1 + uw_1
    (the newest image of each camera, 224 px)."""
    look = look or st.look_id
    poses = ("real", "handheld", "mount", "mount_start", "mount_start_recgap")
    blank = np.zeros((224, 224, 3), np.uint8)
    rows = []
    head = np.zeros((26, 224 * 2 * len(poses), 3), np.uint8)
    for i, p in enumerate(poses):
        cv2.putText(head, f"{p}: main_1 | uw_1", (i * 448 + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    title = np.zeros((26, 224 * 2 * len(poses), 3), np.uint8)
    cv2.putText(title, f"look {look} ({st.station.name}, model {st.doc['model_sha256'][:12]}, camera_look "
                       f"{(st.look_sha256 or 'none')[:12]}); mount / mount_start = robot gap, *_recgap = recorded gap",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    rows += [title, head]
    for ref in st.refs:
        key = ref["key"]
        R = st.real[key]
        tiles = [R["main"][1], R["uw"][1]]
        for p in poses[1:]:
            d = st.out / look / p / key
            for nm in ("main_1", "uw_1"):
                f = d / f"{nm}.png"
                tiles.append(cv2.imread(str(f))[:, :, ::-1] if f.is_file() else blank)
        row = np.concatenate(tiles, axis=1).copy()
        cv2.putText(row, key, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        rows.append(row)
    img = np.concatenate(rows, axis=0)
    path = st.out / f"contact_{look}.png"
    cv2.imwrite(str(path), img[:, :, ::-1])
    return path


def contact_sheets(st: Study) -> list:
    return [contact_sheet(st)]


# ------------------------------------------------------------------------------------------------ main
def _merge_json(path: Path, look: str, value) -> None:
    doc = json.loads(path.read_text()) if path.is_file() else {}
    doc.setdefault("looks", {})[look] = value
    _write_json_atomic(path, doc)


def write_index(st: Study, records: dict) -> Path:
    """index.json: one row per look / pose / reference (this look's rows replaced), written atomically."""
    path = st.out / "index.json"
    idx = json.loads(path.read_text()) if path.is_file() else {"sets": []}
    keep = [s for s in idx.get("sets", []) if s["look"] != st.look_id]
    for (pose, key), rec in sorted(records.items()):
        row = {"look": st.look_id, "pose": pose, "reference": key,
                     "folder": _rel(rec["folder"]),
                     "obs": _rel(rec["folder"] / "obs.npz"),
                     "station": str(st.station.relative_to(REPO)) if st.station.is_relative_to(REPO) else str(st.station),
                     "model": _rel(st.s.model), "model_sha256": st.doc["model_sha256"],
                     "camera_look": st.look_path, "camera_look_sha256": st.look_sha256,
                     "lowdim": (str((B_DIR / "reference" / key / "obs.npz").relative_to(REPO)) if pose != "robot_trials"
                                else "episode start (eef_pos 0, rot6d identity, width 0.090)")}
        if st.blur:
            row["motion_blur"] = ({"exposure_s": st.blur["exposure_s"], "k": st.blur["k"]}
                                  if POSES.get(pose, ("static",))[0] == "per_row" else None)
        if pose == "robot_trials":
            row["obs_variants"] = {"0.090": row["obs"], "0.080": _rel(rec["folder"] / "obs_w0.080.npz"),
                                   "0.087": _rel(rec["folder"] / "obs_w0.087.npz")}
            row["real_static_sets"] = _rel(rec["folder"] / "real_static_sets.json")
        keep.append(row)
    keep.sort(key=lambda s: (s["look"], s["pose"], s["reference"]))
    looks = sorted({s["look"] for s in keep})
    doc = {"schema": "camera_study.B.sets.index/v3", "looks": looks, "poses": POSE_NOTES, "gripper_map": GRIPPER_MAP,
           "sets": keep,
           "note": "obs.npz = the C4 Observation keys; low-dim identical to the reference (robot_trials: an episode "
                   "start), only the images differ. 2026-11-20 ~21:00: 'mount' / 'mount_start' now use the ROBOT gap; "
                   "the earlier recorded-gap mount sets are 'mount_recgap' / 'mount_start_recgap'."}
    _write_json_atomic(path, doc)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.camera_study.render_sets", description=__doc__.splitlines()[0])
    ap.add_argument("--station", type=Path, default=DEFAULT_STATION)
    ap.add_argument("--out", type=Path, default=OUT, help=f"results folder (default {OUT})")
    ap.add_argument("--look-id", default=None, help="sets folder name (default: the station's id)")
    ap.add_argument("--refit", action="store_true", help="recompute <out>/placement.json")
    ap.add_argument("--only", nargs="*", default=None, help="reference keys (default: all 15)")
    ap.add_argument("--no-robot-trials", action="store_true", help="skip the robot_trials set (robot_trials.py)")
    ap.add_argument("--motion-blur-s", type=float, default=None, metavar="EXPOSURE_S",
                    help="opt-in motion blur: exposure time in s (window centred on each frame's row; default off)")
    ap.add_argument("--motion-blur-k", type=int, default=9, metavar="K", help="sub-frames per blurred frame (default 9)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = Study(a.station, out=out, look_id=a.look_id, motion_blur_s=a.motion_blur_s, motion_blur_k=a.motion_blur_k)
    if st.blur:
        print(f"[sets] motion blur: exposure {st.blur['exposure_s'] * 1e3:.1f} ms, {st.blur['k']} sub-frames", flush=True)
    print(f"[sets] look {st.look_id}: model {st.s.model} sha {st.doc['model_sha256'][:16]} (camera_look "
          f"{st.look_path}, sha {st.look_sha256}); out {out}; real cache {next(iter(st.real.values()))['_cache']}",
          flush=True)
    pl_path = out / "placement.json"
    if a.refit or not pl_path.is_file():
        placement = st.fit_placement()
        placement["kit_check"] = kit_check(st)
        placement["b1_reference_check"] = b1_reference_check(st.real)
        placement["model_sha256"] = st.doc["model_sha256"]
        _write_json_atomic(pl_path, placement)
        print(f"[sets] wrote {pl_path}", flush=True)
    placement = json.loads(pl_path.read_text())
    for i, e in placement["episodes"].items():
        f = e["fits"][e["chosen"]]
        print(f"[sets] ep{int(i):02d} eraser: before rms {e['resid_before']['rms_px']} px -> {e['chosen']} "
              f"dx {f['dx_mm']} dy {f['dy_mm']} mm dyaw {f['dyaw_deg']} deg dz {f['dz_mm']} mm: rms "
              f"{f['resid']['rms_px']} px, centre mean {f['resid']['centre_mean_abs_px']} px", flush=True)
    t1 = time.perf_counter()
    scenes = plan_scenes(st, placement)
    only = set(a.only) if a.only else None
    records = render_all(st, scenes, only)
    t2 = time.perf_counter()
    vram = records.pop("_vram", {})
    geo = geometry(st, records)
    geo["scene_checks"] = scene_checks(scenes)
    geo["vram"] = vram
    geo["timing_s"] = {"setup_fit": round(t1 - t0, 1), "render": round(t2 - t1, 1)}
    geo["model_sha256"] = st.doc["model_sha256"]
    geo["camera_look_sha256"] = st.look_sha256
    _merge_json(out / "geometry.json", st.look_id, geo)
    write_index(st, records)
    sheets = contact_sheets(st)
    if not a.no_robot_trials and not only:
        from sim_deploy.camera_study import robot_trials as RT
        t3 = time.perf_counter()
        rt = RT.run(st, out, refit=a.refit)
        records.update(rt)
        write_index(st, records)
        print(f"[sets] robot_trials: {len(rt)} sets in {time.perf_counter() - t3:.1f} s", flush=True)
    print(json.dumps(geo["summary"], indent=1))
    print(f"[sets] scene checks: {geo['scene_checks']['worst']}, penetrating contacts "
          f"{geo['scene_checks']['penetrating_contacts']}")
    print(f"[sets] {len(records)} sets, contact sheet {sheets[0]}; total {time.perf_counter() - t0:.1f} s "
          f"(render {t2 - t1:.1f} s); VRAM {vram}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
