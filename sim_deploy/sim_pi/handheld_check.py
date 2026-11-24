"""Check G3 (PLAN.md section 6): the sim cameras in the HANDHELD pose against real training frames.

    python -m sim_deploy.sim_pi.handheld_check --out DIR [--zarr ZIP --index 14] [--frames N]

The handheld training gripper is rebuilt from what training itself assumes, then rendered through the sim Pi's full
chain and prepared by the copy's ImagePrep, exactly as the laptop would see it:

* fingers + finger holders: finger kit ``iphumi_umi_soft_v1`` (franka_sim), whose meshes are in the UMI assembly
  frame (x across, the fingers close along x; y toward the fingertips; z up to the top cover), the frame of
  umi-processing's ``rig_geometry_from_cad.py``; each side slides along x to the frame's recorded width;
* ArUco tags DICT_4X4_50 ids 0 (left) / 1 (right), real patterns (16 mm marker on a 20 mm white sticker) on the kit's
  tag frames;
* cameras: training's TCP (fingertip point (0, 201.35, -2.2) mm, TCP axes x across, y down, z forward) and the copy's
  rig file: ``T_cam_tcp`` (IMX219), ``T_imu_cam`` of both cameras (IMX519 relative to the IMX219);
* the rig parts that ride with the cameras (E2's cover, Pi 5, LED and button boards, IMX519/IMU boards; not the
  Franka mount and plate), placed rigidly by E2's IMX219 pose;
* rendering: sim_pi.cameras (native size, measured lens, INTER_AREA to 640x480), JPEG q90 and back, then
  ``ImagePrep.prep_ultrawide`` / ``prep_main`` (224x224).

The reference is the packaged training episode (default PICK_TEST1 ``pick1-v1`` index 14 = episode_000152): its
224x224 ultrawide images (10 Hz) and recorded widths, exported by handheld_reference.py in the torch env.

Width (E7, DESIGN.md D8 revised): umiproc's recorded width = tag x-distance - min_width = the fingertip gap, and the
real stickers sit 1.855 mm further out on each holder than the CAD outline (tag distance = gap + 44.29 mm). Variant
``E7`` places the fingers that way (gap = width, real sticker positions) with the camera at training's T_cam_tcp;
``E7+camera`` also moves the camera by the translation that best explains the real tag corners (``fit_geometry``:
least squares over every frame through the copy's virtual camera, no rendering); ``D8_old`` is wave 2's assumption
(gap = width - 0.0443, CAD tag positions), kept for comparison. ``fit_geometry`` also reports which other freedoms
(tag distance offset, camera rotation, tag size) explain the rest. Tags are detected in both the training and the sim
224 images (4x cubic upscale), corners matched up to the marker's rotation, offsets reported in 224 px.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import sim_deploy.sim_pi.render as R_  # noqa: F401  (EGL + NVIDIA before mujoco)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_deploy.sim_pi import _deploy as D  # noqa: E402
from sim_deploy.sim_pi.cameras import CameraChain  # noqa: E402
from sim_deploy.sim_pi.render import GLRenderer, Lighting, finish_model, prepare_spec  # noqa: E402

REPO = D.REPO
KIT = REPO / "controlSW" / "franka_sim" / "franka_sim" / "assets" / "finger_kits" / "iphumi_umi_soft_v1"
RIG_DIR = REPO / "sim_deploy" / "rig"
DEFAULT_ZARR = Path("~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip").expanduser()
TORCH_PY = Path("~/envs/sim_policy_deploy_torch/bin/python").expanduser()
# Training's TCP in the assembly frame (umi-processing scripts/rig_geometry_from_cad.py TCP_POS, TCP_AXES)
TCP_POS = np.array([0.0, 0.20135, -0.0022])
TCP_AXES = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)
CLOSED_WIDTH_M = 0.0443          # handheld min_width (gripper_range.json): wave 2's D8 took it as gap 0 + 44.3
D8_TAG_OFFSET_M = 0.040577       # CAD tag-centre distance minus fingertip gap (kit: 120 - 79.42 mm; E3's 40.58)
TAG_OUTWARD_M = 0.001855         # E7: the real stickers, per side, outward of the CAD outline (adapter.py Holder)
E7_TAG_DIST = lambda w: w + D8_TAG_OFFSET_M + 2 * TAG_OUTWARD_M   # noqa: E731  gap = width, real stickers
D8_TAG_DIST = lambda w: w - CLOSED_WIDTH_M + D8_TAG_OFFSET_M      # noqa: E731  wave 2's assumption
STICKER_M = 0.020                # engraved sticker outline on the holder (kit tags_pose)
RIG_PARTS = ("cover", "pi5", "led_strip", "button_board", "imx519_board", "imx219_board", "imu_board")
COLORS = {"finger": [0.93, 0.93, 0.92, 1], "holder": [0.22, 0.30, 0.62, 1], "cover": [0.30, 0.32, 0.55, 1],
          "pi5": [0.12, 0.42, 0.20, 1], "board": [0.10, 0.25, 0.30, 1], "table": [0.70, 0.55, 0.36, 1]}


def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    return _T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q


def camera_poses(rig) -> dict[str, np.ndarray]:
    """Assembly-frame poses (OpenCV axes) of both cameras from training's TCP and the rig file."""
    A_tcp = _T(TCP_AXES, TCP_POS)
    A_uw = A_tcp @ _inv(np.asarray(rig.T_cam_tcp, float))
    T_imu_uw = np.asarray(rig.cameras["ultrawide"].T_imu_cam, float)
    T_imu_main = np.asarray(rig.cameras["main"].T_imu_cam, float)
    return {"ultrawide": A_uw, "main": A_uw @ _inv(T_imu_uw) @ T_imu_main}


def tag_texture(dictionary, tag_id: int, px: int = 750) -> np.ndarray:
    """RGB sticker: the 16 mm marker centred on a 20 mm white square (px per side)."""
    inner = int(round(px * 0.016 / STICKER_M))
    marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), inner)
    img = np.full((px, px), 255, np.uint8)
    o = (px - inner) // 2
    img[o:o + inner, o:o + inner] = marker
    return np.ascontiguousarray(np.repeat(img[:, :, None], 3, axis=2))


class HandheldScene:
    """The handheld gripper (kit + tags + rig parts) as a MuJoCo model; ``set_width`` moves the two sides."""

    def __init__(self, rig, chains: dict, tex_dir: Path, tag_scale: float = 1.0, lighting: Lighting | None = None,
                 camera_dt_m=(0.0, 0.0, 0.0), tag_outward_m: float = 0.0):
        """camera_dt_m: a translation of both cameras (and the rig parts with them), IMX219 axes. tag_outward_m: the
        tags sit this much further out (along x, away from the other finger) than the kit's CAD outline."""
        from franka_sim import finger_kit as FK
        self.kit = FK.load_kit(KIT)
        nominal = camera_poses(rig)
        A_uw = nominal["ultrawide"] @ _T(None, np.asarray(camera_dt_m, float))
        self.cam = {"ultrawide": A_uw, "main": A_uw @ _inv(nominal["ultrawide"]) @ nominal["main"]}
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        tex_dir.mkdir(parents=True, exist_ok=True)
        self.sides = {}
        self.tag_native_x = {}
        for side_name, side in self.kit.sides.items():
            body = spec.worldbody.add_body()
            body.name = f"side_{side_name}"
            body.mocap = True                       # moved along x by set_tag_distance (planes need a static body)
            for pn in side.parts:
                part = self.kit.parts[pn]
                mesh = spec.add_mesh()
                mesh.name = pn
                mesh.uservert = part.vertices.astype(np.float32).ravel().tolist()
                mesh.userface = part.faces.astype(np.int32).ravel().tolist()
                g = body.add_geom()
                g.name, g.type, g.meshname = pn, mujoco.mjtGeom.mjGEOM_MESH, pn
                g.rgba = COLORS["finger" if part.role == "contact" else "holder"]
                g.contype = g.conaffinity = 0
            for t in side.tags:
                png = tex_dir / f"tag_{t.id}.png"
                cv2.imwrite(str(png), tag_texture(d, t.id))
                tex = spec.add_texture()
                tex.name, tex.file, tex.type = f"tag{t.id}", str(png), mujoco.mjtTexture.mjTEXTURE_2D
                mat = spec.add_material()
                mat.name = f"tag{t.id}"
                mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag{t.id}"
                mat.texrepeat = [1, 1]
                mat.texuniform = False
                mat.specular = 0.2
                n, up = np.asarray(t.normal, float), np.asarray(t.up, float)
                x = np.cross(up, n)
                Rt = np.column_stack([x, up, n])              # plane: z = tag normal, y = tag up
                g = body.add_geom()
                g.name, g.type, g.material = f"tag_{t.id}", mujoco.mjtGeom.mjGEOM_PLANE, f"tag{t.id}"
                half = 0.5 * STICKER_M * tag_scale
                g.size = [half, half, 0.001]
                c = np.asarray(t.centre, float) + [np.sign(t.centre[0]) * tag_outward_m, 0.0, 0.0]
                g.pos = c + 0.0002 * n
                g.quat = _quat(Rt)
                g.contype = g.conaffinity = 0
                self.tag_native_x[t.id] = float(c[0])
            self.sides[side_name] = side
        # rig parts, rigid with the IMX219 (E2's hand-frame meshes, through E2's IMX219 pose)
        from sim_deploy.rig.add_rig import hand_T_cam, load_rig
        rdoc = load_rig(RIG_DIR)
        A_hand = self.cam["ultrawide"] @ _inv(hand_T_cam(rdoc, "cam_imx219"))
        self.A_hand = A_hand
        self.E2_uw_T_main = _inv(hand_T_cam(rdoc, "cam_imx219")) @ hand_T_cam(rdoc, "cam_imx519")
        self.own = {}
        for pn in RIG_PARTS:
            f = RIG_DIR / "assets" / f"{pn}.stl"
            mesh = spec.add_mesh()
            mesh.name, mesh.file = f"rig_{pn}", str(f)
            g = spec.worldbody.add_geom()
            g.name, g.type, g.meshname = f"rig_{pn}", mujoco.mjtGeom.mjGEOM_MESH, f"rig_{pn}"
            g.pos, g.quat = A_hand[:3, 3], _quat(A_hand[:3, :3])
            g.rgba = COLORS["cover" if pn == "cover" else "pi5" if pn == "pi5" else "board"]
            g.contype = g.conaffinity = 0
        tb = spec.worldbody.add_geom()                         # a table 30 cm below the fingertips' plane
        tb.name, tb.type, tb.size = "table", mujoco.mjtGeom.mjGEOM_PLANE, [1.5, 1.5, 0.05]
        tb.pos, tb.rgba = [0, 0.35, -0.12], COLORS["table"]
        tb.contype = tb.conaffinity = 0
        W = max(c.max_view[0] for c in chains.values())
        H = max(c.max_view[1] for c in chains.values())
        self.lighting = lighting or Lighting(centre=(0.0, 0.2, 0.0), shadow_extent_m=1.5, direction=(0.1, 0.25, -1.0))
        prepare_spec(spec, self.lighting, W, H, keep_model_lights=False)
        self.m = spec.compile()
        finish_model(self.m, self.lighting)
        self.d = mujoco.MjData(self.m)
        for role, gname in (("main", "rig_imx519_board"), ("ultrawide", "rig_imx219_board")):
            self.own[role] = self.m.geom(gname).id
        self.W, self.H = W, H

    def set_tag_distance(self, dist_m: float) -> None:
        """Slide both sides symmetrically (along x) so the two tag centres are dist_m apart."""
        for side, tid in (("left", 0), ("right", 1)):
            native = self.tag_native_x[tid]
            dx = -np.sign(native) * (abs(native) - dist_m / 2)
            self.d.mocap_pos[self.m.body_mocapid[self.m.body(f"side_{side}").id]] = [dx, 0.0, 0.0]
        mujoco.mj_kinematics(self.m, self.d)

    def tips_world(self) -> np.ndarray:
        """The two kit fingertip points at the current slide positions (assembly frame)."""
        out = []
        for side, s in self.sides.items():
            x = self.d.mocap_pos[self.m.body_mocapid[self.m.body(f"side_{side}").id]][0]
            out.append(np.asarray(s.tip, float) + [x, 0, 0])
        return np.array(out)


class TagDetector:
    def __init__(self, scale: int = 4):
        self.s = scale
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                           cv2.aruco.DetectorParameters())

    def __call__(self, rgb: np.ndarray) -> dict[int, np.ndarray]:
        big = cv2.resize(rgb, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_CUBIC)
        c, ids, _ = self.det.detectMarkers(cv2.cvtColor(big, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return {}
        return {int(i): (cc.reshape(4, 2) + 0.5) / self.s - 0.5 for i, cc in zip(ids.ravel(), c)}


def match_corners(sim: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-corner offsets sim - real (4, 2) for the cyclic order that fits best (marker rotation), and that shift."""
    best = None
    for k in range(4):
        dv = np.roll(sim, k, axis=0) - real
        e = np.linalg.norm(dv, axis=1).mean()
        if best is None or e < best[0]:
            best = (e, dv, k)
    return best[1], best[2]


def _side(a: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(a[i] - a[(i + 1) % 4]) for i in range(4)]))


FIT_CASES = (("nominal", (0, 0, 0, 0, 0, 0, 0, 0)), ("distance offset", (0, 0, 0, 0, 0, 0, 1, 0)),
             ("camera translation", (1, 1, 1, 0, 0, 0, 0, 0)), ("distance + camera translation", (1, 1, 1, 0, 0, 0, 1, 0)),
             ("distance + camera pose", (1, 1, 1, 1, 1, 1, 1, 0)), ("distance + camera pose + tag size", (1,) * 8))


def tag_corners_3d(tag, dist_m: float, size_m: float) -> np.ndarray:
    """ArUco corner order (marker top-left, top-right, bottom-right, bottom-left; marker top = the tag's up)."""
    n, up = np.asarray(tag.normal, float), np.asarray(tag.up, float)
    x = np.cross(up, n)
    c = np.asarray(tag.centre, float).copy()
    c[0] = np.sign(c[0]) * dist_m / 2
    h = size_m / 2
    return np.array([c - h * x + h * up, c + h * x + h * up, c + h * x - h * up, c - h * x - h * up])


def fit_geometry(rig, prep, kit, real_tags: dict, widths: np.ndarray, base=E7_TAG_DIST) -> dict:
    """Least-squares explanations of the real tag corners (224 px) through the copy's virtual camera.
    Parameters: camera translation (camera axes, m), rotation vector (camera axes), tag-centre distance offset
    relative to ``base(width)`` (m), tag size (m)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    virt = prep.ultrawide.virt
    A0 = camera_poses(rig)["ultrawide"]
    tags = {t.id: t for sd in kit.sides.values() for t in sd.tags}
    frames = sorted(real_tags)

    def resid(p):
        A = A0 @ _T(Rotation.from_rotvec(p[3:6]).as_matrix(), p[0:3])
        Ai = _inv(A)
        out = []
        for j in frames:
            d = base(widths[j]) + p[6]
            for tid in (0, 1):
                P = tag_corners_3d(tags[tid], d, p[7])
                pc = (np.c_[P, np.ones(4)] @ Ai.T)[:, :3]
                uv = prep.ultrawide.virtual_to_out(virt.project(pc)[0])
                out.append((uv - real_tags[j][tid]).ravel())
        return np.concatenate(out)

    p_nom = np.array([0, 0, 0, 0, 0, 0, 0, 0.016])
    scale = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.01, 0.004])
    res = {}
    for name, free in FIT_CASES:
        mask = np.array(free, bool)
        p = p_nom.copy()
        if mask.any():
            def f(x, mask=mask):
                q = p_nom.copy()
                q[mask] = x
                return resid(q)
            p[mask] = least_squares(f, p_nom[mask], x_scale=scale[mask]).x
        e = np.linalg.norm(resid(p).reshape(-1, 2), axis=1)
        res[name] = {"camera_dt_mm": (p[:3] * 1e3).round(1).tolist(), "camera_rot_deg": np.degrees(p[3:6]).round(2).tolist(),
                     "tag_distance_offset_vs_base_mm": round(float(p[6] * 1e3), 1), "tag_size_mm": round(float(p[7] * 1e3), 2),
                     "corner_mean_px": round(float(e.mean()), 2), "corner_max_px": round(float(e.max()), 2)}
    return {"frames": len(frames), "fits": res}



def run(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = out / f"reference_{Path(a.zarr).stem}_{a.index}.npz"
    if not ref.is_file():
        cmd = [str(a.ref_python), str(Path(__file__).with_name("handheld_reference.py")), str(a.zarr), str(a.index),
               str(ref)]
        print("[g3] exporting the training reference:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    R = np.load(ref)
    rig = D.load_live_rig(D.DEFAULT_RIG_FILE)
    prep = D.ImagePrep(rig)
    chains = {r: CameraChain(r, rig) for r in ("main", "ultrawide")}
    detect = TagDetector()
    real_tags = [detect(im) for im in R["uw"]]
    usable = [j for j, t in enumerate(real_tags) if {0, 1} <= set(t)]
    if a.frames and len(usable) > a.frames:
        usable = [usable[i] for i in np.linspace(0, len(usable) - 1, a.frames).round().astype(int)]
    print(f"[g3] {R['name']}: {len(R['uw'])} training ultrawide images, both tags detected in "
          f"{sum({0, 1} <= set(t) for t in real_tags)}; using {len(usable)}", flush=True)
    from franka_sim import finger_kit as FK
    both = {j: real_tags[j] for j in range(len(real_tags)) if {0, 1} <= set(real_tags[j])}
    fit = fit_geometry(rig, prep, FK.load_kit(KIT), both, R["width_m"])
    fit["base"] = "E7: tag distance = width + %.2f mm (gap = width, real stickers)" % ((D8_TAG_OFFSET_M + 2 * TAG_OUTWARD_M) * 1e3)
    dt_cam = np.array(fit["fits"]["camera translation"]["camera_dt_mm"]) / 1e3
    variants = {"E7": dict(tag_scale=1.0, dist=E7_TAG_DIST, tag_outward_m=TAG_OUTWARD_M),
                "E7+camera": dict(tag_scale=1.0, dist=E7_TAG_DIST, tag_outward_m=TAG_OUTWARD_M, camera_dt_m=dt_cam),
                "D8_old": dict(tag_scale=1.0, dist=D8_TAG_DIST)}
    report = {"episode": str(R["name"]), "frames": [int(j) for j in usable], "fit": fit, "variants": {}}
    for vname, v in variants.items():
        scene = HandheldScene(rig, chains, out / "textures", tag_scale=v["tag_scale"],
                              camera_dt_m=v.get("camera_dt_m", (0.0, 0.0, 0.0)),
                              tag_outward_m=v.get("tag_outward_m", 0.0))
        for r, c in chains.items():
            c.set_hide([scene.own[r]])
        gl = GLRenderer(scene.m, scene.W, scene.H)
        if vname == "E7":
            report["camera_check"] = _camera_consistency(scene)
            report["gl"] = gl.gl_renderer
        rows = []
        saved = 0
        for j in usable:
            w = float(R["width_m"][j])
            scene.set_tag_distance(v["dist"](w))
            gl.update(scene.d)
            imgs = {}
            for role, c in chains.items():
                A = scene.cam[role]
                rgb640 = c.frame(gl, A[:3, :3], A[:3, 3])
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb640, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                dec = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                imgs[role] = (dec, prep.prep_main(dec) if role == "main" else prep.prep_ultrawide(dec))
            st = detect(imgs["ultrawide"][1])
            row = {"frame": int(j), "width_m": w, "sim_ids": sorted(st)}
            for tid in (0, 1):
                if tid not in st:
                    continue
                dv, k = match_corners(st[tid], real_tags[j][tid])
                row[f"tag{tid}"] = {"corner_mean_px": float(np.linalg.norm(dv, axis=1).mean()),
                                    "corner_max_px": float(np.linalg.norm(dv, axis=1).max()),
                                    "centre_dxdy_px": (st[tid].mean(0) - real_tags[j][tid].mean(0)).tolist(),
                                    "side_ratio": _side(st[tid]) / _side(real_tags[j][tid]), "shift": k}
            rows.append(row)
            if saved < 3 and j in usable[:: max(1, len(usable) // 3)]:
                _save_compare(out / f"g3_{vname}_frame{j:03d}.png", R["uw"][j], imgs["ultrawide"][1], real_tags[j], st,
                              R["main"][j], imgs["main"][1])
                cv2.imwrite(str(out / f"g3_{vname}_frame{j:03d}_sim_uw640.png"), imgs["ultrawide"][0][:, :, ::-1])
                cv2.imwrite(str(out / f"g3_{vname}_frame{j:03d}_sim_main640.png"), imgs["main"][0][:, :, ::-1])
                saved += 1
        gl.close()
        report["variants"][vname] = {"summary": _summarise(rows), "rows": rows}
    (out / "g3_report.json").write_text(json.dumps(report, indent=1))
    _print(report)
    print(f"[g3] wrote {out / 'g3_report.json'} and the g3_*.png side-by-sides", flush=True)
    return 0


def _camera_consistency(scene: HandheldScene) -> dict:
    """IMX519 relative to the IMX219: training's rig (T_imu_cam) vs E2's CAD (both cameras on one cover)."""
    T_tr = _inv(scene.cam["ultrawide"]) @ scene.cam["main"]
    T_e2 = scene.E2_uw_T_main
    dR = T_tr[:3, :3].T @ T_e2[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return {"uw_T_main_training_mm": (T_tr[:3, 3] * 1e3).round(2).tolist(),
            "uw_T_main_E2_mm": (T_e2[:3, 3] * 1e3).round(2).tolist(),
            "translation_diff_mm": float(np.linalg.norm(T_tr[:3, 3] - T_e2[:3, 3]) * 1e3), "rotation_diff_deg": ang}


def _summarise(rows: list[dict]) -> dict:
    out = {}
    for tid in (0, 1):
        vals = [r[f"tag{tid}"] for r in rows if f"tag{tid}" in r]
        if not vals:
            out[f"tag{tid}"] = None
            continue
        cm = np.array([v["corner_mean_px"] for v in vals])
        dxy = np.array([v["centre_dxdy_px"] for v in vals])
        out[f"tag{tid}"] = {"n": len(vals), "detected_in_sim": len(vals), "corner_mean_px": round(float(cm.mean()), 2),
                            "corner_median_px": round(float(np.median(cm)), 2),
                            "corner_max_px": round(float(max(v["corner_max_px"] for v in vals)), 2),
                            "centre_dxdy_mean_px": dxy.mean(0).round(2).tolist(),
                            "side_ratio_sim_over_real": round(float(np.mean([v["side_ratio"] for v in vals])), 3),
                            "shifts": sorted({v["shift"] for v in vals})}
    out["frames"] = len(rows)
    out["both_tags_in_sim"] = sum(1 for r in rows if {0, 1} <= set(r["sim_ids"]))
    return out


def _print(rep: dict) -> None:
    print(f"[g3] GL {rep.get('gl')}; IMX519 vs IMX219, training rig vs E2 CAD: {rep['camera_check']}")
    print(f"[g3] projection fits over {rep['fit']['frames']} frames (corners in 224 px):")
    for name, f in rep["fit"]["fits"].items():
        print(f"       {name:36s} mean {f['corner_mean_px']:6.2f} max {f['corner_max_px']:6.2f} | tag distance "
              f"{f['tag_distance_offset_vs_base_mm']:+6.1f} mm vs E7, camera dt {f['camera_dt_mm']} mm rot "
              f"{f['camera_rot_deg']} deg, tag {f['tag_size_mm']} mm")
    for vname, v in rep["variants"].items():
        s = v["summary"]
        print(f"[g3] variant {vname}: {s['frames']} frames, both tags found in sim {s['both_tags_in_sim']}")
        for tid in (0, 1):
            t = s[f"tag{tid}"]
            if t is None:
                print(f"       tag {tid}: never detected in sim")
                continue
            print(f"       tag {tid}: corner offset mean {t['corner_mean_px']} / median {t['corner_median_px']} / max "
                  f"{t['corner_max_px']} px (224), centre (dx, dy) {t['centre_dxdy_mean_px']} px, size sim/real "
                  f"{t['side_ratio_sim_over_real']}")


def _save_compare(path: Path, real_uw, sim_uw, real_t, sim_t, real_main, sim_main) -> None:
    def marks(img, tags, col):
        im = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST).copy()
        for c in tags.values():
            cv2.polylines(im, [((c + 0.5) * 2 - 0.5).round().astype(np.int32).reshape(-1, 1, 2)], True, col, 1)
        return im
    diff = np.clip(np.abs(real_uw.astype(np.int16) - sim_uw.astype(np.int16)) * 2, 0, 255).astype(np.uint8)
    blend = (0.5 * real_uw + 0.5 * sim_uw).astype(np.uint8)
    row1 = np.concatenate([marks(real_uw, real_t, (255, 0, 0)), marks(sim_uw, sim_t, (0, 255, 0)),
                           marks(blend, {**{k + 10: v for k, v in real_t.items()}}, (255, 0, 0)),
                           cv2.resize(diff, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)], axis=1)
    for c in sim_t.values():                                   # sim corners onto the blend tile too
        cv2.polylines(row1[:, 896:1344], [((c + 0.5) * 2 - 0.5).round().astype(np.int32).reshape(-1, 1, 2)], True,
                      (0, 255, 0), 1)
    big = lambda x: cv2.resize(x, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)  # noqa: E731
    mb = (0.5 * real_main + 0.5 * sim_main).astype(np.uint8)
    row2 = np.concatenate([big(real_main), big(sim_main), big(mb), np.zeros_like(big(mb))], axis=1)
    lab = np.zeros((26, row1.shape[1], 3), np.uint8)
    for i, s in enumerate(("training UW (red: tags)", "sim UW (green)", "blend", "|diff| x2")):
        cv2.putText(lab, s, (i * 448 + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    img = np.concatenate([lab, row1, row2], axis=0)
    cv2.imwrite(str(path), img[:, :, ::-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.sim_pi.handheld_check", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    ap.add_argument("--index", type=int, default=14)
    ap.add_argument("--frames", type=int, default=0, help="use at most this many frames (0: every usable one)")
    ap.add_argument("--ref-python", type=Path, default=TORCH_PY, help="python with zarr + JPEG-XL codecs")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
