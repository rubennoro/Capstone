"""W2 "robot_trials": the sim at the real arm's trial starts (arm at q_start, fingertip gap 0.040), the eraser placed by
fitting its blob in the REAL MOUNT images of each trial's static start sets (PLAN.md 11, coordinator addendum
"real-arm mount images").

    unset PYTHONPATH; cd ~/Capstone-main-sim
    ~/envs/sim_deploy/bin/python -m sim_deploy.camera_study.robot_trials [--station sim_deploy/stations/<id>.yaml]
        [--out sim_evals/_pipeline/W2/sets] [--look-id ID] [--refit]

(``render_sets`` calls ``run(st, out)`` itself after its own sets, so normally this is not run by hand.)

Real data (read-only): ``~/Capstone/data/<task>/policy/<run>/deploy/<date>_<HH-MM>_trial<N>_probe_live/
NNNNNN_{main,ultrawide}_{0,1}.png`` = the policy's four 224 inputs on the real arm (after the real Pi's JPEG and the
deployment's ImagePrep). The arm is at the start pose for the first ~45 sets of every trial (motion starts at set
46-75; ``STATIC_SETS`` = 0..39 are used). Trial 4 of run_022 had a black object (not the eraser): not fitted, not
rendered (listed in ``robot_trials_fit.json`` as skipped).

Fit (``fit_trial``), per trial, once per ``--out`` (``robot_trials_fit.json``, reused by every look; ``--refit``):
the arm at ``q_start`` (stations/start_pose.yaml), fingertip gap 0.040 m, the model's own mount cameras; the eraser
flat on the table at x, y, yaw with a free box length / width (thickness 12 mm). Its predicted blob per image = the
box hull projected through the copy's camera models into the 224 image (rasterised at 8x, INTER_AREA), times
(1 - the occluder coverage): an exact render of everything but the table, floor and eraser at that arm pose (the
fingers occlude the eraser in several trials). Real blob: the yellow core (HSV hue 18-40, S >= 110, V >= 90), then the
soft alpha of ``render_sets.colour_alpha``. Least squares (soft_l1, 1.5 px) of centroid, sqrt area and sqrtm of the
second moments over sets 0, 13, 26, 39 x main_0, main_1, uw_0, uw_1. Initial x, y from the IMX519 blob centroid's
ray on the table, yaw from its major axis.

Sets (``robot_trials/<trial>/``): rendered through the sim Pi chain as ``render_sets`` does (the look's camera_look,
JPEG, ImagePrep), with the eraser at the fitted pose AND the fitted length / thickness, its width always
sim_deploy/eraser.py WIDTH_M (30 mm, as every sim eraser; the fitted width is reported only) (visual only: the private
render model's eraser geom is resized; physics is not involved). ``obs.npz`` = the C4 keys with the low-dim of an episode start:
eef_pos zeros, eef_rot6d and eef_rot6d_wrt_start (1, 0, 0, 0, 1, 0), gripper_width_m 0.090 (= gripper_map^-1(0.040));
``obs_w0.080.npz`` / ``obs_w0.087.npz`` the same images with width 0.080 / 0.087. ``real_static_sets.json`` lists the
real counterpart (the static sets of the trial). ``meta.json`` as render_sets' sets.

Geometry check (``geometry`` in ``robot_trials_fit.json``): the ArUco tags detected in the real mount IMX219 images
vs the sim's mount projection at q_start / 0.040 (the real robot's camera-to-finger geometry vs the model's E2 CAD).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import zlib
from pathlib import Path

from sim_deploy.camera_study import render_sets as RS   # noqa: E402  (EGL + NVIDIA before mujoco: via render_sets)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_deploy import eraser as eraser_mod  # noqa: E402
from sim_deploy.sim_pi.cameras import CameraChain  # noqa: E402
from sim_deploy.sim_pi.model_file import CAMERAS, OWN_BOARD, RenderModel, read_doc  # noqa: E402
from sim_deploy.sim_pi.render import Lighting, finish_model, prepare_spec  # noqa: E402

DATA = Path("~/Capstone/data").expanduser()
TRIAL_GLOB = "*/policy/*/deploy/*_probe_live"
STATIC_SETS = tuple(range(40))
FIT_SETS = (0, 13, 26, 39)
GAP_M = 0.040                                   # the robot's fingertip gap at the start (fully open)
WIDTHS = (0.090, 0.080, 0.087)                  # obs widths: gripper_map^-1(0.040) = 0.090, plus two checks
SKIP = {"PICK_TEST1_run_022_trial4": "black object, not the eraser"}
IMG = {"main": "main", "ultrawide": "ultrawide"}


def trials() -> list[dict]:
    out = []
    for d in sorted(glob.glob(str(DATA / TRIAL_GLOB))):
        d = Path(d)
        task, run = d.parts[-5], d.parts[-3]
        trial = d.name.split("_")[2]                      # <date>_<HH-MM>_trial<N>_probe_live
        tid = f"{task}_{'_'.join(run.split('_')[:2])}_{trial}"
        notes = (d / "NOTES.md").read_text() if (d / "NOTES.md").is_file() else ""
        out.append({"id": tid, "folder": str(d), "task": task, "run": run, "trial": trial,
                    "n_sets": len(list(d.glob("*_main_1.png"))), "notes_first_line": notes.splitlines()[0] if notes else ""})
    return out


def real_img(folder, i: int, role: str, j: int) -> np.ndarray:
    return cv2.imread(str(Path(folder) / f"{i:06d}_{IMG[role]}_{j}.png"))[:, :, ::-1].copy()


def yellow_alpha(rgb: np.ndarray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = (hsv[..., k].astype(int) for k in range(3))
    core = ((H >= 18) & (H <= 40) & (S >= 110) & (V >= 90)).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if core.sum() < 4:
        return None
    return RS.colour_alpha(rgb, core)


def hull_alpha(uv: np.ndarray, ss: int = 8) -> np.ndarray:
    """Coverage (224x224 float) of the convex hull of points uv (224 px, pixel centres at integers)."""
    from scipy.spatial import ConvexHull
    uv = uv[np.isfinite(uv).all(axis=1)]
    out = np.zeros((224 * ss, 224 * ss), np.uint8)
    if len(uv) >= 3:
        h = uv[ConvexHull(uv).vertices]
        pts = np.round(((h + 0.5) * ss - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(out, [pts], 255, lineType=cv2.LINE_8, shift=4)
    return cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0


def load_mount_render_model(scene_xml: Path, lighting: Lighting, W: int, H: int, keep_model_lights: bool,
                            eraser_box=None) -> RenderModel:
    """model_file.load_render_model's steps on a spec whose eraser geom has the given visual size (L, W, T m)."""
    scene_xml = Path(scene_xml)
    doc, sha = read_doc(scene_xml)
    ref = mujoco.MjModel.from_xml_path(str(scene_xml))
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    if eraser_box is not None:
        spec.geom("eraser").size = (np.asarray(eraser_box, float) / 2).tolist()
    prepare_spec(spec, lighting, W, H, keep_model_lights=keep_model_lights)
    m = spec.compile()
    finish_model(m, lighting)
    if m.nq != ref.nq or not np.array_equal(m.jnt_qposadr, ref.jnt_qposadr):
        raise RuntimeError("render model's qpos layout differs from the file's")
    cam_id = {r: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n) for r, n in CAMERAS.items()}
    own = {r: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for r, g in OWN_BOARD.items()}
    own = {r: (g if g >= 0 else None) for r, g in own.items()}
    rm = RenderModel(scene_xml, doc, sha, m, mujoco.MjData(m), cam_id, own, np.asarray(doc["home"]["qpos"], float))
    rm.set_qpos(rm.home_qpos)
    return rm


def occluder_alpha(rend, role: str, qpos: np.ndarray) -> np.ndarray:
    """224 coverage of everything but the world-body geoms (table, floor, eraser): what can hide the eraser."""
    rm, gl = rend.rm, rend.gl
    rm.set_qpos(qpos)
    gl.update(rm.data)
    Rw, pw = rm.camera_pose(role)
    m = rm.model
    world = {m.geom("table").id, m.geom("eraser").id}
    world |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
    scn = gl.scn
    for i in range(scn.ngeom):
        g = scn.geoms[i]
        occ = not (g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in world)
        a = float(g.rgba[3])
        g.rgba[:] = [1.0, 1.0, 1.0, a] if occ else [0.0, 0.0, 0.0, a]
        g.emission = 1.0 if occ else 0.0
        g.specular = 0.0
        g.reflectance = 0.0
        g.matid = -1
    sky, sh = scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX], scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]
    scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    try:
        mask = rend._chain_frame(role, Rw, pw)
    finally:
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = sky
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = sh
    m224 = rend.prep.prep_main(mask) if role == "main" else rend.prep.prep_ultrawide(mask)
    return m224[..., 1].astype(np.float64) / 255.0


def _ray_to_table(st, role, O_cam, uv224, z):
    """Table point (z) seen at a 224 pixel (IMX519 only: inverse letterbox + the physical model)."""
    from policy_deploy.frames._dataproc.imageprep import letterbox_params
    size, top, _, left, _ = letterbox_params(st.proj.main_res)
    s = 224 / size
    u = (uv224[0] + 0.5) / s - 0.5 - left
    v = (uv224[1] + 0.5) / s - 0.5 - top
    rays, ok = st.proj.main.unproject(np.array([[u, v]]))
    d = O_cam[:3, :3] @ rays[0]
    o = O_cam[:3, 3]
    t = (z - o[2]) / d[2]
    return o + t * d


def fit_trial(st, tr: dict, occ: dict) -> dict:
    from scipy.optimize import least_squares
    kin = st.kin
    T_m = float(kin.box[2])
    kin.set(st.q_start, GAP_M, kin.eraser_pose(5.0, 0.0, 0.0))
    cams = {r: kin.cam_T(r) for r in RS.ROLES}
    images = [(i, role, j) for i in FIT_SETS for role in RS.ROLES for j in (0, 1)]
    real = {}
    for i, role, j in images:
        al = yellow_alpha(real_img(tr["folder"], i, role, j))
        real[(i, role, j)] = RS.alpha_blob(al) if al is not None else None
    if real[(0, "main", 1)] is None:
        return {"error": "no yellow blob in the IMX519 image"}
    # initial guess: IMX519 centroid ray on the table, yaw from its major axis
    b = real[(0, "main", 1)]
    c = _ray_to_table(st, "main", cams["main"], b["c"], T_m / 2)
    w, V = np.linalg.eigh(b["C"])
    ax = V[:, 1] * np.sqrt(w[1])
    c2 = _ray_to_table(st, "main", cams["main"], b["c"] + ax, T_m / 2)
    yaw0 = float(np.degrees(np.arctan2(c2[1] - c[1], c2[0] - c[0])))
    x0 = np.array([c[0] * 1e3, c[1] * 1e3, yaw0, 100.0, 50.0])

    def blobs(p):
        pose = kin.eraser_pose(p[0] * 1e-3, p[1] * 1e-3, p[2])
        box = np.array([p[3] * 1e-3, p[4] * 1e-3, T_m])
        P = kin.eraser_points(pose, box=box)
        out = {}
        for role in RS.ROLES:
            a = hull_alpha(st.proj(role, cams[role], P)) * (1.0 - occ[role])
            out[role] = RS.alpha_blob(a)
        return out

    def resid(p):
        bb = blobs(p)
        r = []
        for k in images:
            if real[k] is None or bb[k[1]] is None:
                r.append(np.full(6, 20.0))
                continue
            r.append(RS.blob_resid(bb[k[1]], real[k]))
        return np.concatenate(r)

    f = least_squares(resid, x0, loss="soft_l1", f_scale=1.5, x_scale=[5, 5, 2, 5, 3], diff_step=[1e-3, 1e-2, 1e-2, 5e-3, 5e-3])
    e = f.fun.reshape(-1, 6)
    per_cam = {}
    for role in RS.ROLES:
        sel = [n for n, k in enumerate(images) if k[1] == role]
        per_cam[role] = RS._resid_summary(e[sel])
    # the model's box (its station's size_m) at the fitted centre / yaw: how far off its blob is
    bm = {}
    pose = kin.eraser_pose(f.x[0] * 1e-3, f.x[1] * 1e-3, f.x[2])
    for role in RS.ROLES:
        a = hull_alpha(st.proj(role, cams[role], kin.eraser_points(pose))) * (1.0 - occ[role])
        rb = real[(0, role, 1)]
        bm[role] = RS.blob_compare(RS.alpha_blob(a), rb)
    yaw = RS._wrap90(float(f.x[2]))
    pose = kin.eraser_pose(f.x[0] * 1e-3, f.x[1] * 1e-3, yaw)
    pose[2] = kin.table_z + T_m / 2
    return {"x0": RS._r(x0, 2), "x_mm": round(float(f.x[0]), 2), "y_mm": round(float(f.x[1]), 2),
            "yaw_deg": round(yaw, 2), "box_LWT_mm": [round(float(f.x[3]), 1), round(float(f.x[4]), 1), round(T_m * 1e3, 1)],
            "eraser_pose_O": RS._r(pose, 6), "resid": RS._resid_summary(e), "resid_per_camera": per_cam,
            "model_box_at_fitted_pose_vs_real_set0": bm,
            "real_blobs_set0": {role: RS.blob_json(real[(0, role, 1)]) for role in RS.ROLES},
            "status": int(f.status), "nfev": int(f.nfev)}


def tag_check(st, trs: list[dict]) -> dict:
    """Real mount IMX219 tags (sets 0, 20, 39 of every fitted trial) vs the model's mount projection at q_start, 0.040."""
    det = RS.HC.TagDetector()
    kin = st.kin
    kin.set(st.q_start, GAP_M, kin.eraser_pose(5.0, 0.0, 0.0))
    out = {}
    for role in RS.ROLES:
        proj = {tid: st.proj(role, kin.cam_T(role), P) for tid, P in kin.tag_corners_world().items()}
        rows = []
        for tr in trs:
            for i in (0, 20, 39):
                t = det(real_img(tr["folder"], i, role, 1))
                if t:
                    rows.append(RS.tags_compare(proj, t))
        cm = [v["corner_mean_px"] for r in rows for v in r.values() if v]
        dd = [v["centre_dxdy_px"] for r in rows for v in r.values() if v]
        sr = [v["size_ratio"] for r in rows for v in r.values() if v]
        ids = sorted({k for r in rows for k, v in r.items() if v})
        out[role] = None if not cm else {
            "n_tags": len(cm), "ids": ids, "corner_mean_px": round(float(np.mean(cm)), 3),
            "corner_max_px": round(float(np.max([v["corner_max_px"] for r in rows for v in r.values() if v])), 3),
            "centre_dxdy_mean_px_sim_minus_real": RS._r(np.mean(dd, 0), 3),
            "size_ratio_sim_over_real": round(float(np.mean(sr)), 4)}
    out["note"] = "sim = the model's mount cameras (E2 CAD) at q_start, gap 0.040; real = probe_live sets (224)"
    return out


def fit_all(st, out: Path, refit: bool = False) -> dict:
    path = out / "robot_trials_fit.json"
    if path.is_file() and not refit:
        doc = json.loads(path.read_text())
        if doc.get("model_sha256_physics_note") and doc.get("trials"):
            return doc
    trs = trials()
    chains = {r: CameraChain(r, st.rig, density=st.s.density) for r in RS.ROLES}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    rm = load_mount_render_model(st.s.model, Lighting(shadows=st.s.shadows), W, H, st.s.keep_model_lights)
    rend = RS.Renderer(rm, chains, st.s, st.prep, None)
    try:
        q = RS.qpos_for(st.kin, st.q_start, GAP_M, st.kin.eraser_pose(5.0, 0.0, 0.0))
        occ = {r: occluder_alpha(rend, r, q) for r in RS.ROLES}
    finally:
        rend.close()
    doc = {"schema": "w2.robot_trials.fit/v1", "gap_m": GAP_M, "static_sets": [STATIC_SETS[0], STATIC_SETS[-1]],
           "fit_sets": list(FIT_SETS), "q_start_rad": RS._r(st.q_start, 7),
           "model_sha256": st.doc["model_sha256"],
           "model_sha256_physics_note": "fit uses the model's kinematics / cameras only (same in every look)",
           "trials": {}, "skipped": {}}
    for tr in trs:
        if tr["id"] in SKIP:
            doc["skipped"][tr["id"]] = {"why": SKIP[tr["id"]], **tr}
            continue
        t0 = time.perf_counter()
        r = fit_trial(st, tr, occ)
        r.update(tr)
        r["fit_s"] = round(time.perf_counter() - t0, 1)
        doc["trials"][tr["id"]] = r
        print(f"[robot_trials] {tr['id']}: eraser O ({r.get('x_mm')}, {r.get('y_mm')}) mm yaw {r.get('yaw_deg')} box "
              f"{r.get('box_LWT_mm')} mm: rms {r.get('resid', {}).get('rms_px')} px, centre mean "
              f"{r.get('resid', {}).get('centre_mean_abs_px')} px", flush=True)
    doc["geometry"] = tag_check(st, [t for t in trs if t["id"] not in SKIP])
    RS._write_json_atomic(path, doc)
    return doc


def render_box_m(tr: dict) -> np.ndarray:
    """The rendered eraser box (L, W, T m): the fitted length and thickness, the one sim width (sim_deploy/eraser.py)."""
    L, _, T = (float(v) * 1e-3 for v in tr["box_LWT_mm"])
    return np.array(eraser_mod.size_m(L, T))


def lowdim(width: float) -> dict:
    r6 = np.array([1, 0, 0, 0, 1, 0], np.float32)
    return {"eef_pos": np.zeros((2, 3), np.float32), "eef_rot6d": np.stack([r6, r6]),
            "eef_rot6d_wrt_start": np.stack([r6, r6]), "gripper_width_m": np.full((2, 1), width, np.float32)}


def render(st, out: Path, fit: dict) -> dict:
    """Renders robot_trials for this look; returns {('robot_trials', trial id): record} for render_sets' index."""
    chains = {r: CameraChain(r, st.rig, density=st.s.density) for r in RS.ROLES}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    records = {}
    rm = load_mount_render_model(st.s.model, Lighting(shadows=st.s.shadows), W, H, st.s.keep_model_lights)
    rend = RS.Renderer(rm, chains, st.s, st.prep, st.cam_look)
    eid = rm.model.geom("eraser").id
    size0 = rm.model.geom_size[eid].copy()
    vram = RS._vram()
    try:
        for tid, tr in fit["trials"].items():
            if "eraser_pose_O" not in tr:
                continue
            box = render_box_m(tr)
            rm.model.geom_size[eid] = box / 2                  # visual only (this private render model)
            rm.model.geom_rbound[eid] = float(np.linalg.norm(box / 2))
            q = RS.qpos_for(st.kin, st.q_start, GAP_M, tr["eraser_pose_O"])
            folder = out / st.look_id / "robot_trials" / tid
            folder.mkdir(parents=True, exist_ok=True)
            imgs, cams = {}, {}
            for role in RS.ROLES:
                for j in (0, 1):
                    rng = np.random.default_rng([zlib.crc32(tid.encode()), j, RS.ROLES.index(role)])
                    img, jpg, _, _ = rend.frame(role, q, rng)
                    nm = f"{RS.SHORT[role]}_{j}"
                    cv2.imwrite(str(folder / f"{nm}.png"), img[:, :, ::-1])
                    (folder / f"jpg640_{nm}.jpg").write_bytes(jpg)
                    imgs[(role, j)] = img
                    Rw, pw = rm.camera_pose(role)
                    cams[nm] = RS._r(RS._T(Rw, pw), 6)
            _write_trial(st, out, tid, tr, folder, imgs, cams)
            records[("robot_trials", tid)] = {"folder": folder}
        vram_end = RS._vram()
    finally:
        rm.model.geom_size[eid] = size0
        rend.close()
    print(f"[robot_trials] {st.look_id}: VRAM {vram} after the context, {vram_end} at the end", flush=True)
    return records


def _write_trial(st, out, tid, tr, folder, imgs, cams) -> None:
    """obs.npz (+ width variants), real_static_sets.json, meta.json of one trial."""
    ims = {"main": np.stack([imgs[("main", 0)], imgs[("main", 1)]]),
           "ultrawide": np.stack([imgs[("ultrawide", 0)], imgs[("ultrawide", 1)]])}
    for k, w in enumerate(WIDTHS):
        name = "obs.npz" if k == 0 else f"obs_w{w:.3f}.npz"
        np.savez_compressed(folder / name, **lowdim(w), **ims)
    RS._write_json_atomic(folder / "real_static_sets.json", {
        "folder": tr["folder"], "static_sets": list(STATIC_SETS),
        "files": "NNNNNN_{main,ultrawide}_{0,1}.png (224, post-ImagePrep); same low-dim as obs.npz"})
    meta = {"schema": "w2.robot_trials.set/v1", "trial": tid, "real_folder": tr["folder"], "look": st.look_id,
            "station": RS._rel(st.station), "model": RS._rel(st.s.model), "model_sha256": st.doc["model_sha256"],
            "camera_look": st.look_path, "camera_look_sha256": st.look_sha256,
            "tool_sha256": {"robot_trials.py": RS.sha256_file(__file__), "render_sets.py": RS.sha256_file(RS.__file__)},
            "pose": "robot_trials", "q_arm_rad": RS._r(st.q_start, 7), "gap_m": GAP_M,
            "eraser_pose_O": tr["eraser_pose_O"], "eraser_visual_box_mm": RS._r(render_box_m(tr) * 1e3, 1),
            "eraser_fitted_box_mm": tr["box_LWT_mm"],
            "eraser_note": ("visual-only size (render model only): length / thickness from the fit to the real mount "
                            "images, width sim_deploy/eraser.py WIDTH_M (every sim eraser)"),
            "obs": {"obs.npz": WIDTHS[0], **{f"obs_w{w:.3f}.npz": w for w in WIDTHS[1:]}},
            "lowdim": "episode start: eef_pos 0, eef_rot6d = eef_rot6d_wrt_start = (1,0,0,0,1,0)",
            "images": "main = [main_0, main_1], ultrawide = [uw_0, uw_1]; static arm (both rows the same scene)",
            "cameras_O_T_cam_cv": cams, "strip_h": RS.STRIP_H}
    RS._write_json_atomic(folder / "meta.json", meta)


def contact_sheet(st, out: Path, fit: dict) -> Path:
    rows = []
    for tid, tr in fit["trials"].items():
        d = out / st.look_id / "robot_trials" / tid
        if not (d / "main_1.png").is_file():
            continue
        t = [real_img(tr["folder"], 10, "main", 1), real_img(tr["folder"], 10, "ultrawide", 1),
             cv2.imread(str(d / "main_1.png"))[:, :, ::-1], cv2.imread(str(d / "uw_1.png"))[:, :, ::-1]]
        bl = [(0.5 * t[0] + 0.5 * t[2]).astype(np.uint8), (0.5 * t[1] + 0.5 * t[3]).astype(np.uint8)]
        row = np.concatenate(t + bl, axis=1).copy()
        cv2.putText(row, tid, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        rows.append(row)
    head = np.zeros((26, 224 * 6, 3), np.uint8)
    cv2.putText(head, f"robot_trials [{st.look_id}]: real main | real uw | sim main | sim uw | blend main | blend uw",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    img = np.concatenate([head] + rows, axis=0)
    p = out / f"contact_{st.look_id}_robot_trials.png"
    cv2.imwrite(str(p), img[:, :, ::-1])
    return p


def run(st, out: Path, refit: bool = False) -> dict:
    fit = fit_all(st, out, refit)
    recs = render(st, out, fit)
    contact_sheet(st, out, fit)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.camera_study.robot_trials", description=__doc__.splitlines()[0])
    ap.add_argument("--station", type=Path, default=RS.DEFAULT_STATION)
    ap.add_argument("--out", type=Path, default=RS.OUT)
    ap.add_argument("--look-id", default=None)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--fit-only", action="store_true")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = RS.Study(a.station, out=out, look_id=a.look_id)
    if a.fit_only:
        fit_all(st, out, a.refit)
        return 0
    recs = run(st, out, a.refit)
    _index_rows(st, out, recs)
    print(f"[robot_trials] {len(recs)} sets for look {st.look_id}")
    return 0


def _index_rows(st, out: Path, recs: dict) -> None:
    """Standalone run: put this look's robot_trials rows into <out>/index.json, keeping the look's other rows
    (render_sets.write_index replaces all rows of a look, so the others are added back)."""
    path = out / "index.json"
    old = json.loads(path.read_text())["sets"] if path.is_file() else []
    other = [s for s in old if s["look"] == st.look_id and s["pose"] != "robot_trials"]
    RS.write_index(st, recs)
    doc = json.loads(path.read_text())
    doc["sets"] = sorted(doc["sets"] + other, key=lambda s: (s["look"], s["pose"], s["reference"]))
    RS._write_json_atomic(path, doc)


if __name__ == "__main__":
    sys.exit(main())
