"""W2 step 1 "look": build the realistic-look assets in ``sim_deploy/look/assets/`` from the real PICK_TEST1 data.

Run from ``~/Capstone-main-sim`` with ``unset PYTHONPATH``. Reads ``~/data`` only (read-only)::

    # 1. decode the real IMX519 frames (PyAV rgb24, as umiproc; light env)
    ~/envs/sim_policy_deploy/bin/python -m sim_deploy.look.make_assets frames --work DIR
    # 2. the table texture (sim_deploy env)
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.make_assets table --work DIR [--out sim_deploy/look/assets]
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.make_assets tile --work DIR    # re-cut the tile only
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.make_assets eraser --work DIR [--out sim_deploy/look/assets]
    # 3. the workspace mesh (sim_deploy env; needs step 2's table_real.json for the colour match)
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.make_assets mesh --work DIR [--out sim_deploy/look/assets]
    # 4. the lab backdrop walls behind the desk (sim_deploy env; the B1 references' IMX219 640 frames)
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.make_assets backdrop --work DIR [--out sim_deploy/look/assets]

``frames``: for every episode of the package (``pick1-v1``, 30 episodes) the IMX519 1600x1200 frames of the rows
0, 12, 24, ... before the first grasp starts (``placement.grasp_rows``), mapped row -> frame exactly as the B1
reference tool (``camera_study/reference.py _Episode``: umiproc's grid, piece trim, mid-exposure times, nearest
frame) -> ``DIR/frames/ep<ii>_r<row>.png`` + ``DIR/frames/index.json``.

``table``: each frame is rectified onto the table plane O z 0 (top-down, ``MM_PER_PX``; column = O x, row = -O y)
through the copy's IMX519 model at 1600x1200, camera pose = the package TCP (tag frame) x training's handheld
``T_cam_tcp`` with E7's camera fix (``render_sets.handheld_tcp_T_cam``, the same pose the sim's handheld sets use)
-> O by ``layout.json T_O_tag``. Masked out: the image below the fingertip line, the eraser (colour) + a margin for
its shadow, non-table colours (keyboard, cables: far from the frame's robust table colour in Lab), saturated
pixels. Each frame's exposure gain to the median is divided out (auto exposure), each frame is re-registered to the
median by a translation in the table plane (phase correlation; the shifts are reported: the pose consistency),
then the per-texel median over all frames is the real table at its real place. A ``TILE_M`` rectangle with full
coverage is cut out and made seamless by cross-fading a ``SEAM_M`` band with the matching band one period away
(only that band is a blend) -> ``table_real.png`` + ``table_real.json`` (its O placement, size, colour, checks).
Scale check, independent of the texture: the eraser (127 x 51 mm) measured in frames rectified at its top face
(z 12 mm) by its colour blob.

``mesh``: umiproc's workspace reconstruction (``~/data/umi/tasks/PICK_TEST1/analysis/mesh_3d.glb``, scene frame =
the SLAM world turned gravity-up with the tag centre at 0; glTF y-up) -> ``workspace_mesh.obj`` in the O frame
(T_O_mesh = layout ``T_O_tag`` x inv(tx_slam_tag) x inv(T_scene_W), applied to the vertices; the station's pose is
then the identity) + ``workspace_mesh.png`` (the atlas halved, colour-matched: per-channel gains so the mesh's own
table faces match ``table_real.png`` over the same O area, i.e. the mesh texture in IMX519 colours like the table
texture). Faces dropped: the table's own faces (within ``TABLE_BAND_M`` of the fitted table plane: the sim's
textured table is drawn there; the mesh table carries the 160 mm map tag where the eraser lies), anything below
the table, anything within ``ROBOT_CLEAR_M`` of the robot base axis (O z axis), and anything within ``ARM_PAD_M``
of the arm / hand / rig / finger geoms at the start pose and the home pose (the cameras' near field).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TASK = Path("~/data/umi/tasks/PICK_TEST1").expanduser()
DATASET = Path("~/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip").expanduser()
LAYOUT = REPO / "sim_evals" / "_camera_study" / "B" / "layout.json"
ASSETS = REPO / "sim_deploy" / "look" / "assets"
MM_PER_PX = 0.5                                # texture resolution
ERASER_MM = (127.0, 51.0, 12.0)
RECT_BOX_O = ((0.15, 1.00), (-0.45, 0.45))     # O x, O y range rectified
ROW_STEP = 12                                  # 60 Hz rows between frames (5 Hz)
FINGER_LINE = 0.52                             # image rows below this fraction of the height: fingers / their shadow
TILE_M = (0.50, 0.20)                          # tile [O x (along the strips), O y]
SEAM_M = 0.04                                  # cross-faded band at each wrap seam
MIN_FRAMES = 15                                # every tile texel seen by at least this many frames
TILE_CENTRE_O = (0.50, 0.05)                   # tile chosen nearest this O point (the start pose's view)
TABLE_BAND_M = 0.012                           # mesh faces this close to the table plane are the table
ROBOT_CLEAR_M = 0.30                           # mesh faces within this of the robot base axis are dropped
#: W2 (2026-11-21): mesh faces over the WORK AREA of the desk (O x, y box) lower than WORK_CLEAR_H_M above the table
#: plane are dropped: they are the mapping session's objects / tag edges / reconstruction noise (dark blobs next to the
#: eraser in the handheld IMX519 renders), not part of the demo scene (the real frames show bare wood there).
WORK_CLEAR_BOX_O = ((0.20, -0.28), (0.90, 0.25))
WORK_CLEAR_H_M = 0.06
UW_ROWS_FRAC = 0.42                            # backdrop: the upper part of each IMX219 frame (above the table)
ARM_PAD_M = 0.08                               # mesh faces this close to the arm's geoms (start pose, home) are dropped


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- frames (light env)


def cmd_frames(a) -> dict:
    from sim_deploy.camera_study import reference as RF
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode

    RF._ensure_deploy_path()
    out = Path(a.work) / "frames"
    out.mkdir(parents=True, exist_ok=True)
    pkg = RF._package_episodes(DATASET)
    train_rig = RF._train_rig()
    index = []
    for i, pe in enumerate(pkg):
        e = RF._Episode(pe["folder"], pe["slot"], pe["pi_name"], int(pe["n_steps"]), train_rig)
        ep = load_episode(DATASET, i)
        g = grasp_rows(ep["widths"])
        rows = list(range(0, max(1, int(0.85 * g["start"])), ROW_STEP))
        rows = [r for r in rows if abs(float(e.main_dt_ms[r])) <= 25.0]
        fidx = [int(e.main_idx[r]) for r in rows]
        fr = RF._decode(e.video("main"), fidx)
        for r, f in zip(rows, fidx):
            name = f"ep{i:02d}_r{r:03d}.png"
            cv2.imwrite(str(out / name), cv2.cvtColor(fr[f], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            index.append({"index": i, "slot": pe["slot"], "pi_name": pe["pi_name"], "row": r, "frame": f,
                          "main_dt_ms": round(float(e.main_dt_ms[r]), 3), "file": name,
                          "grasp_start_row": int(g["start"])})
        print(f"[frames] ep{i:02d} {pe['slot']}: rows {rows}", flush=True)
    doc = {"dataset": str(DATASET), "row_step": ROW_STEP, "mapping": "camera_study/reference.py _Episode (umiproc)",
           "decode": "PyAV rgb24", "frames": index}
    (out / "index.json").write_text(json.dumps(doc, indent=1))
    return {"n_frames": len(index), "dir": str(out)}


# ----------------------------------------------------------------------------------------------------- table


def eraser_mask(rgb: np.ndarray) -> np.ndarray:
    """The lime eraser in IMX519 frames (OpenCV HSV hue 38-70 of 180 with S > 45: about (160, 215, 145) RGB, S ~85;
    the table is nearly neutral, S < 20), largest blob, holes filled."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 38) & (hsv[..., 0] <= 70) & (hsv[..., 1] > 45) & (hsv[..., 2] > 70)).astype(np.uint8)
    k = max(3, int(round(min(rgb.shape[:2]) / 200)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool)
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob = (lab == j).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(blob, cnts, -1, 1, cv2.FILLED)
    return blob.astype(bool)


def _rect_grid(box=RECT_BOX_O, mm=MM_PER_PX):
    (x0, x1), (y0, y1) = box
    xs = x0 + (np.arange(int(round((x1 - x0) * 1000 / mm))) + 0.5) * mm / 1000.0
    ys = y1 - (np.arange(int(round((y1 - y0) * 1000 / mm))) + 0.5) * mm / 1000.0
    return xs, ys


def rectify_maps(model, T_O_cam, z, wh, box=RECT_BOX_O, mm=MM_PER_PX, shift_px=(0.0, 0.0)):
    """remap maps of the top-down image of the plane O z = ``z`` (column = O x increasing right, row = O y
    decreasing downward, texel centres) from an image of size ``wh`` taken by ``model`` at ``T_O_cam`` (OpenCV
    axes). ``shift_px``: the texture grid shifted by (dcol, drow) texels (registration)."""
    xs, ys = _rect_grid(box, mm)
    xs = xs + shift_px[0] * mm / 1000.0
    ys = ys - shift_px[1] * mm / 1000.0
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    Ti = _inv(T_O_cam)
    Pc = P @ Ti[:3, :3].T + Ti[:3, 3]
    uv, ok = model.project(Pc)
    ok = np.asarray(ok, bool) & (Pc[:, 2] > 0.05)
    mx = np.where(ok, uv[:, 0], -1e4).reshape(X.shape).astype(np.float32)
    my = np.where(ok, uv[:, 1], -1e4).reshape(X.shape).astype(np.float32)
    w, h = wh
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return mx, my, valid


def _lab(rgb_f):
    return cv2.cvtColor(np.clip(rgb_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def table_pixel_mask(rgb: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Pixels of ``rgb`` (inside ``base``) whose Lab colour is close to the frame's robust table colour: drops the
    keyboard, cables, dark objects, the eraser's grey foam, saturated highlights."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    v = lab[base]
    if len(v) < 1000:
        return np.zeros(rgb.shape[:2], bool)
    med = np.median(v, 0)
    mad = np.median(np.abs(v - med), 0) * 1.4826 + 1.0
    d = np.abs(lab - med) / mad
    ok = base & (d[..., 0] < 4.0) & (d[..., 1] < 4.0) & (d[..., 2] < 4.0) & (rgb.max(-1) < 252)
    ok = cv2.erode(ok.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return ok


def _shift_phase(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> tuple[float, float, float]:
    """Translation of ``a`` relative to ``b`` (both gray float, masked by ``m``), texels, by phase correlation on the
    high-passed images (grain) within the bounding box of ``m``."""
    ys, xs = np.nonzero(m)
    if len(ys) < 20000:
        return 0.0, 0.0, 0.0
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    A, B, M = a[r0:r1, c0:c1], b[r0:r1, c0:c1], m[r0:r1, c0:c1].astype(np.float32)

    def hp(x):
        x = np.where(M > 0, x, 0)
        lo = cv2.GaussianBlur(x, (0, 0), 6) / np.maximum(cv2.GaussianBlur(M, (0, 0), 6), 1e-3)
        return (x - lo) * M
    Mw = cv2.GaussianBlur(M, (0, 0), 8)
    (dx, dy), resp = cv2.phaseCorrelate(hp(B).astype(np.float64), hp(A).astype(np.float64), Mw.astype(np.float64))
    return float(dx), float(dy), float(resp)


def cmd_table(a) -> dict:
    import sim_deploy.sim_pi.render  # noqa: F401  (EGL vendor before mujoco, imported by render_sets)
    from sim_deploy.camera_study.render_sets import E7_DT_M, handheld_tcp_T_cam
    from sim_deploy.scene.zarr_lowdim import load_episode
    from sim_deploy.sim_pi import _deploy as D

    work = Path(a.work)
    scratch = work / "table"
    scratch.mkdir(parents=True, exist_ok=True)
    fdoc = json.loads((work / "frames" / "index.json").read_text())
    rig = D.load_live_rig(D.DEFAULT_RIG_FILE)
    native = D.physical_model(rig, "main").scaled(1600, 1200)
    tcp_T_cam = handheld_tcp_T_cam(rig, E7_DT_M)["main"]
    T_O_tag = np.asarray(json.loads(LAYOUT.read_text())["T_O_tag"], float)
    xs, ys = _rect_grid()
    H, W = len(ys), len(xs)
    eps = {}
    frames = fdoc["frames"]
    if a.max_frames:
        frames = frames[:: max(1, len(frames) // a.max_frames)]
    rep = {"n_frames": len(frames), "mm_per_px": MM_PER_PX, "rect_box_O": RECT_BOX_O, "camera":
           "copy's IMX519 physical model at 1600x1200; pose = package TCP x render_sets.handheld_tcp_T_cam (E7)",
           "frames": []}

    # pass 1: rectify (uint8 + mask, cropped to the valid bounding box to keep memory low)
    recs = []
    er_checks = []
    for k, fr in enumerate(frames):
        i, row = int(fr["index"]), int(fr["row"])
        if i not in eps:
            eps[i] = load_episode(DATASET, i)
        rgb = cv2.cvtColor(cv2.imread(str(work / "frames" / fr["file"])), cv2.COLOR_BGR2RGB)
        T_O_cam = T_O_tag @ eps[i]["poses"][row] @ tcp_T_cam
        h, w = rgb.shape[:2]
        base = np.ones((h, w), bool)
        base[int(FINGER_LINE * h):] = False
        em = eraser_mask(rgb)
        if em.any():
            em_d = cv2.dilate(em.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))).astype(bool)
        else:
            em_d = em
        ok = table_pixel_mask(rgb, base & ~em_d) & ~em_d
        mx, my, valid = rectify_maps(native, T_O_cam, 0.0, (w, h))
        img = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        okr = cv2.remap(ok.astype(np.uint8) * 255, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        okr &= valid
        rr, cc = np.nonzero(okr)
        if len(rr) < 20000:
            continue
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        recs.append({"fr": fr, "T": T_O_cam, "box": (r0, r1, c0, c1), "img": img[r0:r1, c0:c1].copy(),
                     "ok": okr[r0:r1, c0:c1].copy(), "cam_h": float(T_O_cam[2, 3])})
        # scale check: the eraser's top face (z = 12 mm) rectified, its colour blob measured
        if em.sum() > 2000 and not em[int(FINGER_LINE * h):].any():
            ys_e, xs_e = np.nonzero(em)
            box_e = None
            mx2, my2, v2 = rectify_maps(native, T_O_cam, ERASER_MM[2] / 1000.0, (w, h))
            er = cv2.remap(rgb, mx2, my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            m2 = eraser_mask(er) & v2
            if m2.sum() > 500:
                cnts, _ = cv2.findContours(m2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                c = max(cnts, key=cv2.contourArea)
                (cx, cy), (bw, bh), ang = cv2.minAreaRect(c)
                touches = (xs_e.min() < 3 or ys_e.min() < 3 or xs_e.max() > w - 4)
                er_checks.append({"file": fr["file"], "length_mm": round(max(bw, bh) * MM_PER_PX, 1),
                                  "width_mm": round(min(bw, bh) * MM_PER_PX, 1),
                                  "area_mm2": round(float(m2.sum()) * MM_PER_PX ** 2, 0),
                                  "cam_height_m": round(float(T_O_cam[2, 3]), 3),
                                  "touches_image_edge": bool(touches),
                                  "centre_O_m": [round(float(xs[0] + (cx - 0.5) * MM_PER_PX / 1000), 4),
                                                 round(float(ys[0] - (cy - 0.5) * MM_PER_PX / 1000), 4)]})
                del box_e
        if k % 20 == 0:
            print(f"[table] rectified {k + 1}/{len(frames)}", flush=True)

    def stack_median(recs, gains, shifts=None, min_n=3):
        """Per-texel median over the frames (memory: one row band at a time)."""
        med = np.full((H, W, 3), np.nan, np.float32)
        cnt = np.zeros((H, W), np.int32)
        band = 200
        for b0 in range(0, H, band):
            b1 = min(H, b0 + band)
            vals, oks = [], []
            for j, r in enumerate(recs):
                r0, r1, c0, c1 = r["box"]
                if r1 <= b0 or r0 >= b1:
                    continue
                img = r["img"].astype(np.float32) * gains[j]
                ok = r["ok"]
                if shifts is not None and (shifts[j][0] or shifts[j][1]):
                    Mt = np.float32([[1, 0, -shifts[j][0]], [0, 1, -shifts[j][1]]])
                    img = cv2.warpAffine(img, Mt, (c1 - c0, r1 - r0), flags=cv2.INTER_LINEAR)
                    ok = cv2.warpAffine(ok.astype(np.uint8), Mt, (c1 - c0, r1 - r0), flags=cv2.INTER_NEAREST) > 0
                full = np.full((b1 - b0, W, 3), np.nan, np.float32)
                fo = np.zeros((b1 - b0, W), bool)
                s0, s1 = max(b0, r0), min(b1, r1)
                full[s0 - b0:s1 - b0, c0:c1] = np.where(ok[s0 - r0:s1 - r0, :, None], img[s0 - r0:s1 - r0], np.nan)
                fo[s0 - b0:s1 - b0, c0:c1] = ok[s0 - r0:s1 - r0]
                vals.append(full)
                oks.append(fo)
            if not vals:
                continue
            V = np.stack(vals)
            n = np.stack(oks).sum(0)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m = np.nanmedian(V, axis=0)
            m[n < min_n] = np.nan
            med[b0:b1] = m
            cnt[b0:b1] = n
        return med, cnt

    def gains_to(med, recs, shifts=None):
        g = []
        for j, r in enumerate(recs):
            r0, r1, c0, c1 = r["box"]
            M = med[r0:r1, c0:c1]
            img = r["img"].astype(np.float32)
            if shifts is not None and (shifts[j][0] or shifts[j][1]):
                Mt = np.float32([[1, 0, -shifts[j][0]], [0, 1, -shifts[j][1]]])
                img = cv2.warpAffine(img, Mt, (c1 - c0, r1 - r0), flags=cv2.INTER_LINEAR)
            both = r["ok"] & np.isfinite(M[..., 0])
            if both.sum() < 5000:
                g.append(np.ones(3, np.float32))
                continue
            g.append((np.median(M[both], 0) / np.maximum(np.median(img[both], 0), 1)).astype(np.float32))
        return g

    ones = [np.ones(3, np.float32)] * len(recs)
    med0, cnt0 = stack_median(recs, ones)
    gains = gains_to(med0, recs)
    med1, cnt1 = stack_median(recs, gains)
    # registration: each frame vs the median (translation in the table plane)
    shifts, resp = [], []
    g1 = cv2.cvtColor(np.nan_to_num(med1).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    have1 = np.isfinite(med1[..., 0])
    for j, r in enumerate(recs):
        r0, r1, c0, c1 = r["box"]
        a_ = cv2.cvtColor(r["img"], cv2.COLOR_RGB2GRAY).astype(np.float32) * float(np.mean(gains[j]))
        m_ = r["ok"] & have1[r0:r1, c0:c1]
        dx, dy, rs = _shift_phase(a_, g1[r0:r1, c0:c1], m_)
        if rs < 0.05 or np.hypot(dx, dy) > 30:
            dx, dy = 0.0, 0.0
        shifts.append((dx, dy))
        resp.append(rs)
    med2, cnt2 = stack_median(recs, gains, shifts)
    gains = gains_to(med2, recs, shifts)
    med, cnt = stack_median(recs, gains, shifts)

    # agreement of each frame with the final median, before / after registration
    agree = []
    for j, r in enumerate(recs):
        r0, r1, c0, c1 = r["box"]
        M = med[r0:r1, c0:c1]
        both = r["ok"] & np.isfinite(M[..., 0])
        img = r["img"].astype(np.float32) * gains[j]
        Mt = np.float32([[1, 0, -shifts[j][0]], [0, 1, -shifts[j][1]]])
        imgs = cv2.warpAffine(img, Mt, (c1 - c0, r1 - r0), flags=cv2.INTER_LINEAR)
        oks = cv2.warpAffine(r["ok"].astype(np.uint8), Mt, (c1 - c0, r1 - r0), flags=cv2.INTER_NEAREST) > 0
        both_s = oks & np.isfinite(M[..., 0])
        agree.append({"file": r["fr"]["file"], "cam_height_m": round(r["cam_h"], 3),
                      "gain_rgb": np.round(gains[j], 3).tolist(),
                      "shift_mm": [round(shifts[j][0] * MM_PER_PX, 2), round(shifts[j][1] * MM_PER_PX, 2)],
                      "phase_resp": round(resp[j], 3),
                      "mad_before": round(float(np.median(np.abs(img[both] - M[both]))), 2) if both.any() else None,
                      "mad_after": round(float(np.median(np.abs(imgs[both_s] - M[both_s]))), 2) if both_s.any() else None})
    rep["frames"] = agree
    sh = np.array([np.hypot(*s) for s in shifts]) * MM_PER_PX
    rep["registration_shift_mm"] = {"median": round(float(np.median(sh)), 2), "p90": round(float(np.percentile(sh, 90)), 2),
                                    "max": round(float(sh.max()), 2), "n_zeroed": int(sum(1 for s in shifts if s == (0.0, 0.0)))}
    rep["n_frames_used"] = len(recs)
    rep["eraser_check"] = er_checks
    good = [e for e in er_checks if not e["touches_image_edge"]]
    if good:
        L = np.array([e["length_mm"] for e in good])
        Wd = np.array([e["width_mm"] for e in good])
        rep["eraser_check_summary"] = {"n": len(good), "length_mm_median": round(float(np.median(L)), 1),
                                       "length_mm_p10_p90": np.round(np.percentile(L, [10, 90]), 1).tolist(),
                                       "width_mm_median": round(float(np.median(Wd)), 1),
                                       "width_mm_p10_p90": np.round(np.percentile(Wd, [10, 90]), 1).tolist(),
                                       "true_mm": list(ERASER_MM[:2])}

    have = cnt >= 3
    vis = np.nan_to_num(med).clip(0, 255).astype(np.uint8)
    vis[~have] = 0
    cv2.imwrite(str(scratch / "table_mosaic.png"), vis[:, :, ::-1])
    cv2.imwrite(str(scratch / "table_count.png"), np.clip(cnt * 4, 0, 255).astype(np.uint8))
    cv2.imwrite(str(scratch / "table_mosaic_reg0.png"),
                np.nan_to_num(med1).clip(0, 255).astype(np.uint8)[:, :, ::-1])

    cv2.imwrite(str(scratch / "table_count16.png"), np.clip(cnt, 0, 65535).astype(np.uint16))
    rep["tile"] = cut_tile(vis, cnt, Path(a.out), scratch)
    rep["mosaic_coverage_m2"] = round(float(have.sum()) * (MM_PER_PX / 1000) ** 2, 4)
    (scratch / "table_report.json").write_text(json.dumps(rep, indent=1))
    (out / "table_real.json").write_text(json.dumps({k: rep[k] for k in ("tile", "eraser_check_summary",
                                                                         "registration_shift_mm", "n_frames_used")
                                                     if k in rep}, indent=1))
    return rep


def cmd_eraser(a) -> dict:
    """The real eraser's top-face colour as the IMX519 records it: median of the eraser core (mask eroded 9 px)
    over every decoded frame where the eraser is fully in view above the finger line -> ``eraser_real.json``."""
    work = Path(a.work)
    fdoc = json.loads((work / "frames" / "index.json").read_text())
    cols, n_px = [], []
    for fr in fdoc["frames"]:
        rgb = cv2.cvtColor(cv2.imread(str(work / "frames" / fr["file"])), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        m = eraser_mask(rgb)
        ys, xs = np.nonzero(m)
        if m.sum() < 8000 or ys.max() >= FINGER_LINE * h or xs.min() < 5 or ys.min() < 5 or xs.max() > w - 6:
            continue
        core = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        cols.append(np.median(rgb[core].astype(np.float64), 0))
        n_px.append(int(core.sum()))
    C = np.array(cols)
    rep = {"n_frames": len(C), "median_rgb": np.round(np.median(C, 0), 1).tolist(),
           "p10_rgb": np.round(np.percentile(C, 10, axis=0), 1).tolist(),
           "p90_rgb": np.round(np.percentile(C, 90, axis=0), 1).tolist(),
           "method": "IMX519 1600x1200 frames (PyAV rgb24), eraser_mask eroded 9 px, per-frame median, then the "
                     "median over frames"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eraser_real.json").write_text(json.dumps(rep, indent=1))
    return rep


def cut_tile(mosaic_u8: np.ndarray, cnt: np.ndarray, out: Path, scratch: Path) -> dict:
    """The TILE_M rectangle (+ SEAM_M) seen by >= MIN_FRAMES frames everywhere, nearest TILE_CENTRE_O, made seamless
    -> ``out/table_real.png``; returns its placement in O."""
    H, W = cnt.shape
    xs, ys = _rect_grid()
    have = cnt >= MIN_FRAMES
    nc, nr = (int(round(v * 1000 / MM_PER_PX)) for v in TILE_M)
    sm = int(round(SEAM_M * 1000 / MM_PER_PX))
    ii = cv2.integral(have.astype(np.uint8))
    best = None
    for r0 in range(0, H - nr - sm, 4):
        for c0 in range(0, W - nc - sm, 4):
            s = ii[r0 + nr + sm, c0 + nc + sm] - ii[r0, c0 + nc + sm] - ii[r0 + nr + sm, c0] + ii[r0, c0]
            cx_, cy_ = xs[c0 + (nc + sm) // 2], ys[r0 + (nr + sm) // 2]
            score = s - 20.0 * np.hypot(cx_ - TILE_CENTRE_O[0], cy_ - TILE_CENTRE_O[1]) * 1000
            if best is None or score > best[0]:
                best = (score, s, r0, c0)
    _, s, r0, c0 = best
    b8 = mosaic_u8[r0:r0 + nr + sm, c0:c0 + nc + sm].copy()
    hole = ~have[r0:r0 + nr + sm, c0:c0 + nc + sm]
    if hole.any():
        b8 = cv2.inpaint(b8, hole.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    t8 = seamless_band(b8.astype(np.float32), nr, nc, sm).clip(0, 255).round().astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "table_real.png"), t8[:, :, ::-1])
    cv2.imwrite(str(scratch / "table_tile_2x2.png"), np.tile(t8, (2, 2, 1))[:, :, ::-1])
    x0 = float(RECT_BOX_O[0][0] + c0 * MM_PER_PX / 1000)          # texel edges: column 0 left edge, row 0 top edge
    y1 = float(RECT_BOX_O[1][1] - r0 * MM_PER_PX / 1000)
    return {"file": "table_real.png", "sha256": sha256(out / "table_real.png"), "px": [nc, nr],
            "size_m": [nc * MM_PER_PX / 1000, nr * MM_PER_PX / 1000], "mm_per_px": MM_PER_PX,
            "origin_O_m": [round(x0, 6), round(y1 - nr * MM_PER_PX / 1000, 6)],
            "O_x_m": [round(x0, 6), round(x0 + nc * MM_PER_PX / 1000, 6)],
            "O_y_m": [round(y1 - nr * MM_PER_PX / 1000, 6), round(y1, 6)],
            "orientation": "column = O +x (right), row = O -y (down); origin_O_m = O (x, y) of the image's "
                           "bottom-left corner (column 0, last row); the seam band (cross-faded) is the first "
                           "SEAM_M of columns and rows",
            "seam_band_m": SEAM_M, "min_frames_per_texel": MIN_FRAMES,
            "inpainted_frac": round(float(hole.mean()), 5),
            "frames_per_texel_min_median": [int(cnt[r0:r0 + nr, c0:c0 + nc].min()), int(np.median(cnt[r0:r0 + nr, c0:c0 + nc]))],
            "mean_rgb": np.round(t8.reshape(-1, 3).mean(0), 2).tolist(),
            "std_rgb": np.round(t8.reshape(-1, 3).std(0), 2).tolist(),
            "colour": "as the IMX519 records the table (PyAV rgb24 of the raw frames, per-frame exposure gain to the "
                      "median divided out; no vignetting correction)"}


def cmd_tile(a) -> dict:
    """Re-cut the tile from the table stage's mosaic (``table/table_mosaic.png``, ``table_count16.png``)."""
    scratch = Path(a.work) / "table"
    med = cv2.cvtColor(cv2.imread(str(scratch / "table_mosaic.png")), cv2.COLOR_BGR2RGB)
    cp = scratch / "table_count16.png"
    cnt = (cv2.imread(str(cp), cv2.IMREAD_UNCHANGED).astype(np.int32) if cp.is_file()
           else cv2.imread(str(scratch / "table_count.png"), 0).astype(np.int32) // 4)
    tile = cut_tile(med, cnt, Path(a.out), scratch)
    out = Path(a.out)
    jp = out / "table_real.json"
    doc = json.loads(jp.read_text()) if jp.is_file() else {}
    rep_p = Path(a.work) / "table" / "table_report.json"
    if rep_p.is_file():
        r = json.loads(rep_p.read_text())
        doc.update({k: r[k] for k in ("eraser_check_summary", "registration_shift_mm", "n_frames_used") if k in r})
    doc["tile"] = tile
    jp.write_text(json.dumps(doc, indent=1))
    return doc


def seamless_band(big: np.ndarray, nr: int, nc: int, sm: int) -> np.ndarray:
    """A (nr, nc) tile that wraps: ``big`` is (nr + sm, nc + sm); the first ``sm`` columns (rows) of the tile are a
    linear cross-fade of big's columns [0, sm) (weight rising 0 -> 1) with columns [nc, nc + sm) (falling), so
    column nc - 1 continues into column 0 of the next copy; everything else is big as it is."""
    w = ((np.arange(sm) + 0.5) / sm).astype(np.float32)
    t = big.copy()
    t[:, :sm] = big[:, :sm] * w[None, :, None] + big[:, nc:nc + sm] * (1 - w)[None, :, None]
    t = t[:, :nc]
    t2 = t.copy()
    t2[:sm] = t[:sm] * w[:, None, None] + t[nr:nr + sm] * (1 - w)[:, None, None]
    return t2[:nr]


# ------------------------------------------------------------------------------------------------------ mesh


def read_glb(path: Path):
    b = path.read_bytes()
    l0, _ = struct.unpack("<II", b[12:20])
    j = json.loads(b[20:20 + l0])
    o1 = 20 + l0
    l1, _ = struct.unpack("<II", b[o1:o1 + 8])
    binc = b[o1 + 8:o1 + 8 + l1]

    def view(i):
        bv = j["bufferViews"][i]
        o = bv.get("byteOffset", 0)
        return binc[o:o + bv["byteLength"]]
    if len(j["meshes"]) != 1 or len(j["meshes"][0]["primitives"]) != 1:
        raise ValueError("expected one mesh with one primitive")
    if j.get("nodes") and any(("matrix" in n or "rotation" in n or "translation" in n or "scale" in n)
                              for n in j["nodes"]):
        raise ValueError("glb node transforms are not handled")
    prim = j["meshes"][0]["primitives"][0]
    pa, ta = j["accessors"][prim["attributes"]["POSITION"]], j["accessors"][prim["attributes"]["TEXCOORD_0"]]
    pos = np.frombuffer(view(pa["bufferView"]), np.float32).reshape(-1, 3).astype(np.float64)
    uv = np.frombuffer(view(ta["bufferView"]), np.float32).reshape(-1, 2).astype(np.float64)
    img = cv2.imdecode(np.frombuffer(view(j["images"][0]["bufferView"]), np.uint8), cv2.IMREAD_COLOR)
    if "indices" in prim:
        idx = np.frombuffer(view(j["accessors"][prim["indices"]]["bufferView"]),
                            {5125: np.uint32, 5123: np.uint16}[j["accessors"][prim["indices"]]["componentType"]])
        pos, uv = pos[idx], uv[idx]
    scene = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], 1)       # glTF y-up -> scene z-up
    return scene, uv, img


def T_tag_scene() -> np.ndarray:
    m = json.loads((TASK / "analysis" / "mesh_3d.json").read_text())
    t = json.loads((TASK / "map" / m["map"] / "tx_slam_tag.json").read_text())
    return _inv(np.asarray(t["tx_slam_tag"], float)) @ _inv(np.asarray(m["T_scene_W"], float))


def arm_boxes(pad: float = ARM_PAD_M) -> list:
    """World AABBs (padded) of every visible arm / hand / rig / finger geom of ``build/sim_rig`` at the start pose
    (start_pose.yaml q_start) and at the home key: no part of the background mesh may sit there."""
    import mujoco
    import yaml

    scene = REPO / "sim_deploy" / "build" / "sim_rig" / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    sp = yaml.safe_load((REPO / "sim_deploy" / "stations" / "start_pose.yaml").read_text())
    q_start = _find_key(sp, "q_start_rad")
    out = []
    for q in ("home", q_start):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        if q != "home":
            d.qpos[:7] = np.asarray(q, float)
        mujoco.mj_forward(m, d)
        for g in range(m.ngeom):
            if m.geom_group[g] not in (1, 2, 3) or m.geom_rgba[g, 3] <= 0:
                continue
            c = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ m.geom_aabb[g, :3]
            h = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_aabb[g, 3:]
            out.append((c - h - pad, c + h + pad))
    return out


def _find_key(doc, key):
    if isinstance(doc, dict):
        if key in doc:
            return doc[key]
        for v in doc.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def cmd_mesh(a) -> dict:
    out = Path(a.out)
    work = Path(a.work) / "mesh"
    work.mkdir(parents=True, exist_ok=True)
    pos, uv, img = read_glb(TASK / "analysis" / "mesh_3d.glb")
    T_ts = T_tag_scene()
    T_O_tag = np.asarray(json.loads(LAYOUT.read_text())["T_O_tag"], float)
    T_O_mesh = T_O_tag @ T_ts
    P = pos @ T_O_mesh[:3, :3].T + T_O_mesh[:3, 3]                     # O frame
    tri = P.reshape(-1, 3, 3)
    uvt = uv.reshape(-1, 3, 2)
    # the table plane in O (robust fit on the near-plane vertices)
    s = np.abs(P[:, 2]) < 0.02
    A = np.c_[P[s, 0], P[s, 1], np.ones(s.sum())]
    coef = np.linalg.lstsq(A, P[s, 2], rcond=None)[0]
    for _ in range(3):
        res = P[s, 2] - A @ coef
        keep = np.abs(res) < 2.5 * np.median(np.abs(res)) + 1e-4
        coef = np.linalg.lstsq(A[keep], P[s, 2][keep], rcond=None)[0]
    res = P[s, 2] - A @ coef
    dz = tri[..., 2] - (coef[0] * tri[..., 0] + coef[1] * tri[..., 1] + coef[2])
    table = np.all(np.abs(dz) < TABLE_BAND_M, axis=1)
    below = np.all(dz < -TABLE_BAND_M, axis=1) | np.any(tri[..., 2] < -0.05, axis=1) & np.all(dz < TABLE_BAND_M, axis=1)
    rxy = np.linalg.norm(tri[..., :2], axis=2)
    near_base = np.any(rxy < ROBOT_CLEAR_M, axis=1)
    boxes = arm_boxes()
    inside = np.zeros(tri.shape[:2], bool)
    for lo, hi in boxes:
        inside |= np.all((tri > lo) & (tri < hi), axis=2)
    in_reach = np.any(inside, axis=1) & ~table
    (wx0, wy0), (wx1, wy1) = WORK_CLEAR_BOX_O
    work = (np.all((tri[..., 0] > wx0) & (tri[..., 0] < wx1) & (tri[..., 1] > wy0) & (tri[..., 1] < wy1), axis=1)
            & np.all(dz < WORK_CLEAR_H_M, axis=1))
    keep_f = ~(table | below | near_base | in_reach | work)

    # colour match: the mesh's table faces vs table_real.png over the same O area (per-channel gain)
    tdoc = json.loads((out / "table_real.json").read_text())["tile"]
    tex = cv2.cvtColor(cv2.imread(str(out / tdoc["file"])), cv2.COLOR_BGR2RGB).astype(np.float32)
    (tx0, tx1), (ty0, ty1) = tdoc["O_x_m"], tdoc["O_y_m"]
    c = tri.mean(1)
    on_tile = table & (c[:, 0] > tx0) & (c[:, 0] < tx1) & (c[:, 1] > ty0) & (c[:, 1] < ty1)
    rgb_atlas = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    ah, aw = rgb_atlas.shape[:2]
    tc = uvt[on_tile].mean(1)
    au = np.clip((tc[:, 0] * aw).astype(int), 0, aw - 1)
    av = np.clip((tc[:, 1] * ah).astype(int), 0, ah - 1)              # glTF uv: v down, row = v * h
    mesh_cols = rgb_atlas[av, au]
    # tag area (the 160 mm map tag: dark/white) excluded: only table-coloured texels
    lab = cv2.cvtColor(mesh_cols.reshape(-1, 1, 3).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    okc = (lab[:, 0] > 60) & (np.abs(lab[:, 1].astype(int) - 128) < 40) & (lab[:, 2] > 135)
    tex_mean = tex.reshape(-1, 3).mean(0)
    mesh_mean = np.median(mesh_cols[okc], 0)
    gain = tex_mean / np.maximum(mesh_mean, 1)
    half = cv2.resize(rgb_atlas, (aw // 2, ah // 2), interpolation=cv2.INTER_AREA)
    half = (half * gain).clip(0, 255).round().astype(np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "workspace_mesh.png"), half[:, :, ::-1])
    # OBJ in O: unique positions, per-corner texcoords (glTF v down -> OBJ v up)
    Pk = tri[keep_f].reshape(-1, 3)
    Tk = uvt[keep_f].reshape(-1, 2)
    q = np.round(Pk / 1e-6).astype(np.int64)
    _, vi, vinv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    vinv = vinv.ravel()
    lines = ["# sim_deploy W2 look: umiproc mesh_3d (PICK_TEST1 map) in the sim base frame O, without the table's own "
             "faces, below-table pieces and anything near the robot; see sim_deploy/look/make_assets.py"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in Pk[vi]]
    lines += [f"vt {u:.6f} {1.0 - v:.6f}" for u, v in Tk]
    F = vinv.reshape(-1, 3) + 1
    lines += [f"f {a_}/{3 * k + 1} {b}/{3 * k + 2} {c_}/{3 * k + 3}" for k, (a_, b, c_) in enumerate(F)]
    (out / "workspace_mesh.obj").write_text("\n".join(lines) + "\n")
    kept = tri[keep_f]
    rep = {"source": str(TASK / "analysis" / "mesh_3d.glb"), "source_sha256": sha256(TASK / "analysis" / "mesh_3d.glb"),
           "faces_in": int(len(tri)), "faces_table_dropped": int(table.sum()), "faces_below_dropped": int((below & ~table).sum()),
           "faces_near_base_dropped": int((near_base & ~table & ~below).sum()),
           "faces_near_arm_dropped": int((in_reach & ~near_base & ~below).sum()),
           "faces_work_area_dropped": int((work & ~(table | below | near_base | in_reach)).sum()),
           "faces_kept": int(keep_f.sum()), "vertices": int(len(vi)),
           "table_plane_O_z_axb_c": np.round(coef, 6).tolist(),
           "table_plane_tilt_deg": round(float(np.degrees(np.arctan(np.hypot(coef[0], coef[1])))), 3),
           "table_plane_offset_at_O_055_0_mm": round(float((coef[0] * 0.55 + coef[2]) * 1e3), 2),
           "table_plane_resid_mad_mm": round(float(np.median(np.abs(res)) * 1e3), 2),
           "T_tag_scene": np.round(T_ts, 6).tolist(), "T_O_tag": T_O_tag.tolist(),
           "T_O_mesh_applied_to_vertices": np.round(T_O_mesh, 6).tolist(),
           "obj_frame": "O (the station's background_mesh pose is the identity)",
           "kept_bbox_O_m": [np.round(kept.reshape(-1, 3).min(0), 3).tolist(), np.round(kept.reshape(-1, 3).max(0), 3).tolist()],
           "texture_px": list(half.shape[:2]), "colour_gain_rgb": np.round(gain, 4).tolist(),
           "colour_match": {"mesh_table_median_rgb": np.round(mesh_mean, 1).tolist(),
                            "table_real_mean_rgb": np.round(tex_mean, 1).tolist(), "n_faces": int(okc.sum())},
           "params": {"TABLE_BAND_M": TABLE_BAND_M, "ROBOT_CLEAR_M": ROBOT_CLEAR_M, "ARM_PAD_M": ARM_PAD_M,
                      "arm_boxes": len(boxes), "WORK_CLEAR_BOX_O": WORK_CLEAR_BOX_O, "WORK_CLEAR_H_M": WORK_CLEAR_H_M},
           "obj_mb": round((out / "workspace_mesh.obj").stat().st_size / 1e6, 1),
           "obj_sha256": sha256(out / "workspace_mesh.obj"), "png_sha256": sha256(out / "workspace_mesh.png")}
    (out / "workspace_mesh.json").write_text(json.dumps(rep, indent=1))
    return rep


BACKDROP_WALLS = {                                   # name: (p0, u, v) in O, proxies beyond the lab desk
    "wall_back": ((-0.80, 0.72, 1.20), (2.40, 0.0, 0.0), (0.0, 0.0, -1.20)),
    "wall_left": ((-0.65, -0.80, 1.20), (0.0, 1.52, 0.0), (0.0, 0.0, -1.20)),
    "wall_right": ((1.45, 0.72, 1.20), (0.0, -1.52, 0.0), (0.0, 0.0, -1.20)),
}


def cmd_backdrop(a) -> dict:
    """``backdrop_real.obj`` / ``.png``: the training lab beyond the desk (monitors, PC, wall) as proxy walls painted
    from the real HANDHELD IMX219 640 frames (the 15 references' raw640_uw_{0,1} + the 10 raw_extra frames; camera
    = the package TCP x E7 handheld pose; the copy's fisheye model), the upper ``UW_ROWS_FRAC`` of each frame only
    (no rig / fingers), the eraser masked; IMX219 colours divided by a fisheye-radius gain fitted on the same frames'
    table pixels against table_real.png (so the panels are in IMX519 colours like every texture)."""
    import sim_deploy.sim_pi.render  # noqa: F401
    from sim_deploy.camera_study.render_sets import E7_DT_M, handheld_tcp_T_cam
    from sim_deploy.look import backdrop as BD
    from sim_deploy.scene.zarr_lowdim import load_episode
    from sim_deploy.sim_pi import _deploy as D

    ref = REPO / "sim_evals" / "_camera_study" / "B" / "reference"
    work = Path(a.work) / "backdrop"
    work.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    rig = D.load_live_rig(D.DEFAULT_RIG_FILE)
    fish = D.physical_model(rig, "ultrawide")
    tcp_T_cam = handheld_tcp_T_cam(rig, E7_DT_M)["ultrawide"]
    T_O_tag = np.asarray(json.loads(LAYOUT.read_text())["T_O_tag"], float)
    frames = []                                           # (rgb 640, O_T_cam)
    eps = {}
    for d in sorted(p for p in ref.iterdir() if p.is_dir() and p.name.startswith("ep")):
        meta = json.loads((d / "meta.json").read_text())
        i = int(meta["package_index"])
        eps.setdefault(i, load_episode(DATASET, i))
        rows = meta["uw_frames_first_row_in_episode"]
        for k in (0, 1):
            img = cv2.cvtColor(cv2.imread(str(d / f"raw640_uw_{k}.png")), cv2.COLOR_BGR2RGB)
            frames.append((d.name + f"_{k}", img, T_O_tag @ eps[i]["poses"][int(rows[k])] @ tcp_T_cam))
    xi = json.loads((ref / "raw_extra" / "index.json").read_text())
    for f in xi["frames"]:
        img = cv2.cvtColor(cv2.imread(str(ref / "raw_extra" / f["files"][1])), cv2.COLOR_BGR2RGB)
        frames.append((f["files"][1], img, T_O_tag @ np.asarray(f["tcp_tag_row"], float) @ tcp_T_cam))

    def projector(T):
        def f(P):
            pc = (P - T[:3, 3]) @ T[:3, :3]
            uv, ok = fish.project(pc)
            return uv, np.asarray(ok, bool) & (pc[:, 2] > -1.0)
        return f

    def radius(T, P):
        pc = (P - T[:3, 3]) @ T[:3, :3]
        uv = fish.project(pc)[0]
        return np.hypot(uv[:, 0] - fish.cx, uv[:, 1] - fish.cy)

    # the radial gain from the table: IMX219 pixel vs table_real.png at the same O point (tile repeated)
    tdoc = json.loads((out / "table_real.json").read_text())["tile"]
    tile = cv2.cvtColor(cv2.imread(str(out / tdoc["file"])), cv2.COLOR_BGR2RGB)
    th, tw = tile.shape[:2]
    (ox, oy), (sx, sy) = tdoc["origin_O_m"], tdoc["size_m"]
    big = cv2.GaussianBlur(tile, (0, 0), 6)

    def table_rgb(P):
        u = ((P[:, 0] - ox) / sx) % 1.0 * tw
        v = (1.0 - ((P[:, 1] - oy) / sy) % 1.0) * th
        return big[np.clip(v.astype(int), 0, th - 1), np.clip(u.astype(int), 0, tw - 1)].astype(np.float64)
    X, Y = np.meshgrid(np.arange(0.15, 0.95, 0.004), np.arange(-0.25, 0.45, 0.004))
    Pt = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    rr, meas, expct = [], [], []
    views = []
    for name, img, T in frames:
        em = eraser_mask_uw(img)
        em = cv2.dilate(em.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        tab = np.zeros(img.shape[:2], bool)
        tab[int(0.30 * 480):int(0.62 * 480)] = True                  # the table band between horizon and fingers
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        c, ok = BD.sample(BD.View(name, projector(T), blur, tab & ~em), Pt)
        hsv = cv2.cvtColor(c.clip(0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        ok &= (hsv[:, 2] > 90)                                        # no keyboard / dark objects on the desk
        gain = np.median(c[ok], 0) / np.maximum(np.median(table_rgb(Pt[ok]), 0), 1) if ok.sum() > 500 else None
        if ok.sum() > 500:
            rr.append(radius(T, Pt[ok]))
            meas.append(c[ok])
            expct.append(table_rgb(Pt[ok]))
        up = np.zeros(img.shape[:2], bool)
        up[:int(UW_ROWS_FRAC * 480)] = True
        views.append((name, img, T, up & ~em))
        del gain
    fit = BD.fit_radial_gain(np.concatenate(rr), np.concatenate(meas), np.concatenate(expct),
                             r_max=float(np.hypot(fish.cx, fish.cy)))
    quads = [BD.Quad(n, np.array(p0), np.array(u), np.array(v), 5.0) for n, (p0, u, v) in BACKDROP_WALLS.items()]
    texs = {}
    for q in quads:
        P = q.points().reshape(-1, 3)
        cols, oks = [], []
        for name, img, T, msk in views:
            c, ok = BD.sample(BD.View(name, projector(T), img, msk), P)
            c = c / BD.radial_gain(fit, radius(T, P))
            cols.append(np.where(ok[:, None], c, np.nan))
            oks.append(ok)
        C = np.stack(cols)
        n = np.stack(oks).sum(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(C, axis=0)
        h, w = q.shape
        seen = (n > 0).reshape(h, w)
        base = np.nanmedian(med[n > 0], 0) if seen.any() else np.array([128.0] * 3)
        img = np.where(seen[..., None], np.nan_to_num(med.reshape(h, w, 3)), base).clip(0, 255).astype(np.uint8)
        if seen.any() and not seen.all():
            near = cv2.dilate(seen.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~seen
            img = cv2.inpaint(img, near.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        texs[q.name] = (img, float(seen.mean()))
        cv2.imwrite(str(work / f"quad_{q.name}.png"), img[:, :, ::-1])
    rep = BD.write_obj(out / "backdrop_real.obj", out / "backdrop_real.png", quads, texs,
                       header="sim_deploy W2 real_bg look: projective backdrop of the training lab (O frame)")
    rep.update({"n_views": len(views), "views": [v[0] for v in views], "imx219_gain_handheld": fit,
                "rows_used_frac": UW_ROWS_FRAC, "obj_sha256": sha256(out / "backdrop_real.obj"),
                "png_sha256": sha256(out / "backdrop_real.png"),
                "colour": "IMX219 pixels / imx219_gain_handheld (fisheye radius): IMX519 colours"})
    (out / "backdrop_real.json").write_text(json.dumps(rep, indent=1))
    return rep


def eraser_mask_uw(rgb: np.ndarray) -> np.ndarray:
    """The lime eraser in IMX219 frames (more saturated and yellower than in the IMX519: hue 25-50, S > 90)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 50) & (hsv[..., 1] > 90) & (hsv[..., 2] > 80)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("frames", "table", "tile", "eraser", "mesh", "backdrop"))
    ap.add_argument("--work", required=True, help="scratch folder (frames/, table/, mesh/)")
    ap.add_argument("--out", default=str(ASSETS))
    ap.add_argument("--max-frames", type=int, default=0, help="table: use about this many frames (0: all)")
    ap.add_argument("--report", help="write the numbers here (json)")
    a = ap.parse_args(argv)
    cv2.setNumThreads(4)
    rep = {"frames": cmd_frames, "table": cmd_table, "tile": cmd_tile, "eraser": cmd_eraser, "mesh": cmd_mesh, "backdrop": cmd_backdrop}[a.what](a)
    txt = json.dumps(rep, indent=1)
    if a.report:
        Path(a.report).write_text(txt)
    print(txt[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
