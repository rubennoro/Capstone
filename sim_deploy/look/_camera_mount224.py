"""The ``mount`` stage of ``fit_camera_look.py`` (W2 step 2): the fitted camera model vs the REAL-ARM mount frames.

    ~/envs/sim_deploy/bin/python -m sim_deploy.look.fit_camera_look mount-labels     # once (GL): finger masks
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.fit_camera_look mount [--yaml FILE]

Real: the 224x224 policy inputs of the 2026-11-20 real-Franka trials (``~/Capstone/data/<task>/policy/<run>/deploy/
*_probe_live/NNNNNN_{main,ultrawide}_1.png``: the real Pi's JPEG + the deployment's ImagePrep), only the static
sets at the start (arm at the start pose; consecutive main frames identical within 0.8 DN mean abs). The robot cell's
table is plain white: a flat field seen by both cameras at once. Table mask per trial: the sim's mount label image at
the start pose (``mount-labels``: fingers, holders, tags and arm out, eroded) AND the real median frame bright
(V >= 120) and unsaturated (HSV S <= 70), the largest connected piece (the wall / floor beyond the table edge drop
out), minus the eraser (saturated yellow) and the LED glow (dilated).

Model side, the same 224 pipeline: a UNIFORM 640x480 frame (one RGB value everywhere = a flat white table under
even light) through ``CameraLook.apply`` (the camera's shading, tone, noise, blur), JPEG q90, the copy's ImagePrep.
The uniform input colour per camera is solved so the model's 224 output matches the real table's centre-band mean
(``c_in``). Compared per radial band (rho of the 640 sensor position, ImagePrep applied to coordinate images):
mean RGB, chromaticity ratios R/G and B/G (decoded with gamma 2.2) relative to the centre band, relative luminance,
and the noise (Immerkaer block sigma at 224, 8x8 blocks inside the mask). "One camera model for both cameras" check:
the same physical white table needs ONE scene colour; ``c_in(main)`` vs ``c_in(ultrawide)`` says how far apart the two
fitted models put it (the handheld wood table gives the same number for the training scene, from the handheld fit).
Output: ``sim_evals/_pipeline/W2/camera/mount224.json`` and ``mount224_profiles.png``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from sim_deploy.look.camera_look import load_camera_look
from sim_deploy.look.fit_camera_look import OUT, REPO, ROLES, SHORT

DATA = Path("~/Capstone/data").expanduser()
FILE_ROLE = {"main": "main", "ultrawide": "ultrawide"}
BANDS = {"main": [0.0, 0.3, 0.55, 0.8, 1.01], "ultrawide": [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01]}
NORM_PX = 400.0
SEED = 2026


def trials() -> list[Path]:
    return sorted(DATA.glob("*/policy/*/deploy/*_probe_live"))


def static_sets(d: Path, max_n: int = 90, thr: float = 0.8) -> list[int]:
    idx = sorted({int(p.name[:6]) for p in d.glob("*_main_1.png")})[:max_n]
    out, prev = [], None
    for i in idx:
        im = cv2.imread(str(d / f"{i:06d}_main_1.png")).astype(np.float32)
        if prev is not None and np.abs(im - prev).mean() > thr:
            break
        out.append(i)
        prev = im
    return out


def load_set(d: Path, ids: list[int], role: str) -> np.ndarray:
    return np.stack([cv2.imread(str(d / f"{i:06d}_{FILE_ROLE[role]}_1.png"))[:, :, ::-1] for i in ids])


def coords224(prep) -> dict:
    """role -> (224, 224, 2) float 640-sensor (u, v) per 224 pixel (NaN outside), via ImagePrep on coordinate images."""
    u = np.arange(640, dtype=np.float64)
    v = np.arange(480, dtype=np.float64)
    U, V = np.meshgrid(u, v)
    out = {}
    for role in ROLES:
        f = prep.prep_main if role == "main" else prep.prep_ultrawide
        hi = np.zeros((480, 640, 3), np.uint8)
        hi[..., 0] = np.round(U / 639 * 254).astype(np.uint8) + 1
        hi[..., 1] = np.round(V / 479 * 254).astype(np.uint8) + 1
        hi[..., 2] = 255
        o = f(hi).astype(np.float64)
        valid = o[..., 2] > 250
        uv = np.stack([(o[..., 0] - 1) / 254 * 639, (o[..., 1] - 1) / 254 * 479], -1)
        uv[~valid] = np.nan
        out[role] = uv
    return out


def rho_of(role: str, uv: np.ndarray, centre) -> np.ndarray:
    cx, cy = centre[role]
    return np.hypot(uv[..., 0] - cx, uv[..., 1] - cy) / NORM_PX


def table_mask(med: np.ndarray, lab224: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(med, cv2.COLOR_RGB2HSV)
    S, V = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    sim_tab = cv2.erode((lab224 == 1).astype(np.uint8), np.ones((5, 5), np.uint8))
    m = ((V >= 120) & (S <= 70) & (sim_tab > 0)).astype(np.uint8)
    bad = ((S > 90) | ((hsv[..., 0] > 100) & (hsv[..., 0] < 160) & (S > 50))).astype(np.uint8)   # eraser, LED
    m &= 1 - cv2.dilate(bad, np.ones((7, 7), np.uint8))
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    n, labs, st, _ = cv2.connectedComponentsWithStats(m, 4)
    if n < 2:
        return m.astype(bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return labs == k


_IMM = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def block_noise(imgs: np.ndarray, mask: np.ndarray, blk: int = 8) -> list:
    """Median Immerkaer sigma per channel over 8x8 blocks fully inside ``mask``, pooled over images."""
    h, w = mask.shape[0] // blk, mask.shape[1] // blk
    full = mask[:h * blk, :w * blk].reshape(h, blk, w, blk).all(axis=(1, 3))
    sig = []
    for im in imgs:
        r = np.abs(cv2.filter2D(im.astype(np.float32), -1, _IMM))
        s = math.sqrt(math.pi / 2) / 6 * r[:h * blk, :w * blk].reshape(h, blk, w, blk, 3)[:, 1:-1, :, 1:-1].mean(axis=(1, 3))
        sig.append(s[full])
    s = np.concatenate(sig) if sig else np.zeros((0, 3))
    return [round(float(v), 3) for v in np.median(s, axis=0)] if len(s) else None


def profile(imgs: np.ndarray, mask: np.ndarray, rho: np.ndarray, edges) -> list:
    mean_img = imgs.astype(np.float64).mean(0)
    rows = []
    c0 = None
    for b in range(len(edges) - 1):
        s = mask & (rho >= edges[b]) & (rho < edges[b + 1])
        if s.sum() < 30:
            continue
        m = mean_img[s].mean(0)
        lin = (m / 255.0) ** 2.2
        c0 = lin if c0 is None else c0
        rows.append({"band": b, "rho": [edges[b], edges[b + 1]], "n_px": int(s.sum()), "mean_dn": np.round(m, 1).tolist(),
                     "r_over_g_rel": round(float((lin[0] / lin[1]) / (c0[0] / c0[1])), 4),
                     "b_over_g_rel": round(float((lin[2] / lin[1]) / (c0[2] / c0[1])), 4),
                     "lum_rel": round(float(lin.mean() / c0.mean()), 4)})
    return rows


def model_224(look, role: str, c_in, prep, n: int = 4) -> np.ndarray:
    f = prep.prep_main if role == "main" else prep.prep_ultrawide
    flat = np.empty((480, 640, 3), np.uint8)
    flat[:] = np.clip(np.round(c_in), 0, 255).astype(np.uint8)
    out = []
    for k in range(n):
        x = look.apply(role, flat, np.random.default_rng([SEED, ROLES.index(role), k])) if look else flat
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(f(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
    return np.stack(out)


def solve_input(look, role: str, target: np.ndarray, centre_mask: np.ndarray, prep) -> np.ndarray:
    """Uniform input colour whose model output (no noise) has the target mean in the centre band (float search,
    the uniform frame is quantised to uint8 at the end, so the result is within ~1 DN)."""
    m = look.models[role]

    def out_mean(c):
        flat = np.empty((480, 640, 3), np.uint8)
        flat[:] = np.clip(np.round(c), 0, 255).astype(np.uint8)
        q = cv2.convertScaleAbs(m.sqrt_lin(flat), alpha=255.0)
        y = cv2.LUT(q, m.enc_u8)
        f = prep.prep_main if role == "main" else prep.prep_ultrawide
        return f(y).astype(float)[centre_mask].mean(0)

    c = target.astype(float).copy()
    for _ in range(12):                       # fixed-point on the integer input grid (monotone per channel)
        o = out_mean(c)
        step = target - o
        if np.all(np.abs(step) < 0.6):
            break
        c = np.clip(c + step * 0.9, 0, 255)
    return np.round(c, 1)


def stage_mount(yaml_path: Path, out: Path) -> int:
    from sim_deploy.look._camera_fit import CENTRE
    from sim_deploy.sim_pi._deploy import DEFAULT_RIG_FILE, ImagePrep, load_live_rig
    look = load_camera_look(yaml_path)
    prep = ImagePrep(load_live_rig(DEFAULT_RIG_FILE))
    labs = np.load(OUT / "renders" / "mount_start_labels.npz")
    uvs = coords224(prep)
    res = {"yaml": str(yaml_path), "fitted_against": look.doc.get("fitted_against"), "trials": {}, "cameras": {}}
    pooled = {r: {"imgs": [], "masks": []} for r in ROLES}
    sheet = []
    for d in trials():
        name = f"{d.parts[-5]}/{d.parts[-3]}/{d.name}"
        ids = static_sets(d)
        if len(ids) < 8:
            res["trials"][name] = {"static_sets": len(ids), "used": False}
            continue
        tr = {"static_sets": len(ids), "used": "trial4" not in d.name, "note": "black object" if "trial4" in d.name else ""}
        row = []
        for role in ROLES:
            imgs = load_set(d, ids, role)
            med = np.median(imgs, axis=0).astype(np.uint8)
            mask = table_mask(med, labs[f"label224_{SHORT[role]}"])
            tr[f"{SHORT[role]}_table_px"] = int(mask.sum())
            tr[f"{SHORT[role]}_table_mean_dn"] = np.round(med[mask].astype(float).mean(0), 1).tolist()
            if tr["used"]:
                pooled[role]["imgs"].append(imgs)
                pooled[role]["masks"].append(np.repeat(mask[None], len(imgs), 0))
            ov = med.copy()
            e = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
            ov[e > 0] = (255, 0, 0)
            row.append(ov)
        sheet.append(np.concatenate(row, 1))
        res["trials"][name] = tr
    cv2.imwrite(str(out / "mount224_masks.png"), np.concatenate(sheet, 0)[:, :, ::-1])
    c_in = {}
    for role in ROLES:
        imgs = np.concatenate(pooled[role]["imgs"])
        masks = np.concatenate(pooled[role]["masks"])
        mask_any = masks.all(axis=0)                                 # pixels on the table in every used trial
        rho = rho_of(role, uvs[role], CENTRE)
        edges = BANDS[role]
        real_prof = profile(imgs, mask_any, rho, edges)
        b0 = real_prof[0]["band"]
        cmask = mask_any & (rho >= edges[b0]) & (rho < edges[b0 + 1])
        target = np.asarray(real_prof[0]["mean_dn"], float)
        ci = solve_input(look, role, target, cmask, prep)
        c_in[role] = ci
        mod = model_224(look, role, ci, prep)
        none = model_224(None, role, target, prep, n=1)
        R = {"table_px_all_trials": int(mask_any.sum()), "n_real_frames": int(len(imgs)), "uniform_input_c_in": ci.tolist(),
             "profile_real": real_prof, "profile_model": profile(mod, mask_any, rho, edges),
             "profile_no_model": profile(none, mask_any, rho, edges),
             "noise_224_real": block_noise(imgs[::3], mask_any),
             "noise_224_model": block_noise(mod, mask_any), "noise_224_no_model": block_noise(none, mask_any),
             "rho_max_in_224": round(float(np.nanmax(rho)), 3)}
        res["cameras"][role] = R
    rl = REPO / "sim_deploy" / "look" / "assets" / "robot_look.json"
    if rl.is_file():
        res["step1_imx219_gain_robot_check"] = step1_gain_check(json.loads(rl.read_text()).get("imx219_gain_robot"),
                                                                pooled, uvs, CENTRE, look, c_in["ultrawide"], prep)
    cm, cu = c_in["main"], c_in["ultrawide"]
    lm, lu = (cm / 255) ** 2.2, (cu / 255) ** 2.2
    res["one_model_check_white_table"] = {
        "c_in_main": cm.tolist(), "c_in_ultrawide": cu.tolist(), "diff_dn": np.round(cu - cm, 1).tolist(),
        "chroma_ultrawide_over_main_lin": {"r/g": round(float((lu[0] / lu[1]) / (lm[0] / lm[1])), 4),
                                           "b/g": round(float((lu[2] / lu[1]) / (lm[2] / lm[1])), 4)},
        "lum_ultrawide_over_main_lin": round(float(lu.mean() / lm.mean()), 4),
        "meaning": "the scene colour the main model needs to reproduce the real IMX519 white table vs the one the "
                   "ultrawide model needs for the real IMX219 white table (same physical table): equal = one "
                   "consistent camera model pair; a chroma ratio far from 1 = the cameras' processing (auto white "
                   "balance / exposure) changed with the scene, which a fixed model cannot follow"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "mount224.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({r: {k: res["cameras"][r][k] for k in ("uniform_input_c_in", "noise_224_real", "noise_224_model")}
                      for r in ROLES}, indent=1))
    print(json.dumps(res["one_model_check_white_table"], indent=1))
    for r in ROLES:
        for kind in ("profile_real", "profile_model", "profile_no_model"):
            print(r, kind, [(p["band"], p["r_over_g_rel"], p["b_over_g_rel"], p["lum_rel"]) for p in res["cameras"][r][kind]])
    return 0


def step1_gain_check(g: dict | None, pooled: dict, uvs: dict, centre: dict, look, c_uw, prep) -> dict | None:
    """Step 1's IMX219 radial gain on the robot table (display DN ratio IMX219 / IMX519 vs (r / r_max)^2, r the
    640 fisheye radius) next to the same radial profile of the real 224 frames and of the camera model (uniform
    input), each relative to its first bin: the shape the camera model must put back on the robot backdrop."""
    if not g:
        return None
    imgs = np.concatenate(pooled["ultrawide"]["imgs"])
    mask = np.concatenate(pooled["ultrawide"]["masks"]).all(axis=0)
    cx, cy = centre["ultrawide"]
    r = np.hypot(uvs["ultrawide"][..., 0] - cx, uvs["ultrawide"][..., 1] - cy)
    x = (r / g["r_max"]) ** 2
    xb = np.asarray(g["bins_x"], float)
    edges = np.r_[0.0, 0.5 * (xb[1:] + xb[:-1]), 1e9]
    mod = model_224(look, "ultrawide", c_uw, prep).astype(float).mean(0)
    real = imgs.astype(float).mean(0)
    rows = []
    for i, xc in enumerate(xb):
        s = mask & (x >= edges[i]) & (x < edges[i + 1])
        if s.sum() < 20:
            continue
        rows.append({"x": float(xc), "rho": round(float(math.sqrt(xc) * g["r_max"] / NORM_PX), 3), "n_px": int(s.sum()),
                     "step1_gain": g["bins_gain_rgb"][i], "real_dn": np.round(real[s].mean(0), 1).tolist(),
                     "model_dn": np.round(mod[s].mean(0), 1).tolist()})
    if not rows:
        return None
    s0, r0, m0 = (np.asarray(rows[0][k], float) for k in ("step1_gain", "real_dn", "model_dn"))
    for row in rows:
        row["step1_rel"] = np.round(np.asarray(row["step1_gain"]) / s0, 3).tolist()
        row["real_rel"] = np.round(np.asarray(row["real_dn"]) / r0, 3).tolist()
        row["model_rel"] = np.round(np.asarray(row["model_dn"]) / m0, 3).tolist()
    last = rows[-1]
    return {"method": step1_gain_check.__doc__.split(":")[0], "bins": rows,
            "outermost_rel": {"step1": last["step1_rel"], "real_224": last["real_rel"], "model": last["model_rel"]}}
