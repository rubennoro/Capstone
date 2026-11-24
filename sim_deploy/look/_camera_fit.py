"""The ``fit`` and ``images`` stages of ``fit_camera_look.py`` (W2 step 2); run them through that module::

    ~/envs/sim_deploy/bin/python -m sim_deploy.look.fit_camera_look fit --look <build id> [--write-yaml]
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.fit_camera_look images --look <build id>

CPU only (cv2 4 threads). Inputs: ``sim_evals/_pipeline/W2/camera/renders/<look>/`` (the render stage) and B1's real
``raw640_*.png``. What is measured and fitted: fit_camera_look.py's docstring.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.special import erf

from sim_deploy.look.camera_look import SCHEMA, CameraLook, shading_map
from sim_deploy.look.fit_camera_look import B_REF, OUT, REPO, ROLES, SHORT

cv2.setNumThreads(4)

NORM_PX = 400.0                                     # shading radius unit: half the 640x480 diagonal
ERODE = {"main": 5, "ultrawide": 3}                 # region erosion (px): misregistration + edge blur
UW_TABLE_MAX_M = 0.35                               # IMX219 table cells: table-plane distance from the TCP (real
                                                    # wood all around within it; the lab beyond, step 1's real_bg)
QS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
BANDS = {"main": [0.0, 0.3, 0.55, 0.8, 1.01], "ultrawide": [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01]}
FIT_REGIONS = {"main": {"table": 1.0},                         # sim colours sampled from real IMX519 frames only:
               "ultrawide": {"table": 1.0, "eraser": 0.5}}      # table + eraser (holders / tags / pads are not)
REPORT_REGIONS = ("table", "eraser", "pads", "holders", "tag_white", "tag_black")
YAML_OUT = REPO / "sim_deploy" / "look" / "camera_look_real.yaml"
SEED = 2026
COMPARE_KEYS = ("ep14_first", "ep02_approach_1of3", "ep25_approach_2of3")


def _rig_centres() -> dict:
    from sim_deploy.sim_pi._deploy import DEFAULT_RIG_FILE, load_live_rig, physical_model
    rig = load_live_rig(DEFAULT_RIG_FILE)
    out = {}
    for r in ROLES:
        m = physical_model(rig, r)
        out[r] = (float(m.cx), float(m.cy))
    return out


CENTRE = _rig_centres()


def _r(x, n=4):
    return np.round(np.asarray(x, float), n).tolist()


# ------------------------------------------------------------------------------------------------ data
def load_items(look: str) -> tuple[dict, list]:
    d = OUT / "renders" / look
    meta = json.loads((d / "meta.json").read_text())
    items = []
    for key in sorted(meta["refs"]):
        z = np.load(d / f"{key}.npz")
        for role in ROLES:
            for j in (0, 1):
                nm = f"{SHORT[role]}_{j}"
                real = cv2.imread(str(B_REF / key / f"raw640_{nm}.png"), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
                it = {"key": key, "nm": nm, "role": role, "real": real, "sim": z[f"rgb_{nm}"],
                      "label": z[f"label_{nm}"], "xy": z[f"table_xy_{nm}"], "tcp": z[f"O_T_tcp_{nm}"]}
                it["regions"] = regions(it)
                items.append(it)
    return meta, items


def _disk(r: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _lum(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _otsu_split(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lum = _lum(img)
    v = lum[mask]
    if v.size < 30:
        z = np.zeros_like(mask)
        return z, z
    thr = cv2.threshold(v.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    k = _disk(1)
    blk = cv2.erode((mask & (lum < thr)).astype(np.uint8), k).astype(bool)
    wht = cv2.erode((mask & (lum >= thr)).astype(np.uint8), k).astype(bool)
    return blk, wht


def regions(it: dict) -> dict:
    """name -> (mask on the real image, mask on the sim image)."""
    role, lab = it["role"], it["label"]
    k = _disk(ERODE[role])
    out = {}
    for c, name in ((1, "table"), (2, "eraser"), (3, "pads"), (4, "holders"), (6, "rig")):
        m = cv2.erode((lab == c).astype(np.uint8), k).astype(bool)
        if name == "table" and role == "ultrawide":
            T = it["tcp"]
            xy = it["xy"]
            with np.errstate(invalid="ignore"):
                m &= np.hypot(xy[..., 0] - T[0, 3], xy[..., 1] - T[1, 3]) < UW_TABLE_MAX_M
        out[name] = (m, m)
    tag = cv2.erode((lab == 5).astype(np.uint8), _disk(max(ERODE[role] - 1, 1))).astype(bool)
    rb, rw = _otsu_split(it["real"], tag)
    sb, sw = _otsu_split(it["sim"], tag)
    out["tag_black"] = (rb, sb)
    out["tag_white"] = (rw, sw)
    return out


def rho_quadrant(role: str) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = CENTRE[role]
    X = (np.arange(640) - cx) / NORM_PX
    Y = (np.arange(480) - cy) / NORM_PX
    X, Y = np.meshgrid(X, Y)
    rho = np.hypot(X, Y)
    ang = np.degrees(np.arctan2(Y, X))
    quad = (np.round(ang / 90.0).astype(int) % 4)          # 0 right, 1 down, 2 left, 3 up
    return rho, quad


# ------------------------------------------------------------------------------------------------ edges
def _erf_fit(t: np.ndarray, p: np.ndarray):
    lo, hi = float(np.mean(p[:3])), float(np.mean(p[-3:]))
    if abs(hi - lo) < 1e-6:
        return None

    def f(x):
        return x[0] + (x[1] - x[0]) * 0.5 * (1.0 + erf((t - x[2]) / (math.sqrt(2.0) * max(x[3], 1e-3))))

    try:
        r = least_squares(lambda x: f(x) - p, [lo, hi, 0.0, 1.0], bounds=([-1e3, -1e3, -4.0, 0.08], [1e3, 1e3, 4.0, 6.0]))
    except Exception:  # noqa: BLE001
        return None
    x = r.x
    rms = float(np.sqrt(np.mean(r.fun ** 2)))
    return {"lo": x[0], "hi": x[1], "t0": x[2], "sigma": x[3], "rms": rms, "contrast": abs(x[1] - x[0])}


BLUR_CLASSES = ("pad|table", "tag black|white")   # sharp physical edges (the real eraser and holder edges are
                                                  # rounded / bevelled: wider in reality for geometric reasons)
EDGE_PAIRS = (("eraser|table", 2, 1), ("pad|table", 3, 1), ("holder|table", 4, 1), ("tag black|white", None, None))


def edge_points(it: dict, max_per_class: int = 120, seed: int = 0) -> list:
    """(class name, y, x, ny, nx) boundary points on the sim label image with the outward normal of side A."""
    lab = it["label"]
    g = np.random.default_rng(seed)
    pts = []
    for name, a, b in EDGE_PAIRS:
        if a is None:
            tag = (lab == 5)
            tag_in = cv2.erode(tag.astype(np.uint8), _disk(2)).astype(bool)
            lum = _lum(it["sim"])
            v = lum[tag_in]
            if v.size < 30:
                continue
            thr = cv2.threshold(v.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            A = (tag & (lum < thr)).astype(np.float32)
            B = (tag & (lum >= thr))
            near = tag_in
        else:
            A = (lab == a).astype(np.float32)
            B = (lab == b)
            near = np.ones_like(B)
        Bd = cv2.dilate(B.astype(np.uint8), _disk(2)).astype(bool)
        Ad = cv2.dilate(A.astype(np.uint8), _disk(2)).astype(bool)
        cand = (A > 0) & Bd & ~cv2.erode(A.astype(np.uint8), _disk(1)).astype(bool) & near
        if a is not None:
            cand &= Ad
        ys, xs = np.nonzero(cand)
        keep = (ys > 10) & (ys < 470) & (xs > 10) & (xs < 630)
        ys, xs = ys[keep], xs[keep]
        if len(ys) == 0:
            continue
        sm = cv2.GaussianBlur(A, (0, 0), 1.5)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
        n = np.stack([-gy[ys, xs], -gx[ys, xs]], axis=1)
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 0.05
        ys, xs, n = ys[ok], xs[ok], n[ok] / nn[ok, None]
        if len(ys) > max_per_class:
            sel = g.choice(len(ys), max_per_class, replace=False)
            ys, xs, n = ys[sel], xs[sel], n[sel]
        pts += [(name, int(y), int(x), float(v[0]), float(v[1])) for y, x, v in zip(ys, xs, n)]
    return pts


def edge_widths(img: np.ndarray, pts: list, half: float) -> list:
    """erf fit per point: profile along the normal (A inside -> B outside), RGB projected on the contrast vector."""
    if not pts:
        return []
    t = np.arange(-half, half + 1e-9, 0.5)
    P = np.array([[p[1], p[2], p[3], p[4]] for p in pts], float)
    my = (P[:, 0:1] + t[None, :] * P[:, 2:3]).astype(np.float32)
    mx = (P[:, 1:2] + t[None, :] * P[:, 3:4]).astype(np.float32)
    prof = cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = []
    for i in range(len(pts)):
        pr = prof[i]
        d = pr[-4:].mean(0) - pr[:4].mean(0)
        nd = float(np.linalg.norm(d))
        if nd < 1e-6:
            out.append(None)
            continue
        s = pr @ (d / nd)
        f = _erf_fit(t, s)
        if f is None or f["contrast"] < 30 or abs(f["t0"]) > 3.0 or f["rms"] > 0.12 * f["contrast"] or \
                not (0.15 <= f["sigma"] <= 4.0):
            out.append(None)
            continue
        out.append(f)
    return out


def measure_edges(items: list, role: str, images_key: str, blur: float = 0.0, look: CameraLook | None = None) -> dict:
    """Per edge class: list of sigma (px) for the image kind ('real' | 'sim'), or the sim through ``look``."""
    res = {}
    for i, it in enumerate(items):
        if it["role"] != role:
            continue
        pts = edge_points(it, seed=i)
        img = it[images_key]
        if look is not None:
            img = look.apply(role, img, np.random.default_rng([SEED, i]))
        elif blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur, borderType=cv2.BORDER_REFLECT_101)
        fits = edge_widths(img, pts, 4.0 if role == "ultrawide" else 5.0)
        for p, f in zip(pts, fits):
            res.setdefault(p[0], {})[(it["key"], it["nm"], p[1], p[2])] = f["sigma"] if f else None
    return res


def edge_summary(real: dict, sim: dict) -> dict:
    out = {}
    allp = []
    for cls in sorted(set(real) | set(sim)):
        a, b = real.get(cls, {}), sim.get(cls, {})
        pairs = [(a[k], b[k]) for k in a if a[k] is not None and b.get(k) is not None]
        if not pairs:
            out[cls] = {"n_pairs": 0}
            continue
        pr = np.array(pairs)
        allp.append(pr)
        out[cls] = {"n_pairs": len(pr), "real_sigma_median_px": round(float(np.median(pr[:, 0])), 3),
                    "sim_sigma_median_px": round(float(np.median(pr[:, 1])), 3),
                    "real_10_90_width_px": round(2.563 * float(np.median(pr[:, 0])), 3),
                    "sim_10_90_width_px": round(2.563 * float(np.median(pr[:, 1])), 3),
                    "median_sigma2_diff": round(float(np.median(pr[:, 0] ** 2 - pr[:, 1] ** 2)), 4)}
    if allp:
        pr = np.concatenate(allp)
        out["all"] = {"n_pairs": len(pr), "real_sigma_median_px": round(float(np.median(pr[:, 0])), 3),
                      "sim_sigma_median_px": round(float(np.median(pr[:, 1])), 3),
                      "real_10_90_width_px": round(2.563 * float(np.median(pr[:, 0])), 3),
                      "sim_10_90_width_px": round(2.563 * float(np.median(pr[:, 1])), 3),
                      "median_sigma2_diff": round(float(np.median(pr[:, 0] ** 2 - pr[:, 1] ** 2)), 4)}
    return out


# ------------------------------------------------------------------------------------------------ noise
_IMM = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
BLK = 8


def _block_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per 8x8 block: mean level (3), Immerkaer sigma (3), and the filtered response (for correlations)."""
    f = img.astype(np.float32)
    resp = cv2.filter2D(f, -1, _IMM, borderType=cv2.BORDER_REFLECT_101)
    h, w = img.shape[0] // BLK, img.shape[1] // BLK
    lev = f[:h * BLK, :w * BLK].reshape(h, BLK, w, BLK, 3).mean(axis=(1, 3))
    inner = np.abs(resp[:h * BLK, :w * BLK].reshape(h, BLK, w, BLK, 3)[:, 1:-1, :, 1:-1])
    sig = math.sqrt(math.pi / 2) / 6.0 * inner.mean(axis=(1, 3))
    return lev, sig, resp


def flat_blocks(it: dict) -> np.ndarray:
    """(h, w) bool: blocks fully inside a non-table region (or the IMX219 table cells) and flat in the sim render
    (Immerkaer sigma < 0.6 DN on every channel: no texture or edge there in the scene)."""
    R = it["regions"]
    m = np.zeros((480, 640), bool)
    for name in ("eraser", "pads", "holders", "tag_white", "tag_black", "table"):
        m |= R[name][1] & R[name][0]
    h, w = 480 // BLK, 640 // BLK
    full = m[:h * BLK, :w * BLK].reshape(h, BLK, w, BLK).all(axis=(1, 3))
    _, sig, _ = _block_stats(it["sim"])
    return full & (sig < 0.6).all(axis=2)


LEVEL_BINS = np.arange(0, 257, 16)


def noise_curve(items: list, role: str, kind: str, look: CameraLook | None = None, blur: float = 0.0) -> dict:
    """Per channel and 16-DN level bin: median block sigma over the flat blocks (n >= 12)."""
    levs, sigs, resp_all = [], [], []
    for i, it in enumerate(items):
        if it["role"] != role:
            continue
        fb = flat_blocks(it)
        img = it[kind]
        if look is not None:
            img = look.apply(role, img, np.random.default_rng([SEED, 7, i]))
        elif blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur, borderType=cv2.BORDER_REFLECT_101)
        lev, sig, resp = _block_stats(img)
        levs.append(lev[fb])
        sigs.append(sig[fb])
        bm = np.kron(fb, np.ones((BLK, BLK), bool))
        r = resp[:bm.shape[0], :bm.shape[1]][bm]
        resp_all.append(r)
    lev = np.concatenate(levs)
    sig = np.concatenate(sigs)
    resp = np.concatenate(resp_all)
    out = {"n_blocks": int(len(lev)), "bins": {}}
    for c in range(3):
        rows = []
        for lo, hi in zip(LEVEL_BINS[:-1], LEVEL_BINS[1:]):
            sel = (lev[:, c] >= lo) & (lev[:, c] < hi)
            if sel.sum() >= 12:
                rows.append([float(0.5 * (lo + hi)), float(np.median(sig[sel, c])), int(sel.sum())])
        out["bins"]["rgb"[c]] = rows
    C = np.corrcoef(resp.T) if len(resp) > 10 else np.eye(3)
    out["channel_corr"] = {"rg": round(float(C[0, 1]), 3), "rb": round(float(C[0, 2]), 3),
                           "gb": round(float(C[1, 2]), 3)}
    return out


def spatial_ratio(items: list, role: str, kind: str, look: CameraLook | None = None) -> float:
    """Median over flat blocks of Immerkaer sigma / std(img - GaussianBlur(img, 1.5)): lower = more correlated."""
    rat = []
    for i, it in enumerate(items):
        if it["role"] != role:
            continue
        fb = flat_blocks(it)
        img = it[kind]
        if look is not None:
            img = look.apply(role, img, np.random.default_rng([SEED, 7, i]))
        f = img.astype(np.float32)
        _, sig, _ = _block_stats(img)
        hp = f - cv2.GaussianBlur(f, (0, 0), 1.5)
        h, w = 480 // BLK, 640 // BLK
        sd = hp[:h * BLK, :w * BLK].reshape(h, BLK, w, BLK, 3)[:, 1:-1, :, 1:-1].std(axis=(1, 3))
        ok = fb & (sd > 0.3).all(axis=2)
        rat.append((sig[ok] / sd[ok]).mean(axis=1))
    r = np.concatenate(rat) if rat else np.array([np.nan])
    return float(np.median(r))


def bank_ratio(sigma_px: float) -> float:
    from sim_deploy.look.camera_look import noise_bank
    n = noise_bank(5, sigma_px, 0.0)[:480, :640] * 4.0 + 128.0
    f = n.astype(np.float32)
    _, sig, _ = _block_stats(f)
    hp = f - cv2.GaussianBlur(f, (0, 0), 1.5)
    h, w = 480 // BLK, 640 // BLK
    sd = hp[:h * BLK, :w * BLK].reshape(h, BLK, w, BLK, 3)[:, 1:-1, :, 1:-1].std(axis=(1, 3))
    return float(np.median((sig / sd).mean(axis=2)))


def bank_block_factor(sigma_px: float) -> float:
    """Median Immerkaer block sigma of unit-sd bank noise (the estimator's response to the model's noise)."""
    from sim_deploy.look.camera_look import noise_bank
    n = noise_bank(6, sigma_px, 0.0)[:480, :640]
    _, sig, _ = _block_stats(n)
    return float(np.median(sig))


# ------------------------------------------------------------------------------------------------ colour model
class Param:
    """Parameter vector <-> camera_look block for one camera."""

    def __init__(self, role: str):
        self.role = role
        if role == "main":
            self.names = (["g_r", "g_g", "g_b", "o_r", "o_g", "o_b", "gam_r", "gam_g", "gam_b"] +
                          [f"rad{k}_{c}" for c in "rgb" for k in (1, 2)])
        else:
            self.names = ([f"M{i}{j}" for i in range(3) for j in range(3)] + ["o_r", "o_g", "o_b",
                          "gam_r", "gam_g", "gam_b"])
        self.fixed_shading = None                       # IMX219: the real cross-camera shading (fit_uw_shading)
        self.centre = None                              # prior centre (IMX219: the real cross-camera transfer)

    def x0(self) -> np.ndarray:
        if self.centre is not None:
            return np.asarray(self.centre, float)
        v = []
        for n in self.names:
            if n.startswith("g_") or n in ("M00", "M11", "M22"):
                v.append(1.0)
            elif n.startswith("gam_"):
                v.append(2.2)
            else:
                v.append(0.0)
        return np.array(v)

    def bounds(self):
        lo, hi = [], []
        for n in self.names:
            if n.startswith("g_") or n in ("M00", "M11", "M22"):
                lo.append(0.3), hi.append(2.5)
            elif n.startswith("M"):
                lo.append(-0.8), hi.append(0.8)
            elif n.startswith("o_"):
                lo.append(-0.1), hi.append(0.1)
            elif n.startswith("gam_"):
                lo.append(1.2), hi.append(4.0)
            elif n.startswith("rad"):
                lo.append(-3.0), hi.append(3.0)
            else:
                lo.append(-1.5), hi.append(1.5)
        return np.array(lo), np.array(hi)

    def prior(self, x: np.ndarray) -> np.ndarray:
        """Regularisation residuals (toward identity), in 'DN-like' units."""
        x0 = self.x0()
        w = []
        for n in self.names:
            # weight = DN of quantile error one unit of the parameter is worth (IMX519 strong: the scene, sampled
            # from IMX519 frames, must not be fitted into the camera; IMX219 moderate)
            if self.role == "main":
                w.append({"g": 60.0, "o": 300.0, "gam": 10.0, "rad": 6.0}[n.split("_")[0].rstrip("12")])
            else:
                if n.startswith("M"):
                    w.append(10.0 if n[1] == n[2] else 20.0)
                elif n.startswith("o_"):
                    w.append(100.0)
                else:
                    w.append(4.0)
        return (x - x0) * np.array(w)

    def block(self, x: np.ndarray) -> dict:
        d = dict(zip(self.names, x))
        if self.role == "main":
            M = np.diag([d["g_r"], d["g_g"], d["g_b"]])
            tilt = [[0.0, 0.0]] * 3
            shading = {"centre_px": list(CENTRE[self.role]), "norm_px": NORM_PX,
                       "radial": [[d[f"rad1_{c}"], d[f"rad2_{c}"]] for c in "rgb"], "tilt": tilt}
        else:
            M = np.array([[d[f"M{i}{j}"] for j in range(3)] for i in range(3)])
            shading = self.fixed_shading
        return {"decode_gamma": 2.2, "matrix": M.tolist(), "offset": [d["o_r"], d["o_g"], d["o_b"]],
                "shading": shading, "tone_gamma": [d["gam_r"], d["gam_g"], d["gam_b"]]}


def model_values(block: dict, rgb: np.ndarray, uv: np.ndarray, z: np.ndarray | None, noise: dict | None) -> np.ndarray:
    """camera_look's chain in float (no quantisation) on sample pixels (N, 3) at pixel positions uv (N, 2)."""
    lin = (rgb.astype(np.float64) / 255.0) ** block["decode_gamma"]
    M = np.asarray(block["matrix"], float)
    lin = lin @ M.T + np.asarray(block["offset"], float)
    sh = block["shading"]
    cx, cy = sh["centre_px"]
    X = (uv[:, 0] - cx) / sh["norm_px"]
    Y = (uv[:, 1] - cy) / sh["norm_px"]
    if sh.get("rho_max"):
        f = np.minimum(1.0, float(sh["rho_max"]) / np.maximum(np.hypot(X, Y), 1e-12))
        X, Y = X * f, Y * f
    r2 = X * X + Y * Y
    for c in range(3):
        e = np.zeros_like(r2)
        p = np.ones_like(r2)
        for a in sh["radial"][c]:
            p = p * r2
            e += a * p
        e += sh["tilt"][c][0] * X + sh["tilt"][c][1] * Y
        lin[:, c] *= np.exp(e)
    s = np.sqrt(np.clip(lin, 0.0, 1.0))
    if noise is not None and z is not None:
        s = s + (np.asarray(noise["s0"]) + np.asarray(noise["s1"]) * s) * z
    s = np.clip(s, 0.0, 1.0)
    tg = np.asarray(block["tone_gamma"], float)
    return 255.0 * s ** (2.0 / tg)


def build_cells(items: list, role: str, blur: float, max_n: int = 5000) -> dict:
    """cell -> {'real': (N,3), 'sim': (N,3), 'uv': (N,2), 'z': (N,3)} pooled over the camera's images."""
    rho, quad = rho_quadrant(role)
    edges = BANDS[role]
    g = np.random.default_rng(SEED)
    acc: dict = {}
    vv, uu = np.mgrid[0:480, 0:640]
    for it in items:
        if it["role"] != role:
            continue
        sim = cv2.GaussianBlur(it["sim"], (0, 0), blur, borderType=cv2.BORDER_REFLECT_101) if blur > 0 else it["sim"]
        for name, (mr, ms) in it["regions"].items():
            if name == "rig":
                continue
            if name == "table":
                for b in range(len(edges) - 1):
                    band = (rho >= edges[b]) & (rho < edges[b + 1])
                    for q in (range(4) if edges[b] >= 0.3 else (None,)):
                        cell_m = band if q is None else band & (quad == q)
                        cname = f"table/b{b}" + ("" if q is None else f"q{q}")
                        _acc(acc, cname, it["real"][mr & cell_m], sim[ms & cell_m],
                             np.stack([uu[ms & cell_m], vv[ms & cell_m]], 1))
            else:
                _acc(acc, name, it["real"][mr], sim[ms], np.stack([uu[ms], vv[ms]], 1))
    cells = {}
    for k, v in acc.items():
        R = np.concatenate(v["real"]) if v["real"] else np.zeros((0, 3))
        S = np.concatenate(v["sim"]) if v["sim"] else np.zeros((0, 3))
        U = np.concatenate(v["uv"]) if v["uv"] else np.zeros((0, 2))
        if len(R) < 200 or len(S) < 200:
            continue
        n_real, n_sim = len(R), len(S)
        if len(R) > max_n:
            R = R[g.choice(len(R), max_n, replace=False)]
        if len(S) > max_n:
            sel = g.choice(len(S), max_n, replace=False)
            S, U = S[sel], U[sel]
        cells[k] = {"real": R.astype(float), "sim": S.astype(float), "uv": U.astype(float),
                    "z": g.standard_normal(S.shape), "n_real": n_real, "n_sim": n_sim,
                    "q_real": np.quantile(R, QS, axis=0)}
    return cells


def _acc(acc, name, real, sim, uv):
    a = acc.setdefault(name, {"real": [], "sim": [], "uv": []})
    if len(real):
        a["real"].append(real)
    if len(sim):
        a["sim"].append(sim)
        a["uv"].append(uv)


def cell_weights(cells: dict, role: str) -> dict:
    regs = FIT_REGIONS[role]
    tot = {}
    for k, c in cells.items():
        reg = k.split("/")[0]
        if reg in regs:
            tot[reg] = tot.get(reg, 0.0) + math.sqrt(c["n_real"])
    return {k: regs[k.split("/")[0]] * math.sqrt(c["n_real"]) / tot[k.split("/")[0]]
            for k, c in cells.items() if k.split("/")[0] in regs}


def fit_colour(cells: dict, role: str, noise: dict | None, x_init=None, shading=None,
               centre=None) -> tuple[np.ndarray, Param, dict]:
    P = Param(role)
    P.fixed_shading = shading
    P.centre = centre
    W = cell_weights(cells, role)
    keys = sorted(W)
    nw = {k: W[k] * math.sqrt(len(keys)) for k in keys}       # mean weight ~1

    def resid(x):
        b = P.block(x)
        out = []
        for k in keys:
            c = cells[k]
            y = model_values(b, c["sim"], c["uv"], c["z"], noise)
            q = np.quantile(y, QS, axis=0)
            out.append(((q - c["q_real"]) * nw[k]).ravel() / math.sqrt(QS.size))
        out.append(P.prior(x))
        return np.concatenate(out)

    lo, hi = P.bounds()
    if centre is not None:
        lo, hi = np.minimum(lo, P.x0() - 0.3), np.maximum(hi, P.x0() + 0.3)
    x0 = P.x0() if x_init is None else np.clip(x_init, lo, hi)
    r0 = resid(P.x0())
    f = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", diff_step=1e-3, max_nfev=400)
    info = {"cost_at_prior_centre": round(float(np.sum(r0 ** 2)), 2), "cost_fit": round(float(np.sum(f.fun ** 2)), 2),
            "nfev": int(f.nfev), "status": int(f.status), "cells": keys,
            "weights": {k: round(W[k], 4) for k in keys}}
    return f.x, P, info


# ------------------------------------------------------------------------------------------------ evaluation
def w1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per channel Earth mover's distance (= mean |CDF_a - CDF_b| integrated over levels), DN."""
    q = np.linspace(0.005, 0.995, 100)
    return np.abs(np.quantile(a, q, axis=0) - np.quantile(b, q, axis=0)).mean(axis=0)


def region_stats(items: list, role: str, look: CameraLook | None, blur_only: float = 0.0) -> dict:
    """Per region (and IMX219/IMX519 table bands): real vs sim means and W1 per channel, sim = render (+ look)."""
    rho, _ = rho_quadrant(role)
    edges = BANDS[role]
    acc: dict = {}
    for i, it in enumerate(items):
        if it["role"] != role:
            continue
        sim = it["sim"]
        if look is not None:
            sim = look.apply(role, sim, np.random.default_rng([SEED, 11, i]))
        for name, (mr, ms) in it["regions"].items():
            if name == "rig":
                continue
            cells = [(name, mr, ms)]
            if name == "table":
                for b in range(len(edges) - 1):
                    band = (rho >= edges[b]) & (rho < edges[b + 1])
                    cells.append((f"table/band{b} rho {edges[b]:.2f}-{edges[b + 1]:.2f}", mr & band, ms & band))
            for cn, a, bm in cells:
                d = acc.setdefault(cn, {"real": [], "sim": []})
                d["real"].append(it["real"][a])
                d["sim"].append(sim[bm])
    out = {}
    for cn, d in acc.items():
        R = np.concatenate(d["real"]).astype(float)
        S = np.concatenate(d["sim"]).astype(float)
        if len(R) < 100 or len(S) < 100:
            continue
        out[cn] = {"n_real": len(R), "n_sim": len(S), "mean_real": _r(R.mean(0), 2), "mean_sim": _r(S.mean(0), 2),
                   "sd_real": _r(R.std(0), 2), "sd_sim": _r(S.std(0), 2), "w1_dn": _r(w1(S, R), 2),
                   "w1_mean_dn": round(float(w1(S, R).mean()), 2)}
    return out


def per_image_spread(items: list, role: str) -> dict:
    """Real table mean per image (auto exposure / white balance changes between frames)."""
    m = []
    for it in items:
        if it["role"] == role:
            mr = it["regions"]["table"][0]
            if mr.sum() > 500:
                m.append(it["real"][mr].astype(float).mean(0))
    m = np.array(m)
    return {"n_images": len(m), "table_mean_real_per_image_sd": _r(m.std(0), 2), "min": _r(m.min(0), 1),
            "max": _r(m.max(0), 1)}


# ------------------------------------------------------------------------------------------------ cross-camera
def cross_camera_samples(items: list, meta: dict) -> dict:
    """Real frames alone: each IMX219 table pixel (UW_TABLE_MAX_M cells) -> its table point -> projected into the
    IMX519 frames of the same episode (the copy's 640 model, each frame's camera pose from the render stage; first
    IMX519 frame whose table region holds it). Both sides Gaussian-smoothed (sigma 2 px) against grain and
    misregistration. Returns pooled samples: uw RGB, main RGB, uw pixel (u, v)."""
    from sim_deploy.sim_pi._deploy import DEFAULT_RIG_FILE, load_live_rig, physical_model
    rig = load_live_rig(DEFAULT_RIG_FILE)
    mm = physical_model(rig, "main")
    by_ep: dict = {}
    for it in items:
        by_ep.setdefault(it["key"].split("_")[0], []).append(it)
    renders = OUT / "renders" / meta["look"]
    U, Mn, UV = [], [], []
    for ep, its in by_ep.items():
        mains = [it for it in its if it["role"] == "main"]
        poses = {(it["key"], it["nm"]): np.load(renders / f"{it['key']}.npz")[f"O_T_cam_{it['nm']}"] for it in mains}
        smooth = {(it["key"], it["nm"]): cv2.GaussianBlur(it["real"], (0, 0), 2.0) for it in mains}
        for it in its:
            if it["role"] != "ultrawide":
                continue
            ys, xs = np.nonzero(it["regions"]["table"][0])
            if len(ys) == 0:
                continue
            P = np.c_[it["xy"][ys, xs], np.zeros(len(ys))]
            got = np.full((len(ys), 3), np.nan)
            for mt in mains:
                T = poses[(mt["key"], mt["nm"])]
                pc = (np.c_[P, np.ones(len(P))] @ np.linalg.inv(T).T)[:, :3]
                uv, ok = mm.project(pc)
                uv = np.asarray(uv, float)
                ok = np.asarray(ok, bool) & (pc[:, 2] > 0) & (uv[:, 0] > 8) & (uv[:, 0] < 631) & \
                    (uv[:, 1] > 8) & (uv[:, 1] < 471)
                if not ok.any():
                    continue
                ui, vi = np.round(uv[ok, 0]).astype(int), np.round(uv[ok, 1]).astype(int)
                good = mt["regions"]["table"][0][vi, ui]
                idx = np.flatnonzero(ok)[good]
                vals = smooth[(mt["key"], mt["nm"])][vi[good], ui[good]].astype(float)
                fill = np.isnan(got[idx, 0])
                got[idx[fill]] = vals[fill]
            ok = ~np.isnan(got[:, 0])
            if not ok.any():
                continue
            U.append(cv2.GaussianBlur(it["real"], (0, 0), 2.0)[ys[ok], xs[ok]].astype(float))
            Mn.append(got[ok])
            UV.append(np.stack([xs[ok], ys[ok]], 1).astype(float))
    return {"uw": np.concatenate(U), "main": np.concatenate(Mn), "uv": np.concatenate(UV)}


def _xcells(uv: np.ndarray, role: str = "ultrawide") -> list:
    """(name, bool index) cells: radial bands, split in 4 quadrants from rho 0.2 on."""
    cx, cy = CENTRE[role]
    X, Y = (uv[:, 0] - cx) / NORM_PX, (uv[:, 1] - cy) / NORM_PX
    rho = np.hypot(X, Y)
    quad = np.round(np.degrees(np.arctan2(Y, X)) / 90.0).astype(int) % 4
    e = BANDS[role]
    out = []
    for b in range(len(e) - 1):
        band = (rho >= e[b]) & (rho < e[b + 1])
        if e[b] < 0.2:
            out.append((f"b{b}", band))
        else:
            for q in range(4):
                out.append((f"b{b}q{q}", band & (quad == q)))
    return out


def cross_camera_table(xs: dict, shading: dict | None = None, min_n: int = 300) -> dict:
    """Per cell: n, real IMX219 / IMX519 ratio of the decoded (gamma 2.2) mean RGB relative to the centre cell b0,
    and (given a shading block) the model's mean gain ratio over the same IMX219 pixels."""
    lu, lm = (xs["uw"] / 255.0) ** 2.2, (xs["main"] / 255.0) ** 2.2
    G = shading_map(shading) if shading else None
    rows, r0, g0 = {}, None, None
    for name, sel in _xcells(xs["uv"]):
        if sel.sum() < min_n:
            continue
        ratio = lu[sel].mean(0) / lm[sel].mean(0)
        r0 = ratio if r0 is None else r0
        row = {"n": int(sel.sum()), "ratio_lin": _r(ratio, 4), "ratio_rel_b0": _r(ratio / r0, 4),
               "uw_mean_dn": _r(xs["uw"][sel].mean(0), 1), "main_mean_dn": _r(xs["main"][sel].mean(0), 1)}
        if G is not None:
            ui, vi = xs["uv"][sel, 0].astype(int), xs["uv"][sel, 1].astype(int)
            g = G[vi, ui].astype(float).mean(0)
            g0 = g if g0 is None else g0
            row["model_gain_rel_b0"] = _r(g / g0, 4)
        rows[name] = row
    return rows


def fit_uw_shading(xs: dict) -> tuple[dict, dict]:
    """IMX219 colour shading from the real cross-camera ratios alone (per channel: exp(a1 rho^2 + a2 rho^4 + tx X +
    ty Y), least squares on log(ratio / ratio_b0) per cell, weights sqrt(n), weak prior on a2)."""
    tab = cross_camera_table(xs)
    names = [n for n in tab if n != "b0"]
    cells = dict(_xcells(xs["uv"]))
    cx, cy = CENTRE["ultrawide"]
    X = (xs["uv"][:, 0] - cx) / NORM_PX
    Y = (xs["uv"][:, 1] - cy) / NORM_PX
    r2 = X * X + Y * Y
    b0 = cells["b0"]
    radial, tilt, info = [], [], {}
    for c in range(3):
        tgt = np.array([math.log(tab[n]["ratio_rel_b0"][c]) for n in names])
        w = np.sqrt([tab[n]["n"] for n in names])

        def res(p):
            g = np.exp(p[0] * r2 + p[1] * r2 * r2 + p[2] * X + p[3] * Y)
            m0 = g[b0].mean()
            out = [(math.log(g[cells[n]].mean() / m0) - t) * wi for n, t, wi in zip(names, tgt, w)]
            return np.r_[out, 0.3 * np.sqrt(w.mean()) * p[1]]

        f = least_squares(res, [0.0, 0.0, 0.0, 0.0])
        radial.append([round(float(f.x[0]), 5), round(float(f.x[1]), 5)])
        tilt.append([round(float(f.x[2]), 5), round(float(f.x[3]), 5)])
        info["rgb"[c]] = {"rms_log": round(float(np.sqrt(np.mean((f.fun[:-1] / w) ** 2))), 4)}
    rho_max = round(float(np.percentile(np.sqrt(r2), 99.5)), 3)
    shading = {"centre_px": list(CENTRE["ultrawide"]), "norm_px": NORM_PX, "rho_max": rho_max, "radial": radial,
               "tilt": tilt}
    return shading, {"rho_max_note": "the map is held constant beyond rho_max (99.5 % of the cross-camera samples)",
                     "fit_rms_log_ratio": info, "cells_used": names, "n_samples": int(len(r2))}


def fit_uw_transfer(items: list, xs: dict, shading: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """The IMX219's colour response relative to the IMX519 from the real frames alone: decoded (gamma 2.2) mean
    RGB of the same physical things seen by both cameras at the same instant, the IMX219 side divided by its colour
    shading (fit_uw_shading) at those pixels: the table (cross-camera cells, weight 1), the eraser (each
    reference's newest pair, weight 0.5), the blue holders (weight 0.3). lin_uw = M lin_main + o, least squares in
    sqrt(lin) (about display DN / 255), weak prior toward a diagonal M."""
    G = shading_map(shading)
    X, Y, W, what = [], [], [], []
    lu, lm = (xs["uw"] / 255.0) ** 2.2, (xs["main"] / 255.0) ** 2.2
    ui, vi = xs["uv"][:, 0].astype(int), xs["uv"][:, 1].astype(int)
    g = G[vi, ui].astype(float)
    cells = [(n, s) for n, s in _xcells(xs["uv"]) if s.sum() >= 300]
    ntot = sum(math.sqrt(s.sum()) for _, s in cells)
    for n, s in cells:
        X.append(lm[s].mean(0))
        Y.append((lu[s] / g[s]).mean(0))
        W.append(1.0 * math.sqrt(s.sum()) / ntot)
        what.append(f"table/{n}")
    pairs = {}
    for it in items:
        if it["nm"].endswith("_1"):
            pairs.setdefault(it["key"], {})[it["role"]] = it
    for reg, wt in (("eraser", 0.5), ("holders", 0.3)):
        rows = []
        for key, pr in sorted(pairs.items()):
            if len(pr) < 2:
                continue
            mm, mu = pr["main"]["regions"][reg][0], pr["ultrawide"]["regions"][reg][0]
            if mm.sum() < 100 or mu.sum() < 50:
                continue
            xm = ((pr["main"]["real"][mm] / 255.0) ** 2.2).mean(0)
            yu = (((pr["ultrawide"]["real"][mu] / 255.0) ** 2.2) / G[mu]).mean(0)
            rows.append((xm, yu, key))
        for xm, yu, key in rows:
            X.append(xm)
            Y.append(yu)
            W.append(wt / len(rows))
            what.append(f"{reg}/{key}")
    X, Y, W = np.array(X), np.array(Y), np.array(W)
    sw = np.sqrt(W / W.sum() * len(W))

    def pred(p):
        M = p[:9].reshape(3, 3)
        return X @ M.T + p[9:12]

    def res(p):
        M = p[:9].reshape(3, 3)
        off = M - np.diag(np.diag(M))
        r = (np.sqrt(np.clip(pred(p), 1e-6, None)) - np.sqrt(Y)) * 255.0 * sw[:, None]
        return np.r_[r.ravel(), 2.0 * off.ravel(), 50.0 * p[9:12]]

    p0 = np.r_[np.eye(3).ravel(), 0.0, 0.0, 0.0]
    f = least_squares(res, p0)
    M, o = f.x[:9].reshape(3, 3), f.x[9:12]
    P = pred(f.x)
    err = (np.sqrt(np.clip(P, 1e-6, None)) - np.sqrt(Y)) * 255.0
    info = {"method": fit_uw_transfer.__doc__.split("\n")[0], "n_pairs": len(X),
            "rms_dn_sqrt_domain": {k: round(float(np.sqrt(np.mean(err[[w.startswith(k) for w in what]] ** 2))), 2)
                                   for k in ("table", "eraser", "holders")},
            "pairs": {w: {"main_dn": _r(255 * x ** (1 / 2.2), 1), "uw_unshaded_dn": _r(255 * y ** (1 / 2.2), 1),
                          "pred_dn": _r(255 * np.clip(p, 0, None) ** (1 / 2.2), 1)}
                      for w, x, y, p in zip(what, X, Y, P) if not w.startswith("table/b") or w == "table/b0"}}
    return M, o, info


def model_shading_bands(block: dict, role: str) -> list:
    G = shading_map(block["shading"])
    rho, _ = rho_quadrant(role)
    edges = BANDS[role]
    out = []
    g0 = None
    for b in range(len(edges) - 1):
        s = (rho >= edges[b]) & (rho < edges[b + 1])
        g = G[s].mean(0)
        g0 = g if g0 is None else g0
        out.append({"band": b, "gain_lin": _r(g, 4), "gain_rel_band0": _r(g / g0, 4)})
    return out


# ------------------------------------------------------------------------------------------------ stages
def _timing(look: CameraLook, items: list) -> dict:
    cv2.setNumThreads(1)
    out = {}
    try:
        for role in ROLES:
            img = next(it["sim"] for it in items if it["role"] == role)
            rng = np.random.default_rng(1)
            for _ in range(5):
                look.apply(role, img, rng)
            ts = []
            for _ in range(100):
                t0 = time.perf_counter()
                look.apply(role, img, rng)
                ts.append((time.perf_counter() - t0) * 1e3)
            out[role] = {"mean_ms": round(float(np.mean(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3),
                         "threads": 1}
    finally:
        cv2.setNumThreads(4)
    return out


def _doc(blocks: dict, meta: dict) -> dict:
    return {"schema": SCHEMA, "seed": 0, "fitted_against": {"look": meta["look"], "model_sha256": meta["model_sha256"]},
            "cameras": blocks}


def _yaml_text(doc: dict) -> str:
    head = ("# W2 step 2: the Pi cameras' image formation, fitted to the real PICK_TEST1 640x480 frames (B1's 15\n"
            "# references) against handheld renders of the build named in fitted_against (sim_deploy/look/\n"
            "# fit_camera_look.py; results: sim_evals/_pipeline/W2/camera/camera_look.json). Applied by the sim Pi to\n"
            "# each 640x480 frame before JPEG when a station's sim_pi.camera_look names this file (camera_look.py).\n"
            "# main = IMX519, ultrawide = IMX219. Every block: camera_look.py's module docstring.\n")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(round(float(o), 6))
        if isinstance(o, np.integer):
            return int(o)
        return o

    return head + yaml.safe_dump(clean(doc), sort_keys=False, default_flow_style=None, width=118)


def stage_fit(look_id: str, out: Path, write_yaml: bool) -> int:
    t0 = time.perf_counter()
    meta, items = load_items(look_id)
    print(f"[camfit] {look_id}: {len(items)} images (model sha {meta['model_sha256'][:16]}...)", flush=True)
    res = {"look": look_id, "model": meta["model"], "model_sha256": meta["model_sha256"],
           "placement_source": meta["placement_source"], "real": "B1 raw640_*.png (Pi recording decode, INTER_AREA)",
           "settings": {"erode_px": ERODE, "uw_table_max_m": UW_TABLE_MAX_M, "bands_rho": BANDS,
                        "norm_px": NORM_PX, "centre_px": CENTRE, "fit_regions": FIT_REGIONS,
                        "quantiles": QS.tolist()},
           "cameras": {}}
    blocks, real_edges = {}, {}
    xs = cross_camera_samples(items, meta)
    print(f"[camfit] cross-camera samples: {len(xs['uw'])}", flush=True)
    for role in ROLES:
        R = {}
        # 1. edges -> blur
        er = real_edges[role] = measure_edges(items, role, "real")
        es = measure_edges(items, role, "sim")
        e_before = edge_summary(er, es)
        d2s = [e_before[c]["median_sigma2_diff"] for c in BLUR_CLASSES if e_before.get(c, {}).get("n_pairs", 0) >= 50]
        d2 = float(np.median(d2s)) if d2s else 0.0
        blur = round(math.sqrt(max(d2, 0.0)), 3)
        R["edges_before"] = e_before
        R["blur_sigma_px"] = blur
        print(f"[camfit] {role}: edges real {e_before.get('all')}, blur sigma {blur} px", flush=True)
        # 2. noise (real), spatial + channel correlation
        nr = noise_curve(items, role, "real")
        ns_b = noise_curve(items, role, "sim", blur=blur)
        rr = spatial_ratio(items, role, "real")
        grid = [0.0, 0.3, 0.45, 0.6, 0.75, 0.9, 1.1, 1.4]
        br = {g: bank_ratio(g) for g in grid}
        sp = min(grid, key=lambda g: abs(br[g] - rr))
        cc = nr["channel_corr"]
        rho_c = float(np.clip(np.mean([cc["rg"], cc["rb"], cc["gb"]]), 0.0, 0.95))
        kappa = bank_block_factor(sp)
        R["noise_real"] = nr
        R["noise_sim_before"] = ns_b
        R["noise_spatial"] = {"real_ratio": round(rr, 4), "bank_ratio_by_sigma": {str(k): round(v, 4) for k, v in br.items()},
                              "chosen_sigma_px": sp, "estimator_factor_kappa": round(kappa, 4)}
        # 3. colour + shading, then noise in the s domain, then colour again with noise
        cells = build_cells(items, role, blur)
        shading, centre = None, None
        if role == "ultrawide":
            shading, sh_info = fit_uw_shading(xs)
            R["shading_from_real_cross_camera"] = sh_info
            Mt, ot, tr_info = fit_uw_transfer(items, xs, shading)
            R["colour_transfer_from_real_cross_camera"] = {"matrix": _r(Mt, 5), "offset": _r(ot, 5), **tr_info}
            centre = np.r_[Mt.ravel(), ot, [2.2, 2.2, 2.2]]
        x, P, info1 = fit_colour(cells, role, None, shading=shading, centre=centre)
        blk = P.block(x)
        noise = fit_noise(nr, ns_b, blk, kappa)
        x, P, info2 = fit_colour(cells, role, noise, x_init=x, shading=shading, centre=centre)
        blk = P.block(x)
        noise = fit_noise(nr, ns_b, blk, kappa)
        blocks[role] = {"blur_sigma_px": blur, **blk,
                        "noise": {"s0": noise["s0"], "s1": noise["s1"], "spatial_sigma_px": sp,
                                  "channel_corr": round(rho_c, 3)}}
        R["colour_fit"] = {"pass1": info1, "pass2": info2, "params": dict(zip(P.names, _r(x, 5)))}
        print(f"[camfit] {role}: colour fit cost {info2['cost_at_prior_centre']} -> {info2['cost_fit']}; noise {noise}", flush=True)
        res["cameras"][role] = R
    doc = _doc(blocks, meta)
    look = CameraLook(doc)
    # 4. verify: noise, edges, regions after (the real implementation, apply())
    for role in ROLES:
        R = res["cameras"][role]
        na = noise_curve(items, role, "sim", look=look)
        R["noise_sim_after"] = na
        R["noise_spatial"]["sim_after_ratio"] = round(spatial_ratio(items, role, "sim", look=look), 4)
        ea = measure_edges(items, role, "sim", look=look)
        R["edges_after"] = edge_summary(real_edges[role], ea)
        R["regions_before"] = region_stats(items, role, None)
        R["regions_after"] = region_stats(items, role, look)
        R["real_per_image_spread"] = per_image_spread(items, role)
        R["model_shading_bands"] = model_shading_bands(blocks[role], role)
        R["noise_model_dn"] = {str(L): _r(look.models[role].noise_dn(L)[0], 3) for L in (16, 48, 96, 160, 224)}
        R["noise_compare"] = noise_compare(R["noise_real"], R["noise_sim_before"], na)
    res["cross_camera_real"] = {"method": cross_camera_samples.__doc__.split(".")[0] + "; ratio IMX219 / IMX519 of "
                                          "the decoded mean RGB per IMX219 cell, relative to the centre cell b0",
                                "cells": cross_camera_table(xs, blocks["ultrawide"]["shading"])}
    res["timing_per_frame"] = _timing(look, items)
    res["fit_s"] = round(time.perf_counter() - t0, 1)
    res["yaml"] = doc
    if write_yaml:
        YAML_OUT.write_text(_yaml_text(doc))
        res["yaml_file"] = str(YAML_OUT.relative_to(REPO))
        print(f"[camfit] wrote {YAML_OUT}", flush=True)
    else:
        p = out / f"camera_look_{look_id}.yaml"
        p.write_text(_yaml_text(doc))
        res["yaml_file"] = str(p)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("camera_look.json" if write_yaml else f"camera_look_{look_id}.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps(summary(res), indent=1))
    return 0


def noise_compare(real: dict, before: dict, after: dict) -> dict:
    out = {}
    for c in "rgb":
        rb = {round(r[0]): r[1] for r in real["bins"][c]}
        bb = {round(r[0]): r[1] for r in before["bins"][c]}
        ab = {round(r[0]): r[1] for r in after["bins"][c]}
        out[c] = [[L, round(rb[L], 3), round(bb[L], 3) if L in bb else None, round(ab[L], 3) if L in ab else None]
                  for L in sorted(rb)]
    out["columns"] = "level DN, real, sim before, sim after (median Immerkaer block sigma, DN)"
    return out


def fit_noise(nr: dict, ns_b: dict, blk: dict, kappa: float) -> dict:
    """s0, s1 per channel: the model noise sd (DN) at each real level bin should give sqrt(sim_before^2 +
    (kappa sd)^2) = real (median block estimator; kappa = the estimator's response to unit model noise)."""
    s0, s1 = [], []
    tg = np.asarray(blk["tone_gamma"], float)
    for ci, c in enumerate("rgb"):
        rows = nr["bins"][c]
        sb = {round(r[0]): r[1] for r in ns_b["bins"][c]}
        L = np.array([r[0] for r in rows])
        sr = np.array([r[1] for r in rows])
        sbv = np.array([sb.get(round(l), np.nan) for l in L])
        sbv = np.where(np.isnan(sbv), np.nanmin(sbv) if np.isfinite(sbv).any() else 0.0, sbv)
        target = np.sqrt(np.maximum(sr ** 2 - sbv ** 2, 0.0)) / kappa
        wts = np.sqrt(np.array([r[2] for r in rows], float))
        e = 2.0 / tg[ci]
        s = (np.clip(L, 0.5, 254.5) / 255.0) ** (1.0 / e)
        dyds = 255.0 * e * s ** (e - 1.0)

        def res(p):
            return (dyds * np.abs(p[0] + p[1] * s) - target) * wts

        def res2(p):                                        # p = sd at s = 0 and at s = 1 (both >= 0)
            return res([p[0], p[1] - p[0]])

        f = least_squares(res2, [0.003, 0.003], bounds=([0.0, 0.0], [0.05, 0.05]))
        s0.append(round(float(f.x[0]), 6))
        s1.append(round(float(f.x[1] - f.x[0]), 6))
    return {"s0": s0, "s1": s1}


def summary(res: dict) -> dict:
    out = {"model_sha256": res["model_sha256"][:16], "timing": res["timing_per_frame"]}
    for role, R in res["cameras"].items():
        rb, ra = R["regions_before"], R["regions_after"]
        out[role] = {"blur": R["blur_sigma_px"],
                     "edge_10_90_px real/before/after": [R["edges_before"].get("all", {}).get("real_10_90_width_px"),
                                                         R["edges_before"].get("all", {}).get("sim_10_90_width_px"),
                                                         R["edges_after"].get("all", {}).get("sim_10_90_width_px")],
                     "w1_mean_dn before -> after": {k: [rb[k]["w1_mean_dn"], ra.get(k, {}).get("w1_mean_dn")]
                                                    for k in rb}}
    return out


# ------------------------------------------------------------------------------------------------ images
def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 8 + 7 * len(text)), 16), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _jpeg_rt(rgb: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def stage_images(look_id: str, out: Path, yaml_path: Path) -> int:
    from sim_deploy.look.camera_look import load_camera_look
    from sim_deploy.sim_pi._deploy import DEFAULT_RIG_FILE, ImagePrep, load_live_rig
    look = load_camera_look(yaml_path)
    prep = ImagePrep(load_live_rig(DEFAULT_RIG_FILE))
    d = OUT / "renders" / look_id
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key in COMPARE_KEYS:
        z = np.load(d / f"{key}.npz")
        rows640, rows224 = [], []
        for role in ROLES:
            nm = f"{SHORT[role]}_1"
            real = cv2.imread(str(B_REF / key / f"raw640_{nm}.png"))[:, :, ::-1].copy()
            sim = z[f"rgb_{nm}"]
            simc = look.apply(role, sim, np.random.default_rng([SEED, 99, ROLES.index(role)]))
            trip = [("real", real), (f"sim {look_id} (no camera model)", sim), ("sim + camera model", simc)]
            rows640.append(np.concatenate([_label(im, f"{SHORT[role]} 640 {t}") for t, im in trip], 1))
            pf = prep.prep_main if role == "main" else prep.prep_ultrawide
            ims = [pf(_jpeg_rt(im)) for _, im in trip]
            rows224.append(np.concatenate([_label(cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                                                  f"{SHORT[role]} 224 {t}") for (t, _), im in zip(trip, ims)], 1))
        p640 = out / f"compare640_{look_id}_{key}.png"
        p224 = out / f"compare224_{look_id}_{key}.png"
        cv2.imwrite(str(p640), np.concatenate(rows640, 0)[:, :, ::-1])
        cv2.imwrite(str(p224), np.concatenate(rows224, 0)[:, :, ::-1])
        written += [p640, p224]
    print("[camfit] wrote " + ", ".join(str(p) for p in written), flush=True)
    return 0
