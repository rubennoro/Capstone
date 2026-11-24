"""Camera study B, step corr_main: the IMX519 (main) view handheld vs Franka mount, and the crop/zoom of the decoded
640x480 main frame (before the letterbox) that makes the mount view match the handheld view. CPU only, no rendering.

    unset PYTHONPATH; cd ~/Capstone-main-sim
    ~/envs/sim_policy_deploy_torch/bin/python sim_deploy/corrections/export_main_refs.py \
        sim_evals/_camera_study/B/layout.json sim_evals/_camera_study/B/corr_main/refs_main.npz   # overlays only
    ~/envs/sim_deploy/bin/python -m sim_deploy.corrections.fit_main [--out sim_evals/_camera_study/B/corr_main]

What it does (geometry in geometry_main.py):

1. Camera poses in the TCP frame: handheld (training + E7's fix; and nominal), mount (model cam_imx519), with the
   cross-checks (model IMX519 vs model IMX219 + E2's CAD relation; E2's closed formulas; training's T_imu_cam
   relation vs E2's; the mount IMX519 rebuilt from the model IMX219 with training's relation).
2. For the 15 reference frames of layout.json (TCP pose, width, grasp from the package low-dim data): finger, tag,
   table and eraser points, projected through the copy's main camera model + MainPrep letterbox for each camera.
   Offsets handheld vs mount per class in 224 px, apparent scale ratio (RMS spread about the centroid).
3. Least-squares similarity (scale about the 640 centre + rotation + shift) of the mount 640 image onto the
   handheld one, class-balanced weights (each class of each frame weighs the same): all classes, fingers + tags,
   table, eraser; residuals before/after (224 px) and the black border (warp of a white frame).
4. ``main_crop_corr.yaml`` (the all-classes fit) + ``main_crop_corr_fingers_tags.yaml`` / ``_table.yaml`` next to
   main_crop.py; ``fit.json``, overlay PNGs on the real training main images; an end-to-end check: a synthetic 640
   frame of dots at the mount projections -> MainCrop.apply -> ImagePrep.prep_main -> dot centroids vs the fit's
   prediction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from sim_deploy.corrections import geometry_main as G
from sim_deploy.corrections import main_crop as MC
from sim_deploy.scene.placement import grasp_rows
from sim_deploy.scene.zarr_lowdim import load_episode

import cv2  # noqa: E402  (after geometry_main: EGL vendor before mujoco)

REPO = G.REPO
LAYOUT = REPO / "sim_evals" / "_camera_study" / "B" / "layout.json"
OUT = REPO / "sim_evals" / "_camera_study" / "B" / "corr_main"
CORR_DIR = REPO / "sim_deploy" / "corrections"
S224 = 224.0 / 640.0                         # letterbox scale (640x480 -> 224)
CENTRE = np.array([319.5, 239.5])
FITS = {"all": ("fingers", "tags", "table", "eraser"), "fingers_tags": ("fingers", "tags"), "table": ("table",),
        "eraser": ("eraser",)}
TARGETS = ("handheld", "handheld_nominal")
CROP_224 = 0.025 * 224                       # training's eval CenterCrop 0.95: 2.5 % per side = 5.6 px


# ---------------------------------------------------------------------------------------------------- frames

def build_frames(geo: G.MainGeometry, cams: dict, layout: dict) -> list[dict]:
    T_O_tag = np.asarray(layout["T_O_tag"], float)
    frames = []
    for sel in layout["selection"]:
        ep = load_episode(layout["dataset"], int(sel["index"]))
        g = grasp_rows(ep["widths"])["start"]
        O_T_grasp = T_O_tag @ ep["poses"][g]
        O_T_e = geo.eraser_pose(O_T_grasp)
        for row, label in zip(sel["rows_in_episode"], sel["labels"]):
            O_T_tcp = T_O_tag @ ep["poses"][int(row)]
            w = float(ep["widths"][int(row)])
            fp = geo.finger_points(w, cams["handheld"])
            pts = {"fingers": fp["fingers"], "tags": fp["tags"],
                   "table": geo.table_points(O_T_tcp, cams["handheld"]),
                   "eraser": geo.eraser_points(O_T_e, O_T_tcp)}
            side = {"fingers": fp["finger_side"], "tags": fp["tag_side"]}
            name = f"ep{int(sel['index']):02d}_{label}"
            fr = {"name": name, "index": int(sel["index"]), "label": label, "row": int(row), "width_m": w,
                  "grasp_row": int(g), "tcp_height_m": float(O_T_tcp[2, 3]),
                  "tilt_deg": float(np.degrees(np.arccos(np.clip(-O_T_tcp[2, 2], -1, 1)))),
                  "eraser_O": O_T_e[:3, 3].tolist(), "uv": {}, "ok": {}, "finger_side": fp["finger_side"]}
            for cname in ("handheld", "handheld_nominal", "handheld_e2rel", "mount", "mount_trrel"):
                hulls = []
                uvs, oks = {}, {}
                for cls in G.CLASSES:
                    P = pts[cls]
                    if cname.startswith("mount") and cls in side:
                        P = geo.mount_points(P, side[cls])
                    uv640, _, ok = geo.project(cams[cname], P)
                    uvs[cls], oks[cls] = uv640, ok
                for sgn, V in fp["hulls"]:
                    if cname.startswith("mount"):
                        V = geo.mount_points(V, np.full(len(V), sgn))
                    hulls.append(geo.project(cams[cname], V)[0])
                for cls in ("table", "eraser"):          # hidden behind a finger in this view
                    oks[cls] = oks[cls] & ~G.hull_mask(hulls, np.nan_to_num(uvs[cls], nan=-1e6))
                fr["uv"][cname], fr["ok"][cname] = uvs, oks
            frames.append(fr)
    return frames


def pairs(frames: list[dict], src: str, dst: str, classes) -> dict:
    """Stacked correspondences: {'p': src uv640, 'q': dst uv640, 'w': class-balanced weights, 'cls', 'frame'}."""
    P, Q, W, C, F = [], [], [], [], []
    for k, fr in enumerate(frames):
        for cls in classes:
            ok = fr["ok"][src][cls] & fr["ok"][dst][cls]
            n = int(ok.sum())
            if n == 0:
                continue
            P.append(fr["uv"][src][cls][ok])
            Q.append(fr["uv"][dst][cls][ok])
            W.append(np.full(n, 1.0 / n))
            C += [cls] * n
            F += [k] * n
    return {"p": np.vstack(P), "q": np.vstack(Q), "w": np.concatenate(W), "cls": np.array(C), "frame": np.array(F)}


# ---------------------------------------------------------------------------------------------------- fit

def fit_similarity(p: np.ndarray, q: np.ndarray, w: np.ndarray) -> MC.MainCrop:
    """Weighted least squares q ~ A p + t0, A = s R(theta); returned about the 640 centre."""
    n = len(p)
    M = np.zeros((2 * n, 4))
    M[0::2] = np.c_[p[:, 0], -p[:, 1], np.ones(n), np.zeros(n)]
    M[1::2] = np.c_[p[:, 1], p[:, 0], np.zeros(n), np.ones(n)]
    y = q.reshape(-1)
    sw = np.sqrt(np.repeat(w, 2))
    a, b, tx, ty = np.linalg.lstsq(M * sw[:, None], y * sw, rcond=None)[0]
    A = np.array([[a, -b], [b, a]])
    shift = A @ CENTRE + np.array([tx, ty]) - CENTRE
    return MC.MainCrop(scale=float(np.hypot(a, b)), rotation_deg=float(np.degrees(np.arctan2(b, a))),
                       centre_px=(float(CENTRE[0]), float(CENTRE[1])), shift_px=(float(shift[0]), float(shift[1])))


def stats(e224: np.ndarray) -> dict:
    if len(e224) == 0:
        return {"n": 0}
    return {"n": int(len(e224)), "mean": round(float(e224.mean()), 2), "p90": round(float(np.percentile(e224, 90)), 2),
            "max": round(float(e224.max()), 2)}


def residuals(pr: dict, crop: MC.MainCrop | None) -> dict:
    p = pr["p"] if crop is None else crop.map_points(pr["p"])
    e = np.linalg.norm(p - pr["q"], axis=1) * S224
    out = {cls: stats(e[pr["cls"] == cls]) for cls in G.CLASSES}
    out["all_points"] = stats(e)
    return out


def border(crop: MC.MainCrop) -> dict:
    white = np.full((480, 640, 3), 255, np.uint8)
    out = crop.apply(white)[:, :, 0]
    black = out < 128
    row, col = black[:, 320], black[240, :]

    def run(a):
        return int(np.argmax(~a)) if (~a).any() else len(a)
    return {"black_fraction_640": round(float(black.mean()), 5),
            "edge_px_640": {"top": run(row), "bottom": run(row[::-1]), "left": run(col), "right": run(col[::-1])},
            "edge_px_224_max": round(max(run(row), run(row[::-1]), run(col), run(col[::-1])) * S224, 2)}


def class_offsets(frames: list[dict], src: str, dst: str) -> dict:
    """dst - src per class, 224 px: mean |d|, p90, max, mean vector; apparent scale = RMS spread src / dst."""
    out = {}
    for cls in G.CLASSES:
        d_all, ratios = [], []
        for fr in frames:
            ok = fr["ok"][src][cls] & fr["ok"][dst][cls]
            if ok.sum() < 2:
                continue
            a = fr["uv"][src][cls][ok] * S224
            b = fr["uv"][dst][cls][ok] * S224
            d_all.append(b - a)
            sa = np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean())
            sb = np.sqrt(((b - b.mean(0)) ** 2).sum(1).mean())
            ratios.append(sb / sa)
        if not d_all:
            out[cls] = None
            continue
        d = np.vstack(d_all)
        e = np.linalg.norm(d, axis=1)
        out[cls] = {**stats(e), "mean_dxdy": d.mean(0).round(2).tolist(),
                    "scale_dst_over_src": round(float(np.mean(ratios)), 4),
                    "scale_range": [round(float(min(ratios)), 4), round(float(max(ratios)), 4)]}
    return out


# ---------------------------------------------------------------------------------------------------- cameras

def camera_report(geo: G.MainGeometry, cams: dict) -> dict:
    from sim_deploy.rig.add_rig import cam_T_tip, load_rig
    rd = load_rig(G.H.RIG_DIR)
    hh = cams["handheld"]
    rep = {"tcp_T_cam_mm": {}, "vs_handheld": {}}
    for k in ("handheld", "handheld_nominal", "handheld_e2rel", "mount", "mount_trrel"):
        T = cams[k]
        d = T[:3, 3] - hh[:3, 3]
        rep["tcp_T_cam_mm"][k] = {"pos": (T[:3, 3] * 1e3).round(2).tolist(),
                                  "tilt_about_tcp_x_deg": round(float(np.degrees(np.arctan2(T[2, 1], T[1, 1]))), 3)}
        rep["vs_handheld"][k] = {"d_tcp_mm": (d * 1e3).round(2).tolist(),
                                 "d_in_handheld_cam_axes_mm": (hh[:3, :3].T @ d * 1e3).round(2).tolist(),
                                 "dist_mm": round(float(np.linalg.norm(d) * 1e3), 2),
                                 "rot_deg": round(G.rot_deg(hh[:3, :3], T[:3, :3]), 4)}
    # cross-checks
    m519, m219 = cams["mount"], cams["_uw_mount"]
    via_e2 = m219 @ cams["_uw_T_main_e2"]
    z_tip = 0.23565
    tip_cam = G._inv(m519)[:3, 3]                           # TCP origin in the mount IMX519 frame
    tr, e2 = cams["_uw_T_main_training"], cams["_uw_T_main_e2"]
    rep["checks"] = {
        "mount_imx519_vs_imx219_plus_E2_CAD_mm": round(float(np.linalg.norm(via_e2[:3, 3] - m519[:3, 3]) * 1e3), 4),
        "mount_imx519_vs_imx219_plus_E2_CAD_deg": round(G.rot_deg(via_e2[:3, :3], m519[:3, :3]), 5),
        "E2_formula_cam_T_tip_mm": (np.array([8.18, 158.54 - 0.5 * z_tip * 1e3, 55.82 + 0.8660 * z_tip * 1e3])).round(2).tolist(),
        "add_rig_cam_T_tip_mm": (cam_T_tip(rd, "cam_imx519", z_tip)[:3, 3] * 1e3).round(2).tolist(),
        "model_tip_in_cam_imx519_mm": (tip_cam * 1e3).round(2).tolist(),
        "uw_T_main_training_mm": (tr[:3, 3] * 1e3).round(2).tolist(),
        "uw_T_main_E2_mm": (e2[:3, 3] * 1e3).round(2).tolist(),
        "uw_T_main_training_vs_E2_mm": round(float(np.linalg.norm(tr[:3, 3] - e2[:3, 3]) * 1e3), 2),
        "uw_T_main_training_vs_E2_deg": round(G.rot_deg(tr[:3, :3], e2[:3, :3]), 4),
        "mount_trrel_vs_mount_mm": round(float(np.linalg.norm(cams["mount_trrel"][:3, 3] - m519[:3, 3]) * 1e3), 2),
        "handheld_imx219_mount_vs_handheld_mm": ((cams["_uw_mount"][:3, 3] - cams["_uw_handheld"][:3, 3]) * 1e3).round(2).tolist(),
        "mount_fingers_offset_vs_handheld_mm": {k: (np.asarray(v) * 1e3).round(3).tolist() if not isinstance(v, float)
                                                else round(v, 4) for k, v in
                                                ((str(k), v) for k, v in geo.mount["side_offset"].items())},
    }
    return rep


# ---------------------------------------------------------------------------------------------------- overlays

def _dots(img, uv224, col, r=1):
    for u, v in uv224:
        if np.isfinite(u) and np.isfinite(v):
            cv2.circle(img, (int(round((u + 0.5) * 3 - 0.5)), int(round((v + 0.5) * 3 - 0.5))), r, col, -1,
                       cv2.LINE_AA)


def _inside(uv640) -> np.ndarray:
    return (uv640[:, 0] >= -0.5) & (uv640[:, 0] <= 639.5) & (uv640[:, 1] >= -0.5) & (uv640[:, 1] <= 479.5)


def _px3(uv224) -> np.ndarray:
    return np.round((np.asarray(uv224, float) + 0.5) * 3 - 0.5).astype(np.int32)


def overlay(fr: dict, ref: np.ndarray | None, crop: MC.MainCrop, tgt: str = "handheld", title: str = "") -> np.ndarray:
    """3x the 224 training main image: finger outlines (convex hull per side) and eraser outline as polygons, table
    grid as dots with a line mount -> target; green = target (handheld), red = mount, cyan = mount + crop."""
    base = ref if ref is not None else np.zeros((224, 224, 3), np.uint8)
    img = (0.8 * cv2.resize(base, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
    views = {"tgt": (fr["uv"][tgt], fr["ok"][tgt], (60, 255, 60)), "mount": (fr["uv"]["mount"], fr["ok"]["mount"],
             (255, 60, 60)), "corr": (fr["uv"]["mount"], fr["ok"]["mount"], (60, 210, 255))}
    for key, (uv, ok, col) in views.items():
        f = (lambda q: crop.map_points(q)) if key == "corr" else (lambda q: q)
        if key == "corr":                              # only what stays inside the corrected 640 frame
            ok = {c: ok[c] & _inside(crop.map_points(np.nan_to_num(uv[c]))) for c in ok}
        for sgn in (-1.0, 1.0):
            m = ok["fingers"] & (fr["finger_side"] == sgn)
            if m.sum() >= 3:
                hull = cv2.convexHull(_px3(G.D.letterbox_point(f(uv["fingers"][m]), G.IN_RES)).reshape(-1, 1, 2))
                cv2.polylines(img, [hull], True, col, 2, cv2.LINE_AA)
        m = ok["eraser"]
        if m.sum() >= 3:
            hull = cv2.convexHull(_px3(G.D.letterbox_point(f(uv["eraser"][m]), G.IN_RES)).reshape(-1, 1, 2))
            cv2.polylines(img, [hull], True, col, 2, cv2.LINE_AA)
    okt = fr["ok"][tgt]["table"] & fr["ok"]["mount"]["table"]
    h = _px3(G.D.letterbox_point(fr["uv"][tgt]["table"][okt], G.IN_RES))
    mm = _px3(G.D.letterbox_point(fr["uv"]["mount"]["table"][okt], G.IN_RES))
    cc = _px3(G.D.letterbox_point(crop.map_points(fr["uv"]["mount"]["table"][okt]), G.IN_RES))
    for a, b, c in zip(h, mm, cc):
        cv2.line(img, tuple(map(int, b)), tuple(map(int, a)), (255, 90, 90), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, b)), 2, (255, 60, 60), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, c)), 3, (60, 210, 255), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, a)), 2, (60, 255, 60), -1, cv2.LINE_AA)
    cv2.putText(img, f"{fr['name']} {title}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "green handheld (training+E7)  red mount  cyan mount+crop", (6, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------------------------------- e2e check

def e2e_check(geo: G.MainGeometry, frames: list[dict], crop: MC.MainCrop, out: Path) -> dict:
    """Synthetic 640 frame of Gaussian dots at the mount projections -> apply -> prep_main -> centroids (224) vs
    letterbox_point(crop.map_points(p)); also without the crop (detector check)."""
    res = {}
    rng_sigma = 2.2
    for fr in frames[:: max(1, len(frames) // 5)]:
        P = np.vstack([fr["uv"]["mount"][c][fr["ok"]["mount"][c]] for c in G.CLASSES])
        pred = G.D.letterbox_point(crop.map_points(P), G.IN_RES)
        keep, chosen = [], []
        for k, (u, v) in enumerate(pred):
            if not (8 <= u <= 215 and 36 <= v <= 187):        # inside the 224 content (letterbox rows 28..195)
                continue
            q0 = G.D.letterbox_point(P[k:k + 1], G.IN_RES)[0]
            if not (8 <= q0[0] <= 215 and 36 <= q0[1] <= 187):
                continue
            if all(np.hypot(u - pred[j][0], v - pred[j][1]) > 12 for j in keep):
                keep.append(k)
        P = P[keep]
        yy, xx = np.mgrid[0:480, 0:640]
        img = np.zeros((480, 640), np.float64)
        for u, v in P:
            x0, x1, y0, y1 = int(u) - 10, int(u) + 11, int(v) - 10, int(v) + 11
            img[y0:y1, x0:x1] += np.exp(-((xx[y0:y1, x0:x1] - u) ** 2 + (yy[y0:y1, x0:x1] - v) ** 2) / (2 * rng_sigma ** 2))
        rgb = np.repeat(np.clip(img * 255, 0, 255).round().astype(np.uint8)[:, :, None], 3, axis=2)
        row = {}
        for name, c in (("corrected", crop), ("uncorrected", None)):
            src = c.apply(rgb) if c is not None else rgb
            im224 = geo.prep.prep_main(src)[:, :, 0].astype(np.float64)
            exp = G.D.letterbox_point(c.map_points(P) if c is not None else P, G.IN_RES)
            errs = []
            for u, v in exp:
                x0, y0 = int(round(u)) - 4, int(round(v)) - 4
                win = im224[y0:y0 + 9, x0:x0 + 9]
                ys, xs = np.mgrid[y0:y0 + 9, x0:x0 + 9]
                wsum = win.sum()
                cu, cv_ = (win * xs).sum() / wsum, (win * ys).sum() / wsum
                errs.append(np.hypot(cu - u, cv_ - v))
            errs = np.array(errs)
            row[name] = {"dots": int(len(errs)), "mean_px": round(float(errs.mean()), 3),
                         "max_px": round(float(errs.max()), 3)}
            if name == "corrected" and fr is frames[0]:
                cv2.imwrite(str(out / "e2e_dots_640_in.png"), rgb[:, :, ::-1])
                cv2.imwrite(str(out / "e2e_dots_224_corrected.png"),
                            cv2.resize(geo.prep.prep_main(src), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)[:, :, ::-1])
        res[fr["name"]] = row
    allc = [v["corrected"]["max_px"] for v in res.values()]
    return {"frames": res, "corrected_max_px": max(allc),
            "corrected_mean_px": round(float(np.mean([v["corrected"]["mean_px"] for v in res.values()])), 3)}


# ---------------------------------------------------------------------------------------------------- main

def run(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    layout = json.loads(Path(a.layout).read_text())
    geo = G.MainGeometry.load()
    cams = geo.cameras()
    cam_rep = camera_report(geo, cams)
    frames = build_frames(geo, cams, layout)
    report = {"what": "camera study B corr_main: IMX519 handheld vs mount, crop/zoom of the decoded 640 main frame",
              "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "layout": str(Path(a.layout)), "cameras": cam_rep,
              "frames": [{k: fr[k] for k in ("name", "row", "width_m", "grasp_row", "tcp_height_m", "tilt_deg",
                                                 "eraser_O")} | {"n_points": {c: int((fr["ok"]["handheld"][c] & fr["ok"]["mount"][c]).sum()) for c in G.CLASSES}}
                         for fr in frames],
              "offsets_224px": {}, "fits": {}, "training_crop_224px": CROP_224}
    for src in ("handheld", "handheld_nominal", "handheld_e2rel"):
        for dst in ("mount", "mount_trrel"):
            if src != "handheld" and dst != "mount":
                continue
            report["offsets_224px"][f"{dst}_vs_{src}"] = class_offsets(frames, src, dst)
    for tgt in TARGETS:
        for fname, classes in FITS.items():
            pr = pairs(frames, "mount", tgt, classes)
            crop = fit_similarity(pr["p"], pr["q"], pr["w"])
            prall = pairs(frames, "mount", tgt, G.CLASSES)
            by_label = {}
            for lab in ("first", "approach_1of3", "approach_2of3"):
                sel = [k for k, fr in enumerate(frames) if fr["label"] == lab]
                m = np.isin(prall["frame"], sel)
                sub = {k: v[m] for k, v in prall.items()}
                by_label[lab] = {"before": residuals(sub, None), "after": residuals(sub, crop)}
            report["fits"][f"{tgt}/{fname}"] = {
                "classes": list(classes), "scale": round(crop.scale, 5), "zoom_pct": round((crop.scale - 1) * 100, 3),
                "rotation_deg": round(crop.rotation_deg, 4), "shift_px_640": np.round(crop.shift_px, 3).tolist(),
                "shift_px_224": (np.asarray(crop.shift_px) * S224).round(3).tolist(),
                "residual_224px_before": residuals(prall, None), "residual_224px_after": residuals(prall, crop),
                "by_label": by_label, "border": border(crop), "params": crop.to_dict()}
            if tgt == "handheld" and fname in ("all", "fingers_tags", "table"):
                path = CORR_DIR / ("main_crop_corr.yaml" if fname == "all" else f"main_crop_corr_{fname}.yaml")
                hdr = (f"IMX519 crop/zoom of the decoded 640x480 main frame (camera study B corr_main, sim only).\n"
                       f"Fit: mount (model cam_imx519) -> handheld (training rig + E7 fix), classes {'+'.join(classes)},\n"
                       f"class-balanced least squares over the 15 reference frames of layout.json.\n"
                       f"Generated by sim_deploy/corrections/fit_main.py {report['generated_utc']}; load with main_crop.load().")
                MC.dump(crop, path, hdr, {"fit": {"target": tgt, "classes": list(classes),
                                                  "residual_224px_after": residuals(prall, crop)["all_points"],
                                                  "residual_224px_before": residuals(prall, None)["all_points"]}})
                report["fits"][f"{tgt}/{fname}"]["yaml"] = str(path.relative_to(REPO))
    crop = MC.load(CORR_DIR / "main_crop_corr.yaml")
    report["e2e_check"] = e2e_check(geo, frames, crop, out)
    # overlays on the real training main images
    refs = None
    rp = out / "refs_main.npz"
    if rp.is_file():
        refs = np.load(rp)
    crops = {k: MC.load(CORR_DIR / ("main_crop_corr.yaml" if k == "all" else f"main_crop_corr_{k}.yaml"))
             for k in ("all", "fingers_tags", "table")}
    tiles = []
    for fr in frames:
        ref = refs[fr["name"]] if refs is not None and fr["name"] in refs else None
        tiles.append(overlay(fr, ref, crop, title="fit: all"))
    for name in ("ep14_first", "ep14_approach_2of3"):
        fr = next(f for f in frames if f["name"] == name)
        ref = refs[name] if refs is not None and name in refs else None
        row = [overlay(fr, ref, c, title=f"fit: {k}") for k, c in crops.items()]
        cv2.imwrite(str(out / f"overlay_{name}_fits.png"), np.concatenate(row, axis=1)[:, :, ::-1])
    small = [cv2.resize(t, (448, 448), interpolation=cv2.INTER_AREA) for t in tiles]
    rows = [np.concatenate(small[i:i + 3], axis=1) for i in range(0, len(small), 3)]
    cv2.imwrite(str(out / "overlay_all.png"), np.concatenate(rows, axis=0)[:, :, ::-1])
    (out / "fit.json").write_text(json.dumps(report, indent=1))
    _print(report)
    return 0


def _print(r: dict) -> None:
    c = r["cameras"]
    print("[corr_main] IMX519 in the TCP frame (mm):")
    for k, v in c["tcp_T_cam_mm"].items():
        print(f"   {k:17s} {v['pos']} tilt {v['tilt_about_tcp_x_deg']}  vs handheld {c['vs_handheld'][k]}")
    print("[corr_main] checks:", json.dumps(c["checks"]))
    for k, off in r["offsets_224px"].items():
        print(f"[corr_main] offsets {k}:")
        for cls, s in off.items():
            print(f"   {cls:8s} {s}")
    for k, f in r["fits"].items():
        b, a = f["residual_224px_before"], f["residual_224px_after"]
        print(f"[corr_main] fit {k:28s} scale {f['scale']:.4f} rot {f['rotation_deg']:+.3f} shift640 {f['shift_px_640']} "
              f"border {f['border']['black_fraction_640']}")
        for cls in (*G.CLASSES, "all_points"):
            print(f"      {cls:10s} before {b[cls]}  after {a[cls]}")
    print("[corr_main] e2e:", json.dumps(r["e2e_check"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.corrections.fit_main", description=__doc__.splitlines()[0])
    ap.add_argument("--layout", type=Path, default=LAYOUT)
    ap.add_argument("--out", type=Path, default=OUT)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
