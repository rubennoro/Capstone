"""W1 / S6 question 2 (+ the image part of 4): the real robot's policy inputs vs training's first frames.

Real: the probe set written just before each trial's episode start (the arm at the start pose, eraser placed; probe =
a second C1 client of the same frames process, so the same ImagePrep as the manager's) and the ~1 s of sets before it.
Training: the first sample (row 6) of every demo of pick1-double-v2 (``real_trials_train_ref.py``: PICK_TEST1's 30 =
pick1-v1, PICK_TEST2's 32), and the finger tags vs width over the first 6 s of every demo (``tags_vs_width.json``).

Per image (picture rows 28..195 of the letterboxed 224): mean RGB, brightness (luma mean), contrast (luma std),
saturation; the eraser (yellow/lime blob, real_trials_common.eraser: centroid, area, bbox, colour); the finger tags (ArUco DICT_4X4_50 ids 0/1, the
detector of sim_pi/handheld_check.py: centre, mean side in px). Training's tags are read at the robot's opening with a
straight-line fit of tag x / y / side vs recorded width (rms = the tolerance). Cross-check of the tag scale before
ImagePrep: the raw 640x480 IMX219 frames on the arm (``~/policy_deploy_sessions/2026-11-20_16-32-25_pi_collect/jpeg``)
vs B1's training ``raw640_uw_*.png`` (the same INTER_AREA 640x480 the Pi streams). Eraser track after the start
(main image, 0..6 s) for question 4.

Montage per trial (``montage_<task>_<run>_trialN.png``): row 1 the real start set (main T-50, main T, IMX219 T-100,
IMX219 T), row 2 the nearest training first frame (eraser position in the main image, among the checkpoint's
demos), row 3 overlays (real images; training's eraser centroids = cyan dots, real eraser = yellow cross; training's
tag 1 at the robot width = red square of the fitted size, real tag 1 = green).

Run (light env, CPU, < 1 min)::

    cd ~/Capstone-main-sim && unset PYTHONPATH && ~/envs/sim_policy_deploy/bin/python -m sim_deploy.offline.real_trials_images

Needs ``inventory.json`` (real_trials_inventory) and ``train_first/pick1-double-v2/`` (real_trials_train_ref, both
modes). Output: ``sim_evals/_pipeline/W1/real_trials/images.json`` + ``montage_*.png`` + ``montage_all_starts.png``.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from sim_deploy.offline.real_trials_common import (B1_REF, OUT, SESSIONS, TagDetector, appearance, copy_version,
                                                   eraser, set_paths, set_times, tags_of, wall, write_json)

cv2.setNumThreads(4)
PKG = "pick1-double-v2"
TF = OUT / "train_first" / PKG
ROBOT_WIDTH_TRAINING_UNITS = 0.090      # gripper_map.to_model(0.040), the open robot hand as the model got it
KEYS = (("main_0", "main", 0), ("main_1", "main", 1), ("ultrawide_0", "ultrawide", 0), ("ultrawide_1", "ultrawide", 1))


def rgb(p: Path) -> np.ndarray:
    im = cv2.imread(str(p))
    if im is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def measure(det, ims: dict) -> dict:
    out = {}
    for k, im in ims.items():
        out[k] = {"appearance": appearance(im), "eraser": eraser(im), "tags": tags_of(det, im)}
    return out


def agg(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    return {"mean": np.round(a.mean(0), 2).tolist(), "std": np.round(a.std(0), 2).tolist(),
            "min": np.round(a.min(0), 2).tolist(), "max": np.round(a.max(0), 2).tolist(), "n": int(len(a))}


def summarize(meas: list[dict], cam: str) -> dict:
    ap = [m[cam]["appearance"] for m in meas]
    er = [m[cam]["eraser"] for m in meas]
    s = {"mean_rgb": agg([a["mean_rgb"] for a in ap]), "luma_mean": agg([a["luma_mean"] for a in ap]),
         "luma_std": agg([a["luma_std"] for a in ap]), "sat_mean": agg([a["sat_mean"] for a in ap]),
         "eraser_centroid": agg([e["centroid"] if e else None for e in er]),
         "eraser_area_px": agg([e["area_px"] if e else None for e in er]),
         "eraser_rgb": agg([e["mean_rgb"] if e else None for e in er]),
         "eraser_found": f"{sum(e is not None for e in er)}/{len(er)}"}
    for tid in ("0", "1"):
        tg = [m[cam]["tags"].get(tid) for m in meas]
        s[f"tag{tid}_centre"] = agg([t["centre"] if t else None for t in tg])
        s[f"tag{tid}_side_px"] = agg([t["side_px"] if t else None for t in tg])
        s[f"tag{tid}_found"] = f"{sum(t is not None for t in tg)}/{len(tg)}"
    return s


def tag_fit() -> dict:
    d = json.loads((TF / "tags_vs_width.json").read_text())["frames"]
    out = {}
    for tid in ("0", "1"):
        F = np.array([(f["width_m"], *f["tags_uw"][tid]["centre"], f["tags_uw"][tid]["side_px"])
                      for f in d if tid in f["tags_uw"]])
        A = np.c_[F[:, 0], np.ones(len(F))]
        fit = {"n": int(len(F)), "width_range_m": [round(float(F[:, 0].min()), 4), round(float(F[:, 0].max()), 4)]}
        for j, n in ((1, "x"), (2, "y"), (3, "side")):
            c = np.linalg.lstsq(A, F[:, j], rcond=None)[0]
            r = F[:, j] - A @ c
            fit[n] = {"slope_px_per_m": round(float(c[0]), 2), "at_0_m": round(float(c[1]), 2),
                      "rms_px": round(float(r.std()), 3), "max_abs_resid_px": round(float(np.abs(r).max()), 2)}
        fit["predict"] = {f"{w:.3f}": {n: round(fit[n]["slope_px_per_m"] * w + fit[n]["at_0_m"], 2)
                                       for n in ("x", "y", "side")} for w in (0.040, 0.051, 0.061, 0.090)}
        out[tid] = fit
    return out


def raw640_check(det2) -> dict:
    """Tag side at the Pi's 640x480 (before ImagePrep): robot (pi_collect, 16:32) vs training (B1 raw640_uw)."""
    robot = []
    for f in sorted(glob.glob(str(SESSIONS / "2026-11-20_16-32-25_pi_collect/jpeg/*_ultrawide.jpg"))):
        robot.append({"file": Path(f).name, "tags": tags_of(det2, rgb(Path(f)))})
    train = []
    for f in sorted(glob.glob(str(B1_REF / "ep*_first/raw640_uw_1.png"))):
        train.append({"file": "/".join(Path(f).parts[-2:]), "tags": tags_of(det2, rgb(Path(f)))})
    w_train = {}
    for f in sorted(glob.glob(str(B1_REF / "ep*_first/meta.json"))):
        w_train[Path(f).parent.name] = json.loads(Path(f).read_text())["widths_m"][-1]

    def sides(rows, tid):
        return agg([r["tags"][tid]["side_px"] for r in rows if tid in r["tags"]])

    def cen(rows, tid):
        return agg([r["tags"][tid]["centre"] for r in rows if tid in r["tags"]])
    return {"detector_scale": 2, "robot": robot, "training": train, "training_widths_m": w_train,
            "summary": {f"tag{t}": {"robot_side": sides(robot, t), "training_side": sides(train, t),
                                    "robot_centre": cen(robot, t), "training_centre": cen(train, t)}
                        for t in ("0", "1")}}


def montage(path: Path, real: dict, tr: dict, tr_label: str, real_label: str, tr_eraser_main, fit1, real_m) -> None:
    def up(im):
        return cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    r1 = [up(real[k]) for k, _, _ in KEYS]
    r2 = [up(tr[c][i]) for _, c, i in KEYS]
    r3 = []
    for k, cam, _ in KEYS:
        im = up(real[k]).copy()
        if cam == "main":
            for c in tr_eraser_main:
                cv2.circle(im, (int(2 * c[0]), int(2 * c[1])), 3, (0, 255, 255), -1)
            e = real_m[k]["eraser"]
            if e:
                x, y = (int(2 * v) for v in e["centroid"])
                cv2.drawMarker(im, (x, y), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        else:
            p = fit1["predict"]["0.090"]
            s = p["side"]
            cv2.rectangle(im, (int(2 * (p["x"] - s / 2)), int(2 * (p["y"] - s / 2))),
                          (int(2 * (p["x"] + s / 2)), int(2 * (p["y"] + s / 2))), (255, 0, 0), 2)
            t = real_m[k]["tags"].get("1")
            if t:
                pts = (np.array(t["corners"]) * 2).astype(np.int32)
                cv2.polylines(im, [pts], True, (0, 255, 0), 2)
        r3.append(im)
    rows = [np.hstack(r) for r in (r1, r2, r3)]
    W = rows[0].shape[1]
    bars = []
    for txt in (f"REAL start set: {real_label} (main T-50, main T, IMX219 T-100, IMX219 T)",
                f"TRAINING nearest first frame: {tr_label}",
                "overlay on real: cyan = training eraser centroids, yellow + = real eraser; red = training tag 1 at "
                "open width (fit), green = real tag 1"):
        b = np.zeros((22, W, 3), np.uint8)
        cv2.putText(b, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        bars.append(b)
    out = np.vstack([bars[0], rows[0], bars[1], rows[1], bars[2], rows[2]])
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def main() -> int:
    det = TagDetector(4)
    inv = json.loads((OUT / "inventory.json").read_text())
    idx = json.loads((TF / "index.json").read_text())["episodes"]
    tr_meas, tr_imgs = [], []
    for e in idx:
        z = np.load(TF / f"ep{e['index']:02d}.npz")
        ims = {"main": z["main"], "ultrawide": z["ultrawide"]}
        tr_imgs.append(ims)
        tr_meas.append({"main": measure(det, {"x": z["main"][1]})["x"],
                        "ultrawide": measure(det, {"x": z["ultrawide"][1]})["x"],
                        "folder": e["folder"], "index": e["index"], "width": float(z["gripper_width_m"][1, 0])})
    groups = {"PICK_TEST1": [m for m in tr_meas if m["folder"] == "PICK_TEST1"],
              "PICK_TEST2": [m for m in tr_meas if m["folder"] == "PICK_TEST2"], "both": tr_meas}
    training = {g: {cam: summarize(ms, cam) for cam in ("main", "ultrawide")} for g, ms in groups.items()}
    fit = tag_fit()
    trials_out = []
    all_starts = []
    for t in inv["trials"]:
        times = set_times(t["folder"])
        i0 = t["probe"]["set_at_start"]
        ws = t["episode"]["wall_start_epoch"]
        ims = {k: rgb(p) for k, p in set_paths(t["folder"], i0).items()}
        m0 = measure(det, ims)
        pre = [i for i in range(len(times)) if ws - 1.0 <= times[i] < ws]
        pre_m = [measure(det, {k: rgb(p) for k, p in set_paths(t["folder"], i).items()}) for i in pre]
        pre_s = {cam: summarize([{cam: m[f"{cam}_1"]} for m in pre_m], cam) for cam in ("main", "ultrawide")}
        # eraser track (main T image) and tag 1 (IMX219 T) from the start to +6 s
        track = []
        for i in range(len(times)):
            dt = times[i] - ws
            if -1.0 <= dt <= 6.0:
                im = rgb(set_paths(t["folder"], i)["main_1"])
                e = eraser(im)
                track.append({"set": i, "t_s": round(float(dt), 2), "eraser_main": e and
                              {"centroid": e["centroid"], "area_px": e["area_px"]}})
        # nearest training first frame by the eraser centroid in the main image (the checkpoint's demos)
        pool = tr_meas if t["task"] == "PICK_DOUBLE" else groups["PICK_TEST1"]
        er = m0["main_1"]["eraser"]
        if er:
            dist = [np.linalg.norm(np.subtract(m["main"]["eraser"]["centroid"], er["centroid"]))
                    if m["main"]["eraser"] else 1e9 for m in pool]
            near = pool[int(np.argmin(dist))]
            near_d = float(min(dist))
        else:
            near, near_d = pool[0], None
        tr_er = [m["main"]["eraser"]["centroid"] for m in pool if m["main"]["eraser"]]
        name = f"montage_{t['task']}_{t['run'][:7]}_trial{t['trial']}.png"
        montage(OUT / name, ims, tr_imgs[near["index"]],
                f"{near['folder']} package index {near['index']} (eraser {near_d and round(near_d, 1)} px away)",
                f"{t['label']} set {i0} ({t['probe']['set_at_start_age_s']} s before the start)", tr_er, fit["1"], m0)
        all_starts.append(np.hstack([ims["main_1"], ims["ultrawide_1"]]))
        tag1 = m0["ultrawide_1"]["tags"].get("1")
        p = fit["1"]["predict"]
        trials_out.append({
            "label": t["label"], "folder": t["folder"], "set_at_start": i0, "montage": str(OUT / name),
            "start_set": m0, "pre_start_1s": pre_s, "n_pre_sets": len(pre),
            "nearest_training": {"index": near["index"], "folder": near["folder"], "eraser_dist_px": near_d},
            "tag1_vs_training": None if tag1 is None else {
                "real": {"centre": tag1["centre"], "side_px": tag1["side_px"]},
                "training_at_0.090": p["0.090"], "training_at_0.051": p["0.051"],
                "d_y_px_vs_0.090": round(tag1["centre"][1] - p["0.090"]["y"], 2),
                "side_ratio_vs_0.090": round(tag1["side_px"] / p["0.090"]["side"], 3),
                "side_ratio_vs_0.051": round(tag1["side_px"] / p["0.051"]["side"], 3)},
            "eraser_track_main": track})
        print(t["label"], "eraser main", er and er["centroid"], "area", er and er["area_px"], "tag1", tag1 and
              (tag1["centre"], tag1["side_px"]), "near", near["index"], flush=True)
    g = [np.hstack(all_starts[i:i + 3]) for i in range(0, len(all_starts), 3)]
    w = max(x.shape[1] for x in g)
    g = [np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in g]
    cv2.imwrite(str(OUT / "montage_all_starts.png"), cv2.cvtColor(np.vstack(g), cv2.COLOR_RGB2BGR))
    out = {"copy_version": copy_version(), "package": PKG, "training_row": 6,
           "robot_width_training_units": ROBOT_WIDTH_TRAINING_UNITS,
           "training": training, "training_per_episode": tr_meas, "tag_fit_uw_vs_width": fit,
           "raw640_tag_check": raw640_check(TagDetector(2)), "trials": trials_out,
           "notes": ["picture rows 28..195 only (letterbox)", "tags: ArUco DICT_4X4_50 ids 0/1, x4 upscale",
                     "eraser: G>=R, G-B>60, G>110, largest blob >= 60 px; run_022 trial 4 shows a dark object (not found)"]}
    write_json(OUT / "images.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
