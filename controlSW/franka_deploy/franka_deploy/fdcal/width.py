"""Gripper width map: sweep the servo, measure the training width from the finger AR tags in RigCamera ultrawide
frames with the SAME detector as iPhUMI (iphumi detect_localize_aruco_tags_iphone: ArUco + solvePnP per tag, marker
sizes from iPhUMI's aruco_config.yaml) and the same width definition (right tag x - left tag x, minus the closed
value, so 0 = closed). Fit a monotonic table servo position <-> width for gripper.driver.WidthCalibration.
Checks: monotonic, 0 at closed, covers >= --min-coverage-m (default 0.085 m; with a station: tooling.width_map.coverage_required_m).

    python -m franka_deploy fdcal width --data DIR [--left-id 0 --right-id 1] [--out calib.yaml]
        [--min-coverage-m 0.085] [--deploy-config D --profile P]   # with a station: K and coverage from it
DIR/sweep.json = [{"position": p, "image": "uw_000.png"}, ...] (RigCamera ultrawide, rectified); DIR/K.json optional
(default: the virtual ultrawide at 960x720).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

K_UW_960 = np.array([[388.5231, 0, 480.6903], [0, 388.5231, 359.7618], [0, 0, 1.0]])
ARUCO_CFG = (Path(__file__).resolve().parents[4] / "iPhUMI" / "python_package" / "iphumi" / "demonstration_processing"
             / "calibration" / "aruco_config.yaml")


def load_aruco():
    cfg = yaml.safe_load(open(ARUCO_CFG))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["aruco_dict"]["predefined"]))
    sizes = {int(k): float(v) for k, v in cfg["marker_size_map"].items() if k != "default"}
    return d, sizes


def raw_width(img, K, left_id, right_id, aruco_dict, sizes):
    from iphumi.demonstration_processing.scripts.detect_aruco_iphone import detect_localize_aruco_tags_iphone

    tags = detect_localize_aruco_tags_iphone(img, aruco_dict, sizes, K)
    if left_id in tags and right_id in tags:
        return float(tags[right_id]["tvec"][0] - tags[left_id]["tvec"][0])
    return None


def fit_map(positions, raw_widths, min_coverage_m: float = 0.085) -> dict:
    p = np.asarray(positions, float)
    w = np.asarray(raw_widths, float)
    ok = np.isfinite(w)
    p, w = p[ok], w[ok]
    order = np.argsort(w)
    w, p = w[order], p[order]
    closed = w[0]
    width = w - closed
    # enforce strict monotonicity of position in width (pool adjacent violators by direction of the trend)
    inc = np.polyfit(width, p, 1)[0] > 0
    pm = np.maximum.accumulate(p) if inc else np.minimum.accumulate(p)
    keep = np.r_[True, np.diff(width) > 1e-5] & np.r_[True, np.abs(np.diff(pm)) > 1e-9]
    covers = bool(width.max() >= min_coverage_m)
    table = {"width_m": width[keep].round(6).tolist(), "position": pm[keep].round(3).tolist()}
    table["checks"] = {"zero_at_closed": bool(abs(width[0]) < 1e-9), "max_width_m": float(width.max()),
                       "covers_required": covers, "covers_0p085": covers,   # covers_0p085: old name, same value
                       "min_coverage_m": float(min_coverage_m), "monotonic": True, "n_points": int(keep.sum())}
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--left-id", type=int, default=0)
    ap.add_argument("--right-id", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-coverage-m", type=float, default=None,
                    help="width the sweep must reach (default 0.085; with a station: tooling.width_map.coverage_required_m)")
    ap.add_argument("--deploy-config", default=None, help="with --profile: K and coverage come from the station")
    ap.add_argument("--profile", default=None)
    a = ap.parse_args(argv)
    if bool(a.deploy_config) != bool(a.profile):
        ap.error("--deploy-config and --profile go together")
    K_default, min_cov = K_UW_960, 0.085
    if a.deploy_config:
        from franka_deploy.launch.launcher import load_config
        from franka_deploy.station import station_from_deploy_cfg

        st = station_from_deploy_cfg(load_config(a.deploy_config, a.profile))
        if st is None:
            raise SystemExit(f"profile '{a.profile}' of {a.deploy_config} has no station: block")
        K_default = np.array(st.get("sensor.streams.ultrawide.K"), float)
        min_cov = float(st.get("tooling.width_map.coverage_required_m"))
    if a.min_coverage_m is not None:
        min_cov = a.min_coverage_m
    sweep = json.load(open(os.path.join(a.data, "sweep.json")))
    kp = os.path.join(a.data, "K.json")
    K = np.array(json.load(open(kp))) if os.path.exists(kp) else K_default
    d, sizes = load_aruco()
    pos, raw = [], []
    for s in sweep:
        img = cv2.imread(os.path.join(a.data, s["image"]), cv2.IMREAD_GRAYSCALE)
        r = raw_width(img, K, a.left_id, a.right_id, d, sizes)
        pos.append(s["position"])
        raw.append(np.nan if r is None else r)
    table = fit_map(pos, raw, min_cov)
    print(json.dumps(table["checks"]))
    if a.out:
        yaml.safe_dump({"calibration": {k: table[k] for k in ("width_m", "position")}}, open(a.out, "w"))
    c = table["checks"]
    return 0 if c["zero_at_closed"] and c["covers_required"] else 2
