"""Gripper width map: sweep the servo, measure the training width from the finger AR tags in RigCamera ultrawide
frames with the SAME detector as iPhUMI (iphumi detect_localize_aruco_tags_iphone: ArUco + solvePnP per tag, marker
sizes from iPhUMI's aruco_config.yaml) and the same width definition (right tag x - left tag x, minus the closed
value, so 0 = closed). Fit a monotonic table servo position <-> width for gripper.driver.WidthCalibration.
Checks: monotonic, 0 at closed, covers >= --min-coverage-m (default 0.085 m; with a station: tooling.width_map.coverage_required_m).

    python -m franka_deploy fdcal width --data DIR [--left-id 0 --right-id 1] [--out calib.yaml]
        [--min-coverage-m 0.085] [--deploy-config D --profile P]   # with a station: K and coverage from it
DIR/sweep.json = [{"position": p, "image": "uw_000.png"}, ...] (RigCamera ultrawide, rectified); DIR/K.json optional
(default: the virtual ultrawide at 960x720).

Franka Hand (--linear): the sweep comes from `fdcal capture width --hand-endpoint E` (entries carry gap_m, the hand's
own read-back, and position = the commanded gap). The map is LINEAR, `train = a * gap + b` (tooling.width_map), not a table:
    python -m franka_deploy fdcal width --linear --data DIR --deploy-config D --profile P [--anchor closed|raw] [--out fit.json]
- raw tag separation per image (same detector as above), fitted raw = a * gap + c by least squares (c = the tag
  separation at gap 0 of THESE fingers, a = the slope: 1.0 if the hand's gap and the tag geometry agree).
- b: `--anchor closed` (default) sets b = tooling.width_map.train_closed_m (0.0431, the TRAINING data's width at its
  closed gripper; never the sweep minimum), so the policy sees the training convention whatever the new tags'
  closed separation c is. `--anchor raw` sets b = c (the policy sees the raw tag separation of these fingers).
  The difference c - train_closed_m is reported: it is the tag-geometry offset between these fingers and the training
  gripper (> 3 mm is a finding).
- checks: monotonic (no drop > 1 mm between neighbouring gaps), residual p95 <= 1 mm, the gaps that map to the training
  range 0.0447-0.0844 lie inside the sweep and inside 0-0.080. Also printed: the gap the demo hold widths
  0.046-0.062 map to, to compare with the eraser side the demos grasp (20 mm eraser: contact at gap 0.012).
The tool prints a `--set tooling.width_map.a=..` / `.b=..` snippet and a provenance line; it never edits station YAML.
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


TRAIN_HOLD_RANGE = (0.046, 0.062)     # demo hold widths (A5: firstreal-single), used for the re-check printout


def fit_linear(gaps, raw_widths, train_closed_m: float, anchor: str = "closed", train_range=(0.0447, 0.0844),
               stroke_m: float = 0.080, mono_tol_m: float = 0.001) -> dict:
    """Linear Franka Hand width map from a gap sweep. raw = a*gap + c (least squares); b = train_closed_m (anchor
    "closed") or c (anchor "raw"). Pure numpy, so it is tested with synthetic tag separations."""
    if anchor not in ("closed", "raw"):
        raise ValueError("anchor must be closed or raw")
    g = np.asarray(gaps, float)
    w = np.asarray(raw_widths, float)
    ok = np.isfinite(w) & np.isfinite(g)
    g, w = g[ok], w[ok]
    if len(g) < 4 or np.ptp(g) <= 0:
        return {"error": f"only {len(g)} usable points (tags missing in the others?)", "n_points": int(len(g))}
    order = np.argsort(g)
    g, w = g[order], w[order]
    a, c = (float(x) for x in np.polyfit(g, w, 1))
    b = float(train_closed_m) if anchor == "closed" else c
    res = w - (a * g + c)                      # residual of the raw fit: the map is a*gap + (b - c + c), same shape
    drops = np.diff(w)
    lo, hi = train_range
    gl, gh = (lo - b) / a, (hi - b) / a
    sweep_lo, sweep_hi = float(g.min()), float(g.max())
    hold_gap = [(x - b) / a for x in TRAIN_HOLD_RANGE]
    checks = {"monotonic": bool(np.all(drops > -mono_tol_m)), "max_drop_mm": float(max(0.0, -drops.min()) * 1e3),
              "residual_p95_mm": float(np.percentile(np.abs(res), 95) * 1e3),
              "residual_ok": bool(np.percentile(np.abs(res), 95) <= 0.001),
              "train_range_gaps_m": [float(gl), float(gh)], "train_range_in_stroke": bool(gl >= -1e-4 and gh <= stroke_m + 1e-4),
              "train_range_in_sweep": bool(gl >= sweep_lo - 1e-3 and gh <= sweep_hi + 1e-3)}
    checks["ok"] = bool(checks["monotonic"] and checks["residual_ok"] and checks["train_range_in_stroke"]
                        and checks["train_range_in_sweep"] and a > 0)
    return {"a": float(a), "b": float(b), "c_raw_at_gap0_m": c, "anchor": anchor, "train_closed_m": float(train_closed_m),
            "tag_offset_vs_training_mm": float((c - train_closed_m) * 1e3), "n_points": int(len(g)),
            "sweep_gap_range_m": [sweep_lo, sweep_hi], "hold_width_gaps_m": [float(x) for x in hold_gap],
            "checks": checks}


def linear_snippet(fit: dict, out: str, today: str | None = None) -> dict:
    import time

    return {"set": [f"tooling.width_map.a={fit['a']:.4f}", f"tooling.width_map.b={fit['b']:.4f}"],
            "provenance_line": f'width_map: {{status: measured, source: "fdcal width --linear {today or time.strftime("%Y-%m-%d")} {out}"}}'}


def _main_linear(a, ap) -> int:
    if not (a.deploy_config and a.profile):
        ap.error("--linear needs --deploy-config and --profile (K and train_closed_m come from the station)")
    from franka_deploy.launch.launcher import load_config
    from franka_deploy.station import station_from_deploy_cfg

    st = station_from_deploy_cfg(load_config(a.deploy_config, a.profile))
    if st is None:
        raise SystemExit(f"profile '{a.profile}' of {a.deploy_config} has no station: block")
    K_default = np.array(st.get("sensor.streams.ultrawide.K"), float)
    wm = st.get("tooling.width_map")
    sweep = json.load(open(os.path.join(a.data, "sweep.json")))
    kp = os.path.join(a.data, "K.json")
    K = np.array(json.load(open(kp))) if os.path.exists(kp) else K_default
    d, sizes = load_aruco()
    gaps, raw = [], []
    for s in sweep:
        img = cv2.imread(os.path.join(a.data, s["image"]), cv2.IMREAD_GRAYSCALE)
        r = raw_width(img, K, a.left_id, a.right_id, d, sizes)
        gaps.append(s.get("gap_m", s.get("position")) if a.gap_key == "gap_m" else s["position"])
        raw.append(np.nan if r is None else r)
    fit = fit_linear(gaps, raw, wm["train_closed_m"], a.anchor, (wm["train_min_m"], wm["train_max_m"]), wm["gap_max_m"])
    if "error" in fit:
        print(json.dumps(fit))
        return 1
    fit["snippet"] = linear_snippet(fit, a.out or "<out>")
    c = fit["checks"]
    print(f"raw = a*gap + c: a {fit['a']:.4f}, c {fit['c_raw_at_gap0_m'] * 1e3:.2f} mm "
          f"(training closed {fit['train_closed_m'] * 1e3:.2f} mm; tag offset {fit['tag_offset_vs_training_mm']:+.2f} mm); "
          f"n {fit['n_points']}, residual p95 {c['residual_p95_mm']:.2f} mm, monotonic {c['monotonic']}")
    print(f"width_map with anchor {a.anchor}: a {fit['a']:.4f}, b {fit['b']:.4f}; training range 0.0447-0.0844 = gap "
          f"{c['train_range_gaps_m'][0] * 1e3:.1f}-{c['train_range_gaps_m'][1] * 1e3:.1f} mm "
          f"(in stroke {c['train_range_in_stroke']}, in sweep {c['train_range_in_sweep']})")
    print(f"demo hold widths 0.046-0.062 -> gap {fit['hold_width_gaps_m'][0] * 1e3:.1f}-{fit['hold_width_gaps_m'][1] * 1e3:.1f} mm: "
          "compare with the eraser side the demos grasp (20 mm eraser: pad contact at gap 12 mm); watch a demo video to say which side")
    if abs(fit["tag_offset_vs_training_mm"]) > 3.0:
        print("FINDING: the tag separation at gap 0 differs from the training closed value by > 3 mm: these fingers' tag geometry "
              "is not the training gripper's; weigh --anchor raw against --anchor closed")
    for line in fit["snippet"]["set"]:
        print(f"  --set {line}")
    print(f"  provenance  {fit['snippet']['provenance_line']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(fit, indent=1))
    return 0 if c["ok"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy fdcal width", description=__doc__.split("\n")[0])
    ap.add_argument("--linear", action="store_true", help="Franka Hand: fit train = a*gap + b (needs --deploy-config/--profile)")
    ap.add_argument("--anchor", choices=["closed", "raw"], default="closed", help="--linear: b = train_closed_m (default) or the fitted c")
    ap.add_argument("--gap-key", choices=["gap_m", "position"], default="gap_m", help="--linear: the hand's read-back gap or the commanded one")
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
    if a.linear:
        return _main_linear(a, ap)
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
