"""Intrinsics of the DEPLOYMENT stream (960x720 over Wi-Fi), compared with the station K and the training variants.

Why: the policy was trained on frames whose K came from the iPhone's own calibration (two variants, 39 + 6 demos,
`sensor.training_intrinsics_variants`). At deployment the app streams a resized, JPEG-compressed 960x720 image whose
effective K may differ (cropping, a different format, UW at deployment is not the training 640x480 video). A mismatch
shifts every pixel the policy sees; it is a finding, not something the snippet fixes (the policy keeps its training K).

Data: `fdcal capture stream` saves frames as DIR/<stream>_NNN.png (the camera server's output, i.e. after the receiver:
exactly what the policy sees). A fixed ChArUco board (the hand-eye board, handeye.make_board), 20-30 views per stream
with the board near every part of the frame and tilted +-30 degrees.

Per stream: cv2.calibrateCamera at the image size, started from the station K. --distortion none (default: the iPhone
streams are expected to be rectified) or k1k2 (to test that). Reported: f, c, reprojection RMS, relative f and c
differences against the station K and against both training variants scaled to 960x720 (half-pixel rule,
iphone_replay_sender.scale_K; a variant without uw_c_px compares f only), the nearest variant, and the 3x3 coverage of
the frame by detected corners.

Verdict: `match` if |df| <= 0.5 % and |dc| <= 3 px against the station K; otherwise a `--set` snippet for
`sensor.streams.<s>.K` (apply by hand; the tool never edits station YAML) and a provenance line.

    python -m franka_deploy fdcal capture stream --out DIR --n 30 [--camera tcp://localhost:16702] [--stream main|ultrawide|both]
    python -m franka_deploy fdcal stream-intrinsics --data DIR --config deploy.yaml --profile P [--squares-x 9 --squares-y 7
        --square-m 0.03 --marker-m 0.022 --dict DICT_5X5_100] [--stream main|ultrawide|both] [--distortion none|k1k2] --out DIR
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

F_TOL_PCT, C_TOL_PX = 0.5, 3.0
MIN_VIEWS, MIN_CORNERS = 5, 6
SIZE_960 = (960, 720)


def detect_views_from_arrays(images, board, min_corners: int = MIN_CORNERS, names=None):
    """[(name, obj_pts (N,3), img_pts (N,2), image_size_wh)] for every grey image where >= min_corners corners were found."""
    det = cv2.aruco.CharucoDetector(board)
    out = []
    for i, img in enumerate(images):
        if img is None:
            continue
        corners, ids, _, _ = det.detectBoard(img)
        if ids is None or len(ids) < min_corners:
            continue
        obj = board.getChessboardCorners()[ids.reshape(-1)].astype(np.float32)
        out.append((str(names[i]) if names else str(i), obj, corners.reshape(-1, 2).astype(np.float32), (img.shape[1], img.shape[0])))
    return out


def detect_views(files, board, min_corners: int = MIN_CORNERS):
    """Same for image files (read as grey)."""
    files = list(files)
    return detect_views_from_arrays([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in files], board, min_corners, files)


def coverage(views, size_wh) -> dict:
    """3x3 grid counts of detected corners over all views, and the number of cells hit per view."""
    w, h = size_wh
    grid = np.zeros((3, 3), int)
    per_view = []
    for _, _, ip, _ in views:
        g = np.zeros((3, 3), int)
        cx = np.clip((ip[:, 0] / w * 3).astype(int), 0, 2)
        cy = np.clip((ip[:, 1] / h * 3).astype(int), 0, 2)
        for x, y in zip(cx, cy):
            g[y, x] += 1
        grid += g
        per_view.append(int((g > 0).sum()))
    return {"grid_corner_counts": grid.tolist(), "cells_hit": int((grid > 0).sum()), "cells_hit_per_view": per_view}


def calibrate(views, size_wh, K0, distortion: str = "none") -> dict:
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if distortion == "none":
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    elif distortion != "k1k2":
        raise ValueError("distortion must be none or k1k2")
    obj = [v[1] for v in views]
    img = [v[2].reshape(-1, 1, 2) for v in views]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(size_wh), np.asarray(K0, np.float64).copy(), np.zeros(5), flags=flags)
    return {"rms_px": float(rms), "K": K.tolist(), "dist": dist.reshape(-1).tolist(), "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "f": float((K[0, 0] + K[1, 1]) / 2), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "n_views": len(views)}


def compare_K(fit: dict, K_ref, label: str) -> dict:
    K_ref = np.asarray(K_ref, float)
    f_ref = (K_ref[0, 0] + K_ref[1, 1]) / 2
    return {"ref": label, "df_pct": float(100 * (fit["f"] - f_ref) / f_ref), "dcx_px": float(fit["cx"] - K_ref[0, 2]),
            "dcy_px": float(fit["cy"] - K_ref[1, 2]), "f_ref": float(f_ref)}


def variant_Ks(variants: list, stream: str, size_wh=SIZE_960) -> list[dict]:
    """Training variants as 960x720 K (f, and c when recorded). main: 1920x1440 source; ultrawide: 640x480."""
    from franka_deploy.stream.iphone_replay_sender import scale_K

    out = []
    for i, v in enumerate(variants):
        if stream == "main":
            src, f, c = (1920, 1440), v.get("main_1920_f_px"), v.get("main_c_px")
        else:
            src, f, c = (640, 480), v.get("uw_640_f_px"), v.get("uw_c_px")
        if f is None:
            continue
        K = [[f, 0, (c or [src[0] / 2, src[1] / 2])[0]], [0, f, (c or [src[0] / 2, src[1] / 2])[1]], [0, 0, 1]]
        out.append({"variant": i, "n_demos": v.get("n_demos"), "K": scale_K(K, src, size_wh), "has_c": c is not None})
    return out


def judge(fit: dict, K_station, variants: list, stream: str) -> dict:
    st = compare_K(fit, K_station, "station")
    dc = max(abs(st["dcx_px"]), abs(st["dcy_px"]))
    verdict = "match" if abs(st["df_pct"]) <= F_TOL_PCT and dc <= C_TOL_PX else "mismatch"
    comps = []
    for v in variant_Ks(variants, stream):
        c = compare_K(fit, v["K"], f"variant {v['variant']}")
        c.update({"variant": v["variant"], "n_demos": v["n_demos"], "has_c": v["has_c"], "K": v["K"]})
        c["score"] = abs(c["df_pct"]) / F_TOL_PCT + (max(abs(c["dcx_px"]), abs(c["dcy_px"])) / C_TOL_PX if v["has_c"] else 0.0)
        comps.append(c)
    nearest = min(comps, key=lambda c: c["score"])["variant"] if comps else None
    return {"verdict": verdict, "vs_station": {**st, "max_dc_px": dc}, "vs_variants": comps, "nearest_variant": nearest,
            "tolerances": {"f_pct": F_TOL_PCT, "c_px": C_TOL_PX}}


def snippet(stream: str, fit: dict, out_dir: str, today: str | None = None) -> dict:
    K = np.round(np.asarray(fit["K"]), 4).tolist()
    return {"set": [f"sensor.streams.{stream}.K={json.dumps(K)}"],
            "provenance_line": f'streams.{stream}.K: {{status: measured, source: "fdcal stream-intrinsics {today or time.strftime("%Y-%m-%d")} {out_dir}"}}'}


def run_stream(stream: str, files, board, K_station, variants, distortion, out_dir) -> dict:
    views = detect_views(files, board)
    if len(views) < MIN_VIEWS:
        return {"stream": stream, "error": f"only {len(views)} usable views of {len(list(files))} images (need >= {MIN_VIEWS}); "
                                           "check the board is in view, sharp and the --squares/--dict settings"}
    size = views[0][3]
    fit = calibrate(views, size, K_station, distortion)
    res = {"stream": stream, "size_wh": list(size), "n_images": len(list(files)), "distortion": distortion, "fit": fit,
           "coverage": coverage(views, size), **judge(fit, K_station, variants, stream)}
    if size != SIZE_960:
        res["warning"] = f"image size {size} is not 960x720: the station K and the variants are for 960x720"
    if res["verdict"] == "mismatch":
        res["snippet"] = snippet(stream, fit, str(out_dir))
    return res


def _files(data: Path, stream: str):
    pat = re.compile(rf"^{stream}_\d+\.png$")
    return sorted(p for p in data.iterdir() if pat.match(p.name))


def print_result(r: dict) -> None:
    if "error" in r:
        print(f"[{r['stream']}] {r['error']}")
        return
    f = r["fit"]
    st = r["vs_station"]
    print(f"[{r['stream']}] {r['fit']['n_views']} views, rms {f['rms_px']:.3f} px, f {f['f']:.2f} (fx {f['fx']:.2f} fy {f['fy']:.2f}), "
          f"c ({f['cx']:.2f}, {f['cy']:.2f}), dist {r['distortion']}")
    print(f"      vs station K: df {st['df_pct']:+.2f} % dc ({st['dcx_px']:+.2f}, {st['dcy_px']:+.2f}) px  -> {r['verdict'].upper()} "
          f"(tol {F_TOL_PCT} % / {C_TOL_PX} px)")
    for c in r["vs_variants"]:
        print(f"      vs training variant {c['variant']} ({c['n_demos']} demos): df {c['df_pct']:+.2f} % "
              + (f"dc ({c['dcx_px']:+.2f}, {c['dcy_px']:+.2f}) px" if c["has_c"] else "(f only, no recorded c)"))
    print(f"      nearest training variant: {r['nearest_variant']}; coverage {r['coverage']['cells_hit']}/9 cells, "
          f"grid {r['coverage']['grid_corner_counts']}")
    if r["coverage"]["cells_hit"] < 8:
        print("      WARNING: the board never reached some regions of the frame; c and distortion are poorly constrained")
    if "warning" in r:
        print(f"      WARNING: {r['warning']}")
    if "snippet" in r:
        print("      snippet (apply by hand): --set " + r["snippet"]["set"][0])
        print("      provenance: " + r["snippet"]["provenance_line"])
        print("      note: the policy was trained at the training K; a mismatch is a finding to weigh, the snippet does not fix it")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_deploy fdcal stream-intrinsics", description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="DIR from `fdcal capture stream` (<stream>_NNN.png)")
    ap.add_argument("--config", required=True, help="deploy.yaml (station K and training variants)")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--set", action="append", default=[], help="station override group.path=value")
    ap.add_argument("--stream", choices=["main", "ultrawide", "both"], default="both")
    ap.add_argument("--squares-x", type=int, default=9)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-m", type=float, default=0.03)
    ap.add_argument("--marker-m", type=float, default=0.022)
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--distortion", choices=["none", "k1k2"], default="none")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from franka_deploy.fdcal.handeye import make_board
    from franka_deploy.launch.launcher import load_config
    from franka_deploy.station import station_from_deploy_cfg

    cfg = load_config(a.config, a.profile)
    cfg["station_set"] = list(cfg.get("station_set", [])) + list(a.set)
    st = station_from_deploy_cfg(cfg)
    if st is None or "sensor" not in st.groups:
        raise SystemExit(f"profile '{a.profile}' of {a.config} has no station sensor group")
    variants = st.get("sensor.training_intrinsics_variants", [])
    board = make_board({"squares_x": a.squares_x, "squares_y": a.squares_y, "square_m": a.square_m,
                        "marker_m": a.marker_m, "dict": a.dict})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results, worst = {}, 0
    for s in (("main", "ultrawide") if a.stream == "both" else (a.stream,)):
        files = _files(Path(a.data), s)
        if not files:
            print(f"[{s}] no {s}_NNN.png in {a.data}")
            worst = max(worst, 1)
            continue
        r = run_stream(s, files, board, st.get(f"sensor.streams.{s}.K"), variants, a.distortion, out)
        results[s] = r
        print_result(r)
        worst = max(worst, 1 if "error" in r else (2 if r["verdict"] == "mismatch" else 0))
    (out / "stream_intrinsics.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out}/stream_intrinsics.json")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
