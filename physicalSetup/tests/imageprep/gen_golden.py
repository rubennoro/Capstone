#!/usr/bin/env python3
"""Generates the T4 golden data (image prep + camera-model parity) from the Python references.

References (read-only, loaded by file path so that no package __init__ runs):
  * iPhUMI/python_package/iphumi/common/cv_util.py:5-34  get_image_transform_with_border (letterbox + INTER_AREA)
  * sensor_rig/sensor_rig/camera_model.py               PinholeRadtan, FisheyeEquidistant, VirtualPinhole, build_remap

Environment: the T4 scratch venv (numpy 2.2.6, opencv-python-headless 4.12.0.88 = iPhUMI environment.yml).
Run with `python -I`.

  gen_golden.py fixtures                      -> small committed fixtures (tests/imageprep/data, tests/camera_model/data)
  gen_golden.py extended <out_dir> <raw_dir>  -> full-size real-frame set + dense model parity (scratch only, not
                                                 committed); the C++ tests read it when RIG_T4_EXTENDED_DIR=<out_dir>.

Golden image outputs are stored as FNV-1a-64 hashes of the C-contiguous bytes (tests/imageprep/data/golden.txt) to
keep the committed fixtures small; the extended set also stores the expected PNGs for diagnosis.
"""

from __future__ import annotations

import glob
import importlib.util
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PS = HERE.parents[1]          # physicalSetup/
REPO = PS.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cv_util = _load("iphumi_cv_util", REPO / "iPhUMI/python_package/iphumi/common/cv_util.py")
cm = _load("sensor_rig_camera_model", REPO / "sensor_rig/sensor_rig/camera_model.py")


def fnv1a64(a: np.ndarray) -> str:
    """FNV-1a 64-bit over the C-contiguous bytes (mirrored in C++: tests/imageprep/golden_util.cpp)."""
    h = 0xCBF29CE484222325
    for byte in np.ascontiguousarray(a).tobytes():
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def letterbox(img: np.ndarray, out: int, bgr_to_rgb: bool) -> np.ndarray:
    """iPhUMI cv_util.get_image_transform_with_border(mode='rgb') exactly as gen_replay_buffer_zarr.py calls it."""
    ih, iw = img.shape[:2]
    if img.ndim == 2:
        # cv_util mode "rgb" asserts 3 channels; this is its call sequence (cv_util.py:13-31) on one channel.
        size = max(iw, ih)
        top, left = (size - ih) // 2, (size - iw) // 2
        p = cv2.copyMakeBorder(img, top, size - ih - top, left, size - iw - left, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(p, (out, out), interpolation=cv2.INTER_AREA)
    tf = cv_util.get_image_transform_with_border((iw, ih), (out, out), mode="rgb", bgr_to_rgb=bgr_to_rgb)
    return np.ascontiguousarray(tf(img))


def synth_image(w: int, h: int) -> np.ndarray:
    """Deterministic integer test pattern (mirrored in C++: tests/imageprep/golden_util.cpp)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    out = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        out[:, :, c] = ((x * 7 + y * 13 + c * 85 + ((x * y) >> 5)) & 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------------------------------------ models

UW_VIRT = dict(width=640, height=480, fx=246.64919, fy=246.64919, cx=320.16495, cy=239.53679)  # firstreal JSON


def model_specs():
    """(name, python model, C++ line) for the parity sets. C++ line: kind w h fx fy cx cy fov n d..."""
    specs = []

    def add(name, pym, kind, fov, dist):
        line = [kind, pym.width, pym.height, pym.fx, pym.fy, pym.cx, pym.cy, fov, len(dist), *dist]
        specs.append((name, pym, " ".join(repr(float(v)) if isinstance(v, float) else str(v) for v in line)))

    m = cm.PinholeRadtan(1920, 1080, 1478.0, 1478.0, 959.5, 539.5, -0.28, 0.07, 0.0, 0.0)
    add("radtan_radial", m, 0, 0.0, [m.k1, m.k2, m.p1, m.p2])
    m = cm.PinholeRadtan(1920, 1080, 1450.3, 1452.1, 951.2, 545.7, -0.12, 0.31, 8e-4, -5e-4)
    add("radtan_tangential", m, 0, 0.0, [m.k1, m.k2, m.p1, m.p2])
    m = cm.FisheyeEquidistant(1640, 1232, 735.0, 735.0, 819.5, 615.5, theta_max_deg=80.0)
    add("kb4_nominal", m, 1, 160.0, [0.0, 0.0, 0.0, 0.0])
    m = cm.FisheyeEquidistant(1640, 1232, 712.4, 713.9, 822.1, 611.3, 0.031, -0.012, 0.004, -0.0011)
    add("kb4_distorted_fov0", m, 1, 0.0, [m.k1, m.k2, m.k3, m.k4])
    m = cm.VirtualPinhole(**UW_VIRT)
    add("virtual_uw", m, 4, 0.0, [])
    return specs


def random_points(rng, n, max_angle_deg):
    """Directions up to max_angle_deg off-axis (some beyond any model's range), random depth."""
    theta = np.radians(rng.uniform(0.0, max_angle_deg, n))
    phi = rng.uniform(-math.pi, math.pi, n)
    d = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1)
    return d * rng.uniform(0.05, 20.0, n)[:, None]


def random_pixels(rng, m, n):
    u = rng.uniform(-0.25 * m.width, 1.25 * m.width, n)
    v = rng.uniform(-0.25 * m.height, 1.25 * m.height, n)
    return np.stack([u, v], 1)


def _f(v) -> str:
    """Shortest round-trip text of a float64 ('nan' for NaN; numpy 2 repr would print 'np.float64(...)')."""
    return repr(float(v))


def write_parity(path: Path, specs, n_pts: int, seed: int):
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        f.write(f"# T4 camera-model parity vs sensor_rig/sensor_rig/camera_model.py (gen_golden.py, seed {seed})\n")
        f.write(f"{len(specs)}\n")
        for name, pym, line in specs:
            P = random_points(rng, n_pts, 120.0)
            uv, ok = pym.project(P)
            # VirtualPinhole.project also requires in-image (camera_model.py:372); store the front-only flag the
            # C++ model reports, and the in-image flag separately.
            uvp = random_pixels(rng, pym, n_pts)
            rays, okr = pym.unproject(uvp)
            f.write(f"{name}\n{line}\n{n_pts}\n")
            for p, q, o in zip(P, uv, ok):
                f.write(" ".join([_f(p[0]), _f(p[1]), _f(p[2]), str(int(o)), _f(q[0]), _f(q[1])]) + "\n")
            f.write(f"{n_pts}\n")
            for p, r, o in zip(uvp, rays, okr):
                f.write(" ".join([_f(p[0]), _f(p[1]), str(int(o)), _f(r[0]), _f(r[1]), _f(r[2])]) + "\n")


# ------------------------------------------------------------------------------------------------ remap cases

def remap_cases():
    """(name, phys python model, C++ phys line, R_phys_virt) for the remap golden (virtual = iPhone ultrawide)."""
    virt = cm.VirtualPinhole(**UW_VIRT)
    cases = []
    phys = cm.FisheyeEquidistant(1640, 1232, 735.0, 735.0, 819.5, 615.5, theta_max_deg=80.0)
    cases.append(("kb4_nominal_identity", phys, "1 1640 1232 735.0 735.0 819.5 615.5 160.0 4 0.0 0.0 0.0 0.0",
                  np.eye(3)))
    phys2 = cm.FisheyeEquidistant(1640, 1232, 712.4, 713.9, 822.1, 611.3, 0.031, -0.012, 0.004, -0.0011,
                                  theta_max_deg=80.0)
    a = math.radians(4.0)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    b = math.radians(-2.5)
    R = R @ np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    cases.append(("kb4_distorted_rotated", phys2,
                  "1 1640 1232 712.4 713.9 822.1 611.3 160.0 4 0.031 -0.012 0.004 -0.0011", R))
    phys3 = cm.PinholeRadtan(1920, 1080, 1478.0, 1478.0, 959.5, 539.5, -0.28, 0.07, 0.0, 0.0)
    cases.append(("radtan_partial", phys3, "0 1920 1080 1478.0 1478.0 959.5 539.5 0.0 4 -0.28 0.07 0.0 0.0",
                  np.eye(3)))
    return virt, cases


def r_line(R):
    return " ".join(repr(float(v)) for v in np.asarray(R).ravel())


# ------------------------------------------------------------------------------------------------ main

def crops_from_frame(img: np.ndarray):
    """Small crops of a real frame covering the INTER_AREA paths: non-integer down, integer 2x, identity, upscale."""
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    return {
        "crop_211x158": img[cy - 79:cy + 79, cx - 105:cx + 106],     # landscape, odd padding, scale 1.06
        "crop_158x211": img[cy - 105:cy + 106, cx - 79:cx + 79],     # portrait
        "crop_448x336": img[cy - 168:cy + 168, cx - 224:cx + 224],   # integer 2x (resizeAreaFast)
        "crop_224x150": img[cy - 75:cy + 75, cx - 112:cx + 112],     # pad only, no resize
        "crop_97x61": img[cy - 30:cy + 31, cx - 48:cx + 49],         # upscale (INTER_AREA -> bilinear)
    }


def fixtures():
    raw = sorted(glob.glob("/home/ruben/Capstone/data/raw/2026-10-01/*firstreal_demonstration*"))[0]
    base = os.path.join(raw, os.path.basename(raw))
    cap = cv2.VideoCapture(base + "_rgb.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, main = cap.read()
    assert ok
    cap = cv2.VideoCapture(base + "_ultrawidergb.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, uw = cap.read()
    assert ok
    d = HERE / "data"
    d.mkdir(exist_ok=True)
    lines = ["# name out_size swap_rb rows cols channels fnv1a64   (gen_golden.py fixtures; cv2 " + cv2.__version__ + ")"]
    cases = {}
    for k, v in crops_from_frame(main).items():
        cases["main_" + k] = v
    cases["uw_crop_320x240"] = uw[120:360, 160:480]
    cases["uw_gray_crop_161x120"] = cv2.cvtColor(uw[180:300, 240:401], cv2.COLOR_BGR2GRAY)
    for name, img in cases.items():
        img = np.ascontiguousarray(img)
        cv2.imwrite(str(d / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        back = cv2.imread(str(d / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(back, img)
        for out in (224, 64):
            for swap in ((False, True) if img.ndim == 3 else (False,)):
                o = letterbox(img, out, swap)
                lines.append(f"{name} {out} {int(swap)} {o.shape[0]} {o.shape[1]} "
                             f"{o.shape[2] if o.ndim == 3 else 1} {fnv1a64(o)}")
    # Remap golden on the synthetic 1640x1232 / 1920x1080 pattern (generated identically in C++).
    virt, rcases = remap_cases()
    rl = ["# name | phys line | R (9) | coverage n_valid | fnv map1 map2 valid | fnv remap | fnv lb224 | fnv lb224 swap"]
    for name, phys, line, R in rcases:
        t = cm.build_remap(phys, virt, R)
        src = synth_image(phys.width, phys.height)
        out = t.apply(src)
        lb = letterbox(out, 224, False)
        lbs = letterbox(out, 224, True)
        rl.append(" | ".join([name, line, r_line(R), f"{t.coverage!r} {int(t.valid.sum())}",
                              f"{fnv1a64(t.map1)} {fnv1a64(t.map2)} "
                              f"{fnv1a64(t.valid.astype(np.uint8) * 255)}",
                              fnv1a64(out), fnv1a64(lb), fnv1a64(lbs)]))
        print(name, "coverage", t.coverage)
    (d / "golden.txt").write_text("\n".join(lines) + "\n")
    (d / "remap_golden.txt").write_text("\n".join(rl) + "\n")
    cmd = PS / "tests/camera_model/data"
    cmd.mkdir(parents=True, exist_ok=True)
    write_parity(cmd / "parity.txt", model_specs(), 120, seed=20261027)
    print("coverage of nominal 160 deg KB4 onto the iPhone uw:", cm.coverage(rcases[0][1], virt))


def extended(out_dir: Path, raw_dir: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = sorted(glob.glob(os.path.join(raw_dir, "*firstreal_demonstration*")))[:3]
    lines = ["# file out_size swap_rb rows cols channels fnv1a64 expected_png"]
    virt, rcases = remap_cases()
    rl = []
    n = 0
    for e in eps:
        base = os.path.join(e, os.path.basename(e))
        for suffix, tag in (("_rgb.mp4", "main"), ("_ultrawidergb.mp4", "uw")):
            cap = cv2.VideoCapture(base + suffix)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for idx in (0, total // 2, max(0, total - 5)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, img = cap.read()
                if not ok:
                    continue
                name = f"{tag}_{n:02d}"
                n += 1
                cv2.imwrite(str(out_dir / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                for swap in (False, True):
                    o = letterbox(img, 224, swap)
                    ep = f"{name}_lb224_{int(swap)}.png"
                    cv2.imwrite(str(out_dir / ep), o)
                    lines.append(f"{name}.png 224 {int(swap)} 224 224 3 {fnv1a64(o)} {ep}")
                if tag == "main":
                    # A realistic texture as the physical fisheye source: the frame resized to each phys size.
                    for rname, phys, line, R in rcases:
                        src = cv2.resize(img, (phys.width, phys.height), interpolation=cv2.INTER_AREA)
                        sname = f"{name}_src_{phys.width}x{phys.height}.png"
                        if not (out_dir / sname).exists():
                            cv2.imwrite(str(out_dir / sname), src, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                        t = cm.build_remap(phys, virt, R)
                        rem = t.apply(src)
                        lb = letterbox(rem, 224, True)
                        ep = f"{name}_{rname}_lb224_1.png"
                        cv2.imwrite(str(out_dir / ep), lb)
                        rl.append(" | ".join([rname, line, r_line(R), sname, fnv1a64(rem), fnv1a64(lb), ep]))
    (out_dir / "golden_ext.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "remap_ext.txt").write_text("\n".join(rl) + "\n")
    # Dense remap maps (float32 x/y before convertMaps) for a map-level comparison.
    for rname, phys, line, R in rcases:
        t = cm.build_remap(phys, virt, R, fixed_point=False)
        t.map1.astype("<f4").tofile(out_dir / f"map_{rname}_x.f32")
        t.map2.astype("<f4").tofile(out_dir / f"map_{rname}_y.f32")
    write_parity(out_dir / "parity_dense.txt", model_specs(), 20000, seed=7)
    print("extended:", n, "frames ->", out_dir)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "fixtures":
        fixtures()
    elif len(sys.argv) >= 4 and sys.argv[1] == "extended":
        extended(Path(sys.argv[2]), sys.argv[3])
    else:
        sys.exit(__doc__)
