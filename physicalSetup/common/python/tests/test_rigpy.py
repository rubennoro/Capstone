#!/usr/bin/env python3
"""rigpy binding test (T4), run by ctest (rigpy.bindings) with the T4 venv interpreter and `-I`.

Checks that rigpy reproduces
  * the committed golden bytes (tests/imageprep/data/golden.txt, remap_golden.txt): the same hashes the C++ tests
    assert, so rigpy output == C++ output == Python cv2 4.12 output;
  * in-process: iPhUMI cv_util.py:5-34 on random images, and sensor_rig camera_model.py on random points / remaps;
  * the extended real-frame set when RIG_T4_EXTENDED_DIR is set.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.module_dir)
    import rigpy  # noqa: E402
    from rigpy import camera_model as rcm, imageprep as rip  # noqa: E402

    src = Path(a.source_dir)
    g = load("t4_gen_golden", src / "tests/imageprep/gen_golden.py")  # loads cv_util.py + camera_model.py
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # 1. committed letterbox golden
    data = src / "tests/imageprep/data"
    n = 0
    for line in (data / "golden.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, out, swap, rows, cols, ch, h = line.split()
        img = cv2.imread(str(data / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        o = rip.letterbox_square_resize(img, int(out), bool(int(swap)))
        check(o.shape[:2] == (int(rows), int(cols)), f"{name}: shape {o.shape}")
        check(g.fnv1a64(o) == h, f"{name} out={out} swap={swap}: hash differs from golden")
        n += 1
    print(f"committed letterbox cases: {n}")

    # 2. in-process vs cv_util on random images of random sizes (both orientations, up/down scaling)
    rng = np.random.default_rng(5)
    for i in range(60):
        h, w = int(rng.integers(1, 700)), int(rng.integers(1, 700))
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        out = int(rng.choice([224, 96, 300]))
        swap = bool(i % 2)
        exp = g.letterbox(img, out, swap)
        got = rip.letterbox_square_resize(img, out, swap)
        check(np.array_equal(exp, got), f"random {w}x{h} -> {out} swap={swap}: differs from cv_util")
    gm = rip.letterbox_geometry(211, 158)
    check(gm == {"size": 211, "top": 26, "bottom": 27, "left": 0, "right": 0}, f"geometry {gm}")

    # 3. remap golden (maps, mask, remap output, letterbox), committed synthetic pattern
    uw = g.UW_VIRT
    virt = rcm.VirtualPinhole.create(rcm.CameraModelParams(rcm.CameraModelKind.PINHOLE, uw["width"], uw["height"],
                                                           uw["fx"], uw["fy"], uw["cx"], uw["cy"]))
    for line in (data / "remap_golden.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(" | ")
        name, mline, rline = parts[0], parts[1].split(), parts[2]
        kind, w, h, fx, fy, cx, cy, fov, nd = mline[:9]
        dist = [float(v) for v in mline[9:9 + int(nd)]]
        params = rcm.CameraModelParams(rcm.CameraModelKind(int(kind)), int(w), int(h), float(fx), float(fy),
                                       float(cx), float(cy), dist, float(fov))
        phys = rcm.make_camera_model(params)
        R = np.array([float(v) for v in rline.split()]).reshape(3, 3)
        t = rcm.build_remap(phys, virt, R)
        h1, h2, hv = parts[4].split()
        check(g.fnv1a64(t.map1) == h1 and g.fnv1a64(t.map2) == h2 and g.fnv1a64(t.valid_mask) == hv,
              f"{name}: maps differ")
        check(t.map1.dtype == np.int16 and t.map1.shape == (480, 640, 2) and not t.map1.flags.writeable,
              f"{name}: map1 layout {t.map1.dtype} {t.map1.shape}")
        img = g.synth_image(int(w), int(h))
        rem = t.apply(img)
        check(g.fnv1a64(rem) == parts[5].strip(), f"{name}: remap output differs")
        check(g.fnv1a64(rip.remap_letterbox(t, img, 224, False)) == parts[6].strip(), f"{name}: lb differs")
        check(g.fnv1a64(rip.remap_letterbox(t, img, 224, True)) == parts[7].strip(), f"{name}: lb swap differs")
        # vs the Python reference in-process (also the float maps)
        pv = g.cm.VirtualPinhole(**uw)
        _, pcases = g.remap_cases()
        pphys = next(c for c in pcases if c[0] == name)[1]
        pt = g.cm.build_remap(pphys, pv, R, fixed_point=False)
        tf = rcm.build_remap(phys, virt, R, False)
        check(np.array_equal(pt.map1, tf.map1) and np.array_equal(pt.map2, tf.map2), f"{name}: float maps differ")
        rep = rcm.coverage_report(phys, virt, R)
        check(rep.coverage == t.coverage == pt.coverage, f"{name}: coverage {rep.coverage} {t.coverage} {pt.coverage}")
        print(f"remap {name}: coverage {t.coverage:.6f} (python {pt.coverage:.6f})")

    # 4. model parity in-process (project bit-identical in C++ tests; here 1e-9)
    for name, pym, line in g.model_specs():
        v = line.split()
        params = rcm.CameraModelParams(rcm.CameraModelKind(int(v[0])), int(v[1]), int(v[2]), float(v[3]),
                                       float(v[4]), float(v[5]), float(v[6]), [float(x) for x in v[9:]], float(v[7]))
        m = rcm.make_camera_model(params)
        P = g.random_points(rng, 5000, 120.0)
        uv_p, ok_p = pym.project(P)
        uv_c, ok_c, J = m.project(P, jacobian=True)
        if name == "virtual_uw":
            ok_c = ok_c & (uv_c[:, 0] >= 0) & (uv_c[:, 1] >= 0) & (uv_c[:, 0] <= m.width - 1) & (uv_c[:, 1] <= m.height - 1)
        check(np.array_equal(ok_p, ok_c), f"{name}: project flags differ")
        check(np.nanmax(np.abs(uv_p[ok_p] - uv_c[ok_p]), initial=0) <= 1e-9, f"{name}: project values differ")
        check(J.shape == (5000, 2, 3), f"{name}: J shape")
        pix = g.random_pixels(rng, pym, 5000)
        r_p, okr_p = pym.unproject(pix)
        r_c, okr_c = m.unproject(pix)
        check(np.array_equal(okr_p, okr_c), f"{name}: unproject flags differ")
        check(np.nanmax(np.abs(r_p[okr_p] - r_c[okr_p]), initial=0) <= 1e-9, f"{name}: unproject values differ")

    # 5. buffer semantics
    img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    out = rip.letterbox_square_resize(img)
    check(out.flags.owndata is False and out.base is not None, "output should be a capsule-owned view (no copy)")
    for bad in (img[:, ::2], img.astype(np.float32), np.zeros((0, 4, 3), np.uint8)):
        try:
            rip.letterbox_square_resize(bad)
            fails.append(f"no ValueError for {bad.dtype} {bad.shape} contiguous={bad.flags.c_contiguous}")
        except ValueError:
            pass
    try:
        t.apply(img)
        fails.append("apply with wrong size should raise")
    except ValueError:
        pass
    try:
        phys.scaled(100, 33)
        fails.append("non-uniform scaled should raise")
    except ValueError:
        pass

    # 6. extended real frames
    ext = os.environ.get("RIG_T4_EXTENDED_DIR")
    if ext:
        ne = 0
        for line in (Path(ext) / "golden_ext.txt").read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            f, out, swap, _, _, _, h, _ = line.split()
            o = rip.letterbox_square_resize(cv2.imread(os.path.join(ext, f)), int(out), bool(int(swap)))
            check(g.fnv1a64(o) == h, f"extended {f} swap={swap}")
            ne += 1
        print(f"extended letterbox cases: {ne}")

    print("rigpy", rigpy.__file__)
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
