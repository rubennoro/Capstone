"""A7 Renderer tests: wire shapes and K, the half-pixel principal-point fix, fingers in view, determinism, the rig
fidelity mode (B0179 coverage) and render + JPEG timing.

Run: cd controlSW/franka_sim && MUJOCO_GL=egl env -u PYTHONPATH $RE -m pytest -q tests/test_render.py -s
"""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
if os.environ.get("MUJOCO_GL") != "egl":
    pytest.skip("set MUJOCO_GL=egl", allow_module_level=True)

import cv2  # noqa: E402

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import model as M  # noqa: E402
from franka_sim.render import Renderer  # noqa: E402

RIG_B0179 = M.CONTROLSW.parent / "sensor_rig" / "sensor_rig" / "configs" / "rig_jetson_imx219_2p1_b0179.yaml"
PROFILES_UNDER_TEST = ("sim-iphone", "sim-rig")


def _station(profile: str, sets=()):
    return load_station({k: dict(v) for k, v in PROFILES[profile].items()}, sets=list(sets))


@pytest.fixture(scope="module")
def iphone():
    st = _station("sim-iphone")
    sm = M.build_model(st)
    r = Renderer(sm, st)
    yield st, sm, r
    r.close()


def test_shapes_dtype_and_K_both_sensors():
    Ks = {}
    for prof in PROFILES_UNDER_TEST:
        st = _station(prof)
        sm = M.build_model(st)
        r = Renderer(sm, st)
        try:
            img = r.render(M.forward_at(sm).qpos.copy())
        finally:
            r.close()
        for s in ("main", "ultrawide"):
            assert img[s].shape == (720, 960, 3) and img[s].dtype == np.uint8 and img[s].std() > 1.0
            assert np.allclose(r.K[s], np.array(st.get(f"sensor.streams.{s}.K"), float))
        Ks[prof] = r.K
    for s in ("main", "ultrawide"):   # the rig carries the virtual-iPhone K: identical wire K
        assert np.array_equal(Ks["sim-iphone"][s], Ks["sim-rig"][s])


def _marker_model(st, sm, stream: str, uv_list, depth: float, radius: float):
    cam = {"main": "cam_main", "ultrawide": "cam_uw"}[stream]
    d = M.forward_at(sm)
    W_T_C = M.cam_T_cv(sm.model, d, cam)
    K = np.array(st.get(f"sensor.streams.{stream}.K"), float)
    spec = sm.spec
    names, P_world = [], []
    for i, (u, v) in enumerate(uv_list):
        ray = np.linalg.solve(K, [u, v, 1.0])
        Pc = ray / ray[2] * depth
        Pw = W_T_C[:3, :3] @ Pc + W_T_C[:3, 3]
        g = spec.worldbody.add_geom()
        g.name = f"a7_marker_{stream}_{i}"
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [radius, 0, 0]
        g.pos = Pw
        g.rgba = [1, 0, 1, 1]
        g.contype = g.conaffinity = 0
        names.append(g.name)
        P_world.append(Pw)
    m2 = spec.compile()
    for nm in names:                           # leave sm.spec as it was for the other tests
        spec.geom(nm).delete()
    return dataclasses.replace(sm, model=m2), names, np.array(P_world), W_T_C, K


@pytest.mark.parametrize("stream", ["main", "ultrawide"])
def test_half_pixel_projection(iphone, stream):
    st, sm, _ = iphone
    uv = [(200.0, 150.0), (760.0, 150.0), (200.0, 430.0), (760.0, 430.0), (480.0, 300.0)]
    sm2, names, Pw, W_T_C, K = _marker_model(st, sm, stream, uv, depth=0.10, radius=0.0015)
    r = Renderer(sm2, st, streams=(stream,))
    try:
        seg = r.segmentation(M.forward_at(sm2).qpos.copy())[stream]
    finally:
        r.close()
    C_T_W = M.inv_T(W_T_C)
    errs = []
    for nm, P in zip(names, Pw):
        gid = mujoco.mj_name2id(sm2.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        vv, uu = np.nonzero(seg == gid)
        assert uu.size > 20, f"{nm}: marker not visible ({uu.size} px)"
        Pc = C_T_W[:3, :3] @ P + C_T_W[:3, 3]
        proj, _ = cv2.projectPoints(Pc.reshape(1, 3), np.zeros(3), np.zeros(3), K, None)
        errs.append(np.array([uu.mean(), vv.mean()]) - proj.reshape(2))
    errs = np.array(errs)
    print(f"\n[{stream}] rendered centroid - cv2.projectPoints (px): " + "; ".join(f"({e[0]:+.3f}, {e[1]:+.3f})" for e in errs))
    assert np.abs(errs).max() <= 0.15


def test_fingers_in_ultrawide_bottom_half(iphone):
    _, sm, r = iphone
    seg = r.segmentation(M.forward_at(sm).qpos.copy())["ultrawide"]
    frac = r.mask(seg[360:], ["fingers", "tags"]).mean()
    print(f"\nfinger+tag pixel fraction in the UW bottom half: {frac:.4f}")
    assert frac > 0


def test_render_deterministic(iphone):
    _, sm, r = iphone
    q = M.forward_at(sm, gap_m=0.04).qpos.copy()
    a, b = r.render(q), r.render(q)
    for s in a:
        assert np.array_equal(a[s], b[s])


def test_rig_b0179_mode():
    # plain reference = the same rig station without rig= (the iPhone tooling has its own UW extrinsic since W10B)
    st_rig = _station("sim-rig")
    sm_rig = M.build_model(st_rig)
    r, r_plain = Renderer(sm_rig, st_rig, rig=str(RIG_B0179)), Renderer(sm_rig, st_rig)
    try:
        q = M.forward_at(sm_rig).qpos.copy()
        a, b = r.render(q), r_plain.render(q)
        cov = r.coverage
    finally:
        r.close()
        r_plain.close()
    mae = {s: float(np.abs(a[s].astype(np.int16) - b[s].astype(np.int16)).mean()) for s in a}
    print(f"\nB0179 coverage {cov}; rig= mode MAE vs plain render (blur only): {mae}")
    assert cov["main"] == pytest.approx(1.0) and cov["ultrawide"] == pytest.approx(1.0)
    assert cov["main_deploy"] == pytest.approx(1.0) and cov["ultrawide_deploy"] == pytest.approx(1.0)
    for s in a:
        assert a[s].shape == (720, 960, 3)
        assert 0.0 < mae[s] < 10.0        # resampling blur only: same geometry


def test_render_jpeg_timing(iphone):
    _, sm, r = iphone
    q = M.forward_at(sm).qpos.copy()
    ts = []
    for i in range(40):
        t0 = time.perf_counter()
        imgs = r.render(q)
        for s in imgs:
            cv2.imencode(".jpg", imgs[s], [cv2.IMWRITE_JPEG_QUALITY, 90])
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts[5:]) * 1e3
    p95 = float(np.percentile(ts, 95))
    print(f"\nrender + JPEG q90 of both streams (960x720, EGL): p50 {np.percentile(ts, 50):.1f} ms, p95 {p95:.1f} ms "
          f"(target 25 ms)")
    assert p95 < 100.0            # report-only target; this bound only catches a broken (software) GL path
