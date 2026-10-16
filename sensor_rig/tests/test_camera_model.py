import os
import time

import cv2
import numpy as np
import pytest

from sensor_rig.camera_model import (
    FisheyeEquidistant, PinholeRadtan, VirtualPinhole, build_inverse_remap, build_remap, coverage, coverage_mask,
    model_from_dict, model_from_kalibr, model_to_dict, virtual_border_in_physical, virtual_to_physical,
)
from sensor_rig.transforms import rot_x, rot_y, rot_z

# Rig default placeholder lenses (configs/rig_default.yaml) and the virtual iPhone cameras (brief D3).
MAIN = PinholeRadtan(1640, 1232, 1150.0, 1150.0, 819.5, 615.5, -0.30, 0.09, 0.0, 0.0)
MAIN_TAN = PinholeRadtan(1640, 1232, 1150.0, 1148.0, 821.3, 612.9, -0.30, 0.09, 1.2e-3, -8e-4)
UW = FisheyeEquidistant(1640, 1232, 734.5, 734.5, 819.5, 615.5, 0.020, -0.008, 0.002, -0.0003)
V_MAIN = VirtualPinhole(1920, 1440, 1406.3428, 1406.3428, 953.2503, 723.22003)
V_UW = VirtualPinhole(640, 480, 259.01538, 259.01538, 320.2935, 239.67455)
HALF_DIAG = float(np.hypot(819.5, 615.5))


def _grid(m, step=1):
    u, v = np.meshgrid(np.arange(0, m.width, step, dtype=float), np.arange(0, m.height, step, dtype=float))
    return np.stack([u.ravel(), v.ravel()], 1)


def _texture(h, w, seed=0, sigma=2.0):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = (img - img.mean()) / img.std() * 50 + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def _ecc_shift(a, b, mask=None):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
    m = None if mask is None else mask.astype(np.uint8)
    _, warp = cv2.findTransformECC(ga, gb, warp, cv2.MOTION_TRANSLATION, crit, m, 1)
    return float(np.hypot(warp[0, 2], warp[1, 2]))


def test_radtan_matches_opencv():
    rng = np.random.default_rng(0)
    for m in (MAIN, MAIN_TAN):
        xy = rng.uniform(-0.75, 0.75, (10000, 2))
        P = np.c_[xy, np.ones(len(xy))] * rng.uniform(0.2, 5.0, (len(xy), 1))
        uv, ok = m.project(P)
        ref, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), m.K, m.dist)
        assert ok.all()
        assert np.abs(uv - ref[:, 0]).max() < 1e-6


def test_equi_matches_cv2_fisheye():
    rng = np.random.default_rng(1)
    th = rng.uniform(0, np.deg2rad(85), 10000)
    ph = rng.uniform(0, 2 * np.pi, 10000)
    P = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)] * rng.uniform(0.2, 5.0, (10000, 1))
    uv, ok = UW.project(P)
    ref, _ = cv2.fisheye.projectPoints(P[:, None], np.zeros(3), np.zeros(3), UW.K, UW.dist)
    assert ok.all()
    assert np.abs(uv - ref[:, 0]).max() < 1e-6


def test_undistort_matches_opencv():
    rng = np.random.default_rng(2)
    uv = np.c_[rng.uniform(0, 1639, 2000), rng.uniform(0, 1231, 2000)]
    rays, ok = MAIN_TAN.unproject(uv)
    ref = cv2.undistortPointsIter(uv[:, None], MAIN_TAN.K, MAIN_TAN.dist, None, None,
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14))[:, 0]
    assert ok.all()
    assert np.abs(rays[:, :2] / rays[:, 2:] - ref).max() < 1e-6
    rays, ok = UW.unproject(uv)
    ref = cv2.fisheye.undistortPoints(uv[:, None], UW.K, UW.dist,
                                      criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14))[:, 0]
    assert ok.all()
    assert np.abs(rays[:, :2] / rays[:, 2:] - ref).max() < 1e-6


@pytest.mark.parametrize("m", [MAIN, MAIN_TAN, UW], ids=["radtan", "radtan_tangential", "equidistant"])
def test_unproject_roundtrip(m):
    g = _grid(m)
    rays, ok = m.unproject(g)
    assert ok.mean() >= 0.999
    assert np.allclose(np.linalg.norm(rays[ok], axis=1), 1.0, atol=1e-12)
    uv, ok2 = m.project(rays[ok])
    assert ok2.all()
    assert np.abs(uv - g[ok]).max() < 1e-6


def test_radtan_foldover_invalid():
    m = PinholeRadtan(1640, 1232, 1150.0, 1150.0, 819.5, 615.5, -0.4, 0.0, 0.0, 0.0)
    assert m.r_valid == pytest.approx(np.sqrt(1 / 1.2), rel=1e-9)
    # Rays beyond r_valid: the raw formula folds them back into the image, the model marks them invalid.
    r = np.linspace(m.r_valid * 1.001, 1.5, 200)
    P = np.c_[r, np.zeros_like(r), np.ones_like(r)]
    uv, ok = m.project(P)
    assert not ok.any() and np.isnan(uv).all()
    raw_u = m.fx * r * (1 + m.k1 * r**2) + m.cx
    assert ((raw_u >= 0) & (raw_u <= m.width - 1)).any()          # the fold-over really lands inside
    # A very wide virtual camera: no virtual pixel beyond r_valid may be valid in the forward remap.
    wide = VirtualPinhole(800, 600, 250.0, 250.0, 399.5, 299.5)
    table = build_remap(m, wide, fixed_point=False)
    rays, _ = wide.unproject(_grid(wide))
    beyond = (np.hypot(rays[:, 0], rays[:, 1]) / rays[:, 2] > m.r_valid).reshape(600, 800)
    assert beyond.any() and not table.valid[beyond].any()
    # the valid region is exactly the in-range, in-image part
    uvp, okp = virtual_to_physical(m, wide, _grid(wide))
    assert np.array_equal(table.valid.ravel(), okp & m.in_image(uvp))


def test_scaling_convention():
    """960x720 virtual camera == INTER_AREA 2x2 downsample of the 1920x1440 one (half-pixel convention)."""
    v_half = V_MAIN.scaled(960, 720)
    assert v_half.fx == pytest.approx(703.1714) and v_half.cx == pytest.approx(476.37515)
    assert v_half.cy == pytest.approx(361.360015)
    with pytest.raises(ValueError):
        V_MAIN.scaled(960, 700)
    # Render Gaussian dots in the physical image and remap them to both virtual scales.
    rng = np.random.default_rng(3)
    xy = np.c_[rng.uniform(-0.55, 0.55, 40), rng.uniform(-0.4, 0.4, 40)]
    P = np.c_[xy, np.ones(len(xy))]
    uv_p, _ = MAIN.project(P)
    yy, xx = np.mgrid[0:MAIN.height, 0:MAIN.width].astype(np.float32)
    phys = np.zeros((MAIN.height, MAIN.width), np.float32)
    for u, v in uv_p:
        x0, y0 = int(u), int(v)
        sl = np.s_[max(y0 - 25, 0):y0 + 26, max(x0 - 25, 0):x0 + 26]
        phys[sl] += np.exp(-((xx[sl] - u) ** 2 + (yy[sl] - v) ** 2) / (2 * 4.0**2))
    phys8 = np.clip(phys * 250, 0, 255).astype(np.uint8)
    full = build_remap(MAIN, V_MAIN, fixed_point=False).apply(phys8).astype(np.float64)
    half = build_remap(MAIN, v_half, fixed_point=False).apply(phys8).astype(np.float64)
    down = cv2.resize(full, (960, 720), interpolation=cv2.INTER_AREA)
    uv_half, _ = v_half.project(P)
    def centroid(img, sl):
        ys, xs = np.mgrid[sl]
        w = img[sl]
        return np.array([(w * xs).sum(), (w * ys).sum()]) / w.sum()

    errs = []
    for u, v in uv_half:
        x0, y0 = int(round(u)), int(round(v))
        sl = np.s_[y0 - 12:y0 + 13, x0 - 12:x0 + 13]
        errs.append(np.linalg.norm(centroid(down, sl) - centroid(half, sl)))
    assert max(errs) < 0.02


def test_forward_inverse_roundtrip():
    for phys, virt, R in ((MAIN, V_MAIN.scaled(960, 720), rot_y(1.5)), (UW, V_UW, rot_x(-2.0) @ rot_z(3.0))):
        img_v = _texture(virt.height, virt.width, seed=4, sigma=2.5)
        inv = build_inverse_remap(virt, phys, R)
        fwd = build_remap(phys, virt, R)
        back = fwd.apply(inv.apply(img_v))
        mask = cv2.erode((fwd.valid & coverage_mask(phys, virt, R)).astype(np.uint8), np.ones((7, 7), np.uint8))
        # pixels whose physical sample came from content (not the inverse remap's black border)
        content = cv2.erode(fwd.apply(inv.valid.astype(np.uint8) * 255), np.ones((7, 7), np.uint8)) == 255
        m = (mask > 0) & content
        assert m.mean() > 0.5
        mae = np.abs(back.astype(float) - img_v.astype(float))[m].mean()
        assert mae <= 2.0, mae
        assert _ecc_shift(img_v, back, m) <= 0.05


def test_rotation_shifts_principal_ray():
    R = rot_y(5.0)
    for phys, virt in ((MAIN_TAN, V_MAIN), (UW, V_UW)):
        uv, ok = virtual_to_physical(phys, virt, [[virt.cx, virt.cy]], R)
        expect, ok2 = phys.project(R @ [0.0, 0.0, 1.0])
        assert ok.all() and ok2.all()
        assert np.abs(uv - expect).max() < 1e-6
        # and the float remap agrees at the nearest integer pixel (float32 maps)
        t = build_remap(phys, virt, R, fixed_point=False)
        j, i = 300, 200
        uv_ij, _ = virtual_to_physical(phys, virt, [[j, i]], R)
        assert abs(t.map1[i, j] - uv_ij[0, 0]) < 1e-3 and abs(t.map2[i, j] - uv_ij[0, 1]) < 1e-3


def test_coverage_values():
    assert coverage(MAIN, V_MAIN) == 1.0
    assert coverage(UW, V_UW) == 1.0
    assert build_remap(MAIN, V_MAIN.scaled(960, 720)).coverage == 1.0
    # IMX219 "77 deg" main (f = 2.96 mm -> 1321 px at 1640x1232), ideal pinhole: covers only ~83 %.
    imx77 = PinholeRadtan(1640, 1232, 1321.0, 1321.0, 819.5, 615.5)
    c77 = coverage(imx77, V_MAIN)
    assert 0.80 <= c77 <= 0.86, c77
    # IMX219-120 class fisheye: 120 deg diagonal equidistant.
    f120 = HALF_DIAG / np.deg2rad(60.0)
    uw120 = FisheyeEquidistant(1640, 1232, f120, f120, 819.5, 615.5)
    c120 = coverage(uw120, V_UW)
    assert 0.80 <= c120 <= 0.86, c120
    # coverage() and the remap table agree
    assert build_remap(uw120, V_UW).coverage == pytest.approx(c120, abs=0)


def test_fixed_point_vs_float():
    img = _texture(1232, 1640, seed=5, sigma=1.5)
    a = build_remap(MAIN, V_MAIN.scaled(960, 720), rot_y(1.0)).apply(img)
    b = build_remap(MAIN, V_MAIN.scaled(960, 720), rot_y(1.0), fixed_point=False).apply(img)
    assert np.abs(a.astype(float) - b.astype(float)).mean() <= 0.3


def test_remap_table_contract():
    t = build_remap(UW, V_UW, rot_y(30.0))
    assert t.src_size == (1640, 1232) and t.dst_size == (640, 480)
    assert 0.0 < t.coverage < 1.0 and t.valid.shape == (480, 640)
    img = np.full((1232, 1640, 3), 200, np.uint8)
    out = t.apply(img, border_value=7)
    assert out.shape == (480, 640, 3) and out.dtype == np.uint8
    assert (out[t.valid] == 200).all() and (out[~t.valid] == 7).all()
    gray = t.apply(np.full((1232, 1640), 50, np.uint8))
    assert gray.shape == (480, 640) and (gray[~t.valid] == 0).all()
    buf = np.empty((480, 640, 3), np.uint8)
    assert t.apply(img, out=buf) is not None and (buf[t.valid] == 200).all()
    with pytest.raises(ValueError):
        t.apply(np.zeros((480, 640, 3), np.uint8))
    with pytest.raises(TypeError):
        t.apply(np.zeros((1232, 1640, 3), np.float32))
    with pytest.raises(ValueError):
        t.valid[0, 0] = True                    # tables are immutable (they are cached and shared)
    inv = build_inverse_remap(V_UW, UW)
    assert inv.src_size == (640, 480) and inv.dst_size == (1640, 1232) and 0 < inv.coverage < 1


def test_border_polyline():
    poly = virtual_border_in_physical(UW, V_UW, n_per_side=16)
    assert poly.shape == (65, 2) and np.isfinite(poly).all()
    assert np.allclose(poly[0], poly[-1])
    assert (poly[:, 0] > 0).all() and (poly[:, 0] < 1639).all()


def test_dict_and_kalibr_roundtrip():
    for m in (MAIN_TAN, UW):
        d = model_to_dict(m)
        m2 = model_from_dict(d, m.size)
        assert type(m2) is type(m) and np.array_equal(m2.K, m.K) and np.array_equal(m2.dist, m.dist)
        k = model_from_kalibr(m.to_kalibr())
        assert type(k) is type(m) and np.array_equal(k.K, m.K) and np.array_equal(k.dist, m.dist)
        s = m.scaled(820, 616)
        assert s.fx == m.fx / 2 and s.cx == (m.cx + 0.5) / 2 - 0.5 and np.array_equal(s.dist, m.dist)
    with pytest.raises(NotImplementedError):
        model_from_kalibr({"camera_model": "ds", "intrinsics": [1, 1, 1, 1], "distortion_model": "none"})
    with pytest.raises(NotImplementedError):
        model_from_kalibr({"camera_model": "pinhole", "distortion_model": "fov"})
    with pytest.raises(ValueError, match="^distortion"):
        model_from_dict({"type": "pinhole-radtan", "intrinsics": [1, 1, 1, 1], "distortion": [0, 0, 0]}, (4, 4))
    with pytest.raises(ValueError, match="^type"):
        model_from_dict({"type": "omni", "intrinsics": [1, 1, 1, 1]}, (4, 4))


def test_virtual_K_float32_rows():
    rows = V_UW.K_float32_rows()
    assert rows == [[259.01538, 0, 320.2935], [0, 259.01538, 239.67455], [0, 0, 1]]
    assert np.array_equal(np.float32(rows), np.float32(V_UW.K))
    v = VirtualPinhole.from_K(V_MAIN.K, (1920, 1440))
    assert np.array_equal(v.K, V_MAIN.K)


@pytest.mark.skipif(os.environ.get("SENSOR_RIG_PERF") != "1", reason="set SENSOR_RIG_PERF=1 to run timing checks")
def test_remap_speed():
    t = build_remap(MAIN, V_MAIN.scaled(960, 720))
    img = _texture(1232, 1640, seed=6)
    n_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        t.apply(img)
        t0 = time.perf_counter()
        for _ in range(100):
            t.apply(img)
        ms = (time.perf_counter() - t0) / 100 * 1e3
    finally:
        cv2.setNumThreads(n_threads)
    print(f"remap 1640x1232 -> 960x720, 1 thread: {ms:.2f} ms")
    assert ms <= 5.0
