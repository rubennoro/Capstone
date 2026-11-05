import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataproc.geometry.camera_model import VirtualPinhole, build_inverse_remap, build_remap
from dataproc.geometry.transforms import rot_x, rot_y
from dataproc.raw_episode import RawEpisode
from dataproc.slam.base import ultrawide_times_ns
from dataproc.slam.orbslam3 import camera_model_at
from dataproc.stages.width import (
    GripperRange, GripperRangeError, TagDetection, detect_episode_tags, detect_tags, detect_tags_in_frames,
    episode_widths, fit_gripper_range, gripper_width_offset, width_on_grid,
)
from test_geometry_common import analytic_episode, make_rig

Z, Y, CX = 0.098, 0.045, 0.002          # tag depth, height below the camera, finger-pair centre x (virtual frame)


@pytest.fixture(scope="module")
def rig():
    return make_rig()


def tag_poses(width, cx=CX, z=Z):
    """Finger tags (iPhUMI geometry: facing the camera, tilted 30 deg) with centre distance ``width``."""
    return {0: (rot_y(15) @ rot_x(150), np.array([cx - width / 2, Y, z])),
            1: (rot_y(-15) @ rot_x(150), np.array([cx + width / 2, Y, z]))}


def render_tags(K, size=(640, 480), poses=None, seed=0, marker_size=0.016):
    """Pinhole image of DICT_4X4_50 tags with a white quiet zone on a textured background (sensor_rig pattern)."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.uniform(60, 190, (size[1], size[0])).astype(np.float32), (0, 0), 3)
    img = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.float32)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for tid, (R, t) in poses.items():
        n = 600
        pad = n // 4
        m = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, tid, n), pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=255)
        s, h = marker_size / n, marker_size / 2
        A = np.array([[s, 0, -pad * s - h], [0, -s, h + pad * s], [0, 0, 1]])   # marker px -> plane metres
        H = np.asarray(K) @ np.c_[R[:, 0], R[:, 1], t] @ A
        warped = cv2.warpPerspective(m, H, size, flags=cv2.INTER_AREA).astype(np.float32)
        a = cv2.warpPerspective(np.full_like(m, 255), H, size, flags=cv2.INTER_AREA).astype(np.float32)[..., None]
        img = img * (1 - a / 255) + warped[..., None] * (a / 255)
    return np.clip(img, 0, 255).astype(np.uint8)


def _virtual(rig, width, **kw):
    return render_tags(rig.virtual_ultrawide.K, tuple(rig.virtual_ultrawide.resolution), tag_poses(width, **kw),
                       seed=int(width * 1e4))


CAL_WIDTHS = np.linspace(0.030, 0.090, 13)


@pytest.fixture(scope="module")
def gripper_range(rig):
    dets = detect_tags_in_frames((_virtual(rig, w) for w in CAL_WIDTHS), rig)
    return fit_gripper_range(dets, rig)


def test_detect_tags_pose(rig):
    d = detect_tags(_virtual(rig, 0.06), rig)
    assert set(d) == {0, 1} and isinstance(d[0], TagDetection)
    rvec, tvec, corners = d[1]
    assert corners.shape == (4, 2) and rvec.shape == (3,)
    assert np.allclose(tvec, [CX + 0.03, Y, Z], atol=1.5e-3)
    assert np.allclose(d[0].tvec, [CX - 0.03, Y, Z], atol=1.5e-3)
    assert detect_tags(np.full((480, 640, 3), 128, np.uint8), rig) == {}


def test_fit_gripper_range(gripper_range):
    g = gripper_range
    assert (g.left_finger_tag_id, g.right_finger_tag_id, g.gripper_id, g.x_basis) == (0, 1, 0, "tvec0")
    assert g.min_width == pytest.approx(0.030, abs=1.5e-3) and g.max_width == pytest.approx(0.090, abs=1.5e-3)
    # right.x = 2 cx - left.x
    assert g.left_to_right["b"] == pytest.approx(-1.0, abs=0.05)
    assert g.left_to_right["a"] == pytest.approx(2 * CX, abs=1.5e-3)
    assert g.right_to_left["b"] == pytest.approx(-1.0, abs=0.05)
    assert g.fit_stats["n_both_frames"] == 13 and g.fit_stats["mae_left_to_right"] < 1e-3


def test_episode_widths_and_single_tag_fallback(rig, gripper_range):
    true_w = np.array([0.035, 0.050, 0.070, 0.085, 0.060, 0.045])
    dets = detect_tags_in_frames((_virtual(rig, w) for w in true_w), rig)
    dets[4] = {1: dets[4][1]}                    # only the right finger seen
    dets[5] = {0: dets[5][0]}                    # only the left finger seen
    dets.append({})                              # nothing seen: no width
    t = 1_000 + np.arange(len(dets), dtype=np.int64) * 33_000_000
    tw, w, lp, rp = episode_widths(dets, t, gripper_range, rig)
    assert tw.tolist() == t[:6].tolist()
    assert lp.tolist() == [True] * 4 + [False, True] and rp.tolist() == [True] * 5 + [False]
    assert np.abs(w - (true_w - gripper_range.min_width)).max() < 2e-3
    assert np.abs(w - (true_w - 0.030)).max() < 2e-3          # closed (calibration min) reads ~0


def test_width_offset_semantics():
    l2r, r2l = {"a": 0.01, "b": -1.0}, {"a": 0.01, "b": -1.0}
    det = lambda x, z=Z: {"tvec": [x, 0.0, z]}                 # noqa: E731
    w, l, r = gripper_width_offset({0: det(-0.02), 1: det(0.03)}, 0, 1, Z, left_to_right=l2r, right_to_left=r2l)
    assert (w, l, r) == (pytest.approx(0.05), True, True)
    w, l, r = gripper_width_offset({0: det(-0.02)}, 0, 1, Z, left_to_right=l2r, right_to_left=r2l)
    assert w == pytest.approx((0.01 + 0.02) + 0.02) and (l, r) == (True, False)
    w, *_ = gripper_width_offset({0: det(-0.02), 1: det(0.03, z=Z + 0.09)}, 0, 1, Z, left_to_right=l2r,
                                 right_to_left=r2l)                         # right outside the z window
    assert w == pytest.approx(0.05)
    w, *_ = gripper_width_offset({0: det(-0.2), 1: det(0.2)}, 0, 1, Z, left_to_right=l2r, right_to_left=r2l,
                                 min_width=0.03, max_width=0.09)
    assert w == 0.09                                                        # clipped
    assert gripper_width_offset({}, 0, 1, Z, left_to_right=l2r, right_to_left=r2l) == (None, False, False)


def test_fit_errors(rig):
    with pytest.raises(GripperRangeError, match="no frames"):
        fit_gripper_range([], rig)
    with pytest.raises(GripperRangeError, match="never both"):
        fit_gripper_range([{0: {"tvec": [0, 0, Z]}}] * 5, rig)
    both = {0: {"tvec": [-0.02, 0, Z]}, 1: {"tvec": [0.02, 0, Z]}}
    with pytest.raises(GripperRangeError, match="threshold"):
        fit_gripper_range([both] + [{}] * 9, rig)
    with pytest.raises(GripperRangeError, match="n_both_frames=2"):
        fit_gripper_range([both, both], rig)


def test_gripper_range_json_roundtrip(tmp_path, gripper_range):
    p = gripper_range.save(tmp_path / "right_gripper_range.json")
    d = json.loads(p.read_text())
    assert set(d) == {"gripper_id", "left_finger_tag_id", "right_finger_tag_id", "max_width", "min_width",
                      "x_basis", "left_to_right", "right_to_left", "fit_stats"}           # iPhUMI's fields
    assert set(d["left_to_right"]) == {"a", "b"} and "n_both_frames" in d["fit_stats"]
    back = GripperRange.load(p)
    assert back.to_json_dict() == gripper_range.to_json_dict()
    d["x_basis"] = "corners"
    with pytest.raises(ValueError, match="x_basis"):
        GripperRange.from_json_dict(d)


def test_width_on_grid():
    t = np.array([0, 100, 200, 600], dtype=np.int64) * 1_000_000
    w = np.array([0.0, 0.01, 0.02, 0.06])
    grid = np.array([-60, -10, 50, 200, 400, 600, 640], dtype=np.int64) * 1_000_000
    out, ok = width_on_grid(t, w, grid, max_gap_ns=150_000_000)
    assert out.shape == (7, 1) and out.dtype == np.float32
    assert np.allclose(out[:, 0], [0.0, 0.0, 0.005, 0.02, 0.04, 0.06, 0.06])
    assert ok.tolist() == [True, True, True, True, False, True, True]
    out, ok = width_on_grid(t, w, np.array([-200_000_000]), max_gap_ns=150_000_000)
    assert not ok.any()
    out, ok = width_on_grid([], [], grid, 1)
    assert not ok.any() and np.all(out == 0)


def test_fisheye_episode_through_remap(tmp_path, rig, gripper_range):
    """Virtual render -> physical fisheye (inverse remap) -> MP4 -> decode -> build_remap -> detect -> width."""
    phys, _, _ = camera_model_at(rig)
    virt = VirtualPinhole.from_K(rig.virtual_ultrawide.K, rig.virtual_ultrawide.resolution)
    to_phys = build_inverse_remap(virt, phys, rig.virtual_ultrawide.R_phys_virt)
    to_virt = build_remap(phys, virt, rig.virtual_ultrawide.R_phys_virt)
    true_w = np.array([0.04, 0.06, 0.08, 0.05])
    frames = [to_phys.apply(_virtual(rig, w), border_value=128) for w in true_w]
    ep = RawEpisode.open(analytic_episode(tmp_path, rig, uw_frames=frames, write_truth=False))
    t, dets = detect_episode_tags(ep, rig, to_virt.apply)
    assert np.array_equal(t, ultrawide_times_ns(ep, rig)) and len(dets) == 4
    tw, w, lp, rp = episode_widths(dets, t, gripper_range, rig)
    assert len(tw) == 4 and lp.all() and rp.all()
    assert np.abs(w - (true_w - gripper_range.min_width)).max() < 2e-3
    t2, d2 = detect_episode_tags(ep, rig, to_virt.apply, indices=[1, 3])
    assert t2.tolist() == t[[1, 3]].tolist() and len(d2) == 2
    grid_w, ok = width_on_grid(tw, w, t[0] + np.arange(0, 100_000_001, 16_666_667), max_gap_ns=50_000_000)
    assert ok.all() and grid_w.shape == (6, 1)


# --------------------------------------------------------------------------------------------- physical render

def _render_phys(phys, tags, size):
    """Ray-cast a physical fisheye image of planar ArUco tags given in the PHYSICAL camera frame (reviewer test)."""
    W, H = size
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    rays, ok = phys.unproject(np.stack([u.ravel(), v.ravel()], 1))
    img = np.full(W * H, 128.0)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for tid, (R, t, s) in tags.items():
        m = cv2.aruco.generateImageMarker(d, tid, 600).astype(np.float64)
        n = R[:, 2]
        denom = rays @ n
        lam = (t @ n) / np.where(np.abs(denom) > 1e-9, denom, np.nan)
        P = rays * lam[:, None] - t
        x, y = P @ R[:, 0], P @ R[:, 1]
        q = s / 4                                                     # white quiet zone; marker x right, y up
        inside = ok & (lam > 0) & (np.abs(x) < s / 2 + q) & (np.abs(y) < s / 2 + q)
        img[inside] = 255
        mk = inside & (np.abs(x) < s / 2) & (np.abs(y) < s / 2)
        px = np.clip(((x[mk] + s / 2) / s * 600).astype(int), 0, 599)
        py = np.clip(((s / 2 - y[mk]) / s * 600).astype(int), 0, 599)
        img[mk] = m[py, px]
    return np.ascontiguousarray(np.repeat(img.reshape(H, W, 1), 3, 2).astype(np.uint8))


@pytest.mark.parametrize("R_deg", [(0, 0, 0), (12, -8, 5)])
def test_width_from_physical_render(rig, R_deg):
    """Tags placed in 3D in the PHYSICAL frame; the detected tvec must be R_phys_virt^T t_phys (virtual frame)."""
    phys, w, h = camera_model_at(rig)
    R_pv = Rotation.from_euler("xyz", R_deg, degrees=True).as_matrix()
    virt = VirtualPinhole.from_K(rig.virtual_ultrawide.K, rig.virtual_ultrawide.resolution)
    R_tag = Rotation.from_euler("x", 160, degrees=True).as_matrix()    # facing the camera, tilted
    tv = {0: np.array([-0.03, 0.03, 0.098]), 1: np.array([0.03, 0.03, 0.098])}   # virtual-frame positions
    tags = {k: (R_pv @ R_tag, R_pv @ p, 0.016) for k, p in tv.items()}
    img = _render_phys(phys, tags, (w, h))
    vimg = build_remap(phys, virt, None if R_deg == (0, 0, 0) else R_pv).apply(img)
    dets = detect_tags(vimg, rig)
    assert set(dets) >= {0, 1}
    for k in (0, 1):
        err = np.abs(dets[k].tvec - tv[k])
        assert err[:2].max() < 1.5e-3 and err[2] < 4e-3
    assert abs(dets[1].tvec[0] - dets[0].tvec[0] - 0.06) < 2e-3
