"""umiproc.viz.dense_3d (EXPERIMENTAL TRIAL): stereo on a synthetic plane, the merge/voxel/outlier steps, the PLY."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from umiproc.viz import dense_3d as D  # noqa: E402
from umiproc.viz.slam_3d_images import Camera  # noqa: E402

PLANE_Z = 0.5


def _texture(seed: int = 0, n: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 256, (n, n), dtype=np.uint8)
    return cv2.GaussianBlur(t, (0, 0), 1.2)


def _render(cam: Camera, T_W_cam: np.ndarray, tex: np.ndarray, span: float = 1.6) -> np.ndarray:
    """The fisheye image (k = 0: theta_d = theta) of the textured plane z = PLANE_Z (the texture covers +-span/2)."""
    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    x, y = (u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy
    th = np.hypot(x, y)
    s = np.where(th > 1e-12, np.sin(th) / np.maximum(th, 1e-12), 1.0)
    ray = np.stack([x * s, y * s, np.cos(th)], -1) @ T_W_cam[:3, :3].T
    lam = (PLANE_Z - T_W_cam[2, 3]) / ray[..., 2]
    P = T_W_cam[:3, 3] + lam[..., None] * ray
    n = tex.shape[0]
    mx = ((P[..., 0] + span / 2) / span * n).astype(np.float32)
    my = ((P[..., 1] + span / 2) / span * n).astype(np.float32)
    g = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
    return np.repeat(g[..., None], 3, axis=2)


def _pose(x: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    return T


def test_pair_depth_recovers_a_plane():
    cam = Camera(fx=225.0, fy=225.0, cx=319.5, cy=239.5, k=(0.0, 0.0, 0.0, 0.0), width=640, height=480)
    p = D.Params(width=320, height=240, fov_deg=90.0)
    tex = _texture()
    a = D.Frame(fed=0, T_W_cam=_pose(0.0), keyframe=True)
    b = D.Frame(fed=1, T_W_cam=_pose(0.05), keyframe=True)
    for f in (a, b):
        f.raw = _render(cam, f.T_W_cam, tex)
        f.valid = np.ones((p.height, p.width), bool)
        f.median_depth = PLANE_Z
    inp = SimpleNamespace(cam=cam)
    z = D.pair_depth(a, b, inp, p, np.zeros((cam.height, cam.width), bool))
    ok = np.isfinite(z)
    assert ok.mean() > 0.3
    assert abs(float(np.median(z[ok])) - PLANE_Z) < 0.005
    assert np.percentile(np.abs(z[ok] - PLANE_Z), 90) < 0.02


def test_choose_pairs_prefers_sideways_baselines():
    p = D.Params()
    frames = [D.Frame(fed=k, T_W_cam=_pose(0.0), keyframe=False, median_depth=0.5) for k in range(3)]
    frames[1].T_W_cam = _pose(0.05)                        # sideways: 10 % of the depth
    frames[2].T_W_cam = np.eye(4)
    frames[2].T_W_cam[2, 3] = 0.05                         # straight ahead: never a pair
    assert D.choose_pairs(frames, p) == [(0, 1), (1, 0)]


def test_merge_depths():
    a = np.array([[1.0, 1.0, np.nan, np.nan]], np.float32)
    b = np.array([[1.01, 1.5, 2.0, np.nan]], np.float32)
    m = D.merge_depths([a, b], 0.02)
    assert m[0, 0] == pytest.approx(1.005)
    assert np.isnan(m[0, 1]) and m[0, 2] == pytest.approx(2.0) and np.isnan(m[0, 3])


def test_voxel_thin_and_outliers():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.uniform(0, 0.1, (5000, 3)), [[5.0, 5.0, 5.0]]])
    C = np.full((len(X), 3), 100.0)
    Xv, Cv, cnt = D.voxel_thin(X, C, 0.01, 1)
    assert len(Xv) <= 11 ** 3 + 1 and cnt.sum() == len(X) and (Cv == 100).all()
    keep = D.drop_outliers(Xv, 8, 2.0)
    assert not keep[np.argmax(Xv.sum(axis=1))] and keep.mean() > 0.95


def test_write_ply_round_trip(tmp_path):
    X = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]])
    C = np.array([[255, 0, 10], [1, 2, 3]], np.uint8)
    path = tmp_path / "c.ply"
    D.write_ply(path, X, C)
    data = path.read_bytes()
    head, body = data.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in head and b"element vertex 2" in head
    rec = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                                     ("b", "u1")])
    np.testing.assert_allclose(np.stack([rec["x"], rec["y"], rec["z"]], 1), X, atol=1e-6)
    np.testing.assert_array_equal(np.stack([rec["r"], rec["g"], rec["b"]], 1), C)
