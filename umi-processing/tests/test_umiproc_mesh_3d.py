"""umiproc.viz.mesh_3d (EXPERIMENTAL TRIAL): TSDF + marching cubes on a synthetic plane, the view choice, the atlas,
the GLB container."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")

from umiproc.viz import mesh_3d as M  # noqa: E402
from umiproc.viz.slam_3d_images import Camera  # noqa: E402

K = np.array([[160.0, 0, 159.5], [0, 160.0, 119.5], [0, 0, 1]])
CAM = Camera(fx=300.0, fy=300.0, cx=319.5, cy=239.5, k=(0.0, 0.0, 0.0, 0.0), width=640, height=480)


def _look_down(x: float, y: float, h: float, tilt_deg: float = 0.0) -> np.ndarray:
    """Camera -> scene pose above the plane z = 0, looking down, tilted about the scene x axis."""
    a = np.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    T = np.eye(4)
    T[:3, :3] = Rx @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [x, y, h]
    return T


def _plane_depth(T: np.ndarray, plane_z: float = 0.0, size=(320, 240)) -> np.ndarray:
    v, u = np.mgrid[0:size[1], 0:size[0]].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    dz = (d @ T[:3, :3].T)[..., 2]
    z = (plane_z - T[2, 3]) / dz
    return np.where(z > 0, z, np.nan).astype(np.float32)


def test_tsdf_of_a_plane_gives_a_flat_mesh_in_the_right_place():
    poses = [_look_down(0.0, 0.0, 0.5), _look_down(0.05, 0.0, 0.45, 10), _look_down(-0.04, 0.03, 0.55, -8)]
    tsdf = M.Tsdf([-0.2, -0.15, -0.06], [0.2, 0.15, 0.06], 0.004, 0.016)
    for T in poses:
        assert tsdf.integrate(_plane_depth(T, 0.02), K, T) > 1000
    V, F = M.extract_mesh(tsdf, 2)
    V, F, pieces = M.clean_mesh(V, F, 0.01, 10)
    assert len(F) > 2000 and pieces["pieces_kept"] == 1
    assert np.abs(V[:, 2] - 0.02).max() < 0.001                     # flat, at the plane
    assert V[:, 0].max() - V[:, 0].min() > 0.3                      # covers the seen part of the box
    assert (M.face_normals(V, F)[:, 2] > 0.99).mean() > 0.99        # facing up, towards the cameras


def test_unseen_voxels_make_no_surface():
    tsdf = M.Tsdf([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1], 0.005, 0.02)
    V, F = M.extract_mesh(tsdf, 1)
    assert len(F) == 0


def _quad(z: float = 0.0, h: float = 0.05):
    V = np.array([[-h, -h, z], [h, -h, z], [h, h, z], [-h, h, z]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    return V, F


def test_view_choice_picks_the_head_on_view():
    V, F = _quad()
    p = M.MeshParams(zbuf_div=4, n_cand=3)
    T_head_on = _look_down(0.0, 0.0, 0.4)
    T_oblique = np.eye(4)                                           # 0.4 m away, 60 degrees off the normal
    d = np.array([0.0, np.sin(np.radians(60)), np.cos(np.radians(60))]) * 0.4
    z = -d / np.linalg.norm(d)
    x = np.array([1.0, 0, 0])
    T_oblique[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T_oblique[:3, 3] = d
    T_far = _look_down(0.0, 0.0, 1.0)                               # head-on but farther
    raw = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    views = [M.View(fed=k, T_S_cam=T, raw=raw) for k, T in enumerate([T_oblique, T_far, T_head_on])]
    pts = np.concatenate([V, V[F].mean(axis=1)])
    zbs = [M.zbuffer(CAM, v.T_S_cam, pts, p.zbuf_div) for v in views]
    cand, score = M.view_scores(CAM, views, zbs, V, F, None, p)
    assert (cand[:, 0] == 2).all() and (score[:, 0] > score[:, 1]).all()
    assert set(cand[0]) == {0, 1, 2}
    label = M.choose_views(cand, score, M.face_neighbours(F), 0.6, 5)
    assert (label == 2).all()


def _look_at(eye, target) -> np.ndarray:
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = eye
    return T


def test_view_choice_skips_an_occluded_view():
    V, F = _quad(0.0)
    blocker_V, blocker_F = _quad(0.2, 0.2)                          # a plate between camera 0 and the quad
    allV = np.concatenate([V, blocker_V])
    allF = np.concatenate([F, blocker_F + 4])
    p = M.MeshParams(zbuf_div=4, n_cand=2)
    raw = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    views = [M.View(fed=0, T_S_cam=_look_down(0.0, 0.0, 0.4), raw=raw),               # head-on, but blocked
             M.View(fed=1, T_S_cam=_look_at(np.array([0.6, 0.0, 0.4]), [0, 0, 0]), raw=raw)]  # oblique, free
    g = np.stack(np.meshgrid(np.linspace(-0.2, 0.2, 81), np.linspace(-0.2, 0.2, 81)), -1).reshape(-1, 2)
    splats = np.concatenate([allV, np.c_[g, np.full(len(g), 0.2)], np.c_[g / 4, np.zeros(len(g))]])
    zbs = [M.zbuffer(CAM, v.T_S_cam, splats, p.zbuf_div) for v in views]
    cand, score = M.view_scores(CAM, views, zbs, allV, allF, None, p)
    assert (cand[:2, 0] == 1).all() and (cand[:2, 1] == -1).all()    # the quad: only the free view
    assert (cand[2:, 0] == 0).all()                                 # the plate: head-on from camera 0


def test_face_neighbours_and_atlas_bake():
    V, F = _quad()
    nbr = M.face_neighbours(F)
    assert sorted(nbr[0].tolist()) == [-1, -1, 1] and sorted(nbr[1].tolist()) == [-1, -1, 0]
    cell, cols, W, H = M.atlas_layout(len(F), 8, 8192)
    assert (cell, cols, W, H) == (8, 2, 16, 8)
    raw = np.zeros((CAM.height, CAM.width, 3), np.uint8)
    raw[:, :CAM.width // 2] = (200, 10, 10)                         # left half red, right half blue
    raw[:, CAM.width // 2:] = (10, 10, 200)
    views = [M.View(fed=0, T_S_cam=_look_down(0.0, 0.0, 0.4), raw=raw)]
    atlas = M.bake_atlas(CAM, views, None, V, F, np.array([0, 0]), cell, cols, W, H)
    uv = M.face_uvs(len(F), cell, cols, W, H)
    # face 1 (corners 0, 2, 3) lies mostly at x < 0, i.e. image left (camera x = scene x): its corner 2 is red
    px = np.floor(uv[1, 2] * [W, H]).astype(int)
    assert tuple(atlas[px[1], px[0]]) == (200, 10, 10)
    px = np.floor(uv[0, 1] * [W, H]).astype(int)                    # face 0, corner 1 at x > 0: blue
    assert tuple(atlas[px[1], px[0]]) == (10, 10, 200)


def test_write_glb_container(tmp_path):
    V, F = _quad()
    uv = M.face_uvs(len(F), 8, 2, 16, 8)
    path = tmp_path / "m.glb"
    M.write_glb(path, V, F, uv, b"\xff\xd8fakejpeg\xff\xd9")
    data = path.read_bytes()
    assert data[:4] == b"glTF" and int.from_bytes(data[8:12], "little") == len(data)
    n = int.from_bytes(data[12:16], "little")
    gltf = json.loads(data[20:20 + n])
    assert gltf["accessors"][0]["count"] == 6 and gltf["images"][0]["mimeType"] == "image/jpeg"
    assert gltf["buffers"][0]["byteLength"] == int.from_bytes(data[20 + n:24 + n], "little")


def test_plane_fill_closes_an_enclosed_table_hole_only():
    g = np.arange(-0.2, 0.2001, 0.004)
    X, Y = np.meshgrid(g, g)
    keep = ~((np.abs(X) < 0.05) & (np.abs(Y) < 0.05))              # a 10 cm square hole in a 40 cm table
    V = np.c_[X[keep], Y[keep], np.full(keep.sum(), 0.01)]
    Vn = np.tile([0.0, 0.0, 1.0], (len(V), 1))
    F = np.zeros((0, 3), np.int64)
    Vf, Ff, plane = M.plane_fill(V, F, Vn, 0.008, 0.05)
    assert plane is not None and abs(plane[2] - 0.01) < 1e-6
    c = Vf[Ff].mean(axis=1)
    assert len(Ff) > 0 and np.abs(c[:, :2]).max() < 0.07           # only in (and just around) the hole
    assert (M.face_normals(Vf, Ff)[:, 2] > 0.99).all()
    _, Ff_small, _ = M.plane_fill(V, F, Vn, 0.008, 0.005)          # larger than max_hole_m2: left open
    assert len(Ff_small) == 0
