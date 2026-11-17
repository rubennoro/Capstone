"""The lens model and the building blocks of lens-calibrate: cv2.fisheye == the rig's fisheye-equidistant ==
ORB-SLAM3's KannalaBrandt8, the detector's pixel convention, the motion filter, view selection, the fit's
robustness, the lens file and --apply's backup."""

import datetime as dt

import cv2
import numpy as np
import pytest
import yaml

from umiproc.dp.config import load_rig
from umiproc.dp.geometry.camera_model import FisheyeEquidistant
from umiproc.dp.sim.render import camera_model
from umiproc.lens import detect, fit, lensfile, synth
from umiproc.lens import board as lens_board
from umiproc.paths import repo_config

NOMINAL = load_rig(repo_config("rigs", "rig_pi_v3.yaml")).cameras["ultrawide"]


def _rays(n=3000, max_deg=85.0, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, np.radians(max_deg), n)
    ph = rng.uniform(-np.pi, np.pi, n)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1) * rng.uniform(0.1, 4, (n, 1))


@pytest.mark.parametrize("params", [
    tuple(NOMINAL.model.intrinsics) + tuple(NOMINAL.model.distortion),
    (574.6, 573.1, 643.2, 476.4, 0.021, -0.0052, 0.0011, 0.0),
    (601.0, 598.5, 630.0, 470.0, -0.03, 0.01, -0.004, 0.0007),
])
def test_cv2_fisheye_is_the_rig_model_and_orbslam3_kb8(params):
    fx, fy, cx, cy, *k = params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.array(k)
    P = _rays()
    uv_cv, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    uv_cv = uv_cv.reshape(-1, 2)
    rig = FisheyeEquidistant(1280, 960, fx, fy, cx, cy, *k)
    uv_rig, ok = rig.project(P)
    assert ok.all()
    assert np.abs(uv_cv - uv_rig).max() < 1e-9
    assert np.abs(uv_cv - fit.kannala_brandt8_project(params, P)).max() < 1e-9
    # and the inverse: cv2.fisheye.undistortPoints == the rig model's unproject
    und = cv2.fisheye.undistortPoints(uv_rig.reshape(-1, 1, 2), K, D, criteria=(3, 100, 1e-14)).reshape(-1, 2)
    rays, ok = rig.unproject(uv_rig)
    assert np.abs(und - rays[:, :2] / rays[:, 2:]).max() < 1e-6
    assert fit.model_agreement(K, D, (1280, 960))["agree"]


def test_rig_nominal_model_is_kannala_brandt():
    m = camera_model(NOMINAL)
    assert isinstance(m, FisheyeEquidistant) and NOMINAL.model.type == "fisheye-equidistant"
    assert fit.model_agreement(m.K, m.dist, m.size)["agree"]


def test_detector_offset_and_convention():
    """detectBoard's corners are shifted by the measured offset; after subtracting it they are on the truth."""
    det = detect.make_detector()
    off = detect.detector_offset(det)
    assert off in (0.0, 0.5)
    img = cv2.GaussianBlur(lens_board.board_image(1, 14.0), (0, 0), 0.8)
    keep = np.full(img.shape, 255, np.uint8)
    fc, n_masked = detect.detect_frame(det, img, keep, off)
    truth = (fit.board_points(fc.ids)[:, :2] * 1000.0 + 14.0) * lens_board.EXACT_PX_PER_MM - 0.5
    assert len(fc) == 35 and n_masked == 0
    assert np.abs(fc.uv - truth).max() < 0.01


def test_mask_and_edge_filter():
    det = detect.make_detector()
    img = cv2.GaussianBlur(lens_board.board_image(1, 14.0), (0, 0), 0.8)
    keep = np.full(img.shape, 255, np.uint8)
    keep[:, : img.shape[1] // 2] = 0                         # mask the left half
    fc, n_masked = detect.detect_frame(det, img, keep, detect.detector_offset(det))
    assert 0 < len(fc) < 35 and n_masked == 35 - len(fc)
    assert (fc.uv[:, 0] >= img.shape[1] // 2 - 0.5).all()
    assert list(fc.ids) == sorted(fc.ids)


def _fc(frame, ids, uv):
    return detect.FrameCorners(frame, np.asarray(ids), np.asarray(uv, np.float64))


def test_still_flags():
    ids = np.arange(20)
    base = np.stack([ids * 10.0, ids * 5.0], 1)
    frames = [_fc(0, ids, base), _fc(1, ids, base + 0.4), _fc(2, ids, base + 0.8), _fc(3, ids, base + 3.0),
              _fc(4, ids, base + 3.2), None, _fc(6, ids, base), _fc(7, ids[:5], base[:5]), _fc(8, ids, base)]
    still, motion = detect.still_flags(frames)
    assert list(np.nonzero(still)[0]) == [1]                 # 2 moved 2.2 px to 3; 4 has no next frame (None)
    assert motion[1] == pytest.approx(0.4 * np.sqrt(2))
    assert np.isinf(motion[0]) and np.isinf(motion[8])       # first / last frame: one neighbour only
    assert np.isinf(motion[6])                               # neighbour 7 shares < MIN_COMMON ids


def _view(frame, ids, uv, tilt_bin=0):
    v = fit.View(frame, np.asarray(ids), np.asarray(uv, np.float64), fit.board_points(ids))
    gx = np.clip((v.uv[:, 0] + 0.5) * fit.GRID[0] // 1280, 0, fit.GRID[0] - 1).astype(int)
    gy = np.clip((v.uv[:, 1] + 0.5) * fit.GRID[1] // 960, 0, fit.GRID[1] - 1).astype(int)
    v.cells = np.zeros((fit.GRID[1], fit.GRID[0]), np.int64)
    np.add.at(v.cells, (gy, gx), 1)
    v.tilt_bin = tilt_bin
    return v


def test_select_views_prefers_new_coverage_and_skips_duplicates():
    ids = np.arange(15)
    a = _view(0, ids, np.stack([100 + ids * 5.0, np.full(15, 100.0)], 1))
    a_dup = _view(1, ids, a.uv + 3.0)                         # same pose, 3 px away
    b = _view(2, ids, np.stack([900 + ids * 5.0, np.full(15, 700.0)], 1))
    c = _view(3, ids, a.uv + 40.0, tilt_bin=5)                # same cells as a, but a new tilt
    sel = fit.select_views([a, a_dup, b, c], 10)
    assert a_dup not in sel
    assert [v.frame for v in sel][:2] in ([0, 2], [2, 0])     # new cells first
    assert {v.frame for v in sel} == {0, 2, 3}


def test_hold_out_every_fifth():
    views = [_view(i, np.arange(12), np.stack([50 + 100.0 * i + np.arange(12) * 3, np.full(12, 80.0 * (i % 9) + 40)],
                                              1)) for i in range(12)]
    sel = fit.select_views(views, 11)
    assert len(sel) == 11
    assert [k for k, v in enumerate(sel, 1) if v.role == "held_out"] == [5, 10]


def test_fit_removes_an_outlier_view():
    """A view with wrongly assigned corners is removed (outlier rule); the rest fits the truth."""
    m = synth.truth_model()
    rng = np.random.default_rng(1)
    views = []
    for i, (u, v) in enumerate([(300, 200), (640, 200), (980, 200), (300, 450), (980, 450), (640, 330), (200, 600),
                                (1080, 600), (450, 120), (830, 120), (640, 480), (150, 300)]):
        pose = synth.look_pose(m, (u, v), 0.3 + 0.02 * (i % 4), (-1) ** i * 25, (-1) ** (i // 2) * 20, 5 * i)
        ids = np.arange(35)
        P = fit.board_points(ids) @ pose.R.T + pose.t
        uv, ok = m.project(P)
        ok &= m.in_image(uv)
        uv = uv[ok] + rng.normal(0, 0.05, (ok.sum(), 2))
        views.append(fit.View(i, ids[ok], uv, fit.board_points(ids[ok])))
    bad = views[3]
    bad.uv = bad.uv.copy()
    bad.uv[:6] = bad.uv[[1, 0, 3, 2, 5, 4]]                   # six corners swapped: a mis-detection
    K0 = camera_model(NOMINAL).K
    res = fit.fit_fisheye(views, (1280, 960), K0, np.asarray(NOMINAL.model.distortion))
    assert [v.frame for v in res.removed] == [3] and "outlier" in res.removed[0].removed_reason
    assert res.rms_px < 0.1
    assert res.K[0, 0] == pytest.approx(m.fx, rel=1e-3) and res.K[0, 2] == pytest.approx(m.cx, abs=0.5)


def test_fit_drops_a_degenerate_view():
    """Corners on one line make OpenCV's InitExtrinsics fail with an assertion that names no view: the view is
    screened out before the joint fit instead of failing the calibration."""
    m = synth.truth_model()
    views = []
    for i, (u, v) in enumerate([(300, 200), (640, 200), (980, 200), (300, 450), (980, 450), (640, 330), (200, 600),
                                (1080, 600), (450, 120), (830, 120)]):
        pose = synth.look_pose(m, (u, v), 0.3 + 0.02 * (i % 4), (-1) ** i * 25, (-1) ** (i // 2) * 20, 5 * i)
        uv, ok = m.project(fit.board_points(np.arange(35)) @ pose.R.T + pose.t)
        views.append(fit.View(i, np.arange(35)[ok], uv[ok], fit.board_points(np.arange(35)[ok])))
    ids = np.arange(7)                                         # one row of corners: collinear
    pose = synth.look_pose(m, (640, 280), 0.35)
    uv, _ = m.project(fit.board_points(ids) @ pose.R.T + pose.t)
    views.insert(4, fit.View(99, ids, uv, fit.board_points(ids)))
    K0 = camera_model(NOMINAL).K
    D0 = np.asarray(NOMINAL.model.distortion)
    with pytest.raises(cv2.error):
        fit._calibrate(views, (1280, 960), K0, D0)               # what OpenCV does with it
    res = fit.fit_fisheye(views, (1280, 960), K0, D0)
    assert [v.frame for v in res.removed] == [99] and "degenerate" in res.removed[0].removed_reason
    assert res.rms_px < 1e-3 and res.K[0, 0] == pytest.approx(m.fx, rel=1e-6)


def test_pose_from_model_recovers_the_pose():
    m = synth.truth_model()
    pose = synth.look_pose(m, (400, 300), 0.35, 20, -15, 10)
    ids = np.arange(35)
    uv, _ = m.project(fit.board_points(ids) @ pose.R.T + pose.t)
    rvec, tvec = fit.pose_from_model(fit.board_points(ids), uv, m.K, m.dist)
    R, _ = cv2.Rodrigues(rvec)
    assert np.abs(R - pose.R).max() < 1e-6 and np.abs(tvec - pose.t).max() < 1e-6


def test_synth_rays_match_the_model():
    m = synth.truth_model()
    rng = np.random.default_rng(2)
    u, v = rng.uniform(-0.5, 1279.5, 5000), rng.uniform(-0.5, 959.5, 5000)
    fast = synth._pixel_rays(m, u, v).astype(np.float64)
    exact, ok = m.unproject(np.stack([u, v], 1))
    assert ok.all()
    ang = np.arccos(np.clip(np.sum(fast * exact, 1) / np.linalg.norm(fast, axis=1), -1, 1))
    assert ang.max() < 2e-7                                   # float32 rounding: < 1e-4 px


def _lens_doc(fx=574.0):
    return lensfile.lens_document(rig_id="rig_pi_v3", sensor="imx219", resolution=(1280, 960),
                                  intrinsics=[fx, 573.0, 641.0, 478.0], distortion=[0.02, -0.005, 0.001, 0.0],
                                  date="2026-11-16T12:00:00Z", source={"pi_name": "episode_000001"},
                                  board=lens_board.board_info(), metrics={"fit_rms_px": 0.1})


def test_lens_file_round_trip(tmp_path):
    p = tmp_path / "lens.yaml"
    p.write_text(lensfile.dump_lens(_lens_doc()))
    lf = lensfile.load_lens(p)
    assert isinstance(lf.model, FisheyeEquidistant)
    assert (lf.model.fx, lf.model.cy, lf.model.k2) == (574.0, 478.0, -0.005)
    assert lf.resolution == (1280, 960) and lf.provenance == "measured" and lf.rig_id == "rig_pi_v3"
    d = yaml.safe_load(p.read_text())
    d["schema"] = "umiproc.lens.v0"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(lensfile.LensFileError, match="schema"):
        lensfile.load_lens(p)
    d["schema"], d["model"]["distortion"] = lensfile.LENS_SCHEMA, [0.1]
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(lensfile.LensFileError, match="model.distortion"):
        lensfile.load_lens(p)


def test_apply_backs_up_and_skips_identical(tmp_path):
    new = tmp_path / "new.yaml"
    new.write_text(lensfile.dump_lens(_lens_doc(580.0)))
    dest = tmp_path / "rigs" / "rig_pi_v3.lens.yaml"
    d, backup, changed = lensfile.apply_lens(new, dest)       # nothing there yet
    assert changed and backup is None and dest.read_bytes() == new.read_bytes()
    assert lensfile.apply_lens(new, dest) == (dest, None, False)      # identical: nothing written
    old = dest.read_bytes()
    new.write_text(lensfile.dump_lens(_lens_doc(590.0)))
    now = dt.datetime(2026, 11, 16, 18, 30, 5, tzinfo=dt.timezone.utc)
    _, backup, changed = lensfile.apply_lens(new, dest, now)
    assert changed and backup == dest.with_name("rig_pi_v3.lens.yaml.bak-20261116T183005Z")
    assert backup.read_bytes() == old and dest.read_bytes() == new.read_bytes()
    new.write_text(lensfile.dump_lens(_lens_doc(600.0)))
    _, backup2, _ = lensfile.apply_lens(new, dest, now)        # same second: a distinct backup name
    assert backup2.name == "rig_pi_v3.lens.yaml.bak-20261116T183005Z-2"
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: nope\n")
    with pytest.raises(lensfile.LensFileError):
        lensfile.apply_lens(bad, dest)
    assert lensfile.load_lens(dest).model.fx == 600.0          # a bad file never replaces the applied one
