"""umiproc.map.tag: the table tag's detection through the fisheye lens (on an exactly rendered frame), UMI's sighting
filters (keyframes, 0.3-2 m, centre within 0.6 H/2) at their boundaries, and UMI's aggregation (geometric median,
strict 0.9 quantile, the kept sighting nearest the mean)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from umiproc.map import tag as T
from umiproc.sensors import load_sensors
from umiproc_testlib import render_tag

RIG = "rig_pi_v3"
quiet = lambda *a, **k: None  # noqa: E731


@pytest.fixture(scope="module")
def sensors(tmp_path_factory):
    return load_sensors(tmp_path_factory.mktemp("root"), RIG, log=quiet)


def _pose(rotvec, t):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    M[:3, 3] = t
    return M


@pytest.mark.parametrize("rotvec,t", [
    ([np.pi, 0.0, 0.0], [0.02, -0.03, 0.45]),                     # facing the camera, near the centre
    ([np.pi + 0.5, 0.2, 0.1], [-0.15, -0.10, 0.60]),              # tilted ~30 deg, off-centre (fisheye matters)
])
def test_detection_recovers_the_pose_through_the_fisheye(sensors, rotvec, t):
    det = T.TableTagDetector(sensors)
    truth = _pose(rotvec, t)
    rgb = render_tag(det.K, det.D, truth, det.size_m)
    d = det.observe(7, rgb)
    assert d is not None and d.frame_index == 7 and det.detections == {7: d} and det.n_frames == 1
    est = d.T_cam_tag
    assert np.linalg.norm(est[:3, 3] - truth[:3, 3]) < 0.004           # mm-level on a 16 cm tag
    ang = np.degrees(Rotation.from_matrix(est[:3, :3].T @ truth[:3, :3]).magnitude())
    assert ang < 1.0
    assert est[:3, 2] @ -est[:3, 3] > 0                                  # z points out of the tag, towards the lens
    # a nominal pinhole assumption would be clearly off at the edge: the fisheye path is what makes it right
    assert d.corners.shape == (4, 2)


def test_other_ids_and_no_tag_are_ignored(sensors):
    det = T.TableTagDetector(sensors)
    rgb = render_tag(det.K, det.D, _pose([np.pi, 0, 0], [0, 0, 0.5]), det.size_m, tag_id=12)
    assert det.observe(0, rgb) is None
    assert det.observe(1, np.full((960, 1280, 3), 128, np.uint8)) is None
    assert det.detections == {} and det.n_frames == 2


def test_uses_the_effective_lens(sensors):
    import dataclasses

    from umiproc.dp.config import CameraModel

    cam = dataclasses.replace(sensors.ultrawide, model=CameraModel("fisheye-equidistant", (600.0, 601.0, 650.0, 470.0),
                                                                   (0.01, 0.0, 0.0, 0.0)))
    s2 = SimpleNamespace(slam=sensors.slam, ultrawide=cam)
    det = T.TableTagDetector(s2)
    assert det.K[0, 0] == 600.0 and det.K[1, 2] == 470.0 and det.D[0, 0] == 0.01


# --------------------------------------------------------------------------------------------- filters

def _traj(n, keyframes, posed=None, state=None):
    posed = np.ones(n, bool) if posed is None else np.asarray(posed, bool)
    kf = np.zeros(n, bool)
    kf[list(keyframes)] = True
    Tw = np.tile(np.eye(4), (n, 1, 1))
    Tw[:, 0, 3] = np.arange(n) * 0.01                     # the camera moves along x: T_W_tag = T_W_cam @ T_cam_tag
    return SimpleNamespace(posed=posed, is_keyframe=kf, state=np.full(n, 2) if state is None else np.asarray(state),
                           T_W_cam=Tw)


def _det(i, dist, centre=(640.0, 480.0)):
    c = np.asarray(centre) + np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], float)
    return T.TagDetection(i, np.array([np.pi, 0, 0]), np.array([0.0, 0.0, dist]), c)


def test_distance_limits_are_inclusive():
    dets = {0: _det(0, 0.3), 1: _det(1, 0.2999), 2: _det(2, 2.0), 3: _det(3, 2.0001)}
    st = T.collect_sightings(_traj(4, range(4)), np.arange(4), dets, (1280, 960))
    assert [s.row for s in st.sightings] == [0, 2] and st.rejected_distance == 2
    assert st.keyframes == st.keyframes_with_tag == 4


def test_centre_limit_is_0_6_of_half_height_of_our_image():
    r = 0.6 * 480                                          # 288 px from (640, 480)
    dets = {0: _det(0, 1.0, (640 + r, 480)), 1: _det(1, 1.0, (640 + r + 0.5, 480)),
            2: _det(2, 1.0, (640, 480 - r)), 3: _det(3, 1.0, (1279, 959))}
    st = T.collect_sightings(_traj(4, range(4)), np.arange(4), dets, (1280, 960))
    assert [s.row for s in st.sightings] == [0, 2] and st.rejected_centre == 2
    assert st.sightings[0].centre_fraction == pytest.approx(0.6)


def test_only_posed_keyframes_count_and_rows_map_to_frames():
    n = 6
    traj = _traj(n, keyframes=[0, 2, 4, 5], posed=[1, 1, 1, 1, 0, 1], state=[2, 2, 3, 2, 2, 2])
    frame_of_row = np.array([10, 11, 12, 13, 14, 15])        # row k = video frame k + 10
    dets = {f: _det(f, 1.0) for f in range(10, 16)}
    st = T.collect_sightings(traj, frame_of_row, dets, (1280, 960))
    assert st.keyframes == 3                                 # row 4 is lost, rows 1/3 are not keyframes
    assert [(s.row, s.frame_index) for s in st.sightings] == [(0, 10), (2, 12), (5, 15)]
    assert st.keyframes_not_tracked == 1                     # row 2: RECENTLY_LOST keyframe (UMI keeps it too)
    s = st.sightings[1]
    np.testing.assert_allclose(s.T_world_tag, traj.T_W_cam[2] @ dets[12].T_cam_tag)


# --------------------------------------------------------------------------------------------- aggregation

def test_geometric_median_minimises_the_sum_of_distances():
    rng = np.random.default_rng(3)
    X = np.concatenate([rng.normal(0, 0.01, (30, 3)), [[1.0, 1.0, 1.0], [2.0, -1.0, 0.0]]])
    gm = T.geometric_median(X)
    best = minimize(lambda y: np.linalg.norm(X - y, axis=1).sum(), X.mean(0), method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000}).x
    assert np.linalg.norm(gm - best) < 1e-5
    sq = np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [0, 0, 0]], float)
    np.testing.assert_allclose(T.geometric_median(sq), 0, atol=1e-9)


def _poses(P):
    out = np.tile(np.eye(4), (len(P), 1, 1))
    out[:, :3, 3] = P
    return out


def test_aggregation_is_umis():
    # ten sightings at x = 0..9 mm and one 10 cm away: the 0.9 quantile drops the farthest (strictly below)
    P = np.zeros((11, 3))
    P[:10, 0] = np.arange(10) * 0.001
    P[10] = [0.1, 0, 0]
    a = T.aggregate_sightings(_poses(P))
    med = T.geometric_median(P)
    d = np.linalg.norm(P - med, axis=1)
    assert a["threshold_m"] == pytest.approx(np.quantile(d, 0.9))
    np.testing.assert_array_equal(a["kept"], d < np.quantile(d, 0.9))
    assert not a["kept"][10]
    kept = P[a["kept"]]
    np.testing.assert_allclose(a["std_m"], kept.std(axis=0))
    np.testing.assert_allclose(a["mean"], kept.mean(axis=0))
    nn = int(np.flatnonzero(a["kept"])[np.argmin(np.linalg.norm(kept - kept.mean(0), axis=1))])
    assert a["index"] == nn
    np.testing.assert_allclose(a["T_world_tag"], _poses(P)[nn])


def test_aggregation_of_identical_sightings_keeps_them_all():
    a = T.aggregate_sightings(_poses(np.zeros((1, 3))))
    assert a["kept"].tolist() == [True] and a["index"] == 0
    a = T.aggregate_sightings(_poses(np.ones((5, 3))))
    assert a["kept"].all() and np.allclose(a["std_m"], 0)
    with pytest.raises(ValueError):
        T.aggregate_sightings(np.zeros((0, 4, 4)))


def test_detection_json_round_trip():
    d = _det(4, 0.8)
    back = T.TagDetection.from_json(d.to_json())
    assert back.frame_index == 4
    np.testing.assert_allclose(back.T_cam_tag, d.T_cam_tag)
