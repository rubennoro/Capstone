"""umiproc.viz.slam_3d (``umiproc slam-3d``): reading the front-end's map export, and the HTML page it writes."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from umiproc.cli import build_parser
from umiproc.slam.trajectory_csv import CSV_HEADER
from umiproc.viz import slam_3d as V
from umiproc.viz import slam_3d_images as I

T0 = 1_000_000_000_000
DT = 33_333_000
UP = [0.0, -0.6, -0.8]                     # gravity up in W (the camera looks down and forward)


def _write_trajectory(run: Path, n: int = 6) -> None:
    lines = [",".join(CSV_HEADER), "0,0.000000,1,true,false,0,0,0,0,0,0,0"]          # frame 0 not tracked
    for k in range(1, n):
        lines.append(f"{k},{k * DT / 1e9:.6f},2,false,{str(k == 1).lower()},{0.1 * k},0,0,0,0,0,1")
    (run / "trajectory.csv").write_text("\n".join(lines) + "\n")
    (run / "trajectory.csv.t0_ns").write_text(f"{T0}\n")


def _write_export(run: Path, up=UP) -> Path:
    p = run / "map_points.csv"
    p.write_text("id,x,y,z,observations\n1,0.0,0.5,1.0,4\n2,0.2,0.5,1.1,7\n3,-0.1,0.4,0.9,2\n")
    (run / "map_points_keyframes.csv").write_text(
        "id,frame_id,timestamp,x,y,z,q_x,q_y,q_z,q_w\n"
        "0,1,0.033333,0,0,0,0,0,0,1\n"
        "5,4,0.133332,0.3,0,0,0,0.0499792,0,0.99875026\n")
    (run / "map_points_meta.json").write_text(json.dumps(
        {"format": "umiproc.map_points.v1", "map_id": 0, "keyframes": 2, "map_points": 3, "bad_map_points": 0,
         "first_keyframe_id": 0, "imu_initialized": up is not None, "up_W": up, "T_W_orbworld": None}))
    return p


def _payload(html: str) -> dict:
    start = html.index("const D = ") + len("const D = ")
    return json.loads(html[start:html.index(";\n", start)])


def test_read_map_export(tmp_path):
    ex = V.read_map_export(_write_export(tmp_path))
    assert ex.point_ids.tolist() == [1, 2, 3] and ex.observations.tolist() == [4, 7, 2]
    assert ex.points[1] == pytest.approx([0.2, 0.5, 1.1])
    assert ex.keyframe_ids.tolist() == [0, 5] and ex.keyframe_t_s == pytest.approx([0.033333, 0.133332])
    assert ex.T_W_kf[1, :3, 3] == pytest.approx([0.3, 0, 0])
    assert ex.T_W_kf[1, :3, :3] @ [0, 0, 1] == pytest.approx([np.sin(0.1), 0, np.cos(0.1)], abs=1e-6)  # 0.1 rad yaw
    assert ex.meta["up_W"] == UP


def test_read_map_export_refuses_bad_files(tmp_path):
    p = _write_export(tmp_path)
    p.write_text("id,x,y,z\n1,0,0,0\n")
    with pytest.raises(V.Slam3dError, match="header"):
        V.read_map_export(p)
    p = _write_export(tmp_path)
    (tmp_path / "map_points_meta.json").unlink()
    with pytest.raises(V.Slam3dError, match="_meta.json"):
        V.read_map_export(p)


@pytest.mark.parametrize("up", [UP, None, [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
def test_scene_rotation_turns_up_to_z(up):
    R = V.scene_rotation(up)
    u = np.array([0.0, -1.0, 0.0] if up is None else up)
    assert R @ (u / np.linalg.norm(u)) == pytest.approx([0, 0, 1], abs=1e-9)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-9) and np.linalg.det(R) == pytest.approx(1.0)


def test_page_written(tmp_path):
    _write_trajectory(tmp_path)
    _write_export(tmp_path)
    out, lines = V.make_page(tmp_path, title="t <test>")
    assert out == tmp_path / V.HTML_NAME
    html = out.read_text()
    assert "<title>t &lt;test&gt;</title>" in html and V.PLOTLY_URL in html
    d = _payload(html)
    assert len(d["points"]["xyz"]) == 3 and d["points"]["obs"] == [4, 7, 2]
    assert len(d["keyframes"]["ids"]) == 2 and len(d["frames"]) == 5 and d["frames"][0]["k"] == 1
    assert d["path"]["xyz"][0] == [None, None, None]                 # frame 0 not posed: a break in the line
    R = V.scene_rotation(UP)
    assert d["path"]["xyz"][3] == pytest.approx((R @ [0.3, 0, 0]).tolist(), abs=1e-4)
    assert d["points"]["height"][0] == pytest.approx(float((R @ [0.0, 0.5, 1.0])[2]), abs=1e-4)
    assert len(d["frame_T"]) == 5 and len(d["frame_T"][0]) == 12
    assert any("3 points" in ln for ln in lines)


def test_page_without_points_and_cli(tmp_path, capsys):
    _write_trajectory(tmp_path)
    args = build_parser().parse_args(["slam-3d", str(tmp_path), "--out", str(tmp_path / "x.html")])
    assert args._run(args) == 0
    d = _payload((tmp_path / "x.html").read_text())
    assert d["points"]["xyz"] == [] and d["meta"] is None and len(d["frames"]) == 5
    assert "no map_points.csv" in capsys.readouterr().out
    args = build_parser().parse_args(["slam-3d", str(tmp_path), "--points", str(tmp_path / "missing.csv")])
    assert args._run(args) == 2


# --------------------------------------------------------------------------------------------- declutter

def _pts_payload(tmp_path, filters):
    _write_trajectory(tmp_path)
    _write_export(tmp_path)
    out, lines = V.make_page(tmp_path, filters=filters)
    return _payload(out.read_text()), lines


def test_parse_near():
    assert V.parse_near("1.5") == (1.5, None) and V.parse_near("0.4m") == (None, 0.4)
    assert V.parse_near("0") == (None, None)
    with pytest.raises(ValueError):
        V.parse_near("-1")


def test_filters_min_obs_near_box(tmp_path):
    d, lines = _pts_payload(tmp_path, V.Filters())                 # defaults: min obs 3, near 1 x the path size
    assert len(d["points"]["xyz"]) == 3                            # every point stays in the file
    assert d["hidden_by"]["min_obs"] == 1 and d["filters"]["min_obs"] == 3   # obs 2
    assert d["extent"] == pytest.approx(0.5, abs=1e-3)             # path x 0.1..0.5 and the keyframes 0..0.3
    # point 2 (0.2, 0.5, 1.1) is ~1.2 from the nearest camera position: farther than 1 x 0.5
    assert d["hidden_by"]["near"] == 3 and d["hidden_points"] == 3
    cams = [[0.1 * k, 0, 0] for k in range(1, 6)] + [[0, 0, 0], [0.3, 0, 0]]
    assert d["points"]["dist"][1] == pytest.approx(min(np.linalg.norm(np.subtract([0.2, 0.5, 1.1], c)) for c in cams),
                                                   abs=1e-3)
    assert any("hidden by: fewer than 3 observations 1" in ln for ln in lines)
    d, _ = _pts_payload(tmp_path, V.Filters(min_obs=0, near=None, outliers=False))
    assert d["hidden_points"] == 0 and d["filters"]["near"] is None
    d, _ = _pts_payload(tmp_path, V.Filters(min_obs=0, near=None, near_m=1.15))    # distances 1.12, 1.21, 0.99
    assert d["hidden_by"]["near"] == 1 and d["filters"]["near"] == pytest.approx(2.3)   # 1.15 m / 0.5
    d, _ = _pts_payload(tmp_path, V.Filters(min_obs=0, near=None, box=10.0))
    assert d["hidden_by"]["box"] == 0
    d, _ = _pts_payload(tmp_path, V.Filters(min_obs=0, near=None, box=0.1))
    assert d["hidden_by"]["box"] == 3                              # all ~1 m off the path's box


def test_cli_filter_flags(tmp_path):
    _write_trajectory(tmp_path)
    _write_export(tmp_path)
    args = build_parser().parse_args(["slam-3d", str(tmp_path), "--out", str(tmp_path / "x.html"), "--min-obs",
                                      "5", "--near", "0.5m", "--box", "0.2", "--show-outliers", "--no-images"])
    assert args._run(args) == 0
    f = _payload((tmp_path / "x.html").read_text())["filters"]
    assert f == {"min_obs": 5, "near": pytest.approx(1.0), "box": 0.2, "outliers": False}


# --------------------------------------------------------------------------------------------- image colours

CAM = I.Camera(fx=30.0, fy=30.0, cx=32.0, cy=24.0, k=(0.01, -0.002, 0.0, 0.0), width=64, height=48)


def test_project_matches_cv2_fisheye():
    import cv2

    rng = np.random.default_rng(0)
    pts = rng.uniform([-1, -1, 0.5], [1, 1, 3], (50, 3))
    T = V._poses(np.array([[0.1, -0.2, 0.3]]), Rotation.from_rotvec([0.1, 0.2, -0.05]).as_quat()[None])[0]
    uv, front, d = I.project(CAM, T, pts)
    Tc = np.linalg.inv(T)
    K = np.array([[CAM.fx, 0, CAM.cx], [0, CAM.fy, CAM.cy], [0, 0, 1]])
    ref, _ = cv2.fisheye.projectPoints(pts[front].reshape(-1, 1, 3), cv2.Rodrigues(Tc[:3, :3])[0], Tc[:3, 3], K,
                                       np.array(CAM.k))
    assert uv[front] == pytest.approx(ref.reshape(-1, 2), abs=1e-6)
    assert d == pytest.approx(np.linalg.norm(pts - T[:3, 3], axis=1))


def test_colours_from_a_synthetic_camera():
    img = np.zeros((24, 32, 3), np.uint8)                           # at half the camera's size
    img[:, :16] = (200, 0, 0)                                       # left half red, right half blue
    img[:, 16:] = (0, 0, 200)
    green = np.full((48, 64, 3), (0, 180, 0), np.uint8)
    mask = np.zeros((48, 64), np.uint8)
    mask[36:] = 255                                                 # the gripper: the bottom quarter
    T = np.tile(np.eye(4), (3, 1, 1))
    T[2, :3, 3] = [0.0, 0.0, -3.0]                                  # keyframe 2: same view, 3 m farther back
    pts = np.array([[-0.5, 0, 1], [0.5, 0, 1], [0, 0.6, 1], [0, 0, -4], [-0.4, -0.1, 1]])
    # keyframe 0: red/blue; keyframe 1: no image; keyframe 2: green (farther away)
    rgb, dist = I.sample_colours(CAM, T, pts, {0: img, 2: green}, mask)
    assert np.isnan(rgb[:, 1]).all() and np.isinf(dist[:, 1]).all()
    col, views = I.point_colours(rgb, dist, n_views=1)
    assert col[0].tolist() == [200, 0, 0] and col[1].tolist() == [0, 0, 200]      # the nearer keyframe 0
    assert views[2] == 1 and col[2].tolist() == [0, 180, 0]        # on the mask in 0, not in 2 (farther away)
    assert views[3] == 0                                           # behind every camera
    col, views = I.point_colours(rgb, dist, n_views=5)
    assert views[0] == 2 and col[0].tolist() == [100, 90, 0]       # median of red and green


def _write_recording(root: Path, n: int = 8) -> Path:
    import cv2

    rec = root / "raw"
    rec.mkdir()
    (rec / "meta.json").write_text(json.dumps({"cameras": {"cam1": {"role": "ultrawide"}}}))
    w = cv2.VideoWriter(str(rec / "cam1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for k in range(n):
        w.write(np.full((48, 64, 3), (200, 40, 40) if k == 4 else (40, 40, 200), np.uint8))   # BGR: 4 blue, else red
    w.release()
    (rec / "cam1_frames.csv").write_text("frame_idx,t_ns\n" + "".join(f"{k},{T0 + k * DT}\n" for k in range(n)))
    (root / "orbslam3_settings.yaml").write_text(
        '%YAML:1.0\n---\nCamera.type: "KannalaBrandt8"\nCamera1.fx: 30.0\nCamera1.fy: 30.0\nCamera1.cx: 32.0\n'
        "Camera1.cy: 24.0\nCamera1.k1: 0.0\nCamera1.k2: 0.0\nCamera1.k3: 0.0\nCamera1.k4: 0.0\nCamera.width: 64\n"
        "Camera.height: 48\n")
    return rec


def test_page_with_photos(tmp_path):
    run = tmp_path / "mapping"
    run.mkdir()
    _write_trajectory(run)
    _write_export(run)
    rec = _write_recording(tmp_path)
    assert I.read_camera(tmp_path / "orbslam3_settings.yaml").fx == 30.0
    out, lines = V.make_page(run, recording=rec, settings=tmp_path / "orbslam3_settings.yaml", thumb_width=32,
                             filters=V.Filters(min_obs=0, near=None))
    d = _payload(out.read_text())
    th = d["keyframes"]["thumb"]
    assert len(th) == 2 and all(t.startswith("data:image/jpeg;base64,") for t in th)
    assert d["keyframes"]["frame"] == [1, 4]
    rgb = d["points"]["rgb"]
    assert len(rgb) == 3 and all(c is not None for c in rgb)
    for c in rgb:                                                 # the median of frame 1 (red) and 4 (blue)
        r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
        assert abs(r - 120) < 25 and g < 70 and abs(b - 120) < 25
    assert any("photos: 2 of 2 keyframes" in ln for ln in lines)
    out, lines = V.make_page(run, recording=tmp_path / "nothing", settings=tmp_path / "orbslam3_settings.yaml")
    assert _payload(out.read_text())["points"]["rgb"] is None and any("no photos" in ln for ln in lines)
