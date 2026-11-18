"""umiproc.viz.slam_3d (``umiproc slam-3d``): reading the front-end's map export, and the HTML page it writes."""

import json
from pathlib import Path

import numpy as np
import pytest

from umiproc.cli import build_parser
from umiproc.slam.trajectory_csv import CSV_HEADER
from umiproc.viz import slam_3d as V

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
