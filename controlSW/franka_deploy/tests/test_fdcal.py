"""Synthetic tests of the calibration tools (hand-eye, pivot, width map)."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from franka_deploy.fdcal import handeye as he
from franka_deploy.fdcal import pivot as pv


def T_of(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


BOARD = {"squares_x": 7, "squares_y": 5, "square_m": 0.04, "marker_m": 0.03}
BOARD_HE = {"squares_x": 9, "squares_y": 7, "square_m": 0.03, "marker_m": 0.022}   # hand-eye: more corners


def test_board_detection_and_pnp_on_rendered_view():
    b = he.make_board(BOARD)
    W, H = 1400, 1000
    img = b.generateImage((W, H), marginSize=0)
    s = 7 * 0.04 / W
    Rm = cv2.Rodrigues(np.array([0.2, -0.1, 0.05]))[0]
    t = np.array([-0.12, -0.08, 0.45])
    Hm = he.K_MAIN_960 @ np.column_stack([Rm[:, 0], Rm[:, 1], t]) @ np.diag([s, s, 1.0])
    cam = cv2.warpPerspective(img, Hm, (960, 720), borderValue=255)
    T = he.board_pose_from_image(cam, b, he.K_MAIN_960)
    assert T is not None
    assert np.linalg.norm(T[:3, 3] - t) < 0.002
    assert np.degrees(R.from_matrix(T[:3, :3].T @ Rm).magnitude()) < 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hand_eye_recovers_known_F_T_C(seed):
    rng = np.random.default_rng(seed)
    F_T_C_cad = T_of([0, 0, 0], [0.0, 0.0, 0.10])
    F_T_C = F_T_C_cad @ T_of(rng.normal(0, 0.01, 3), rng.normal(0, 0.004, 3))   # truth = CAD + ~0.6 deg / 4 mm
    O_T_B = T_of([0, 0, 0.3], [0.55, -0.10, 0.0])                                 # board on the table
    board_obj = he.make_board(BOARD_HE).getChessboardCorners()
    F_list, C_list = [], []
    center = O_T_B[:3, :3] @ np.array([0.135, 0.105, 0.0]) + O_T_B[:3, 3]
    while len(F_list) < 40:
        # camera on a hemisphere (r 0.45 m, polar <= 40 deg) looking at the board centre, random roll (+-60 deg):
        # the rotation diversity hand-eye needs
        pol, az, roll = np.radians(rng.uniform(5, 40)), rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(-60, 60))
        pos = center + 0.35 * np.array([np.sin(pol) * np.cos(az), np.sin(pol) * np.sin(az), np.cos(pol)])
        z = (center - pos) / np.linalg.norm(center - pos)
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        Rc = np.column_stack([x, y, z]) @ R.from_rotvec([0, 0, roll]).as_matrix()
        O_T_C = np.eye(4)
        O_T_C[:3, :3], O_T_C[:3, 3] = Rc, pos
        C_T_B = np.linalg.inv(O_T_C) @ O_T_B
        pts_c = (C_T_B[:3, :3] @ board_obj.T).T + C_T_B[:3, 3]
        if np.any(pts_c[:, 2] <= 0.1):
            continue
        uv = (he.K_MAIN_960 @ (pts_c / pts_c[:, 2:3]).T).T[:, :2] + rng.normal(0, 0.3, (len(pts_c), 2))
        if np.any(uv < 0) or np.any(uv[:, 0] > 960) or np.any(uv[:, 1] > 720):
            continue
        C_est = he.pnp(board_obj, uv, he.K_MAIN_960)
        F_list.append(O_T_C @ np.linalg.inv(F_T_C))
        C_list.append(C_est)
    out = he.calibrate(F_list, C_list, cad_F_T_C=F_T_C_cad)
    dt, da = he.pose_diff(np.array(out["F_T_C"]), F_T_C)
    assert dt < 0.0005 and da < 0.1, (dt, da)
    cad_dt, cad_da = he.pose_diff(np.array(out["F_T_C"]), F_T_C_cad)
    assert abs(out["cad_agreement"]["mm"] - cad_dt * 1000) < 1e-9 and abs(out["cad_agreement"]["deg"] - cad_da) < 1e-9
    np.testing.assert_allclose(np.array(out["F_T_TCP"]), np.array(out["F_T_C"]) @ he.C_T_TCP)


def test_hand_eye_cad_disagreement_flagged():
    F = [T_of(R.random(random_state=i).as_rotvec() * 0.5, [0.5 + 0.02 * i, 0.01 * i, 0.4]) for i in range(10)]
    F_T_C = T_of([0, 0, 0], [0, 0, 0.1])
    O_T_B = T_of([0, 0, 0], [0.6, 0, 0])
    C = [np.linalg.inv(f @ F_T_C) @ O_T_B for f in F]
    out = he.calibrate(F, C, cad_F_T_C=T_of([0, 0, 0.05], [0.01, 0, 0.1]))   # CAD off by 10 mm / 2.9 deg
    assert not out["cad_agreement"]["accept"]


def test_pivot_detects_tcp_error():
    F_T_TCP_true = T_of([0.1, 0, 0], [0.0, 0.0644, 0.3114])
    O_T_TCP_center = T_of([np.pi, 0, 0], [0.5, 0.0, 0.1])
    cmds = pv.pivot_commands(O_T_TCP_center, 20.0)
    # the robot rotates about the TRUE TCP; poses recorded = flange poses
    flange = [T @ np.linalg.inv(F_T_TCP_true) for T in cmds]
    ok = pv.analyze(flange, F_T_TCP_true)
    assert ok["accept"] and ok["tcp_excursion_mm"] < 1e-6 and ok["F_T_TCP_translation_error_mm"] < 1e-6
    wrong = F_T_TCP_true.copy()
    wrong[:3, 3] += [0.004, 0.0, 0.0]                      # 4 mm error in the configured TCP
    bad = pv.analyze(flange, wrong)
    assert not bad["accept"] and abs(bad["F_T_TCP_translation_error_mm"] - 4.0) < 1e-6


def render_marker(canvas, K, aruco_dict, mid, size_m, t, px=200, pad=40):
    m = cv2.aruco.generateImageMarker(aruco_dict, mid, px)
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    n = px + 2 * pad
    s = size_m / px
    A = np.array([[s, 0, -(pad + px / 2) * s], [0, -s, (pad + px / 2) * s], [0, 0, 1.0]])
    Rm = np.diag([1.0, -1.0, -1.0])                        # tag facing the camera, upright
    Hm = K @ np.column_stack([Rm[:, 0], Rm[:, 1], t]) @ A
    warped = cv2.warpPerspective(m, Hm, (canvas.shape[1], canvas.shape[0]), borderValue=255)
    return np.minimum(canvas, warped)


def test_width_map_from_rendered_tags():
    pytest.importorskip("iphumi")
    from franka_deploy.fdcal import width as wd
    from franka_deploy.gripper.driver import WidthCalibration

    d, sizes = wd.load_aruco()
    K = wd.K_UW_960
    true_w = np.linspace(0.0, 0.10, 11)
    positions = 3000 - 13636.0 * true_w
    raw = []
    for w in true_w:
        img = np.full((720, 960), 255, np.uint8)
        x0 = 0.022                                         # finger tag offsets from the centre when closed
        img = render_marker(img, K, d, 0, sizes[0], np.array([-(x0 + w / 2), 0.03, 0.08]))
        img = render_marker(img, K, d, 1, sizes[1], np.array([+(x0 + w / 2), 0.03, 0.08]))
        r = wd.raw_width(img, K, 0, 1, d, sizes)
        assert r is not None
        raw.append(r)
    raw = np.array(raw)
    measured = raw - raw[0]
    assert np.max(np.abs(measured - true_w)) < 0.0005      # same detector + width definition as iPhUMI
    table = wd.fit_map(positions, raw)
    c = table["checks"]
    assert c["zero_at_closed"] and c["covers_0p085"] and c["n_points"] == 11
    cal = WidthCalibration(np.array(table["width_m"]), np.array(table["position"]))
    for w in np.linspace(0, 0.1, 21):                      # round trip through the fitted map
        assert abs(cal.to_width(cal.to_position(w)) - w) < 1e-4


def test_pivot_capture_drives_fake_bridge(tmp_path):
    """fdcal capture pivot against the FakeBackend bridge (own process): correct F_T_TCP -> accept, and the arm is
    rotated about the TCP (flange moves, TCP does not)."""
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import yaml

    from franka_deploy.bridge.link import BridgeClient
    from franka_deploy.fdcal import capture

    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(root)
    ep = "tcp://127.0.0.1:5598"
    p = subprocess.Popen([sys.executable, "-m", "franka_deploy.bridge.franka_bridge", "--config",
                          str(root / "configs" / "bridge.yaml"), "--endpoint", ep], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        cli = BridgeClient(ep, timeout_s=1.0)
        cli.request("start")
        F_T_TCP = np.array(yaml.safe_load(open(root / "configs" / "franka_cartesian.yaml"))["F_T_TCP"], float)
        res = capture.capture_pivot(ep, F_T_TCP, angle=20.0, settle_s=0.4, out=str(tmp_path / "piv"), w_deg_s=40.0, v_m_s=0.1)
        assert res["accept"], res
        poses = [np.array(T) for T in __import__("json").load(open(tmp_path / "piv" / "poses.json"))]
        flange_spread = max(np.linalg.norm(T[:3, 3] - poses[0][:3, 3]) for T in poses)
        assert len(poses) == 6 and flange_spread > 0.02 and res["tcp_excursion_mm"] < 0.5
    finally:
        p.terminate()
        p.wait(10)
