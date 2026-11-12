"""The worked check from planning (tests/tcp_check_reference.py, verified 2026-11-12), as tests, plus the hand-frame
camera pose, the real numbers, the rig-file writer and the tool's command line (STRUCTURE.md sections 6.6 and 9)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation as R

from deploy.geometry import F_T_H, GripperFile, RigTcp, TcpTriangle, camera_pose, invert, pose_error
from deploy.geometry.__main__ import main as geometry_main
from deploy import Station
from helpers import F_T_TCP_LIST, REAL_GRIPPER, REAL_RIG, write_yaml


def T(Rm, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = Rm
    M[:3, 3] = t
    return M


# --- fixtures: the reference's numbers ---

@pytest.fixture
def F_T_TCP() -> np.ndarray:
    """Franka Hand axes (-45 deg about flange z), fingertips 150 mm from the plate."""
    return T(R.from_euler("z", -45, degrees=True).as_matrix(), [0, 0, 0.150])


@pytest.fixture
def example_F_T_cam() -> np.ndarray:
    """The reference's generic example mount: 7 cm toward +y, 4 cm out, up = +y, tilted 30 deg (hand frame)."""
    return F_T_H @ camera_pose(np.array([0.0, 0.07, 0.04]), np.array([0.0, 1.0, 0.0]), 30.0)


@pytest.fixture
def real_F_T_cam() -> np.ndarray:
    gripper = GripperFile.load(REAL_GRIPPER)
    return F_T_H @ camera_pose(gripper.position_m, gripper.up_axis, gripper.tilt_deg)


@pytest.fixture(params=["example", "real"])
def F_T_cam(request, example_F_T_cam, real_F_T_cam) -> np.ndarray:
    return example_F_T_cam if request.param == "example" else real_F_T_cam


# --- the reference's five checks ---

def test_triangle_closes(F_T_cam, F_T_TCP):
    T_cam_tcp = invert(F_T_cam) @ F_T_TCP
    np.testing.assert_allclose(F_T_cam @ T_cam_tcp, F_T_TCP, atol=1e-12)


def test_rig_file_matrix_round_trips(F_T_cam, F_T_TCP, tmp_path):
    rig = tmp_path / "rig.yaml"
    rig.write_text(REAL_RIG.read_text())
    T_cam_tcp = invert(F_T_cam) @ F_T_TCP
    RigTcp.write(rig, T_cam_tcp, REAL_GRIPPER)
    np.testing.assert_allclose(RigTcp.read(rig), T_cam_tcp, atol=1e-9)


def test_euler_form_reads_as_scipy_from_euler(F_T_cam, F_T_TCP, tmp_path):
    T_cam_tcp = invert(F_T_cam) @ F_T_TCP
    euler = R.from_matrix(T_cam_tcp[:3, :3]).as_euler("xyz", degrees=True)
    rig = write_yaml(tmp_path / "rig.yaml", {"tcp": {"T_cam_tcp": {
        "rotation": {"euler_xyz_deg": euler.tolist()}, "translation_m": T_cam_tcp[:3, 3].tolist()}}})
    expected = T(R.from_euler("xyz", euler, degrees=True).as_matrix(), T_cam_tcp[:3, 3])
    np.testing.assert_allclose(RigTcp.read(rig), expected, atol=1e-12)
    np.testing.assert_allclose(RigTcp.read(rig), T_cam_tcp, atol=1e-12)


def test_fingertips_in_front_of_and_below_the_camera(F_T_cam, F_T_TCP):
    T_cam_tcp = invert(F_T_cam) @ F_T_TCP
    assert T_cam_tcp[2, 3] > 0 and T_cam_tcp[1, 3] > 0


def test_training_and_robot_agree_for_1000_arm_poses(F_T_cam, F_T_TCP):
    T_cam_tcp = invert(F_T_cam) @ F_T_TCP
    rng = np.random.default_rng(0)
    for _ in range(1000):
        O_T_F = T(R.random(random_state=rng).as_matrix(), rng.uniform(-1, 1, 3))
        training = (O_T_F @ F_T_cam) @ T_cam_tcp           # dataproc: SLAM camera pose x rig T_cam_tcp
        robot = O_T_F @ F_T_TCP                            # franka_control: flange x F_T_TCP
        np.testing.assert_allclose(training, robot, atol=1e-9)


def test_policy_motion_moves_fingertips_identically(F_T_cam, F_T_TCP):
    T_cam_tcp = invert(F_T_cam) @ F_T_TCP
    rng = np.random.default_rng(1)
    dT = T(R.from_euler("xyz", [5, -10, 20], degrees=True).as_matrix(), [0.01, -0.02, 0.005])
    O_T_F = T(R.random(random_state=rng).as_matrix(), [0.5, 0, 0.3])
    tcp_next = (O_T_F @ F_T_TCP) @ dT                      # what message_comms sends (absolute TCP target)
    flange_next = tcp_next @ invert(F_T_TCP)               # what franka_control does internally
    np.testing.assert_allclose(flange_next @ F_T_cam @ T_cam_tcp, tcp_next, atol=1e-9)


def test_camera_pose_matches_the_reference_formula():
    """The reference's camera_pose, copied: the same formula, now applied in the hand frame."""
    pos, up, tilt = np.array([0.0, 0.07, 0.04]), np.array([0.0, 1.0, 0.0]), 30.0
    a = np.radians(tilt)
    z_c = np.cos(a) * np.array([0.0, 0.0, 1.0]) - np.sin(a) * up
    y_c = -(up - (up @ z_c) * z_c)
    y_c /= np.linalg.norm(y_c)
    reference = T(np.column_stack([np.cross(y_c, z_c), y_c, z_c]), pos)
    np.testing.assert_allclose(camera_pose(pos, up, tilt), reference, atol=1e-15)


# --- camera_pose in the hand frame, and F_T_H ---

UP_SIDES = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0]}


@pytest.mark.parametrize("side", UP_SIDES)
@pytest.mark.parametrize("tilt", [0.0, 15.0, 45.0, 89.0])
def test_camera_pose_is_a_right_handed_frame(side, tilt):
    up = np.array(UP_SIDES[side], dtype=float)
    H_T_cam = camera_pose(np.array([0.01, 0.02, 0.03]), up, tilt)
    Rm = H_T_cam[:3, :3]
    np.testing.assert_allclose(Rm.T @ Rm, np.eye(3), atol=1e-12)
    assert np.linalg.det(Rm) == pytest.approx(1.0)
    x, y, z = Rm.T
    np.testing.assert_allclose(np.cross(x, y), z, atol=1e-12)
    assert z @ up == pytest.approx(-np.sin(np.radians(tilt)))            # tilted away from up_side
    assert z @ [0, 0, 1] == pytest.approx(np.cos(np.radians(tilt)))
    assert y @ up < 0                                                     # image y (down) points away from up_side
    np.testing.assert_allclose(H_T_cam[:3, 3], [0.01, 0.02, 0.03])


@pytest.mark.parametrize("side", UP_SIDES)
def test_tilt_zero_looks_along_hand_z(side):
    up = np.array(UP_SIDES[side], dtype=float)
    Rm = camera_pose(np.zeros(3), up, 0.0)[:3, :3]
    np.testing.assert_allclose(Rm[:, 2], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(Rm[:, 1], -up, atol=1e-12)


def test_F_T_H_is_minus_45_about_z():
    np.testing.assert_allclose(F_T_H[:3, :3], R.from_euler("z", -45, degrees=True).as_matrix(), atol=1e-15)
    np.testing.assert_allclose(F_T_H[:3, 3], 0.0)
    np.testing.assert_allclose(F_T_H[:3, :3] @ [1, 0, 0], [np.sqrt(0.5), -np.sqrt(0.5), 0], atol=1e-15)


def test_triangle_F_T_cam_is_F_T_H_times_H_T_cam(fake_root):
    triangle = TcpTriangle(Station.load(fake_root / "station.yaml", False), fake_root / "gripper.yaml")
    gripper = triangle.gripper
    np.testing.assert_allclose(triangle.F_T_cam,
                               F_T_H @ camera_pose(gripper.position_m, gripper.up_axis, gripper.tilt_deg), atol=0)


def test_pose_error():
    A = T(np.eye(3), [0, 0, 0])
    B = T(R.from_euler("x", 4.0, degrees=True).as_matrix(), [0.0123, 0, 0])
    mm, deg = pose_error(A, B)
    assert mm == pytest.approx(12.3) and deg == pytest.approx(4.0)
    assert pose_error(B, B) == pytest.approx((0.0, 0.0), abs=1e-6)


# --- the real numbers ---

def test_real_numbers_reproduce_the_standin_rig_file(real_F_T_cam):
    F_T_TCP = np.array(F_T_TCP_LIST)
    T_cam_tcp = invert(real_F_T_cam) @ F_T_TCP
    np.testing.assert_allclose(T_cam_tcp, RigTcp.read(REAL_RIG), atol=1e-6)
    np.testing.assert_allclose(T_cam_tcp[:3, 3], [0.0, 0.016891, 0.149465], atol=1e-6)
    # a 15 deg tilt toward the fingers: the TCP's z (approach) axis is 15 deg from the camera's viewing axis
    assert np.degrees(np.arccos(T_cam_tcp[:3, 2] @ [0, 0, 1])) == pytest.approx(15.0)
    assert np.linalg.norm(T_cam_tcp[:3, 3]) == pytest.approx(0.1504, abs=1e-4)


# --- RigTcp.write ---

def test_write_keeps_every_other_line_and_adds_header(tmp_path):
    rig = tmp_path / "rig.yaml"
    rig.write_text(REAL_RIG.read_text())
    before = REAL_RIG.read_text().splitlines()
    RigTcp.write(rig, np.eye(4), REAL_GRIPPER)
    after = rig.read_text().splitlines()
    start = next(i for i, line in enumerate(before) if line.startswith("tcp:"))
    end = next(i for i in range(start + 1, len(before)) if before[i] and not before[i][0].isspace())
    new_start = after.index("tcp:") - 2
    assert after[:new_start] == before[:start]
    assert after[new_start + 6:] == before[end:]
    assert after[new_start].startswith("# T_cam_tcp COMPUTED by `python -m deploy.geometry` on ")
    assert after[new_start].endswith(f"from {REAL_GRIPPER}")
    assert after[new_start + 1] == ("# and franka_control tcp.F_T_TCP. Do not edit by hand: change gripper.yaml "
                                    "and rerun the tool.")
    assert after[new_start + 5] == "  provenance: nominal"
    data = yaml.safe_load(rig.read_text())
    assert data["tcp"]["provenance"] == "nominal"
    assert data["imu"] == yaml.safe_load(REAL_RIG.read_text())["imu"]


def test_write_twice_keeps_one_header(tmp_path, real_F_T_cam):
    rig = tmp_path / "rig.yaml"
    rig.write_text(REAL_RIG.read_text())
    RigTcp.write(rig, np.eye(4), REAL_GRIPPER)
    once = rig.read_text()
    RigTcp.write(rig, np.eye(4), REAL_GRIPPER)
    assert rig.read_text() == once
    assert once.count("COMPUTED by `python -m deploy.geometry`") == 1


def test_write_refuses_a_file_without_tcp_block(tmp_path):
    from deploy import GeometryError
    rig = tmp_path / "rig.yaml"
    rig.write_text("schema: x\n")
    with pytest.raises(GeometryError, match="expected one top-level `tcp:` block"):
        RigTcp.write(rig, np.eye(4), REAL_GRIPPER)


# --- the tool's command line ---

def test_tool_writes_the_rig_file(fake_root, capsys):
    rig = fake_root / "process_frames" / "rigs" / "rig_pi_v1_standin.yaml"
    RigTcp.write(rig, np.eye(4), fake_root / "gripper.yaml")
    assert geometry_main([]) == 0
    np.testing.assert_allclose(RigTcp.read(rig), RigTcp.read(REAL_RIG), atol=1e-9)
    out = capsys.readouterr().out
    assert "camera -> fingertips: 150.4 mm" in out and f"wrote tcp.T_cam_tcp to {rig}" in out
    assert f"from {fake_root / 'gripper.yaml'}" in rig.read_text()


def test_tool_dry_run_writes_nothing(fake_root, capsys):
    rig = fake_root / "process_frames" / "rigs" / "rig_pi_v1_standin.yaml"
    RigTcp.write(rig, np.eye(4), fake_root / "gripper.yaml")
    before = rig.read_bytes()
    assert geometry_main(["--station", str(fake_root / "station.yaml"), "--gripper", str(fake_root / "gripper.yaml"),
                          "--dry-run"]) == 0
    assert rig.read_bytes() == before
    assert "dry run" in capsys.readouterr().out


def test_tool_refuses_when_fingertips_behind_or_above(fake_root, capsys):
    gripper = fake_root / "gripper.yaml"
    gripper.write_text(gripper.read_text().replace("up_side: +x", "up_side: -x"))
    rig = fake_root / "process_frames" / "rigs" / "rig_pi_v1_standin.yaml"
    before = rig.read_bytes()
    assert geometry_main([]) == 1
    assert rig.read_bytes() == before
    assert "refusing to write" in capsys.readouterr().err


def test_tool_reports_a_broken_gripper_file(fake_root, capsys):
    (fake_root / "gripper.yaml").write_text("ultrawide: {tilt_deg: 15}\n")
    assert geometry_main([]) == 1
    assert "ultrawide.position_mm: required" in capsys.readouterr().err
