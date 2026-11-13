from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from deploy import Station
from deploy.geometry import RigTcp, TcpTriangle, triangle_check
from helpers import REAL_ROOT, station_data, write_yaml

RIG = "process_frames/rigs/rig_pi_v2.yaml"


def offset(mm: float = 0.0, deg: float = 0.0) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_euler("x", deg, degrees=True).as_matrix()
    M[0, 3] = mm / 1000.0
    return M


@pytest.fixture
def station(fake_root) -> Station:
    return Station.load(fake_root / "station.yaml", False)


def write_rig(fake_root, T: np.ndarray) -> None:
    RigTcp.write(fake_root / RIG, T, fake_root / "gripper.yaml")


@pytest.mark.parametrize("mock", [False, True])
def test_v2_rig_passes_with_real_gripper_and_F_T_TCP(fake_root, station, mock):
    result = TcpTriangle(station, fake_root / "gripper.yaml").check(mock)
    assert (result.process, result.name, result.status) == ("geometry", "tcp triangle", "pass")


def test_consistent_files_pass_after_the_tool(fake_root, station):
    triangle = TcpTriangle(station, fake_root / "gripper.yaml")
    write_rig(fake_root, triangle.compute())
    assert triangle.check(False).status == "pass"


@pytest.mark.parametrize("mm,deg,fails", [(1.9, 0.0, False), (2.1, 0.0, True), (0.0, 0.45, False),
                                          (0.0, 0.55, True)])
@pytest.mark.parametrize("mock", [False, True])
def test_limits_2mm_and_half_degree(fake_root, station, mm, deg, fails, mock):
    triangle = TcpTriangle(station, fake_root / "gripper.yaml")
    write_rig(fake_root, triangle.compute() @ offset(mm, deg))
    result = triangle.check(mock)
    if not fails:
        assert result.status == "pass"
        return
    assert result.status == ("warn" if mock else "fail")
    assert result.detail.startswith("TCP triangle off by")
    for name in ("gripper.yaml", "franka_control/config.yaml (tcp.F_T_TCP)", RIG, "python -m deploy.geometry"):
        assert name in result.detail


@pytest.mark.parametrize("mock", [False, True])
def test_old_iphone_value_fails(fake_root, station, mock):
    iphone = np.eye(4)
    iphone[:3, :3] = R.from_euler("xyz", [15, 0, 0], degrees=True).as_matrix()
    iphone[:3, 3] = [0.0, 0.06, 0.20]
    write_rig(fake_root, iphone)
    result = triangle_check(station, fake_root / "gripper.yaml", mock)
    assert result.status == ("warn" if mock else "fail")
    assert "TCP triangle off by" in result.detail


@pytest.mark.parametrize("mock", [False, True])
def test_missing_gripper_file_fails_naming_it(fake_root, station, mock):
    (fake_root / "gripper.yaml").unlink()
    result = triangle_check(station, fake_root / "gripper.yaml", mock)
    assert result.status == ("warn" if mock else "fail")
    assert result.detail.startswith(f"gripper.yaml: not found: {fake_root / 'gripper.yaml'}")


@pytest.mark.parametrize("mock", [False, True])
def test_broken_gripper_and_missing_rig(fake_root, station, mock):
    (fake_root / RIG).unlink()
    result = triangle_check(station, fake_root / "gripper.yaml", mock)
    assert result.status == ("warn" if mock else "fail") and str(fake_root / RIG) in result.detail
    (fake_root / "gripper.yaml").write_text("ultrawide: {tilt_deg: 100}\n")
    result = triangle_check(station, fake_root / "gripper.yaml", mock)
    assert result.status == ("warn" if mock else "fail") and "ultrawide.tilt_deg" in result.detail


def test_bad_F_T_TCP_fails(fake_root, station):
    write_yaml(fake_root / "franka_control" / "config.yaml", {"tcp": {"F_T_TCP": [[1, 0], [0, 1]]}})
    result = triangle_check(station, fake_root / "gripper.yaml", False)
    assert result.status == "fail" and "tcp.F_T_TCP" in result.detail


def test_station_overrides_are_the_ones_used(fake_root, tmp_path):
    other_rig = tmp_path / "elsewhere" / "rig.yaml"
    other_rig.parent.mkdir()
    other_rig.write_text((fake_root / RIG).read_text())
    longer = np.eye(4)
    longer[:3, :3] = R.from_euler("z", -45, degrees=True).as_matrix()
    longer[2, 3] = 0.170                                    # 20 mm longer fingers
    data = station_data(process_frames={"rig_file": str(other_rig)},
                        franka_control={"tcp": {"F_T_TCP": longer.tolist()}})
    station = Station.load(write_yaml(fake_root / "s.yaml", data), False)
    triangle = TcpTriangle(station, fake_root / "gripper.yaml")
    assert triangle.rig_file == other_rig
    np.testing.assert_allclose(triangle.F_T_TCP, longer)
    result = triangle.check(False)
    assert result.status == "fail" and "off by 20.0 mm" in result.detail and str(other_rig) in result.detail
    RigTcp.write(other_rig, triangle.compute(), fake_root / "gripper.yaml")
    assert triangle.check(False).status == "pass"


def test_real_files_agree():
    """The real gripper.yaml, franka_control/config.yaml and rig file (once those configs exist)."""
    for name in ("franka_control", "process_frames"):
        if not (REAL_ROOT / name / "config.yaml").exists():
            pytest.skip(f"{name}/config.yaml does not exist yet")
    station = Station.load(REAL_ROOT / "station.yaml", True)
    triangle = TcpTriangle(station, REAL_ROOT / "gripper.yaml")
    np.testing.assert_allclose(triangle.F_T_TCP[:3, 3], [0, 0, 0.150], atol=1e-12)
    assert triangle.check(False).status == "pass"
