"""pc_probe --start-pose: the arm's pose as the PC's start pose (q_start_rad) and the demos' start (from
rnoro/final-deployment 927a5ba, policy_deploy/tests/manager/test_pc_probe.py)."""
from __future__ import annotations

import io
import math
from types import SimpleNamespace

import numpy as np
import pytest
from launcher_helpers import make_fake_pc
from station_helpers import write_station
from teleop_helpers import FAST

from teleop_collect.contracts import Station
from teleop_collect.teleop import pc_probe
from teleop_collect.teleop.pc_probe import PcProbe, start_pose_lines
from teleop_collect.teleop.settings import Settings


@pytest.fixture
def probe_for(tmp_path):
    made = []

    def make(**fake_overrides):
        fake = make_fake_pc(tmp_path / "fake", **fake_overrides)
        fake.start()
        station = Station.load(write_station(tmp_path, sections={"teleop": FAST}), mock=True)
        settings = Settings.from_station(station, state_endpoint=fake.state_endpoint,
                                         commands_endpoint=fake.commands_endpoint)
        probe = PcProbe(settings, io.StringIO())
        made.append((fake, probe))
        return fake, probe
    yield make
    for fake, probe in made:
        probe.close()
        fake.close()


def _tilted_forward(deg: float, z_m: float) -> list[list[float]]:
    """The fake PC's pointing-down pose turned about base y: the fingertips (TCP z) tilt `deg` toward forward (+x)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, 0.0, s, 0.45], [0.0, -1.0, 0.0, 0.0], [s, 0.0, -c, z_m], [0.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("pose, width_m, height, direction", [
    (None, 0.04, "z 0.300 m", "base (+0.00, +0.00, -1.00): tilted 0.0 deg from straight down, no horizontal direction"),
    (_tilted_forward(48.0, 0.16), 0.08, "z 0.160 m",
     "base (+0.74, +0.00, -0.67): tilted 48.0 deg from straight down, pointing forward (away from the base)"),
])
def test_start_pose_lines_from_the_fake_pcs_c2(probe_for, pose, width_m, height, direction):
    fake, probe = probe_for(start_width_m=width_m, **({"start_pose": pose} if pose is not None else {}))
    rows = probe.run(0.3, report_every_s=None)
    assert {r.name: r for r in rows}["hello"].status == "pass"
    assert set(fake.request_names()) <= {"c3.hello/v1", "c3.heartbeat/v1"}       # read-only
    lines = start_pose_lines(probe.history.latest())
    assert lines == [
        "q_start_rad: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # paste under franka_control: homing: "
        "in the PC's control_deploy/station.yaml",
        f"fingertip (TCP) height: {height} in the robot base frame",
        f"fingertip direction (TCP z axis) -> {direction}",
        f"gripper width: {width_m * 1000:.1f} mm",
        "demos start: fingertips tilted ~48 deg from straight down (38-64), ~16 cm above the table (6-27), hand open; "
        "object ~16 cm ahead horizontally and ~17 cm below the fingertips"]


@pytest.mark.parametrize("zx, zy, words", [
    (0.0, 0.7, "pointing to the robot's left"), (0.0, -0.7, "pointing to the robot's right"),
    (-0.7, 0.0, "pointing back (toward the base)"),
    (0.5, 0.5, "pointing between forward (away from the base) and to the robot's left"),
])
def test_start_pose_words_and_q(zx, zy, words):
    z = np.array([zx, zy, -math.sqrt(1 - zx * zx - zy * zy)])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[2, 3] = x, np.cross(z, x), z, 0.2
    q = [0.1, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816]
    lines = start_pose_lines(SimpleNamespace(q=np.array(q), O_T_TCP=T, gripper_width_m=0.0))
    assert lines[0].startswith("q_start_rad: [0.1000, -0.7854, 0.0000, -2.3562, 0.0000, 1.5708, 0.7854]   # paste")
    assert lines[2].endswith(words) and f"tilted {math.degrees(math.acos(-z[2])):.1f} deg" in lines[2]
    assert lines[3] == "gripper width: 0.0 mm"


def test_cli_has_start_pose_option(capsys):
    with pytest.raises(SystemExit) as e:
        pc_probe.main(["--help"])
    assert e.value.code == 0
    assert "--start-pose" in capsys.readouterr().out
