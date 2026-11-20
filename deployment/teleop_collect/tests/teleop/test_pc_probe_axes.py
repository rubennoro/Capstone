"""pc_probe --axes: where the measured TCP's axes point, in words (from rnoro/final-deployment ea9f7fb, dc1ee36)."""
from __future__ import annotations

import numpy as np
import pytest

from teleop_collect.teleop import pc_probe
from teleop_collect.teleop.pc_probe import axes_lines


def test_identity_points_along_the_base_axes():
    lines = axes_lines(np.eye(4))
    assert lines == ["TCP x axis -> base (+1.00, +0.00, +0.00): forward (away from the base)",
                     "TCP y axis -> base (+0.00, +1.00, +0.00): to the robot's left",
                     "TCP z axis -> base (+0.00, +0.00, +1.00): up"]


def test_the_expected_home_check():
    """DIRECTIONS: at home with the settled TCP, x to the robot's right, y back, z down."""
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # columns: x, y, z in the base
    words = [line.split(": ", 1)[1] for line in axes_lines(T)]
    assert words == ["to the robot's right", "back (toward the base)", "down"]


@pytest.mark.parametrize("deg, expect", [(20.0, "forward (away from the base)"), (30.0, "between base axes"),
                                         (45.0, "between base axes")])
def test_words_only_within_25_deg_of_a_base_axis(deg, expect):
    a = np.radians(deg)
    T = np.eye(4)
    T[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    assert axes_lines(T)[0].endswith(expect)


def test_cli_has_axes_option(capsys):
    with pytest.raises(SystemExit) as e:
        pc_probe.main(["--help"])
    assert e.value.code == 0
    assert "--axes" in capsys.readouterr().out
