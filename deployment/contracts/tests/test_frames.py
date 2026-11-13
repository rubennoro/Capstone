"""frames.yaml and TcpFrame (STRUCTURE.md 4.2, 4.4)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from deploy_contracts import TcpFrame, frames_file

R = 0.7071067811865476
GOOD = [[R, R, 0.0, 0.0], [-R, R, 0.0, 0.0], [0.0, 0.0, 1.0, 0.150], [0.0, 0.0, 0.0, 1.0]]


def test_loads_the_real_file():
    frame = TcpFrame.load()
    assert frame.F_T_TCP.dtype == np.float64 and frame.F_T_TCP.shape == (4, 4)
    assert np.array_equal(frame.F_T_TCP, np.array(GOOD))
    assert frame.settled is False
    assert TcpFrame.load() is frame  # cached
    assert list(yaml.safe_load(frames_file().read_text())) == ["F_T_TCP", "settled"]


def test_result_is_read_only():
    frame = TcpFrame.load()
    with pytest.raises(ValueError):
        frame.F_T_TCP[0, 3] = 1.0
    assert not TcpFrame.load(frames_file()).F_T_TCP.flags.writeable


def _write(tmp_path, matrix=None, settled=False, **extra):
    d = {"F_T_TCP": GOOD if matrix is None else matrix, "settled": settled, **extra}
    p = tmp_path / "frames.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_a_written_good_file(tmp_path):
    frame = TcpFrame.load(_write(tmp_path, settled=True))
    assert frame.settled is True and np.array_equal(frame.F_T_TCP, np.array(GOOD))
    ints = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert np.array_equal(TcpFrame.load(_write(tmp_path, ints)).F_T_TCP, np.eye(4))  # integers are numbers


@pytest.mark.parametrize("matrix", [
    GOOD[:3],                                   # 3 rows
    [row[:3] for row in GOOD],                  # 3 columns
    GOOD + [[0.0, 0.0, 0.0, 1.0]],              # 5 rows
    "identity",
    [[R, R, 0.0, 0.0], [-R, R, 0.0, 0.0], [0.0, 0.0, 1.0, 0.15], 4],
])
def test_not_4x4_refused(tmp_path, matrix):
    with pytest.raises(ValueError, match="F_T_TCP"):
        TcpFrame.load(_write(tmp_path, matrix))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.15", True, None])
def test_non_finite_or_not_a_number_refused(tmp_path, value):
    m = [list(r) for r in GOOD]
    m[2][3] = value
    with pytest.raises(ValueError, match=r"F_T_TCP\[2\]\[3\]"):
        TcpFrame.load(_write(tmp_path, m))


def test_transposed_refused(tmp_path):
    with pytest.raises(ValueError, match="bottom row"):
        TcpFrame.load(_write(tmp_path, np.array(GOOD).T.tolist()))


def test_reflection_refused(tmp_path):
    m = [list(r) for r in GOOD]
    m[2][2] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        TcpFrame.load(_write(tmp_path, m))


def test_not_orthonormal_refused(tmp_path):
    m = [list(r) for r in GOOD]
    m[0][0] = 0.71
    with pytest.raises(ValueError, match="orthonormal"):
        TcpFrame.load(_write(tmp_path, m))


def test_missing_unknown_and_settled_type(tmp_path):
    p = tmp_path / "frames.yaml"
    p.write_text(yaml.safe_dump({"F_T_TCP": GOOD}))
    with pytest.raises(ValueError, match="settled"):
        TcpFrame.load(p)
    p.write_text(yaml.safe_dump({"settled": False}))
    with pytest.raises(ValueError, match="F_T_TCP"):
        TcpFrame.load(p)
    with pytest.raises(ValueError, match="tcp_name"):
        TcpFrame.load(_write(tmp_path, tcp_name="hand"))
    for bad in (0, "false", None):
        with pytest.raises(ValueError, match="settled"):
            TcpFrame.load(_write(tmp_path, settled=bad))
    p.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match=str(p)):
        TcpFrame.load(p)
