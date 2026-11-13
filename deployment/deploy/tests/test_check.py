from __future__ import annotations

import numpy as np
import pytest

from deploy import CheckResult, Launcher, Station
from deploy.geometry import RigTcp
from fakes import make_process
from helpers import station_data, write_yaml

ORDER = ["franka_control", "process_frames", "policy", "message_comms"]



@pytest.fixture
def station(fake_root) -> Station:
    data = station_data(policy={"chekpoint": "x.ckpt"})                  # a typo: policy's overrides fail
    return Station.load(write_yaml(fake_root / "s.yaml", data), False)


def test_every_failure_reported_grouped_by_process(fake_root, station, capsys):
    rig = fake_root / "process_frames" / "rigs" / "rig_pi_v2.yaml"
    RigTcp.write(rig, np.eye(4), fake_root / "gripper.yaml")          # a rig file that disagrees

    class Crashes(make_process("process_frames", "serve")):
        def checks(self):
            raise RuntimeError("self-check blew up")

    processes = [
        make_process("franka_control", "serve", checks=(CheckResult("franka_control", "arm", "fail", "no arm"),
                                                        CheckResult("franka_control", "rt", "fail", "no RT"),
                                                        CheckResult("franka_control", "ports", "pass", "")))(
            station, False, None),
        Crashes(station, False, None),
        make_process("policy", "serve")(station, False, None),
        make_process("message_comms", "serve", checks=(CheckResult("message_comms", "exec_h", "info", "8"),))(
            station, False, None),
    ]
    assert Launcher(station, False, processes).check() == 2
    out = capsys.readouterr().out
    processes_in_table = [line.split()[0] for line in out.splitlines()[2:-1]]
    assert processes_in_table == ["geometry"] + ["franka_control"] * 5 + ["process_frames"] * 3 + ["policy"] + \
        ["message_comms"] * 3
    assert "TCP triangle off by" in out and "run python -m deploy.geometry" in out
    assert "no arm" in out and "no RT" in out
    assert "policy.chekpoint: not a key of policy/config.yaml" in out
    assert "crashed: RuntimeError: self-check blew up" in out
    assert out.splitlines()[-1] == "deploy check: 5 check(s) failed"


def test_check_passes_with_only_warn_and_info(fake_root, capsys):
    station = Station.load(fake_root / "station.yaml", False)
    processes = [make_process(name, "serve", checks=(CheckResult(name, "backup", "warn", "stand-in"),))(
        station, False, None) for name in ORDER]
    assert Launcher(station, False, processes).check() == 0
    assert capsys.readouterr().out.splitlines()[-1] == "deploy check: ready (4 warning(s))"


def test_missing_launch_files_reported_and_others_loaded(fake_root, capsys):
    (fake_root / "policy" / "launch.py").write_text(
        "from deploy import ManagedProcess\n"
        "import sys\n\n"
        "class PolicyProcess(ManagedProcess):\n"
        "    NAME = 'policy'\n"
        "    READY_HINT = 'warm-up'\n"
        "    def program(self):\n"
        "        return [sys.executable, '-m', 'policy']\n"
        "    def checks(self):\n"
        "        return []\n\n"
        "PROCESS_CLASS = PolicyProcess\n")
    station = Station.load(fake_root / "station.yaml", False)
    launcher = Launcher(station, False)
    assert [p.NAME for p in launcher.processes] == ["policy"]
    assert launcher.check() == 2
    out = capsys.readouterr().out
    for name in ("franka_control", "process_frames", "message_comms"):
        assert f"{name}: no launch.py at" in out
    assert "policy          overrides" in out
