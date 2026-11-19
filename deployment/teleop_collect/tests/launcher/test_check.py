# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_check.py @ 2ffe641
# changes: the teleop_collect ORDER (fake_pc, frames, teleop, ui) and its row names; messages teleop_collect check:;
# imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/deploy/tests/test_check.py @ 91b178b
# changes: the station rows replace the TCP triangle (stubbed here); laptop process names; no launch.py loading.
from __future__ import annotations

import pytest
from launcher_helpers import make_process

from teleop_collect.launcher.check_result import CheckResult
from teleop_collect.launcher.launcher import Launcher

ORDER = ["fake_pc", "frames", "teleop", "ui"]


@pytest.fixture(autouse=True)
def one_station_row(monkeypatch):
    monkeypatch.setattr(Launcher, "station_checks",
                        lambda self: [CheckResult("station", "deploy_contracts", "fail", "from another checkout")])


def test_every_failure_reported_grouped_by_process(station, capsys):
    class Crashes(make_process("frames", "serve")):
        def checks(self):
            raise RuntimeError("rig check blew up")

    processes = [
        make_process("ui", "serve", checks=(CheckResult("ui", "HTTP port", "info", "8765"),))(station, False, None),
        Crashes(station, False, None),
        make_process("fake_pc", "serve", checks=(CheckResult("fake_pc", "C2 port", "fail", "busy"),
                                                 CheckResult("fake_pc", "C3 port", "fail", "busy"),
                                                 CheckResult("fake_pc", "x", "pass", "")))(station, False, None),
        make_process("teleop", "serve")(station, False, None),
    ]
    assert Launcher(station, False, processes).check() == 2
    out = capsys.readouterr().out
    processes_in_table = [line.split()[0] for line in out.splitlines()[2:-1]]
    assert processes_in_table == ["station"] + ["fake_pc"] * 5 + ["frames"] * 3 + ["teleop"] * 2 + ["ui"] * 3
    assert "from another checkout" in out and "busy" in out
    assert "crashed: RuntimeError: rig check blew up" in out
    assert out.splitlines()[-1] == "teleop_collect check: 4 check(s) failed"


def test_check_passes_with_only_warn_and_info(station, capsys, monkeypatch):
    monkeypatch.setattr(Launcher, "station_checks", lambda self: [])
    processes = [make_process(name, "serve", checks=(CheckResult(name, "backup", "warn", "stand-in"),))(
        station, False, None) for name in ORDER]
    assert Launcher(station, False, processes).check() == 0
    assert capsys.readouterr().out.splitlines()[-1] == "teleop_collect check: ready (4 warning(s))"


def test_unknown_process_name_refused(station):
    with pytest.raises(ValueError, match="unknown"):
        Launcher(station, False, [make_process("franka_control", "serve")(station, False, None)])
