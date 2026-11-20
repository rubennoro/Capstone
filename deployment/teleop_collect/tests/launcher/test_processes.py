# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/launcher/test_processes.py @ ea9f7fb
# changes: the four teleop_collect process classes (FakePcProcess, FramesProcess, TeleopProcess, UiProcess); the
# interpreter is swapped with launcher_helpers.use_python (no station python: section); frames has the R1 and
# preview ports (no C1 port, no imports row); teleop (was manager) has the U1 port and pc_probe, no terminal or
# self_check rows; ui (new) has the HTTP port; the policy tests are gone; imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/policy/tests/test_launch.py, process_frames/tests/test_launch.py and
# message_comms/tests/test_launch.py @ 91b178b
# changes: the four process classes of policy_deploy.launcher.processes (no launch.py files); the check subprocesses
# are answered by a fake interpreter (launcher_fakes.write_fake_python); python.torch for diffusion; the manager's
# heartbeat/homing rules are its self_check's (not checked here); the fake_pc process is new.
"""The process classes as the launcher uses them: program(), the interpreter, checks()."""
from __future__ import annotations

import json
import socket
import sys

import pytest
from launcher_helpers import check_line, use_python, write_fake_python

from teleop_collect.contracts import Station
from teleop_collect.launcher.processes import (PROCESS_CLASSES, FakePcProcess, FramesProcess, TeleopProcess,
                                               UiProcess)

RIG_OK = json.dumps({"ok": True, "rig_id": "rig_pi_v3_640", "cameras": {
    "main": {"sensor": "imx519", "width": 640, "height": 480, "fps": 30.0},
    "ultrawide": {"sensor": "imx219", "width": 640, "height": 480, "fps": 30.0}}}) + "\n"
PROBE_OK = check_line("C2", "pass", "100 Hz") + "\n" + check_line("hello", "pass", "rtt 2 ms") + "\n" + \
    check_line("commit", "warn", "PC commit differs") + "\n"


def answers(**change):
    a = {"teleop_collect.frames.rig_file": (RIG_OK, 0), "teleop_collect.teleop.pc_probe": (PROBE_OK, 0)}
    a.update(change)
    return a


@pytest.fixture
def fake_python(tmp_path, monkeypatch):
    """fake_python(**answers) -> path of a fake interpreter, used by every process; calls in tmp_path/calls.jsonl."""
    def make(name="py", **change):
        py = str(write_fake_python(tmp_path / name, answers(**change), tmp_path / "calls.jsonl"))
        use_python(monkeypatch, py)
        return py
    return make


def calls(tmp_path):
    path = tmp_path / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def by_name(results):
    return {r.name: r for r in results}


def load(make_station, mock, **sections):
    return Station.load(make_station(**sections), mock)


def busy_port():
    """A listening socket on 127.0.0.1 (the caller closes it) and its port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s, s.getsockname()[1]


# ---------------------------------------------------------------------------------------------------------- all


def test_names():
    assert {n: c.NAME for n, c in PROCESS_CLASSES.items()} == {n: n for n in PROCESS_CLASSES}
    assert list(PROCESS_CLASSES) == ["fake_pc", "frames", "teleop", "ui"]


@pytest.mark.parametrize("name", ["fake_pc", "frames", "teleop", "ui"])
def test_program_is_python_dash_m(make_station, name):
    station = load(make_station, True)
    proc = PROCESS_CLASSES[name](station, True, None)
    assert proc.program() == [sys.executable, "-m", f"teleop_collect.{name}"]
    assert proc.command()[-1] == "--mock" and "--station" in proc.command()


# ------------------------------------------------------------------------------------------------------- frames


def test_frames_mock_rows(make_station, fake_python, tmp_path):
    fake_python()
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "meta.json").write_text("{}")
    station = load(make_station, True, frames={"mock_episode": str(episode)})
    rows = by_name(FramesProcess(station, True, None).checks())
    assert [r for r in rows.values() if r.status == "fail"] == []
    assert set(rows) == {"rig file", "R1 port", "preview port", "mock episode", "fake Pi port"}
    assert rows["rig file"].detail.startswith("rig_pi_v3_640: main imx519 640x480@30")
    [cmd] = [c for c in calls(tmp_path) if "teleop_collect.frames.rig_file" in c]
    assert cmd[-1] == "--json" and cmd[-2].endswith("contracts/rigs/rig_pi_v3_640.yaml")


def test_frames_mock_episode_missing(make_station, fake_python, tmp_path):
    fake_python()
    station = load(make_station, True, frames={"mock_episode": str(tmp_path)})
    rows = by_name(FramesProcess(station, True, None).checks())
    assert rows["mock episode"].status == "fail" and "no meta.json" in rows["mock episode"].detail


def test_frames_real_pi_unreachable(make_station, fake_python):
    fake_python()
    with socket.socket() as s:                        # a port nobody listens on (closed right after)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    station = load(make_station, False, pi={"address": "127.0.0.1", "port": port})
    rows = by_name(FramesProcess(station, False, None).checks())
    assert rows["Pi reachable"].status == "fail"
    assert rows["Pi reachable"].detail == (f"no TCP connection to 127.0.0.1:{port}: is the Pi on ASUS_5g and its "
                                           f"sender running? (station.yaml pi.address)")
    assert "mock episode" not in rows and "fake Pi port" not in rows


def test_frames_real_pi_reachable(make_station, fake_python):
    fake_python()
    server, port = busy_port()
    with server:
        station = load(make_station, False, pi={"address": "127.0.0.1", "port": port})
        rows = by_name(FramesProcess(station, False, None).checks())
    assert rows["Pi reachable"].status == "pass"


def test_frames_bad_rig(make_station, fake_python):
    fake_python(**{"teleop_collect.frames.rig_file": ('{"ok": false, "error": "cameras.main: missing"}\n', 1)})
    station = load(make_station, True)
    rows = by_name(FramesProcess(station, True, None).checks())
    assert rows["rig file"].status == "fail" and rows["rig file"].detail == "cameras.main: missing"
    assert FramesProcess(station, True, None).ready_hint().startswith("frames from both cameras; check the fake Pi")
    assert FramesProcess(station, False, None).ready_hint() == FramesProcess.READY_HINT


@pytest.mark.parametrize("key,row", [("r1", "R1 port"), ("preview", "preview port"), ("mock_pi", "fake Pi port")])
def test_frames_busy_port_fails(make_station, fake_python, key, row):
    fake_python()
    server, port = busy_port()
    with server:
        station = load(make_station, True, ports={key: port})
        fails = [r for r in FramesProcess(station, True, None).checks() if r.status == "fail"]
    assert [r.name for r in fails if r.name != "mock episode"] == [row]


# ------------------------------------------------------------------------------------------------------- teleop


def test_teleop_mock_rows(make_station, fake_python, tmp_path):
    fake_python()
    station = load(make_station, True)
    rows = by_name(TeleopProcess(station, True, None).checks())
    assert [r for r in rows.values() if r.status == "fail"] == []
    assert set(rows) == {"U1 port", "PC"}
    assert rows["PC"].status == "info" and rows["PC"].detail == "fake PC on 127.0.0.1, started by run"
    assert not any("teleop_collect.teleop.pc_probe" in c for c in calls(tmp_path))     # never under --mock


def test_teleop_mock_cross_test_row(make_station, fake_python):
    fake_python()
    station = load(make_station, True)
    rows = by_name(TeleopProcess(station, True, None, fake_pc=False).checks())
    assert rows["PC"].detail == "PC on 127.0.0.1 started outside the launcher (cross test)"


def test_teleop_real_run_asks_pc_probe(make_station, fake_python, tmp_path):
    fake_python()
    station = load(make_station, False)
    rows = by_name(TeleopProcess(station, False, None).checks())
    assert rows["C2"].detail == "100 Hz" and rows["hello"].status == "pass"
    assert rows["commit"].status == "warn"                            # pc_probe's own rows, as it prints them
    [cmd] = [c for c in calls(tmp_path) if "teleop_collect.teleop.pc_probe" in c]
    assert cmd == ["-m", "teleop_collect.teleop.pc_probe", "--station", str(station.path), "--check", "--json"]


def test_teleop_pc_probe_failure_is_a_pc_row(make_station, fake_python):
    fake_python(**{"teleop_collect.teleop.pc_probe": ("", 2)})
    station = load(make_station, False)
    rows = by_name(TeleopProcess(station, False, None).checks())
    assert rows["PC"].status == "fail" and "control_deploy run" in rows["PC"].detail


def test_teleop_busy_u1_port(make_station, fake_python):
    fake_python()
    server, port = busy_port()
    with server:
        station = load(make_station, True, ports={"u1": port})
        rows = by_name(TeleopProcess(station, True, None).checks())
    assert rows["U1 port"].status == "fail" and str(port) in rows["U1 port"].detail


@pytest.mark.parametrize("on_start", [True, False])
def test_teleop_homes_on_start_from_its_config(make_station, on_start):
    station = load(make_station, False, teleop={"home": {"on_start": on_start}})
    assert TeleopProcess(station, False, None).homes_on_start() is on_start


# ------------------------------------------------------------------------------------------------------------ ui


def test_ui_http_port(make_station):
    station = load(make_station, True)
    rows = by_name(UiProcess(station, True, None).checks())
    assert rows["HTTP port"].status == "pass" and str(station.http_port) in rows["HTTP port"].detail
    server, port = busy_port()
    with server:
        station = load(make_station, True, ports={"http": port})
        assert by_name(UiProcess(station, True, None).checks())["HTTP port"].status == "fail"


# ------------------------------------------------------------------------------------------------------ fake_pc


def test_fake_pc_ports(make_station, link_ports):
    station = load(make_station, True)
    rows = by_name(FakePcProcess(station, True, None).checks())
    assert rows["C2 port"].status == "pass" and str(link_ports.state_port) in rows["C2 port"].detail
    with socket.socket() as s:
        s.bind(("127.0.0.1", link_ports.commands_port))
        s.listen()
        rows = by_name(FakePcProcess(station, True, None).checks())
    assert rows["C3 port"].status == "fail" and rows["C2 port"].status == "pass"
