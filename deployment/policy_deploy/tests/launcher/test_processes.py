# adapted from rnoro/deployment:deployment/policy/tests/test_launch.py, process_frames/tests/test_launch.py and
# message_comms/tests/test_launch.py @ 91b178b
# changes: the four process classes of policy_deploy.launcher.processes (no launch.py files); the check subprocesses
# are answered by a fake interpreter (launcher_fakes.write_fake_python); python.torch for diffusion; the manager's
# heartbeat/homing rules are its self_check's (not checked here); the fake_pc process is new.
"""The process classes as the launcher uses them: program(), USES_TERMINAL, the interpreter, checks()."""
from __future__ import annotations

import json
import socket
import sys

import pytest
from launcher_fakes import check_line, write_fake_python

from policy_deploy.contracts import Station
from policy_deploy.launcher.processes import (PROCESS_CLASSES, FakePcProcess, FramesProcess, ManagerProcess,
                                              PolicyProcess)

RIG_OK = json.dumps({"ok": True, "rig_id": "rig_pi_v3_640", "cameras": {
    "main": {"sensor": "imx519", "width": 640, "height": 480, "fps": 30.0},
    "ultrawide": {"sensor": "imx219", "width": 640, "height": 480, "fps": 30.0}}}) + "\n"
POLICY_OK = check_line("settings", "pass", "policy section valid") + "\n" + \
    check_line("device", "info", "device: cpu") + "\n"
MANAGER_OK = check_line("settings", "pass", "values in range") + "\n" + \
    check_line("worst gap", "pass", "0.57 s < watchdog 1.0 s") + "\n"
PROBE_OK = check_line("C2", "pass", "100 Hz") + "\n" + check_line("hello", "pass", "rtt 2 ms") + "\n"


def answers(**change):
    a = {"policy_deploy.frames.rig_file": (RIG_OK, 0), "policy_deploy.policy.self_check": (POLICY_OK, 0),
         "policy_deploy.manager.self_check": (MANAGER_OK, 0), "policy_deploy.manager.pc_probe": (PROBE_OK, 0)}
    a.update(change)
    return a


@pytest.fixture
def fake_python(tmp_path):
    """fake_python(**answers) -> path of a fake interpreter; its calls are logged to tmp_path/calls.jsonl."""
    def make(name="py", **change):
        return str(write_fake_python(tmp_path / name, answers(**change), tmp_path / "calls.jsonl"))
    return make


def calls(tmp_path):
    path = tmp_path / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def by_name(results):
    return {r.name: r for r in results}


def load(make_station, mock, **sections):
    return Station.load(make_station(**sections), mock)


# ---------------------------------------------------------------------------------------------------------- all


def test_names_and_terminal():
    assert {n: c.NAME for n, c in PROCESS_CLASSES.items()} == {n: n for n in PROCESS_CLASSES}
    assert [n for n, c in PROCESS_CLASSES.items() if c.USES_TERMINAL] == ["manager"]


@pytest.mark.parametrize("name", ["fake_pc", "frames", "policy", "manager"])
def test_program_is_python_dash_m(make_station, name):
    station = load(make_station, True)
    proc = PROCESS_CLASSES[name](station, True, None)
    assert proc.program() == [sys.executable, "-m", f"policy_deploy.{name}"]
    assert proc.command()[-1] == "--mock" and "--station" in proc.command()


# ------------------------------------------------------------------------------------------------------- policy


def test_policy_python_is_torch_only_for_the_kind_that_runs(make_station, tmp_path):
    torch = str(write_fake_python(tmp_path / "torch_py", {}))
    for mock, policy, expected in [
            (True, {"kind": "hold_still"}, sys.executable),
            (True, {"kind": "diffusion"}, sys.executable),                     # --mock runs mock_kind (hold_still)
            (True, {"kind": "hold_still", "mock_kind": "diffusion"}, torch),
            (False, {"kind": "hold_still"}, sys.executable),
            (False, {"kind": "diffusion"}, torch)]:
        station = load(make_station, mock, python={"light": sys.executable, "torch": torch}, policy=policy)
        assert PolicyProcess(station, mock, None).program()[0] == expected, (mock, policy)


def test_hold_still_real_run_passes(make_station, fake_python, tmp_path):
    station = load(make_station, False, python={"light": fake_python()}, policy={"kind": "hold_still"})
    rows = by_name(PolicyProcess(station, False, None).checks())
    assert [r for r in rows.values() if r.status == "fail"] == []
    assert "kind" not in rows and rows["settled"].status == "pass" and rows["C4 port"].status == "pass"
    assert rows["settings"].detail == "policy section valid"                    # from the self-check lines
    [cmd] = [c for c in calls(tmp_path) if "policy_deploy.policy.self_check" in c]
    assert cmd[-1] == "--json" and "--mock" not in cmd and "--station" in cmd


def test_diffusion_on_the_real_arm_refused_while_unsettled(make_station, fake_python, tmp_path):
    torch = fake_python("torch_py")
    station = load(make_station, False, python={"light": sys.executable, "torch": torch},
                   policy={"kind": "diffusion", "checkpoint": "/x.ckpt"})
    rows = by_name(PolicyProcess(station, False, None).checks())
    assert rows["settled"].status == "fail" and "settled is false" in rows["settled"].detail
    assert rows["torch env"].status == "pass"
    assert any("policy_deploy.policy.self_check" in c and c[0] != "-c" for c in calls(tmp_path))


def test_diffusion_with_a_missing_torch_env(make_station):
    station = load(make_station, False, python={"light": sys.executable, "torch": "/no/such/torch/python"},
                   policy={"kind": "diffusion"})
    proc = PolicyProcess(station, False, None)
    assert by_name(proc.base_checks())["program"].status == "fail"
    rows = by_name(proc.checks())
    assert rows["torch env"].status == "fail" and "/no/such/torch/python" in rows["torch env"].detail
    assert rows["self_check"].status == "fail"


def test_unset_kind_fails_and_skips_the_self_check(make_station, fake_python, tmp_path):
    station = load(make_station, False, python={"light": fake_python()}, policy={})
    rows = by_name(PolicyProcess(station, False, None).checks())
    assert rows["kind"].status == "fail" and "policy.kind is not set" in rows["kind"].detail
    assert "settled" not in rows and "settings" not in rows
    assert calls(tmp_path) == []


def test_mock_policy_needs_no_kind_settled_or_torch(make_station, fake_python, tmp_path):
    station = load(make_station, True, python={"light": fake_python()}, policy={"kind": None})
    rows = by_name(PolicyProcess(station, True, None).checks())
    assert [r for r in rows.values() if r.status == "fail"] == []
    assert "settled" not in rows and "kind" not in rows
    [cmd] = [c for c in calls(tmp_path) if "policy_deploy.policy.self_check" in c]
    assert "--mock" in cmd


def test_policy_self_check_crash_is_one_fail(make_station, fake_python):
    py = fake_python(**{"policy_deploy.policy.self_check": ("", 1)})
    station = load(make_station, True, python={"light": py})
    fails = [r for r in PolicyProcess(station, True, None).checks() if r.status == "fail"]
    assert [r.name for r in fails] == ["self_check"]


def test_policy_busy_port_fails(make_station, fake_python):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        station = load(make_station, True, python={"light": fake_python()},
                       ports={"frames": port + 1 if port < 65000 else 20001, "policy": port,
                              "mock_pi": port + 2 if port < 65000 else 20002})
        fails = [r for r in PolicyProcess(station, True, None).checks() if r.status == "fail"]
    assert [r.name for r in fails] == ["C4 port"]


# ------------------------------------------------------------------------------------------------------- frames


def test_frames_mock_rows(make_station, fake_python, tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "meta.json").write_text("{}")
    station = load(make_station, True, python={"light": fake_python()}, frames={"mock_episode": str(episode)})
    rows = by_name(FramesProcess(station, True, None).checks())
    assert [r for r in rows.values() if r.status == "fail"] == []
    assert set(rows) == {"rig file", "imports", "C1 port", "mock episode", "fake Pi port"}
    assert rows["rig file"].detail.startswith("rig_pi_v3_640: main imx519 640x480@30")
    [cmd] = [c for c in calls(tmp_path) if "policy_deploy.frames.rig_file" in c]
    assert cmd[-1] == "--json" and cmd[-2].endswith("contracts/rigs/rig_pi_v3_640.yaml")


def test_frames_mock_episode_missing(make_station, fake_python, tmp_path):
    station = load(make_station, True, python={"light": fake_python()}, frames={"mock_episode": str(tmp_path)})
    rows = by_name(FramesProcess(station, True, None).checks())
    assert rows["mock episode"].status == "fail" and "no meta.json" in rows["mock episode"].detail


def test_frames_real_pi_unreachable(make_station, fake_python):
    with socket.socket() as s:                        # a port nobody listens on (closed right after)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    station = load(make_station, False, python={"light": fake_python()}, pi={"address": "127.0.0.1", "port": port})
    rows = by_name(FramesProcess(station, False, None).checks())
    assert rows["Pi reachable"].status == "fail"
    assert rows["Pi reachable"].detail == (f"no TCP connection to 127.0.0.1:{port}: is the Pi on ASUS_5g and its "
                                           f"sender running? (station.yaml pi.address)")
    assert "mock episode" not in rows


def test_frames_real_pi_reachable(make_station, fake_python):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        station = load(make_station, False, python={"light": fake_python()},
                       pi={"address": "127.0.0.1", "port": port})
        rows = by_name(FramesProcess(station, False, None).checks())
    assert rows["Pi reachable"].status == "pass"


def test_frames_bad_rig_and_missing_imports(make_station, tmp_path):
    py = write_fake_python(tmp_path / "py", answers(**{"policy_deploy.frames.rig_file":
                                                       ('{"ok": false, "error": "cameras.main: missing"}\n', 1)}))
    station = load(make_station, True, python={"light": str(py)})
    rows = by_name(FramesProcess(station, True, None).checks())
    assert rows["rig file"].status == "fail" and rows["rig file"].detail == "cameras.main: missing"
    assert FramesProcess(station, True, None).ready_hint().startswith("frames from both cameras; check the fake Pi")
    assert FramesProcess(station, False, None).ready_hint() == FramesProcess.READY_HINT


# ------------------------------------------------------------------------------------------------------ manager


def test_manager_mock_rows_with_a_script(make_station, fake_python, tmp_path):
    (tmp_path / "script.yaml").write_text("- {key: q}\n")
    station = load(make_station, True, python={"light": fake_python()},
                   manager={"operator_script": str(tmp_path / "script.yaml")})
    rows = by_name(ManagerProcess(station, True, None).checks())
    assert [r for r in rows.values() if r.status == "fail"] == []
    assert rows["operator script"].status == "pass" and rows["worst gap"].status == "pass"
    assert rows["PC"].status == "info" and rows["PC"].detail == "fake PC on 127.0.0.1, started by run"
    assert not any("policy_deploy.manager.pc_probe" in c for c in calls(tmp_path))     # never under --mock


def test_manager_mock_cross_test_row(make_station, fake_python, tmp_path):
    (tmp_path / "s.yaml").write_text("[]\n")
    station = load(make_station, True, python={"light": fake_python()},
                   manager={"operator_script": str(tmp_path / "s.yaml")})
    rows = by_name(ManagerProcess(station, True, None, fake_pc=False).checks())
    assert rows["PC"].detail == "PC on 127.0.0.1 started outside the launcher (cross test)"


def test_manager_real_run_asks_pc_probe(make_station, fake_python, tmp_path):
    (tmp_path / "s.yaml").write_text("[]\n")
    station = load(make_station, False, python={"light": fake_python()},
                   manager={"operator_script": str(tmp_path / "s.yaml")})
    rows = by_name(ManagerProcess(station, False, None).checks())
    assert rows["C2"].detail == "100 Hz" and rows["hello"].status == "pass"
    [cmd] = [c for c in calls(tmp_path) if "policy_deploy.manager.pc_probe" in c]
    assert cmd[-2:] == ["--check", "--json"] and "--mock" not in cmd


def test_manager_pc_probe_failure_is_a_pc_row(make_station, fake_python, tmp_path):
    (tmp_path / "s.yaml").write_text("[]\n")
    py = fake_python(**{"policy_deploy.manager.pc_probe": ("", 2)})
    station = load(make_station, False, python={"light": py}, manager={"operator_script": str(tmp_path / "s.yaml")})
    rows = by_name(ManagerProcess(station, False, None).checks())
    assert rows["PC"].status == "fail" and "control_deploy run" in rows["PC"].detail


def test_manager_no_terminal_without_a_script_fails(make_station, fake_python, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    station = load(make_station, True, python={"light": fake_python()})
    rows = by_name(ManagerProcess(station, True, None).checks())
    assert rows["terminal"].status == "fail" and "no terminal for the keyboard" in rows["terminal"].detail
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    assert by_name(ManagerProcess(station, True, None).checks())["terminal"].status == "pass"


def test_manager_missing_script_fails(make_station, fake_python, tmp_path):
    station = load(make_station, True, python={"light": fake_python()},
                   manager={"operator_script": str(tmp_path / "nope.yaml")})
    assert by_name(ManagerProcess(station, True, None).checks())["operator script"].status == "fail"


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
