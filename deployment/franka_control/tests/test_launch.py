"""launch.py (deploy/STRUCTURE.md §9, franka_control/STRUCTURE.md §11): program(), checks(), the mock choice.

The checks run a stand-in binary (a small shell script) so they don't need a built franka_control; one test also
runs the real binary's --check-config when build/franka_control exists.
"""
from __future__ import annotations

import socket
import stat
import sys
from pathlib import Path

import pytest
import yaml

deploy = pytest.importorskip("deploy")

FOLDER = Path(__file__).resolve().parents[1]
REAL_BINARY = FOLDER / "build" / "franka_control"


@pytest.fixture(scope="module")
def process_class():
    return deploy.load_process_class(FOLDER)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_binary(path: Path, version: str = "0.9.2", problems: list[str] | None = None) -> Path:
    """A stand-in for the binary: --version prints `libfranka <version>`, --check-config prints the problems."""
    problems = problems or []
    lines = "\n".join(f"  echo '{p}'" for p in problems)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n"
                    f"if [ \"$1\" = \"--version\" ]; then echo 'franka_control abc1234 libfranka {version}'; exit 0; fi\n"
                    "if [ \"$1\" = \"--check-config\" ]; then\n"
                    f"{lines if lines else '  :'}\n"
                    f"  exit {2 if problems else 0}\n"
                    "fi\n"
                    "exit 1\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make(process_class, tmp_path: Path, mock: bool, franka_control: dict | None = None):
    endpoints = {name: f"tcp://127.0.0.1:{free_port()}" for name in
                 ("pi_frames", "pi_time_sync", "mock_pi_frames", "mock_pi_time_sync", "frames", "state", "commands",
                  "policy")}
    data = {"station_version": 1, "session_root": str(tmp_path / "sessions"),
            "python": {"light": sys.executable, "policy": sys.executable}, "endpoints": endpoints}
    if franka_control is not None:
        data["franka_control"] = franka_control
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump(data))
    return process_class(deploy.Station.load(station, mock=mock), mock, None)


def by_name(results):
    return {r.name: r for r in results}


def test_name_and_default_program(process_class, tmp_path):
    assert process_class.NAME == "franka_control"
    p = make(process_class, tmp_path, mock=False)
    assert p.program() == [str(deploy.DEPLOYMENT_ROOT / "franka_control" / "build" / "franka_control")]
    cmd = make(process_class, tmp_path, mock=True).command()
    assert cmd[1:3] == ["--station", str(tmp_path / "station.yaml")] and cmd[-1] == "--mock"


def test_a_station_set_binary_resolves_against_the_station_folder(process_class, tmp_path):
    fake_binary(tmp_path / "bin" / "fc")
    p = make(process_class, tmp_path, mock=False, franka_control={"binary": "bin/fc"})
    assert p.program() == [str(tmp_path / "bin" / "fc")]
    assert [r.status for r in p.base_checks()] == ["pass", "pass"]


def test_a_broken_station_section(process_class, tmp_path):
    p = make(process_class, tmp_path, mock=False, franka_control={"bogus": 1})
    assert p.base_checks()[0].status == "fail"
    assert "franka_control.bogus" in p.base_checks()[0].detail
    assert [r.status for r in p.checks()] == ["fail"]


def test_config_problems_reported_with_their_count(process_class, tmp_path):
    fake_binary(tmp_path / "fc", problems=["station.yaml: franka_control.safety.v_max_m_per_s: must be > 0",
                                           "station.yaml: franka_control.watchdog.limit_s: must be >= 0.5 s"])
    r = by_name(make(process_class, tmp_path, mock=True, franka_control={"binary": str(tmp_path / "fc")}).checks())
    assert r["config"].status == "fail"
    assert r["config"].detail == "station.yaml: franka_control.safety.v_max_m_per_s: must be > 0 (2 problems)"


@pytest.mark.skipif(not REAL_BINARY.exists(), reason="build/franka_control not built")
def test_the_real_binary_checks_the_merged_config(process_class, tmp_path):
    p = make(process_class, tmp_path, mock=True, franka_control={"safety": {"v_max_m_per_s": -1}})
    r = by_name(p.checks())
    assert r["config"].status == "fail"
    assert r["config"].detail == "station.yaml: franka_control.safety.v_max_m_per_s: must be > 0"
    assert r["config"].detail
    ok = by_name(make(process_class, tmp_path, mock=True).checks())
    assert ok["config"].status == "pass"


def test_mock_skips_the_arm_rt_kernel_and_rtprio_checks(process_class, tmp_path):
    fake_binary(tmp_path / "fc", version="none")
    r = by_name(make(process_class, tmp_path, mock=True, franka_control={"binary": str(tmp_path / "fc")}).checks())
    assert "arm reachable" not in r and "rt kernel" not in r and "rtprio" not in r and "payload" not in r
    assert r["version"].status == "info" and "rebuild: franka_control/build.sh" in r["version"].detail
    assert r["config"].status == "pass"
    assert r["backups"].detail == "arm: FakeArm, gripper: FakeGripper (--mock)"
    assert all(v.status != "fail" for v in r.values())


def test_realtime_ignore_warns_and_skips_the_rt_checks(process_class, tmp_path):
    fake_binary(tmp_path / "fc")
    p = make(process_class, tmp_path, mock=False,
             franka_control={"binary": str(tmp_path / "fc"), "realtime": "ignore", "robot_ip": "127.0.0.1"})
    r = by_name(p.checks())
    assert r["realtime"].status == "warn" and "loop_late" in r["realtime"].detail
    assert "rt kernel" not in r and "rtprio" not in r
    assert r["version"].status == "pass"
    assert r["arm reachable"].status == "fail"   # nothing listens on 127.0.0.1:1337
    assert r["arm reachable"].detail == ("arm not reachable at 127.0.0.1:1337: check the cable, the PC's static IP, "
                                         "FCI enabled in Desk")
    assert r["payload"].status == "warn"
    assert r["backups"].detail == "arm: LibfrankaArm at 127.0.0.1, realtime ignore"


def test_enforce_runs_the_rt_checks_and_a_binary_without_libfranka_fails(process_class, tmp_path):
    fake_binary(tmp_path / "fc", version="none")
    p = make(process_class, tmp_path, mock=False,
             franka_control={"binary": str(tmp_path / "fc"), "realtime": "enforce", "robot_ip": "127.0.0.1"})
    r = by_name(p.checks())
    assert "rt kernel" in r and "rtprio" in r and "realtime" not in r
    assert r["rt kernel"].status in ("pass", "fail")
    if r["rt kernel"].status == "fail":
        assert "PREEMPT_RT" in r["rt kernel"].detail
    assert r["version"].status == "fail"
    assert r["version"].detail == "built with libfranka none, config.yaml wants 0.9.2: rebuild: franka_control/build.sh"


def test_a_busy_port_fails(process_class, tmp_path):
    fake_binary(tmp_path / "fc")
    p = make(process_class, tmp_path, mock=True, franka_control={"binary": str(tmp_path / "fc")})
    host, port = deploy.parse_endpoint(p.station.endpoint("state"))
    with socket.socket() as s:
        s.bind((host, port))
        s.listen()
        r = by_name(p.checks())
    assert r[f"port {p.station.endpoint('state')}"].status == "fail"
