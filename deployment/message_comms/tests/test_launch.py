"""launch.py (deploy/STRUCTURE.md section 9): program(), USES_TERMINAL, checks()."""
import sys
from pathlib import Path

import pytest
import yaml

deploy = pytest.importorskip("deploy")

FOLDER = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch_class():
    return deploy.load_process_class(FOLDER)


def make(launch_class, tmp_path, section=None, franka=None, script=True, monkeypatch=None):
    doc = {"station_version": 1, "session_root": str(tmp_path / "sessions"),
           "python": {"light": sys.executable},
           "endpoints": {"mock_pi_frames": "tcp://127.0.0.1:5600", "mock_pi_time_sync": "tcp://127.0.0.1:5601",
                         "frames": "tcp://127.0.0.1:5610", "state": "tcp://127.0.0.1:5620",
                         "commands": "tcp://127.0.0.1:5630", "policy": "tcp://127.0.0.1:5640"}}
    section = dict(section or {})
    if script:
        (tmp_path / "script.yaml").write_text("- {key: q}\n")
        section.setdefault("operator_script", "script.yaml")
    doc["message_comms"] = section
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(doc))
    station = deploy.Station.load(path, mock=True)
    proc = launch_class(station, True, None)
    fc = tmp_path / "franka_control"
    fc.mkdir(exist_ok=True)
    (fc / "config.yaml").write_text(yaml.safe_dump(franka or {"watchdog": {"limit_s": 0.5},
                                                              "homing": {"timeout_s": 30.0}}))
    proc.franka_folder = lambda: fc
    return proc


def by_name(results):
    return {r.name: r for r in results}


def test_program_and_terminal(launch_class, tmp_path):
    proc = make(launch_class, tmp_path)
    assert proc.program() == [sys.executable, "-m", "message_comms"]
    assert launch_class.USES_TERMINAL is True and launch_class.NAME == "message_comms"
    cmd = proc.command()
    assert "--station" in cmd and "--mock" in cmd


def test_checks_pass_with_defaults(launch_class, tmp_path):
    results = make(launch_class, tmp_path).checks()
    assert [r for r in results if r.status == "fail"] == []
    infos = {r.name: r.detail for r in results if r.status == "info"}
    assert infos["exec_h"].startswith("exec_h = 8 (0.40 s per step") and infos["tap"] == "tap: off"
    assert infos["pose_source"] == "pose_source = target" and infos["operator"].startswith("operator: script")


def test_missing_operator_script_fails(launch_class, tmp_path):
    results = make(launch_class, tmp_path, {"operator_script": "nope.yaml"}).checks()
    assert any(r.status == "fail" and r.name == "operator_script" for r in results)


def test_no_terminal_without_a_script_fails(launch_class, tmp_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    results = by_name(make(launch_class, tmp_path, script=False).checks())
    assert results["terminal"].status == "fail" and "no terminal for the keyboard" in results["terminal"].detail
    assert results["operator"].detail == "operator: keyboard"


@pytest.mark.parametrize("section", [{"exec_h": 0}, {"exec_h": 17}, {"pose_source": "x"},
                                     {"timeouts": {"home_s": 0}}, {"odd_chunk_max_pos_m": 0},
                                     {"tap_images_every": 0}])
def test_value_checks(launch_class, tmp_path, section):
    results = by_name(make(launch_class, tmp_path, section).checks())
    assert results["settings"].status == "fail"


def test_heartbeat_gap_against_the_watchdog(launch_class, tmp_path):
    proc = make(launch_class, tmp_path, franka={"watchdog": {"limit_s": 0.42}, "homing": {"timeout_s": 30.0}})
    result = by_name(proc.checks())["heartbeat gap"]
    assert result.status == "fail" and "0.420" in result.detail and "0.42 s" in result.detail


def test_home_timeout_against_homing(launch_class, tmp_path):
    proc = make(launch_class, tmp_path, {"timeouts": {"home_s": 30.0}})
    result = by_name(proc.checks())["homing timeout"]
    assert result.status == "fail" and "30.0" in result.detail


def test_real_franka_control_config_passes(launch_class, tmp_path):
    proc = make(launch_class, tmp_path)
    del proc.franka_folder                         # back to the real deployment/franka_control/config.yaml
    if not (deploy.DEPLOYMENT_ROOT / "franka_control" / "config.yaml").exists():
        pytest.skip("franka_control/config.yaml not written yet")
    results = by_name(proc.checks())
    assert results["heartbeat gap"].status == "pass" and results["homing timeout"].status == "pass"


def test_mock_checks_need_no_other_process(launch_class, tmp_path):
    results = make(launch_class, tmp_path).checks()      # nothing listens on the station's ports
    assert all(r.status != "fail" for r in results)
