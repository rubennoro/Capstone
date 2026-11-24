# adapted from rnoro/deployment:deployment/message_comms/tests/test_main.py @ 91b178b
# changes: the common arguments of ProcessArgs (--station optional); FATAL lines say "manager" and name
# config/manager.yaml; no C2 is "PC not answering"; the fakes are FakePc + fake frames/policy.
import dataclasses
import os

import pytest
from manager_servers import FakePolicy, FakeProcessFrames, make_fake_pc
from station_helpers import write_station

from policy_deploy.contracts import ProcessArgs, paths
from policy_deploy.manager.__main__ import main
from policy_deploy.manager.settings import Settings

FAST_CLOCK = {"min_samples_s": 0.2, "min_samples": 5}


def test_the_four_common_arguments():
    args, _ = ProcessArgs.parse("manager", ["--station", "s.yaml", "--session", "d", "--ready-file", "r", "--mock"])
    assert args.station.name == "s.yaml" and str(args.session) == "d" and str(args.ready_file) == "r" and args.mock
    args, _ = ProcessArgs.parse("manager", [])
    assert args.station == paths.default_station() and args.session is None and args.mock is False


def test_unknown_argument_refused():
    with pytest.raises(SystemExit):
        ProcessArgs.parse("manager", ["--station", "s.yaml", "--exec-h", "4"])


def patch_endpoints(monkeypatch, **endpoints):
    original = Settings.load.__func__

    def load(cls, station_path, mock, link=None, frame=None):
        return dataclasses.replace(original(cls, station_path, mock, link, frame), **endpoints)
    monkeypatch.setattr(Settings, "load", classmethod(load))


def test_fatal_line_is_last_on_exit_1(tmp_path, capsys):
    station = write_station(tmp_path, manager={"exec_h": 40})
    assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    err = [x for x in capsys.readouterr().err.splitlines() if x.strip()]
    assert err[-1].startswith("FATAL manager: config: manager.exec_h: must be between 1 and 16")
    assert err[-1].endswith("check station.yaml manager: and config/manager.yaml")


def test_contract_key_in_the_station_is_fatal(tmp_path, capsys):
    station = write_station(tmp_path, manager={"c3_timeout_s": 0.5})
    assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[-1].startswith("FATAL manager: config: ") and "belongs to deployment/contracts/link.yaml" in err[-1]


def test_bad_operator_script_is_fatal(tmp_path, capsys):
    (tmp_path / "bad.yaml").write_text("- {key: s}\n")
    station = write_station(tmp_path, manager={"operator_script": "bad.yaml"})
    assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[-1].startswith("FATAL manager: operator script: ") and "last entry must be {key: q}" in err[-1]


def test_no_fatal_line_on_exit_0(tmp_path, capsys, monkeypatch):
    frames, policy, pc = FakeProcessFrames(), FakePolicy(), make_fake_pc(tmp_path / "pc")
    try:
        patch_endpoints(monkeypatch, frames_endpoint=frames.endpoint, policy_endpoint=policy.endpoint,
                        state_endpoint=pc.state_endpoint, commands_endpoint=pc.commands_endpoint)
        (tmp_path / "script.yaml").write_text("- {wait_phase: idle}\n- {key: q}\n")
        station = write_station(tmp_path, manager={"operator_script": "script.yaml", "clock": FAST_CLOCK})
        ready = tmp_path / "ready"
        code = main(["--station", str(station), "--session", str(tmp_path / "s"), "--ready-file", str(ready),
                     "--mock"])
        assert code == 0 and ready.exists()
        assert not any(x.startswith("FATAL") for x in capsys.readouterr().err.splitlines())
        assert pc.request_names()[-1] == "c3.stop/v1"
        assert (tmp_path / "s" / "manager" / "session.yaml").exists()
    finally:
        for fake in (frames, policy, pc):
            fake.close()


def test_hello_mismatch_without_mock_is_fatal(tmp_path, capsys, monkeypatch):
    """The fake PC says mock true; a manager without --mock refuses at start-up (compare_hello)."""
    frames, policy, pc = FakeProcessFrames(), FakePolicy(), make_fake_pc(tmp_path / "pc")
    try:
        patch_endpoints(monkeypatch, frames_endpoint=frames.endpoint, policy_endpoint=policy.endpoint,
                        state_endpoint=pc.state_endpoint, commands_endpoint=pc.commands_endpoint)
        (tmp_path / "script.yaml").write_text("- {key: q}\n")
        station = write_station(tmp_path, manager={"operator_script": "script.yaml", "clock": FAST_CLOCK})
        assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
        err = capsys.readouterr().err.splitlines()
        assert err[-1].startswith("FATAL manager: hello mismatch: mock: PC true, laptop false")
        assert "c3.start/v1" not in pc.request_names()
    finally:
        for fake in (frames, policy, pc):
            fake.close()


def test_no_c2_fails_startup_naming_the_pc(tmp_path, capsys, monkeypatch):
    frames, policy = FakeProcessFrames(), FakePolicy()
    try:
        patch_endpoints(monkeypatch, frames_endpoint=frames.endpoint, policy_endpoint=policy.endpoint,
                        state_endpoint="tcp://127.0.0.1:1", commands_endpoint="tcp://127.0.0.1:1")
        (tmp_path / "script.yaml").write_text("- {key: q}\n")
        station = write_station(tmp_path, manager={"operator_script": "script.yaml", "timeouts": {"startup_s": 0.5}})
        assert main(["--station", str(station), "--session", str(tmp_path / "s"), "--mock"]) == 1
        err = capsys.readouterr().err.splitlines()
        assert err[-1] == ("FATAL manager: PC not answering: no C2 state from tcp://127.0.0.1:1 within 0.5 s: check "
                           "that control_deploy run is running on the PC, ASUS_5g, and link.yaml pc.address")
    finally:
        frames.close()
        policy.close()


def test_a_script_that_waits_forever_gives_the_fatal_line(tmp_path, capsys, monkeypatch):
    frames, policy, pc = FakeProcessFrames(), FakePolicy(), make_fake_pc(tmp_path / "pc")
    try:
        patch_endpoints(monkeypatch, frames_endpoint=frames.endpoint, policy_endpoint=policy.endpoint,
                        state_endpoint=pc.state_endpoint, commands_endpoint=pc.commands_endpoint)
        script = tmp_path / "script.yaml"
        script.write_text("- {wait_phase: idle}\n- {key: y}\n- {key: q}\n")     # y in idle: refused, forever
        station = write_station(tmp_path, manager={"operator_script": "script.yaml", "operator_script_wait_s": 0.5,
                                                   "clock": FAST_CLOCK})
        assert main(["--station", str(station), "--session", str(tmp_path / "s"), "--mock"]) == 1
        err = capsys.readouterr().err.splitlines()
        assert err[-1] == (f"FATAL manager: operator script: {script}:2: key y: still refused "
                           "(y: no episode running (c start, q quit)) after 0.5 s "
                           "(manager.operator_script_wait_s): check the script (manager.operator_script) "
                           "or raise operator_script_wait_s")
        assert pc.request_names()[-1] == "c3.stop/v1"
    finally:
        for fake in (frames, policy, pc):
            fake.close()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_session_folder_gives_the_fatal_line(tmp_path, capsys):
    (tmp_path / "script.yaml").write_text("- {key: q}\n")
    station = write_station(tmp_path, manager={"operator_script": "script.yaml"})
    folder = tmp_path / "s" / "manager"
    folder.mkdir(parents=True)
    folder.chmod(0o555)
    try:
        assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    finally:
        folder.chmod(0o755)
    err = capsys.readouterr().err.splitlines()
    assert err[-1].startswith(f"FATAL manager: session folder {folder}: cannot write: ")
    assert err[-1].endswith(": check that the session folder (--session, or station.yaml session_root) is writable")


def test_any_unexpected_start_up_error_gives_the_fatal_line(tmp_path, capsys, caplog, monkeypatch):
    import policy_deploy.manager.state_history as sh

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sh, "StateHistory", broken)
    (tmp_path / "script.yaml").write_text("- {key: q}\n")
    station = write_station(tmp_path, manager={"operator_script": "script.yaml"})
    assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[-1] == "FATAL manager: internal error: RuntimeError: boom: see the traceback in the manager log"
    assert any(r.exc_info for r in caplog.records)          # the traceback is in the log
