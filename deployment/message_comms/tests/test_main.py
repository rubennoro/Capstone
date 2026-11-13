import os

import pytest

from conftest import write_station
from fake_servers import FakeFrankaControl, FakePolicy, FakeProcessFrames
from message_comms.__main__ import main, parse_args


def test_the_four_common_arguments(tmp_path):
    args = parse_args(["--station", "s.yaml", "--session", "d", "--ready-file", "r", "--mock"])
    assert str(args.station) == "s.yaml" and str(args.session) == "d" and str(args.ready_file) == "r" and args.mock
    args = parse_args(["--station", "s.yaml"])
    assert args.session is None and args.ready_file is None and args.mock is False


def test_station_is_required(capsys):
    with pytest.raises(SystemExit) as e:
        parse_args([])
    assert e.value.code == 2


def test_unknown_argument_refused():
    with pytest.raises(SystemExit):
        parse_args(["--station", "s.yaml", "--exec-h", "4"])


def test_fatal_line_is_last_on_exit_1(tmp_path, capsys):
    station = write_station(tmp_path, {"exec_h": 40})
    assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    err = [x for x in capsys.readouterr().err.splitlines() if x.strip()]
    assert err[-1].startswith("FATAL message_comms: config: message_comms.exec_h: must be between 1 and 16")
    assert err[-1].endswith("check station.yaml message_comms: and message_comms/config.yaml")


def test_bad_operator_script_is_fatal(tmp_path, capsys):
    (tmp_path / "bad.yaml").write_text("- {key: s}\n")
    station = write_station(tmp_path, {"operator_script": "bad.yaml"})
    assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[-1].startswith("FATAL message_comms: operator script: ") and "last entry must be {key: q}" in err[-1]


def test_no_fatal_line_on_exit_0_and_mock_accepted(tmp_path, capsys):
    frames, policy, franka = FakeProcessFrames(), FakePolicy(), FakeFrankaControl()
    try:
        (tmp_path / "script.yaml").write_text("- {wait_phase: idle}\n- {key: q}\n")
        station = write_station(tmp_path, {"operator_script": "script.yaml"},
                                {"frames": frames.endpoint, "state": franka.state_endpoint,
                                 "commands": franka.commands_endpoint, "policy": policy.endpoint})
        ready = tmp_path / "ready"
        code = main(["--station", str(station), "--session", str(tmp_path / "s"), "--ready-file", str(ready),
                     "--mock"])
        assert code == 0 and ready.exists()
        assert not any(x.startswith("FATAL") for x in capsys.readouterr().err.splitlines())
        assert franka.names()[-1] == "stop"
    finally:
        for fake in (frames, policy, franka):
            fake.close()


def test_missing_process_fails_startup_naming_it(tmp_path, capsys):
    frames, policy = FakeProcessFrames(), FakePolicy()
    try:
        (tmp_path / "script.yaml").write_text("- {key: q}\n")
        station = write_station(tmp_path, {"operator_script": "script.yaml", "timeouts": {"startup_s": 0.5}},
                                {"frames": frames.endpoint, "state": "tcp://127.0.0.1:1",
                                 "commands": "tcp://127.0.0.1:1", "policy": policy.endpoint})
        assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
        err = capsys.readouterr().err.splitlines()
        assert err[-1].startswith("FATAL message_comms: startup: no fresh arm state within 0.5 s (no state yet): "
                                  "check franka_control")
    finally:
        frames.close()
        policy.close()


def test_a_script_that_waits_forever_gives_the_fatal_line(tmp_path, capsys):
    frames, policy, franka = FakeProcessFrames(), FakePolicy(), FakeFrankaControl()
    try:
        script = tmp_path / "script.yaml"
        script.write_text("- {wait_phase: idle}\n- {key: y}\n- {key: q}\n")     # y in idle: refused, forever
        station = write_station(tmp_path, {"operator_script": "script.yaml", "operator_script_wait_s": 0.5},
                                {"frames": frames.endpoint, "state": franka.state_endpoint,
                                 "commands": franka.commands_endpoint, "policy": policy.endpoint})
        assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
        err = capsys.readouterr().err.splitlines()
        assert err[-1] == (f"FATAL message_comms: operator script: {script}:2: key y: still refused "
                           "(y: no episode running (c start, h home, q quit)) after 0.5 s "
                           "(message_comms.operator_script_wait_s): check the script (message_comms.operator_script) "
                           "or raise operator_script_wait_s")
        assert franka.names()[-1] == "stop"
    finally:
        for fake in (frames, policy, franka):
            fake.close()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere")
def test_unwritable_session_folder_gives_the_fatal_line(tmp_path, capsys):
    (tmp_path / "script.yaml").write_text("- {key: q}\n")
    station = write_station(tmp_path, {"operator_script": "script.yaml"})
    folder = tmp_path / "s" / "message_comms"
    folder.mkdir(parents=True)
    folder.chmod(0o555)
    try:
        assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    finally:
        folder.chmod(0o755)
    err = capsys.readouterr().err.splitlines()
    assert err[-1].startswith(f"FATAL message_comms: session folder {folder.resolve()}: cannot write: ")
    assert err[-1].endswith(": check that the session folder (--session, or station.yaml session_root) is writable")


def test_any_unexpected_start_up_error_gives_the_fatal_line(tmp_path, capsys, caplog, monkeypatch):
    import message_comms.__main__ as entry

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(entry, "StateHistory", broken)
    (tmp_path / "script.yaml").write_text("- {key: q}\n")
    station = write_station(tmp_path, {"operator_script": "script.yaml"})
    assert main(["--station", str(station), "--session", str(tmp_path / "s")]) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[-1] == ("FATAL message_comms: internal error: RuntimeError: boom: "
                       "see the traceback in the message_comms log")
    assert any(r.exc_info for r in caplog.records)          # the traceback is in the log
