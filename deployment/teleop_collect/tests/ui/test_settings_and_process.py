"""The ui settings, and `python -m teleop_collect.ui` as a process (free ports only): ready, SIGINT/SIGTERM exit 0,
bad settings -> one FATAL line and exit 1."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest
from ui_helpers import FakeTeleop, wait_until

from station_helpers import write_station
from teleop_collect.ui.settings import Settings, SettingsError


def test_defaults_from_station(tmp_path):
    path = write_station(tmp_path)
    s = Settings.load(path)
    assert (s.status_hz, s.preview_hz, s.keys_resend_ms, s.keys_stale_ms, s.rpc_timeout_s) == (10, 15, 50, 150, 0.5)
    assert s.open_browser is True
    import yaml
    ports = yaml.safe_load(path.read_text())["ports"]
    assert s.http_port == ports["http"]
    assert s.u1_endpoint == f"tcp://127.0.0.1:{ports['u1']}"
    assert s.preview_endpoint == f"tcp://127.0.0.1:{ports['preview']}"


def test_station_overrides(tmp_path):
    s = Settings.load(write_station(tmp_path, sections={"ui": {"keys_stale_ms": 300, "open_browser": False}}))
    assert (s.keys_stale_ms, s.open_browser) == (300.0, False)


@pytest.mark.parametrize("ui,problem", [
    ({"status_hz": 0}, "ui.status_hz"),
    ({"rpc_timeout_s": "fast"}, "ui.rpc_timeout_s"),
    ({"open_browser": 1}, "ui.open_browser"),
    ({"keys_resend_ms": 200}, "must be below ui.keys_stale_ms"),
    ({"colour": "red"}, "not a key of config/ui.yaml"),
])
def test_bad_settings_refused(tmp_path, ui, problem):
    with pytest.raises(SettingsError) as info:
        Settings.load(write_station(tmp_path, sections={"ui": ui}))
    assert problem in str(info.value)


# ------------------------------------------------------------------------------------------------- the process

def _env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _start(station, ready, *extra):
    return subprocess.Popen([sys.executable, "-m", "teleop_collect.ui", "--station", str(station),
                             "--session", str(station.parent / "session"), "--ready-file", str(ready), *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env())


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_process_ready_then_clean_stop(tmp_path, sig):
    station = write_station(tmp_path, pi_address=None)
    from teleop_collect.contracts import Station
    st = Station.load(station, mock=True)
    teleop = FakeTeleop()
    teleop.endpoint = st.u1_endpoint()
    teleop.start()
    ready = tmp_path / "ui.ready"
    proc = _start(station, ready, "--mock")
    try:
        wait_until(lambda: ready.exists() or proc.poll() is not None, 10.0, "the ready file")
        assert ready.exists(), proc.stderr.read() if proc.poll() is not None else ""
        import urllib.request
        with urllib.request.urlopen(st.http_url() + "/", timeout=3) as r:
            assert r.status == 200
        time.sleep(0.2)
        proc.send_signal(sig)
        out, err = proc.communicate(timeout=10)
        assert proc.returncode == 0, err
        assert "FATAL" not in err
        assert (tmp_path / "session" / "ui").is_dir()
    finally:
        if proc.poll() is None:
            proc.kill()
        teleop.stop()


def test_process_bad_settings_one_fatal_line(tmp_path):
    station = write_station(tmp_path, sections={"ui": {"status_hz": -1}})
    proc = _start(station, tmp_path / "ui.ready")
    out, err = proc.communicate(timeout=20)
    assert proc.returncode == 1
    last = err.strip().splitlines()[-1]
    assert last.startswith("FATAL ui: invalid settings:") and "ui.status_hz" in last
    assert not (tmp_path / "ui.ready").exists()


def test_process_port_in_use_one_fatal_line(tmp_path):
    import socket
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        station = write_station(tmp_path, ports={"http": busy.getsockname()[1]})
        proc = _start(station, tmp_path / "ui.ready")
        out, err = proc.communicate(timeout=20)
    assert proc.returncode == 1
    assert err.strip().splitlines()[-1].startswith("FATAL ui: cannot bind the page on 127.0.0.1:")
