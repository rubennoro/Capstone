# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_main.py @ 2ffe641
# changes: python -m teleop_collect.frames; a busy R1 or U2 port instead of C1; + SIGTERM while recording (exit < 5 s,
# files closed) and the ready file of ProcessArgs.write_ready
# adapted from rnoro/deployment:deployment/process_frames/tests/test_main.py @ 91b178b
# changes: python -m policy_deploy.frames; the laptop station (frames: section); FATAL lines name "frames"; the alone
# folder is <session_root>/<stamp>_frames_alone/frames/.
"""``python -m teleop_collect.frames`` as a subprocess (STRUCTURE.md 5.4)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest
from frames_helpers import FOLDER, fast_overrides, wait_for, write_rig, write_station


def start(args):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.Popen([sys.executable, "-m", "teleop_collect.frames", *map(str, args)], cwd=FOLDER, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(proc, timeout_s=20.0):
    out, err = proc.communicate(timeout=timeout_s)
    lines = [ln for ln in err.splitlines() if ln.strip()]
    return proc.returncode, lines


def mock_station(tmp_path, pi_episode, **more):
    return write_station(tmp_path / "station.yaml", frames=fast_overrides(mock_episode=str(pi_episode), **more))


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_ready_file_then_clean_stop(tmp_path, pi_episode, sig):
    station = mock_station(tmp_path, pi_episode)
    ready = tmp_path / "session" / "ready" / "frames.ready"
    proc = start(["--station", station, "--mock", "--session", tmp_path / "session", "--ready-file", ready])
    try:
        wait_for(lambda: ready.exists() or proc.poll() is not None, 30, "the ready file")
        assert ready.exists(), proc.communicate()[1]
    finally:
        proc.send_signal(sig)
    code, lines = finish(proc)
    assert code == 0, lines[-5:]
    assert not any(ln.startswith("FATAL") for ln in lines)
    assert (tmp_path / "session" / "frames").is_dir()


def test_broken_rig_file_exits_1_with_a_fatal_line(tmp_path, pi_episode):
    rig = write_rig(tmp_path, "rig_broken", lambda d: d["cameras"]["main"].update(sensor="imx477"))
    proc = start(["--station", mock_station(tmp_path, pi_episode, rig_file=str(rig)), "--mock"])
    code, lines = finish(proc)
    assert code == 1
    assert lines[-1].startswith("FATAL frames: rig file invalid: ")
    assert "cameras.main.sensor" in lines[-1] and lines[-1].endswith(": check frames.rig_file")


def test_unknown_station_key_exits_1_naming_it(tmp_path, pi_episode):
    proc = start(["--station", mock_station(tmp_path, pi_episode, clock={"intervl_s": 1.0}), "--mock"])
    code, lines = finish(proc)
    assert code == 1
    assert lines[-1].startswith("FATAL frames: invalid settings: ")
    assert "frames.clock.intervl_s" in lines[-1]


@pytest.mark.parametrize("port_key", ["r1", "preview"])
def test_a_busy_r1_or_u2_port_exits_1(tmp_path, pi_episode, port_key):
    import zmq
    import yaml
    station = mock_station(tmp_path, pi_episode)
    port = yaml.safe_load(station.read_text())["ports"][port_key]
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{port}")
    try:
        code, lines = finish(start(["--station", station, "--mock"]))
    finally:
        sock.close()
    assert code == 1 and lines[-1].startswith("FATAL frames: cannot bind R1 on tcp://127.0.0.1:")
    assert str(port) in lines[-1] and lines[-1].endswith(": is another frames process running?")


def test_alone_without_session_writes_into_its_alone_folder(tmp_path, pi_episode):
    root = tmp_path / "sessions"
    station = mock_station(tmp_path, pi_episode)

    def out_dirs():
        return list(root.glob("*_frames_alone/frames")) if root.is_dir() else []

    proc = start(["--station", station, "--mock"])
    try:
        wait_for(lambda: bool(out_dirs()) or proc.poll() is not None, 30, "the _alone folder")
    finally:
        proc.send_signal(signal.SIGTERM)
    code, lines = finish(proc)
    assert code == 0, lines[-5:]
    (d,) = out_dirs()
    assert d.is_dir() and d.parent.name.endswith("_frames_alone")


def test_sigterm_while_recording_exits_within_5_s_with_the_files_closed(tmp_path, pi_episode):
    """The launcher sends SIGTERM and kills after 5 s: frames closes the open recording first (STRUCTURE.md 10)."""
    import time

    import yaml
    from deploy_contracts import Ack, RpcClient
    from frames_helpers import read_csv, read_mkv

    from teleop_collect.contracts import FramesHealth, GetFramesHealth, StartRecording

    station = mock_station(tmp_path, pi_episode)
    ready = tmp_path / "session" / "ready" / "frames.ready"
    proc = start(["--station", station, "--mock", "--session", tmp_path / "session", "--ready-file", ready])
    r1 = RpcClient("frames", f"tcp://127.0.0.1:{yaml.safe_load(station.read_text())['ports']['r1']}")
    ep = tmp_path / "data" / "task" / "episode_000"
    ep.mkdir(parents=True)
    try:
        wait_for(lambda: ready.exists() or proc.poll() is not None, 30, "the ready file")
        assert ready.read_text() == "ready\n"                           # ProcessArgs.write_ready
        assert r1.request(StartRecording(episode_dir=str(ep)), Ack, 5.0).ok
        wait_for(lambda: r1.request(GetFramesHealth(), FramesHealth, 5.0).frames_main >= 10, 10, "frames written")
    finally:
        r1.close()
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
    code, lines = finish(proc, timeout_s=5.0)
    assert time.monotonic() - t0 < 5.0 and code == 0, lines[-5:]
    n = len(read_mkv(ep / "main.mkv"))
    assert n >= 10 and len(read_csv(ep / "main_frames.csv")[1]) == n
    assert any("still open at shutdown" in ln for ln in lines)
