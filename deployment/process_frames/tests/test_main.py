"""``python -m process_frames`` as a subprocess (STRUCTURE.md §12)."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import pytest

from conftest import fast_overrides, wait_for, write_rig, write_station
from process_frames import FOLDER


def start(args):
    return subprocess.Popen([sys.executable, "-m", "process_frames", *map(str, args)], cwd=FOLDER,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(proc, timeout_s=20.0):
    out, err = proc.communicate(timeout=timeout_s)
    lines = [ln for ln in err.splitlines() if ln.strip()]
    return proc.returncode, lines


def mock_station(tmp_path, pi_episode, **more):
    return write_station(tmp_path / "station.yaml",
                         process_frames=fast_overrides(mock_episode=str(pi_episode), **more))


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_ready_file_then_clean_stop(tmp_path, pi_episode, sig):
    station = mock_station(tmp_path, pi_episode)
    ready = tmp_path / "session" / "ready" / "process_frames.ready"
    proc = start(["--station", station, "--mock", "--session", tmp_path / "session", "--ready-file", ready])
    try:
        wait_for(lambda: ready.exists() or proc.poll() is not None, 30, "the ready file")
        assert ready.exists(), proc.communicate()[1]
    finally:
        proc.send_signal(sig)
    code, lines = finish(proc)
    assert code == 0, lines[-5:]
    assert not any(ln.startswith("FATAL") for ln in lines)
    assert (tmp_path / "session" / "process_frames").is_dir()


def test_broken_rig_file_exits_1_with_a_fatal_line(tmp_path, pi_episode):
    rig = write_rig(tmp_path, "rig_broken", lambda d: d["cameras"]["main"].update(sensor="imx477"))
    proc = start(["--station", mock_station(tmp_path, pi_episode, rig_file=str(rig)), "--mock"])
    code, lines = finish(proc)
    assert code == 1
    assert lines[-1].startswith("FATAL process_frames: rig file invalid: ")
    assert "cameras.main.sensor" in lines[-1] and lines[-1].endswith(": check process_frames.rig_file")


def test_unknown_station_key_exits_1_naming_it(tmp_path, pi_episode):
    proc = start(["--station", mock_station(tmp_path, pi_episode, clock={"intervl_s": 1.0}), "--mock"])
    code, lines = finish(proc)
    assert code == 1
    assert lines[-1].startswith("FATAL process_frames: invalid settings: ")
    assert "process_frames.clock.intervl_s" in lines[-1]


def test_alone_without_session_writes_into_its_alone_folder(tmp_path, pi_episode):
    root = tmp_path / "sessions"
    station = write_station(tmp_path / "station.yaml", session_root=root,
                            process_frames=fast_overrides(mock_episode=str(pi_episode)))

    def out_dirs():
        return list(root.glob("*_process_frames_alone/process_frames")) if root.is_dir() else []

    proc = start(["--station", station, "--mock"])
    try:
        wait_for(lambda: bool(out_dirs()) or proc.poll() is not None, 30, "the _alone folder")
    finally:
        proc.send_signal(signal.SIGTERM)
    code, lines = finish(proc)
    assert code == 0, lines[-5:]
    (d,) = out_dirs()
    assert d.is_dir() and d.parent.name.endswith("_process_frames_alone")
