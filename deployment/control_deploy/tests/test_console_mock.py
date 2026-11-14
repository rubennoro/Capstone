"""PLAN §10's "control_deploy run --mock with a scripted console" (control_deploy/STRUCTURE.md 9.2): the real
franka_control (FakeArm, FakeGripper) under `run --mock --local` with stdin from a pipe, driven by
`console --mock --local --yes`; every step passes, the last leaves C2 in fault; a key q on the pipe ends run with 0.

Needs a built binary: franka_control/build/franka_control, or the laptop's libfranka-less build in
franka_control/build-nolibfranka/ (STRUCTURE.md 8.3; the station then overrides `binary`). Skipped without either, and
when the link.yaml ports 5620/5630 are taken on 127.0.0.1.

Also (fix wave): franka_control keeps running when the read end of its log pipe closes (the run wrapper gone): the
writes fail with EPIPE instead of a SIGPIPE killing it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml
from deploy_contracts import ArmState, Subscriber
from helpers import env_without, free_port_pair_in_use, write_station

from control_deploy import paths

BUILDS = ("build", "build-nolibfranka")


def built_binary() -> str | None:
    for b in BUILDS:
        if (paths.franka_control_dir() / b / "franka_control").is_file():
            return f"{b}/franka_control"
    return None


pytestmark = [pytest.mark.needs_binary,
              pytest.mark.skipif(built_binary() is None, reason="franka_control is not built")]


def cli(*args: str) -> list[str]:
    return [sys.executable, "-m", "control_deploy", *args]


def test_a_closed_log_pipe_does_not_kill_franka_control(tmp_path):
    if free_port_pair_in_use():
        pytest.skip("127.0.0.1:5620/5630 are in use (another franka_control or fake PC)")
    station = tmp_path / "fc_station.yaml"
    station.write_text(yaml.safe_dump({"session_root": str(tmp_path / "sessions")}))
    ready = tmp_path / "fc.ready"
    read_fd, write_fd = os.pipe()
    # Popen restores SIGPIPE to its default (kill) in the child, as for a binary started by the wrapper
    proc = subprocess.Popen([str(paths.franka_control_dir() / built_binary()), "--station", str(station), "--mock",
                             "--local", "--ready-file", str(ready)],
                            stdin=subprocess.DEVNULL, stdout=write_fd, stderr=write_fd, cwd=tmp_path,
                            env=env_without("PYTHONPATH"))
    os.close(write_fd)
    try:
        deadline = time.monotonic() + 30.0
        while not ready.exists():
            assert proc.poll() is None, f"franka_control exited {proc.returncode} before ready"
            assert time.monotonic() < deadline, "franka_control not ready within 30 s"
            time.sleep(0.05)
        os.close(read_fd)            # the wrapper is gone; the next log line (the 1 s loop stats) hits a closed pipe
        read_fd = -1
        time.sleep(2.5)
        assert proc.poll() is None, f"franka_control died with {proc.returncode} (-13 is SIGPIPE)"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_run_mock_with_a_scripted_console(tmp_path):
    if free_port_pair_in_use():
        pytest.skip("127.0.0.1:5620/5630 are in use (another franka_control or fake PC)")
    # The console's rotation is gentler than the shipped station (5 deg in 1 s): the impedance law's lag
    # (2 zeta / sqrt(K) = 0.32 s at 40 Nm/rad) puts that at ~2.9 deg of tracking, over the 1 deg pass mark.
    station = write_station(tmp_path, franka_control={"binary": built_binary(), "realtime": "ignore"},
                            console={"rotate_deg": 2.0, "segment_s": 2.0})
    env = env_without("PYTHONPATH")
    run = subprocess.Popen(cli("run", "--mock", "--local", "--station", str(station)), stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=tmp_path)
    lines: list[str] = []
    ready = threading.Event()

    def read() -> None:
        for line in run.stdout:
            lines.append(line)
            if line.startswith("control_deploy: ready:"):
                ready.set()
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert ready.wait(60), "".join(lines)
        console = subprocess.run(cli("console", "--mock", "--local", "--yes", "--station", str(station)),
                                 capture_output=True, text=True, env=env, cwd=tmp_path, timeout=180)
        assert console.returncode == 0, console.stdout + console.stderr
        (folder,) = list((tmp_path / "sessions").glob("*_control_deploy_console_mock"))
        results = yaml.safe_load((folder / "results.yaml").read_text())
        names = [s["name"] for s in results["steps"]]
        assert names == ["hold", "move", "rotate", "gripper", "watchdog", "out_of_box"]
        for s in results["steps"]:
            # warn is allowed only for late 1 kHz cycles of a busy laptop during the hold
            assert s["status"] == "pass" or (s["name"] == "hold" and s["numbers"].get("loop_late_new")), s
        sub = Subscriber("test", "tcp://127.0.0.1:5620", ArmState, 100)
        try:
            msg = None
            end = time.monotonic() + 2.0
            while msg is None and time.monotonic() < end:
                msg = sub.receive(0.1)
            assert msg is not None and msg.mode == "fault" and "safety gate rejected" in msg.error
        finally:
            sub.close()
        run.stdin.write("q")
        run.stdin.flush()
        assert run.wait(10) == 0, "".join(lines)
        reader.join(5)
        text = "".join(lines)
        assert "key q: shutting down" in text and "franka_control stopped cleanly" in text
    finally:
        if run.poll() is None:
            run.terminate()
            run.wait(10)
