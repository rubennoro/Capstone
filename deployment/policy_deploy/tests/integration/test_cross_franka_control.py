"""PLAN 10 "across the two", on the laptop (STRUCTURE.md 12.3; TOP 8.10): the laptop's processes against the REAL
franka_control code with its fake arm (the libfranka-less build of control_deploy/STRUCTURE.md 8.3).

The binary comes from FRANKA_CONTROL_BIN (absolute path); skipped when it is unset, and without
the recorded episode (station_helpers.RECORDED_EPISODE).

1. a temporary franka_control station with only session_root;
2. `$FRANKA_CONTROL_BIN --station <tmp>/fc_station.yaml --mock --local --ready-file <tmp>/fc.ready`, wait for the ready
   file (30 s); skipped if 127.0.0.1:5620/5630 are busy;
3. `python -m policy_deploy run --mock` with launcher.fake_pc: false (frames with the fake Pi, policy hold_still, the
   manager on 127.0.0.1 with the link.yaml ports) and the script c, wait_steps 20, y, h, q;
4. SIGTERM the binary, expect exit 0.

Also (fix wave, S2): a set_waypoints whose failing waypoint is not the first, sent with the laptop's PcClient to the
real binary; the laptop's gate_rejection_detail must name that waypoint and its real reason (the binary lists the other
waypoints with the contract's "not checked: ..." text). Needs no episode.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from deploy_contracts import LinkSettings, SetWaypoints
from integration_helpers import jsonl, need_episode, need_pc_ports_free, run_policy_deploy, session_dir, write_run_station

from policy_deploy.fake_pc.fake_gate import NOT_CHECKED
from policy_deploy.manager.episode_loop import gate_rejection_detail
from policy_deploy.manager.pc_client import PcClient

SCRIPT = [{"key": "c"}, {"wait_steps": 20}, {"key": "y"}, {"key": "h"}, {"key": "q"}]


def _franka_control(tmp_path, with_episode: bool):
    binary = os.environ.get("FRANKA_CONTROL_BIN")
    if not binary:
        pytest.skip("FRANKA_CONTROL_BIN is not set (the libfranka-less franka_control build)")
    binary = Path(binary).expanduser()
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        pytest.fail(f"FRANKA_CONTROL_BIN={binary} is not an executable file")
    if with_episode:
        need_episode()
    need_pc_ports_free()
    fc_dir = tmp_path / "fc"
    fc_dir.mkdir()
    station = fc_dir / "fc_station.yaml"
    station.write_text(yaml.safe_dump({"session_root": str(fc_dir / "sessions")}))
    ready = fc_dir / "fc.ready"
    log = open(fc_dir / "franka_control.log", "w")
    proc = subprocess.Popen([str(binary), "--station", str(station), "--mock", "--local", "--ready-file", str(ready)],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=fc_dir)
    deadline = time.monotonic() + 30.0
    while not ready.exists():
        if proc.poll() is not None:
            log.close()
            pytest.fail(f"franka_control exited {proc.returncode} before ready:\n"
                        f"{(fc_dir / 'franka_control.log').read_text()[-3000:]}")
        if time.monotonic() > deadline:
            proc.kill()
            log.close()
            pytest.fail("franka_control not ready within 30 s:\n" + (fc_dir / "franka_control.log").read_text()[-3000:])
        time.sleep(0.05)
    yield proc, fc_dir
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log.close()


@pytest.fixture
def franka_control(tmp_path):
    yield from _franka_control(tmp_path, with_episode=True)


@pytest.fixture
def franka_control_alone(tmp_path):
    yield from _franka_control(tmp_path, with_episode=False)


def test_laptop_against_the_real_franka_control(tmp_path, franka_control):
    proc, fc_dir = franka_control
    done = run_policy_deploy(write_run_station(tmp_path, SCRIPT, launcher={"fake_pc": False}))
    fc_log = (fc_dir / "franka_control.log").read_text()
    assert done.returncode == 0, (f"exit {done.returncode}\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}\n"
                                  f"franka_control log:\n{fc_log[-3000:]}")
    assert "policy_deploy: done" in done.stdout + done.stderr
    session = session_dir(tmp_path)
    assert not (session / "fake_pc").exists()                       # the launcher did not start the fake PC

    # start-up passed compare_hello against the C++ hello (mock true, schemas, F_T_TCP, limits, watchdog)
    doc = yaml.safe_load((session / "manager/session.yaml").read_text())
    assert doc["hello"]["mock"] is True
    events = jsonl(session / "manager/events.jsonl")
    assert any(e["kind"] == "ready" for e in events)
    assert not any(e["kind"] == "hello_check" for e in events)

    # one episode of 20 steps, chunk ids accepted on the PC's real clock
    episodes = jsonl(session / "manager/episodes.jsonl")
    assert [e["outcome"] for e in episodes] == ["success"] and episodes[0]["steps_sent"] >= 20
    steps = [s for s in jsonl(session / "manager/steps.jsonl") if s["sent"]]
    assert len(steps) >= 20 and all(s["rejected"] == 0 for s in steps)
    assert sum(s["accepted"] for s in steps) > 0
    ids = [s["chunk_id"] for s in steps]
    assert ids == sorted(ids)

    # home answered ok, then the stop of q
    kinds = [(e["kind"], e["detail"]) for e in events]
    assert ("home", "home pose reached") in kinds
    assert ("exit", "operator quit key") in kinds

    # the binary stops cleanly on SIGTERM
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 0


def test_a_rejection_names_the_failing_waypoint_not_the_first(franka_control_alone):
    proc, fc_dir = franka_control_alone
    link = LinkSettings.load()
    pc = PcClient(link.commands_endpoint(local=True), link.c3_timeout_s, 40.0, link.heartbeat_period_s)
    try:
        assert pc.start().ok
        state = pc.get_state()
        T = np.asarray(state.O_T_TCP_target, dtype=np.float64)
        _, sample = pc.hello()
        n = 5
        poses = np.repeat(T[None], n, axis=0)
        poses[3, 2, 3] -= 0.5                                    # waypoint 3 half a metre down: outside the box
        msg = SetWaypoints(chunk_id=1, t_ns=np.array([sample.t_pc_ns + (200 + 100 * i) * 1_000_000 for i in range(n)],
                                                     dtype=np.int64),
                           O_T_TCP=poses, gripper_width_m=np.full(n, float(state.gripper_width_m)))
        reply = pc.set_waypoints(msg)
        assert reply.accepted == 0 and reply.skipped == 0 and [i.index for i in reply.rejected] == list(range(n))
        assert [i.reason for i in reply.rejected if i.index != 3] == [NOT_CHECKED] * (n - 1)   # the contract text
        detail = gate_rejection_detail(reply)
        assert detail.startswith("3: outside workspace box by ") and " on z " in detail, detail
        assert pc.get_state().mode == "fault"
    finally:
        pc.close()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 0, (fc_dir / "franka_control.log").read_text()[-3000:]
