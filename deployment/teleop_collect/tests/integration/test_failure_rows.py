"""The rows of PLAN 7 that need more than one process (STRUCTURE.md 11), on the real laptop processes.

The PC is an in-process fake PC on link.yaml's 127.0.0.1:5620/5630 (so the test can call its hooks) and
`teleop_collect run --mock` runs with `launcher: {fake_pc: false}`. For the camera row the Pi is an in-process fake
Pi too, and frames, teleop and ui are started one by one with the launcher's own process classes (frames without
--mock, reading the fake Pi on the station's pi.port)."""
from __future__ import annotations

import time

import numpy as np
import pytest
from integration_helpers import (InProcessFakePc, InProcessFakePi, LauncherRun, LaptopProcesses, alive, episode_names,
                                 home, jsonl, mock_station, need_episode, need_pc_ports_free, newest_message_id,
                                 open_page, session_dir, start, stop, wait_message, wait_until)
from station_helpers import free_port, write_station

TASK = "rows"


def _ready_task(page) -> None:
    assert page.command("create_task", name=TASK, description="the failure rows")["ok"]
    page.wait_for(lambda s: s["task"] == TASK, 5.0, "the task selected")


def test_a_camera_down_mid_episode_aborts_it_and_the_app_stays_up(tmp_path):
    """PLAN 7: a camera down (2 s) during an episode -> aborted and deleted; between episodes Start is refused while
    the camera is down, and allowed again once it is back."""
    need_episode()
    need_pc_ports_free()
    pi_port = free_port()
    station = write_station(tmp_path, pi_address="127.0.0.1")
    text = station.read_text().replace("port: 5555", f"port: {pi_port}")
    station.write_text(text)
    with InProcessFakePc(tmp_path), InProcessFakePi(pi_port) as pi, LaptopProcesses(station) as procs:
        page = open_page(procs.url)
        try:
            _ready_task(page)
            home(page)
            name = start(page)
            time.sleep(0.5)
            before = newest_message_id(page)
            pi.pause()                                                  # both cameras stop
            s = wait_message(page, "main camera down", before, 10.0)
            msg = next(m for m in s["messages"] if m["id"] > before and "main camera down" in m["text"])
            assert "check the Pi and ASUS_5g" in msg["text"], msg
            s = page.wait_for(lambda s: s["phase"] == "idle" and s["recording_episode"] == "", 10.0, "idle")
            assert name not in episode_names(tmp_path, TASK)
            assert not s["start_allowed"] and "camera" in s["start_refusal"], s["start_refusal"]
            refused = page.command("start_episode")
            assert not refused["ok"] and "camera" in refused["reason"], refused
            assert episode_names(tmp_path, TASK) == []

            pi.resume()                                                 # the app is still up: the next episode works
            page.wait_for(lambda s: s["start_allowed"], 20.0, "start allowed again")
            name2 = start(page)
            assert name2 != name
            time.sleep(0.6)
            stop(page, "success")
            assert episode_names(tmp_path, TASK) == [name2]
            assert all(p.returncode() is None for p in procs.processes)
        finally:
            page.close()
        codes = procs.stop()
    assert codes == {"ui": 0, "teleop": 0, "frames": 0}, codes


def _unfocused_hold(page, seconds: float) -> None:
    """x+ held but the page without focus: the state resent every 50 ms as the page would, with focus=false."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        page.keys(["x+"], focus=False)
        time.sleep(0.05)


def _still(fake, seconds: float) -> list[float]:
    samples = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        samples.append(float(fake.target[0, 3]))
        time.sleep(0.05)
    return samples


def test_blur_stop_arm_and_the_pc_s_key_each_end_motion_or_the_episode(tmp_path):
    """PLAN 5.4, 7: keys held while the page has no focus never move the arm, and losing focus mid-hold stops it
    (the C2 target stops) even while the key state keeps arriving; an operator Stop arm and the PC's `s` during an
    episode -> aborted and deleted; the app stays up through all of them."""
    need_episode()
    need_pc_ports_free()
    with InProcessFakePc(tmp_path) as fake, LauncherRun(mock_station(tmp_path, launcher={"fake_pc": False})) as run:
        page = open_page(run.start())
        try:
            _ready_task(page)

            # --- x+ held without focus from the start: no motion at all
            home(page)
            start(page)
            time.sleep(0.3)
            x_start = float(fake.target[0, 3])
            _unfocused_hold(page, 1.0)
            samples = _still(fake, 0.3)
            assert max(abs(x - x_start) for x in samples + [float(fake.target[0, 3])]) < 1e-9, (x_start, samples)
            assert page.status()["moving"] is False

            # --- focus lost mid-hold, the key state still arriving (focus=false): the target stops
            x0 = float(fake.target[0, 3])
            for _ in range(10):                                      # x+ held 0.5 s with focus
                page.keys(["x+"])
                time.sleep(0.05)
            x_blur = float(fake.target[0, 3])
            assert x_blur - x0 > 0.015, (x0, x_blur)
            _unfocused_hold(page, 0.3)                               # blur: the page keeps sending, focus=false
            x_settled = float(fake.target[0, 3])
            assert x_settled - x_blur < 0.0055, (x_blur, x_settled)  # at most the last plan's 100 ms lead
            end = time.monotonic() + 1.0
            samples = []
            while time.monotonic() < end:                            # still x+ held, unfocused, for 1 s more
                page.keys(["x+"], focus=False)
                samples.append(float(fake.target[0, 3]))
                time.sleep(0.05)
            assert np.ptp(samples) < 1e-9, (min(samples), max(samples))
            assert page.status()["moving"] is False
            page.keys([])
            s = stop(page, "success")
            assert len(s["episodes"]) == 1

            # --- Stop arm
            home(page)
            name = start(page)
            before = newest_message_id(page)
            assert page.command("stop_arm")["ok"]
            wait_message(page, "operator stop", before, 10.0)
            page.wait_for(lambda s: s["phase"] == "idle", 10.0, "idle after Stop arm")
            assert name not in episode_names(tmp_path, TASK)
            assert fake.mode == "stopped"

            # --- the PC's s
            home(page)
            name = start(page)
            time.sleep(0.3)
            before = newest_message_id(page)
            assert fake.keyboard_stop().ok
            wait_message(page, "the PC stopped the arm", before, 10.0)
            page.wait_for(lambda s: s["phase"] == "idle", 10.0, "idle after the PC's s")
            assert name not in episode_names(tmp_path, TASK)
            assert run.poll() is None
            assert len(episode_names(tmp_path, TASK)) == 1
        finally:
            page.close()
        assert run.interrupt() == 0, run.output[-3000:]


def _conclude(tmp_path, hook, fatal: str) -> list[str]:
    """Start an episode, call hook(fake), expect the launcher to end with teleop's FATAL line, exit 1, the open
    episode deleted. Returns the C3 requests the fake PC answered."""
    need_episode()
    need_pc_ports_free()
    with InProcessFakePc(tmp_path) as fake, LauncherRun(mock_station(tmp_path, launcher={"fake_pc": False})) as run:
        page = open_page(run.start())
        try:
            _ready_task(page)
            home(page)
            name = start(page)
            time.sleep(0.5)
            hook(fake)
            code = run.wait(30.0)
        finally:
            page.close()
        out = run.output
        assert code == 1, out[-3000:]
        line = next((ln for ln in out.splitlines() if ln.startswith("teleop_collect: FAILED teleop")), None)
        assert line is not None, out[-3000:]
        assert fatal in line and "FATAL teleop:" in line, line
        assert not [p for p in run.children if alive(p)]
        requests = fake.request_names()
    assert name not in episode_names(tmp_path, TASK)
    assert episode_names(tmp_path, TASK) == []
    events = jsonl(session_dir(tmp_path) / "teleop" / "events.jsonl")
    assert any(fatal in str(e.get("detail")) for e in events), events[-5:]
    return requests


@pytest.mark.parametrize("c3, fatal", [(False, "PC not answering: no C2 state for 2.0 s"),
                                       (True, "PC not answering: no reply to c3.")],
                         ids=["no_c2_for_2_s", "c3_timeout"])
def test_the_pc_going_silent_concludes_pc_not_answering(tmp_path, c3, fatal):
    """PLAN 7: no C2 for 2 s (state_lost_s), or a C3 timeout (c3_timeout_s, 0.3 s) -> conclude, "PC not answering:
    check franka_control on the PC and the link"."""
    _conclude(tmp_path, lambda fake: fake.stall(4.0, c2=True, c3=c3), fatal)


def test_the_pc_restarting_concludes_with_its_fatal_line(tmp_path):
    """PLAN 7: C2 seq going back -> conclude, "franka_control restarted on the PC"."""
    requests = _conclude(tmp_path, lambda fake: fake.restart(), "franka_control restarted on the PC")
    assert requests[-1] == "c3.stop/v1"


def test_a_pc_fault_concludes_with_its_fatal_line(tmp_path):
    """PLAN 7: the PC reports fault -> conclude; the PC holds."""
    requests = _conclude(tmp_path, lambda fake: fake.inject_fault("cartesian reflex"),
                         "the PC reports fault: cartesian reflex")
    assert requests[-1] == "c3.stop/v1"
