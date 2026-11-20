"""PLAN 10 "Across" (STRUCTURE.md 11): the laptop's processes against the REAL franka_control code with its fake arm,
the libfranka-less build already in ~/Capstone-final-deployment (only run, never built or changed; skipped if
missing). Its `git status --porcelain` must be the same before and after each test (nothing here changed it); it
need not be clean, since another session may be editing it. `teleop_collect run --mock` with `launcher: {fake_pc: false}`, so teleop talks to the binary on
127.0.0.1:5620/5630 exactly as to the PC.

The start pose (final-deployment's start-pose work, 927a5ba): the binary's station sets homing.q_start_rad and its
keyboard is a pipe, so the test presses the PC's h and p as the operator does: h, then p before `teleop_collect run`
(teleop learns that pose at start-up; the first start the PC accepts confirms it), and h, then p before the next
episode (Start refused on the page until then). The PC's own refusal away from its start pose is checked too: a C3
start from a second client after only h, and (second test) teleop's candidate path, a session started at home.

Also the speed measurement of PLAN 5.3 (E7 brief (d)): the slope of the C2 O_T_TCP_target while x+ is held, against
v_max."""
from __future__ import annotations

import time

import numpy as np
import pytest
from integration_helpers import (PC_NOT_AT_START_POSE_TEXT, FrankaControl, LauncherRun, check_episode,
                                 final_deployment_status, hold_speed, jsonl, mock_station, need_episode,
                                 need_pc_ports_free, open_page, read_csv, session_dir, start, stop, task_dir, tcp)

from teleop_collect.teleop.episode_flow import NOT_AT_START_POSE, START_REFUSED_HERE

TASK = "cross_check"
V_MAX = 0.05


@pytest.fixture
def franka_control(tmp_path):
    need_episode()
    need_pc_ports_free()
    before = final_deployment_status()
    with FrankaControl(tmp_path) as fc:
        yield fc
    assert final_deployment_status() == before, "~/Capstone-final-deployment changed during the test"


def test_teleop_collect_against_the_real_franka_control(tmp_path, franka_control):
    fc = franka_control
    times = {}
    # the operator before `teleop_collect run`: h, then p on the PC (the fake arm starts at home)
    times["h"] = fc.press("h")
    times["p"] = fc.press("p")
    with LauncherRun(mock_station(tmp_path, launcher={"fake_pc": False})) as run:
        t0 = time.monotonic()
        page = open_page(run.start())
        times["start-up"] = time.monotonic() - t0
        try:
            s = page.status()
            assert s["pc_mock"] is True
            assert s["start_pose_known"] and not s["start_pose_confirmed"] and s["at_start_pose"], s
            assert page.command("create_task", name=TASK, description="the cross test")["ok"]

            assert start(page) == "episode_000"                       # the PC accepts it: at its start pose
            assert page.status()["start_pose_confirmed"]
            time.sleep(1.0)
            p0 = np.array(tcp(page))
            page.hold(["x+"], 1.0)
            time.sleep(0.5)
            p1 = np.array(tcp(page))
            page.gripper("close")
            page.wait_for(lambda s: s["commanded_width_m"] == 0.0, 5.0, "closing")
            time.sleep(1.0)
            page.gripper("open")
            time.sleep(1.0)
            stop(page, "success")

            # the arm moved: Start refused here until the PC's h, then p
            page.wait_for(lambda s: not s["start_allowed"] and s["start_refusal"] == NOT_AT_START_POSE, 5.0,
                          "Start refused away from the start pose")
            times["h2"] = fc.press("h")
            s = page.wait_for(lambda s: s["phase"] == "idle" and s["arm_mode"] == "idle", 5.0, "idle at home")
            assert not s["start_allowed"] and s["start_refusal"] == NOT_AT_START_POSE, s
            refused = fc.start_direct()                               # the PC's own refusal: at home, not the start
            assert not refused.ok and refused.reason == PC_NOT_AT_START_POSE_TEXT, refused
            assert page.status()["start_pose_confirmed"]              # a refusal never moves a confirmed pose
            times["p2"] = fc.press("p")
            t_p = time.monotonic()
            s = page.wait_start_allowed(5.0)
            times["p2 -> Start allowed"] = time.monotonic() - t_p
            assert s["at_start_pose"] and s["start_pose_confirmed"], s
            assert start(page) == "episode_001"
            time.sleep(0.3)
            stop(page, "discard")
        finally:
            page.close()
        assert run.interrupt() == 0, run.output[-4000:]

    assert not (session_dir(tmp_path) / "fake_pc").exists()          # the launcher did not start the fake PC
    folder = task_dir(tmp_path, TASK) / "episode_000"
    meta = check_episode(folder, task=TASK, outcome="success")
    assert meta["pc_hello"]["mock"] is True
    assert meta["counts"]["waypoints_rejected"] == 0
    assert not (task_dir(tmp_path, TASK) / "episode_001").exists()
    sets = [r for r in read_csv(folder / "commands.csv") if r["request"] == "c3.set_waypoints/v1"]
    assert sets and all(r["rejected"] in ("0", "") for r in sets)
    assert sum(int(r["accepted"] or 0) for r in sets) > 0
    assert {0.0, 0.08} <= {float(r["wp0_width_m"]) for r in sets}

    speed = hold_speed(folder)
    print(f"\nfranka_control --mock --local: x+ held {speed['held_s']:.3f} s, target slope "
          f"{speed['slope_m_s'] * 1000:.1f} mm/s ({speed['slope_m_s'] / V_MAX:.0%} of v_max), travel "
          f"{speed['travel_m'] * 1000:.1f} mm; page TCP moved {(p1 - p0)[0] * 1000:.1f} mm")
    print("timings: " + ", ".join(f"{k} {v:.2f} s" for k, v in times.items()))
    assert speed["slope_m_s"] >= 0.9 * V_MAX, speed
    assert speed["slope_m_s"] <= 1.02 * V_MAX, speed                  # never faster than v_max (PLAN 5.3)

    texts = [str(e.get("detail")) for e in jsonl(session_dir(tmp_path) / "teleop" / "events.jsonl")]
    assert any(t.startswith("start pose: the arm's pose now") for t in texts), texts
    assert any(t.startswith("confirmed: the PC accepted a start here") for t in texts), texts
    log = fc.log()
    assert log.count("start pose reached") == 2 and log.count("homing finished") == 2, log[-3000:]
    assert fc.terminate() == 0, fc.log()[-3000:]


def test_a_session_started_away_from_the_pcs_start_pose_takes_the_pose_after_its_p(tmp_path, franka_control):
    """The safeguard (the user's decision 3): `teleop_collect run` with the arm at home (p not pressed). teleop takes
    that pose as the candidate and allows Start; the PC refuses it (not at the start pose), teleop drops the candidate
    and says so on the page; after the PC's p the pose there is the candidate, and the PC's accepting start confirms
    it."""
    fc = franka_control
    with LauncherRun(mock_station(tmp_path, launcher={"fake_pc": False})) as run:
        page = open_page(run.start())
        try:
            s = page.status()
            assert s["start_pose_known"] and not s["start_pose_confirmed"], s
            assert page.command("create_task", name=TASK, description="the cross test")["ok"]
            page.wait_start_allowed(20.0)                              # at the start-up pose: teleop allows it
            ack = page.command("start_episode")
            assert not ack["ok"] and ack["reason"] == f"the PC refused start: {PC_NOT_AT_START_POSE_TEXT}", ack
            s = page.wait_for(lambda s: s["start_refusal"] == START_REFUSED_HERE, 5.0, "the refused-here hint")
            assert not s["start_pose_known"] and not s["start_allowed"] and s["recording_episode"] == "", s

            fc.press("p")
            s = page.wait_start_allowed(5.0)
            assert s["start_pose_known"] and not s["start_pose_confirmed"] and s["at_start_pose"], s
            assert start(page) == "episode_001"                       # episode_000's number went with the refusal
            assert page.status()["start_pose_confirmed"]
            time.sleep(0.3)
            stop(page, "discard")
        finally:
            page.close()
        assert run.interrupt() == 0, run.output[-4000:]

    assert not list(task_dir(tmp_path, TASK).glob("episode_*"))
    texts = [str(e.get("detail")) for e in jsonl(session_dir(tmp_path) / "teleop" / "events.jsonl")]
    assert any("the PC refused a start here (not its start pose): dropped" in t for t in texts), texts
    assert any(t.startswith("candidate: the pose after the PC's move") for t in texts), texts
    assert any(t.startswith("confirmed: the PC accepted a start here") for t in texts), texts
    assert fc.terminate() == 0, fc.log()[-3000:]
