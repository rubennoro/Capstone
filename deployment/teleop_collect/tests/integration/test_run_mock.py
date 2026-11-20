"""`teleop_collect run --mock --no-browser` end to end (STRUCTURE.md 11, PLAN 10 "End to end"): the launcher starts the
fake PC, frames with the fake Pi playing station_helpers.RECORDED_EPISODE, teleop and the ui; ScriptedPage drives the page
over its WebSocket exactly as the browser's app.js does.

The start pose (final-deployment's start-pose work, 927a5ba): no laptop home. The fake PC (config/fake_pc.yaml)
starts at its start pose, as if the operator had pressed h, then p on the PC, and stands in for the operator after
every episode's stop: h, then p by itself auto_h_p_after_stop_s (1 s) later. teleop learns the pose at start-up, the
first start confirms it, and every later Start waits until the stand-in has brought the arm back (refused meanwhile)."""
from __future__ import annotations

import time

import numpy as np
from integration_helpers import (LauncherRun, alive, check_episode, episode_names, hold_speed, jsonl, mock_station,
                                 need_episode, need_pc_ports_free, newest_message_id, next_episode, open_page,
                                 read_csv, session_dir, start, stop, task_dir, tcp, wait_message, wait_stand_in)

from teleop_collect.teleop.episode_flow import NOT_AT_START_POSE

TASK = "pick_block"
#: Start refused while the fake PC's operator stand-in is bringing the arm back (h, then p after the stop).
BACK_SOON = (NOT_AT_START_POSE, "the PC is homing")


def refused_until_back(page) -> None:
    """Right after the stop of an episode that moved the arm: Start is refused (not at the start pose, or the
    stand-in's h/p runs)."""
    s = page.status()
    assert not s["start_allowed"] and s["start_refusal"] in BACK_SOON, s


def test_run_mock_collects_saves_discards_and_deletes(tmp_path):
    need_episode()
    need_pc_ports_free()
    with LauncherRun(mock_station(tmp_path)) as run:
        page = open_page(run.start())
        try:
            s = page.status()
            assert s["pc_mock"] and s["start_refusal"] == "select or create a task first"
            assert s["start_pose_known"] and not s["start_pose_confirmed"] and s["at_start_pose"], s
            assert page.command("create_task", name=TASK, description="pick the block, put it in the bowl")["ok"]
            page.wait_for(lambda s: s["task"] == TASK and TASK in s["tasks"], 5.0, "the new task selected")
            page.wait_for(lambda s: set(page.previews()) == {"main", "ultrawide"}, 10.0, "both previews")

            # --- episode_000: start (at the start-up pose), x+ for 1 s, close, open, save as success
            assert start(page) == "episode_000"
            assert page.status()["start_pose_confirmed"]                # the fake PC accepted it
            time.sleep(1.0)                                         # PLAN 4.3: the still scene first
            p0 = np.array(tcp(page))
            t0 = time.monotonic()
            page.hold(["x+"], 1.0)
            held_s = time.monotonic() - t0
            time.sleep(0.4)
            p1 = np.array(tcp(page))
            assert abs((p1[0] - p0[0]) - 0.05 * held_s) < 0.003, (p0, p1, held_s)   # v_max over the hold, +x only
            assert np.all(np.abs((p1 - p0)[1:]) < 1e-3), (p0, p1)
            page.gripper("close")
            page.wait_for(lambda s: s["commanded_width_m"] == 0.0 and s["gripper_width_m"] < 0.005, 5.0, "closed")
            page.gripper("open")
            page.wait_for(lambda s: s["commanded_width_m"] == 0.08 and s["gripper_width_m"] > 0.075, 5.0, "open")
            s = stop(page, "success")
            assert [e["name"] for e in s["episodes"]] == ["episode_000"] and s["episodes"][0]["outcome"] == "success"
            refused_until_back(page)

            meta = check_episode(task_dir(tmp_path, TASK) / "episode_000", task=TASK, outcome="success")
            folder = task_dir(tmp_path, TASK) / "episode_000"
            states = read_csv(folder / "robot_states.csv")
            xs = [float(r["O_T_TCP_target_03"]) for r in states]
            assert max(xs) - xs[0] > 0.035
            widths = {float(r["wp0_width_m"]) for r in read_csv(folder / "commands.csv")
                      if r["request"] == "c3.set_waypoints/v1"}
            assert {0.0, 0.08} <= widths
            assert any(r["held"] == "x+" for r in read_csv(folder / "keys.csv"))
            assert meta["counts"]["waypoints_rejected"] == 0 and meta["pc_hello"]["mock"] is True
            speed = hold_speed(folder)
            print(f"\nfake PC: x+ held {speed['held_s']:.3f} s, target slope {speed['slope_m_s'] * 1000:.1f} mm/s, "
                  f"travel {speed['travel_m'] * 1000:.1f} mm; page TCP moved {(p1 - p0)[0] * 1000:.1f} mm")
            assert 0.9 * 0.05 <= speed["slope_m_s"] <= 1.02 * 0.05, speed

            # --- episode_001: once the stand-in brought the arm back (h, then p): start, discard -> folder gone
            s = page.wait_start_allowed(10.0)
            assert s["at_start_pose"] and s["start_pose_confirmed"], s
            assert start(page) == "episode_001"
            time.sleep(0.5)
            stop(page, "discard")
            wait_stand_in(page)
            assert episode_names(tmp_path, TASK) == ["episode_000"]

            # --- episode_002: start, Stop arm -> aborted, folder gone, the app stays up
            assert start(page) == "episode_002"
            before = newest_message_id(page)
            assert page.command("stop_arm")["ok"]
            s = wait_message(page, "operator stop", before, 10.0)
            page.wait_for(lambda s: s["phase"] == "idle" and s["recording_episode"] == "", 10.0, "idle after Stop arm")
            wait_stand_in(page)                                         # after a PC stop as well
            assert episode_names(tmp_path, TASK) == ["episode_000"]
            assert run.poll() is None

            # --- episode_003: saved as fail, then deleted; numbers are never reused
            assert start(page) == "episode_003"
            time.sleep(0.8)
            s = stop(page, "fail")
            assert [(e["name"], e["outcome"]) for e in s["episodes"]] == [("episode_000", "success"),
                                                                       ("episode_003", "fail")]
            assert page.command("delete_episode", episode="episode_003")["ok"]
            page.wait_for(lambda s: [e["name"] for e in s["episodes"]] == ["episode_000"], 5.0, "the list after delete")
            assert episode_names(tmp_path, TASK) == ["episode_000"]
            refused = page.command("delete_episode", episode="episode_003")
            assert not refused["ok"] and refused["reason"]
            wait_stand_in(page)

            # --- episode_004 is open when Ctrl-C arrives: deleted, exit 0, nothing left running
            assert start(page) == "episode_004"
            time.sleep(0.5)
        finally:
            page.close()
        children = list(run.children)
        assert len(children) >= 4, children
        code = run.interrupt()
        assert code == 0, run.output[-4000:]
        assert "teleop_collect: stopped by Ctrl-C" in run.output
        assert not [pid for pid in children if alive(pid)]

    assert episode_names(tmp_path, TASK) == ["episode_000"]          # no incomplete folder
    assert next_episode(tmp_path, TASK) == 5
    session = session_dir(tmp_path)
    events = jsonl(session / "teleop" / "events.jsonl")
    texts = " | ".join(str(e.get("detail")) for e in events)
    # the launcher passes Ctrl-C on as SIGTERM; teleop names it that way (teleop/DESIGN.md "SIGTERM text")
    assert "episode_004: stopped by the launcher (Ctrl-C, or another process exited)" in texts, texts[-2000:]
    commands = jsonl(session / "fake_pc" / "commands.jsonl")
    assert commands[-1]["request"] == "c3.stop/v1"                     # Ctrl-C always sends stop
    assert not [c for c in commands if c["request"] == "c3.home/v1"]  # no laptop home: the PC's h and p only
    assert not [c for c in commands if c["request"] == "c3.start/v1" and not c.get("ok", True)]   # never refused
    assert "start pose: the arm's pose now" in texts and "confirmed: the PC accepted a start here" in texts
    assert all(c.get("rejected", 0) == 0 for c in commands if c["request"] == "c3.set_waypoints/v1")
