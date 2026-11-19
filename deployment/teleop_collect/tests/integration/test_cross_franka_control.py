"""PLAN 10 "Across" (STRUCTURE.md 11): the laptop's processes against the REAL franka_control code with its fake arm,
the libfranka-less build already in ~/Capstone-final-deployment (only run, never built or changed; skipped if
missing). `teleop_collect run --mock` with `launcher: {fake_pc: false}`, so teleop talks to the binary on
127.0.0.1:5620/5630 exactly as to the PC.

Also the speed measurement of PLAN 5.3 (E7 brief (d)): the slope of the C2 O_T_TCP_target while x+ is held, against
v_max."""
from __future__ import annotations

import time

import numpy as np
import pytest
from integration_helpers import (FrankaControl, LauncherRun, check_episode, final_deployment_status, hold_speed, home,
                                 jsonl, mock_station, need_episode, need_pc_ports_free, open_page, read_csv,
                                 session_dir, start, stop, task_dir, tcp)

TASK = "cross_check"
V_MAX = 0.05


@pytest.fixture
def franka_control(tmp_path):
    need_episode()
    need_pc_ports_free()
    before = final_deployment_status()
    with FrankaControl(tmp_path) as fc:
        yield fc
    assert final_deployment_status() == before == "", "~/Capstone-final-deployment changed"


def test_teleop_collect_against_the_real_franka_control(tmp_path, franka_control):
    fc = franka_control
    with LauncherRun(mock_station(tmp_path, launcher={"fake_pc": False})) as run:
        page = open_page(run.start())
        try:
            s = page.status()
            assert s["pc_mock"] is True and s["home_known"]           # the start-up home on the real code
            assert page.command("create_task", name=TASK, description="the cross test")["ok"]

            home(page)
            assert start(page) == "episode_000"
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

            home(page)
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
    assert speed["slope_m_s"] >= 0.9 * V_MAX, speed
    assert speed["slope_m_s"] <= 1.02 * V_MAX, speed                  # never faster than v_max (PLAN 5.3)

    fc_events = jsonl(session_dir(tmp_path) / "teleop" / "events.jsonl")
    assert any("home pose reached" in str(e.get("detail")) for e in fc_events)
    assert fc.terminate() == 0, fc.log()[-3000:]
