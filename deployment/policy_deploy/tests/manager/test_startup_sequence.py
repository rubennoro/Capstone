"""The manager's start-up order (STRUCTURE.md 6.3, 12.2) with an in-process FakePc and the real clock."""
from __future__ import annotations

import threading
import time

import pytest
from manager_pc_harness import PcRig, small_link

from policy_deploy.contracts import load_config
from policy_deploy.manager.episode_loop import Phase
from policy_deploy.manager.problems import ProblemCode as C

# a 0.5 s watchdog (the H2 limit is 2 x watchdog); the laptop's worst gap must stay below it
FAST_LINK = small_link(watchdog_limit_s=0.5, c3_timeout_s=0.15)
FAST_GAP = {"link_jitter_allowance_s": 0.05}


@pytest.fixture
def rigs():
    made = []

    def make(tmp_path, **kw):
        rig = PcRig(tmp_path, **kw)
        made.append(rig)
        return rig
    yield make
    for rig in made:
        rig.close()


def test_the_launcher_timeout_exceeds_the_longest_legal_start_up():
    launcher = load_config("launcher", None)["ready_timeout_s"]["manager"]
    manager = load_config("manager", None)["timeouts"]
    homing_timeout_s = load_config("fake_pc", None)["homing_timeout_s"]
    longest = manager["startup_s"] + manager["pc_homing_wait_s"] + homing_timeout_s + 5.0
    assert longest == 115.0 and launcher == 130.0 and launcher > longest


def test_the_order_hellos_then_get_state_then_heartbeats(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    names = rig.requests()
    first_get_state = names.index("c3.get_state/v1")
    assert names[0] == "c3.hello/v1"
    assert set(names[:first_get_state]) == {"c3.hello/v1"}                # a hello every cycle in step 4
    assert first_get_state >= rig.settings.clock.min_samples
    assert rig.wait(lambda: "c3.heartbeat/v1" in rig.requests(), 2.0)     # heartbeats in the other cycles
    kinds = [e["kind"] for e in rig.events()]
    assert kinds[0] == "startup" and "ready" in kinds
    assert (rig.session / "session.yaml").exists()
    assert rig.operator.started and rig.loop.phase is Phase.IDLE


def test_h2_an_orphaned_holding_turns_stopped_and_start_up_goes_on(tmp_path, rigs):
    rig = rigs(tmp_path, link=FAST_LINK, manager=FAST_GAP, fake={"start_mode": "holding"})
    t0 = time.monotonic_ns()
    rig.start()
    first_c3 = rig.fake.requests[0][0]
    assert (first_c3 - t0) / 1e9 >= 0.5 - 0.05                          # nothing on C3 until the watchdog tripped
    assert rig.fake.mode in ("stopped", "holding", "running")
    states = [e["detail"] for e in rig.events() if e["kind"] == "startup"]
    assert "C2 mode holding" in states and "C2 mode stopped" in states


def test_h2_a_holding_that_stays_is_another_client(tmp_path, rigs):
    rig = rigs(tmp_path, link=FAST_LINK, manager=FAST_GAP, fake={"start_mode": "holding"})
    rig.fake.hold_forever()                                             # another client keeps feeding the watchdog
    t0 = time.monotonic()
    rig.startup()
    assert rig.code is C.PC_BUSY, rig.verdict_text()
    assert 1.0 - 0.05 <= time.monotonic() - t0 < 3.0                     # 2 x watchdog
    assert rig.loop.verdict.what_failed == "the PC is driving an episode for another C3 client"
    assert rig.requests() == []                                         # nothing sent, not even stop


def test_homing_at_start_sends_nothing_until_idle(tmp_path, rigs):
    rig = rigs(tmp_path, fake={"start_mode": "homing", "home_duration_s": 0.8})
    t0 = time.monotonic_ns()
    rig.start()
    assert (rig.fake.requests[0][0] - t0) / 1e9 >= 0.7
    assert rig.loop.phase is Phase.IDLE


def test_homing_past_the_wait_limit(tmp_path, rigs):
    rig = rigs(tmp_path, manager={"timeouts": {"pc_homing_wait_s": 0.3}},
               fake={"start_mode": "homing", "home_duration_s": 5.0})
    rig.startup()
    assert rig.code is C.STARTUP and "still homing after 0.3 s" in rig.loop.verdict.what_failed
    assert rig.requests() == []


def test_homing_at_step_6_waits_then_goes_on(tmp_path, rigs):
    rig = rigs(tmp_path, fake={"home_duration_s": 1.0})

    def press_h_on_the_pc() -> None:
        if rig.wait(lambda: len(rig.fake.requests) > 0, 5.0):
            rig.fake.keyboard_home()
    threading.Thread(target=press_h_on_the_pc, daemon=True).start()
    rig.start()
    details = [e["detail"] for e in rig.events()]
    assert "the PC is homing: waiting" in details
    names = rig.requests()
    assert names.count("c3.get_state/v1") >= 2                           # asked again while homing
    assert rig.fake.mode == "idle"


def test_fault_at_start_is_refused_at_once(tmp_path, rigs):
    rig = rigs(tmp_path, fake={"start_mode": "fault"})
    t0 = time.monotonic()
    rig.startup()
    assert rig.code is C.ARM_FAULT and time.monotonic() - t0 < 1.0
    assert "fake fault (start_mode)" in rig.loop.verdict.what_failed
    assert rig.requests() == []


def test_no_c2_is_pc_not_answering(tmp_path, rigs):
    from station_helpers import free_port
    rig = rigs(tmp_path, manager={"timeouts": {"startup_s": 0.4}}, state_endpoint=f"tcp://127.0.0.1:{free_port()}")
    rig.startup()
    v = rig.loop.verdict
    assert rig.code is C.PC_NOT_ANSWERING
    assert v.what_failed.startswith("PC not answering: no C2 state from tcp://127.0.0.1:")
    assert "control_deploy run" in v.what_to_check
    assert rig.requests() == []


def test_hello_mismatch_is_refused(tmp_path, rigs):
    rig = rigs(tmp_path, fake_link=small_link(watchdog_limit_s=0.8))     # the PC's watchdog differs from ours
    rig.startup()
    assert rig.code is C.HELLO_MISMATCH, rig.verdict_text()
    assert rig.loop.verdict.what_failed.startswith("hello mismatch: watchdog_limit_s: PC 0.8")
    assert rig.requests()[-1] == "c3.stop/v1"                           # hellos had started: stop sent once


def test_ready_file_only_after_cameras_ok_and_config_checked(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.frames.ultrawide = "stale"
    ready = tmp_path / "ready" / "manager.ready"
    t = threading.Thread(target=rig.startup, args=(ready,), daemon=True)
    t.start()
    assert rig.wait(lambda: "c3.get_state/v1" in rig.requests(), 5.0)
    time.sleep(0.3)                                                     # config checked, cameras still stale
    assert not ready.exists() and rig.loop.phase is Phase.STARTING
    rig.frames.ultrawide = "ok"
    t.join(timeout=5.0)
    assert ready.exists() and rig.loop.phase is Phase.IDLE


def test_cameras_never_ok_is_a_startup_failure(tmp_path, rigs):
    rig = rigs(tmp_path, manager={"timeouts": {"startup_s": 1.0}})
    rig.frames.ultrawide = "stale"
    rig.startup()
    assert rig.code is C.STARTUP and "no camera ok within 1 s (ultrawide stale)" in rig.loop.verdict.what_failed


def test_settled_rule_with_a_mock_pc_passes(tmp_path, rigs):
    rig = rigs(tmp_path, kind="diffusion")                                 # frames.yaml settled false, PC mock
    rig.start()
    assert rig.loop.phase is Phase.IDLE
