"""Each new failure row of STRUCTURE.md 8.5 end to end against an in-process FakePc (12.2): the real PcClient,
HeartbeatSender, PcListener, clock map and EpisodeLoop; C1/C4 are in-process fakes; the real clock."""
from __future__ import annotations

import json

import pytest
from manager_pc_harness import PcRig, small_link

from policy_deploy.manager.episode_loop import Phase
from policy_deploy.manager.problems import ProblemCode as C


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


def concluded(rig, code, timeout_s=5.0):
    rig.wait_done(timeout_s)
    assert rig.code is code, rig.verdict_text()
    assert rig.exit_code == 1
    return rig.loop.verdict


def stop_was_sent_last_and_episode_aborted(rig):
    assert rig.requests()[-1] == "c3.stop/v1"
    episodes = rig.episodes()
    assert episodes and episodes[-1]["outcome"] == "aborted"


def steps(rig):
    path = rig.session / "steps.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------------------------------------- the PC itself

def test_franka_control_restart_is_pc_restarted(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.fake.restart()
    v = concluded(rig, C.PC_RESTARTED)
    assert f"{v.what_failed}: {v.what_to_check}" == "franka_control restarted on the PC: restart policy_deploy run"


def test_c3_stall_is_pc_not_answering(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.fake.stall(0.4, c2=False)
    v = concluded(rig, C.PC_NOT_ANSWERING)
    assert v.what_failed.startswith("PC not answering: no reply to c3.")
    assert "within 0.3 s" in v.what_failed and v.what_to_check == "check franka_control on the PC and the link"


def test_c2_stall_with_c3_answering_is_state_lost(tmp_path, rigs):
    rig = rigs(tmp_path, link=small_link(state_lost_s=0.6))
    rig.start()
    rig.fake.stall(1.0, c3=False)
    v = concluded(rig, C.STATE_LOST)
    assert v.what_failed == "PC not answering: no C2 state for 0.6 s"


def test_repeated_t_ns_with_c2_arriving_is_not_state_lost(tmp_path, rigs):
    rig = rigs(tmp_path, link=small_link(state_lost_s=0.6))
    rig.start()
    rig.fake.repeat_t_ns(1.0)
    assert not rig.wait(lambda: rig.loop.phase is Phase.DONE, 1.3)
    assert rig.loop.phase is Phase.IDLE


def test_a_burst_of_old_states_after_a_stall_is_state_old_never_fresh(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    rig.fake.burst_after_stall()
    rig.fake.stall(0.4, c3=False)
    assert rig.wait(lambda: any(s["skip_reason"].startswith("state_old") for s in steps(rig)), 3.0)
    n = rig.loop._steps_sent
    assert rig.wait(lambda: rig.loop._steps_sent > n + 3, 3.0)          # resumes by itself after the burst
    assert rig.loop.phase is Phase.RUNNING


def test_watchdog_trip_is_pc_stopped(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    object.__setattr__(rig.fake.s, "watchdog_limit_s", 0.001)          # trips between two keep-alives
    v = concluded(rig, C.PC_STOPPED)
    assert f"{v.what_failed}: {v.what_to_check}" == ("the PC stopped the arm (watchdog or the PC's `s`): check the "
                                                     "link and the PC")
    stop_was_sent_last_and_episode_aborted(rig)


def test_the_pc_keyboard_stop_during_an_episode_is_pc_stopped(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    rig.fake.keyboard_stop()
    concluded(rig, C.PC_STOPPED)
    stop_was_sent_last_and_episode_aborted(rig)


def test_a_stopped_c2_older_than_the_mapped_start_ack_is_ignored(tmp_path, rigs):
    rig = rigs(tmp_path, fake={"start_mode": "stopped"})
    rig.start()
    rig.fake.delay_states(lambda: 0.08)              # stopped states still in flight after the start ack
    rig.start_episode(5)
    assert rig.wait(lambda: rig.loop._steps_sent >= 10, 3.0)
    assert rig.loop.phase is Phase.RUNNING and rig.code is None
    assert rig.loop._start_ack_pc_ns > rig.loop._start_ack_ns + 999 * 10**9    # mapped to the fake PC's +1000 s


# ---------------------------------------------------------------------------------------------- clocks

def test_a_clock_jump_of_2_s_is_clock_mismatch(tmp_path, rigs):
    rig = rigs(tmp_path, manager={"history_s": 4.0})
    rig.start()
    rig.start_episode(3)
    rig.fake.shift_clock(2.0)
    v = concluded(rig, C.CLOCK_MISMATCH)
    assert v.what_failed.startswith("clock mismatch: ")
    assert v.what_to_check == "run pc_probe; check clock.estimator"
    stop_was_sent_last_and_episode_aborted(rig)


def test_a_map_running_behind_is_clock_mismatch_by_future_c2(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    rig.pc_clock.add_hello = lambda s: None                            # the laptop's refits held
    rig.fake.shift_clock(0.1)
    v = concluded(rig, C.CLOCK_MISMATCH)
    assert "ahead of the mapped now" in v.what_failed
    accepted = [s for s in steps(rig) if s["sent"] and s["accepted"] > 0]
    assert accepted                                                     # waypoints accepted, but late


def test_one_skipped_set_is_only_policy_late(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    rig.policy.delays = [0.45]                                         # one step: every waypoint already past
    assert rig.wait(lambda: rig.monitor.counts().get(C.POLICY_LATE, 0) >= 1, 3.0)
    assert rig.wait(lambda: rig.loop._steps_sent >= 8, 3.0)
    assert rig.loop.phase is Phase.RUNNING and rig.code is None
    late = [s for s in steps(rig) if s["sent"] and s["accepted"] == 0]
    assert late and late[0]["skipped"] == 8


# ---------------------------------------------------------------------------------------------- home

def test_home_stopped_on_the_pc_is_normal_and_the_map_rewarms(tmp_path, rigs):
    rig = rigs(tmp_path, fake={"home_duration_s": 2.0})
    rig.start()
    rig.operator.press("h")
    assert rig.wait(lambda: rig.fake.mode == "homing", 3.0)
    rig.fake.keyboard_stop()
    assert rig.wait(lambda: rig.loop.phase is Phase.IDLE and "home_stopped" in [e["kind"] for e in rig.events()], 3.0)
    assert rig.code is None
    assert any(m.startswith("home stopped on the PC: stopped: ") for m in rig.operator.messages)
    assert rig.wait(lambda: "PC clock map ready again" in [e["detail"] for e in rig.events()], 3.0)
    assert not rig.heartbeats.in_startup_mode()
    rig.start_episode(3)                                                # c accepted afterwards
    assert rig.fake.mode in ("holding", "running")


def test_home_refused_is_homing_failed(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.fake.set_mode("holding")                                       # e.g. another client started an episode
    assert rig.wait(lambda: rig.history.latest().mode == "holding", 2.0)
    rig.operator.press("h")
    v = concluded(rig, C.HOMING_FAILED)
    assert v.what_failed == "the PC's homing failed: refused: mode is holding"


def test_home_timed_out_is_homing_failed(tmp_path, rigs):
    rig = rigs(tmp_path, manager={"timeouts": {"home_s": 0.4}},
               fake={"homing_timeout_s": 0.2, "home_duration_s": 3.0})
    rig.start()
    rig.operator.press("h")
    v = concluded(rig, C.HOMING_FAILED)
    assert v.what_failed == "the PC's homing failed: no reply within 0.4 s"


@pytest.mark.parametrize("hook", ["keyboard_home", "keyboard_hand_home"])
def test_c_and_h_refused_locally_while_the_pc_homes(tmp_path, rigs, hook):
    rig = rigs(tmp_path, fake={"home_duration_s": 1.0})
    rig.start()
    assert getattr(rig.fake, hook)().ok
    assert rig.wait(lambda: rig.history.latest().mode == "homing", 2.0)
    rig.operator.press("c")
    rig.operator.press("h")
    assert rig.wait(lambda: len(rig.operator.refused) >= 2, 2.0)
    hints = [h for _, h in rig.operator.refused]
    assert "c: the PC is homing (its h or g key): wait for idle" in hints
    assert "h: the PC is already homing (its h or g key)" in hints
    names = rig.requests()
    assert "c3.start/v1" not in names and "c3.home/v1" not in names
    assert rig.loop.phase is Phase.IDLE and rig.code is None
    assert rig.wait(lambda: rig.history.latest().mode == "idle", 5.0)
    rig.start_episode(2)                                                 # accepted again after idle


def test_start_refused_by_the_pc_is_a_hint(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.fake.set_mode("holding")
    rig.operator.press("c")
    assert rig.wait(lambda: "start_refused" in [e["kind"] for e in rig.events()], 3.0)
    assert rig.loop.phase is Phase.IDLE and rig.code is None
    assert "c: the PC refused start: refused: mode is holding" in rig.operator.messages


# ---------------------------------------------------------------------------------------------- refusals

def test_error_without_field_is_pc_refused(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    rig.fake.set_mode("idle")                                          # set_waypoints now refused by the mode
    v = concluded(rig, C.PC_REFUSED)
    assert v.what_failed.startswith("the PC refused c3.set_waypoints/v1: ")
    assert v.what_to_check == "check the PC terminal"
    stop_was_sent_last_and_episode_aborted(rig)


def test_gate_rejection_is_gate_rejected(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    rig.fake._gate._max_lead_ns = 1_000_000                            # every live waypoint is too far ahead
    v = concluded(rig, C.GATE_REJECTED)
    assert v.what_failed.startswith("the PC's gate rejected waypoint ") and "max_lead_s" in v.what_failed
    stop_was_sent_last_and_episode_aborted(rig)


def test_fault_during_an_episode_is_arm_fault(tmp_path, rigs):
    rig = rigs(tmp_path)
    rig.start()
    rig.start_episode(3)
    rig.fake.inject_fault("reflex: cartesian force")
    v = concluded(rig, C.ARM_FAULT)
    assert v.what_failed == "the PC reports fault: reflex: cartesian force"
    stop_was_sent_last_and_episode_aborted(rig)
