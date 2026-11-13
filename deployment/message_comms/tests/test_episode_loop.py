"""EpisodeLoop with fakes, FakeClock, ManualWorker and scripted keys, driving cycle() (STRUCTURE.md section 11)."""
import io
import json

import numpy as np
import pytest

from contracts import Ack, ContractError
from fakes import health, policy_config, rejecting_reply
from harness import LoopHarness
from message_comms.episode_loop import Phase
from message_comms.problems import Kind, ProblemCode as C
from message_comms.scripted_operator import ScriptedOperator, parse_script

MS = 1_000_000


@pytest.fixture
def h(settings):
    harness = LoopHarness(settings)
    harness.start()
    assert harness.loop.phase is Phase.IDLE, harness.loop.verdict
    return harness


def episodes(settings):
    path = settings.session_dir / "episodes.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def set_cameras(h, main="ok", ultrawide="ok"):
    h.frames.report = h.health.report = health(main, ultrawide)


def stop_times(h):
    return [t for name, t in h.franka.timed if name == "stop"]


# ---- continuous control -------------------------------------------------------------------------------------------

def test_next_chunk_asked_while_the_current_one_runs(h):
    h.start_episode()
    h.tick(10)
    sent_at = [t for t, _ in h.franka.sent]
    step_jobs = {job[2] for job in h.worker.jobs if job[0] == "step"}
    assert len(sent_at) >= 5
    for t, msg in h.franka.sent:
        assert t in step_jobs                       # the next step was submitted in the same cycle (same time)
        assert t < msg.t_ns[-1]                     # ... before the last waypoint's time


def test_waypoints_replace_correctly(h):
    h.start_episode()
    h.tick(15)
    ids = [msg.chunk_id for _, msg in h.franka.sent]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    jobs = [job[1].chunk_id for job in h.worker.jobs if job[0] == "step"]
    assert len(jobs) == len(set(jobs))


def test_a_result_from_an_ended_episode_is_dropped(h):
    h.start_episode()
    h.worker.hold = True
    h.tick(2)
    sent = len(h.franka.sent)
    assert h.worker.busy()
    h.operator.press("y")
    h.tick()
    assert h.loop.phase is Phase.IDLE
    h.worker.release()
    h.tick(3)
    assert len(h.franka.sent) == sent


def test_heartbeats_during_a_slow_step(h):
    h.start_episode()
    h.worker.delay_s = 2.0
    h.tick(3)
    before = h.count("heartbeat")
    h.tick(100)                                     # 2 s: the step is still held back
    assert h.count("heartbeat") - before == 100
    assert h.loop.phase is Phase.RUNNING


@pytest.mark.parametrize("key", ["s", "q"])
def test_stop_in_the_same_cycle_while_a_step_runs(h, key):
    h.start_episode()
    h.worker.hold = True
    h.tick(2)
    assert h.worker.busy()
    h.operator.press(key)
    before = h.clock()
    h.tick()
    assert stop_times(h)[-1] >= before and stop_times(h)[-1] == h.clock()


# ---- keys and episodes --------------------------------------------------------------------------------------------

def test_s_ends_the_episode_and_the_run_goes_on(h, settings):
    h.start_episode()
    h.tick(5)
    h.operator.press("s")
    h.tick()
    assert h.loop.phase is Phase.IDLE and h.franka.calls.count("stop") == 1
    assert [e["outcome"] for e in episodes(settings)] == ["stopped"]
    assert episodes(settings)[0]["reason"] == "operator stop key"
    h.run_until(lambda: not h.worker.busy())
    h.operator.press("h")
    h.tick()
    assert "home" in h.calls and h.loop.phase is Phase.IDLE
    resets = h.policy.calls.count("reset")
    h.start_episode()
    assert h.loop.progress().episode_id == 2 and h.policy.calls.count("reset") == resets + 1
    h.tick(5)
    assert h.loop.phase is Phase.RUNNING and len(h.franka.sent) > 5


def test_s_in_resetting_cancels_the_start(h, settings):
    h.worker.delay_s = 1.0
    h.operator.press("c")
    h.tick()
    assert h.loop.phase is Phase.RESETTING
    h.operator.press("s")
    h.tick()
    assert h.loop.phase is Phase.IDLE and "stop" in h.calls
    h.worker.delay_s = 0.0
    h.tick(5)
    assert "start" not in h.calls and episodes(settings) == [] and h.loop.progress().episode_id == 0


def test_s_in_idle_only_sends_stop(h, settings):
    h.operator.press("s")
    h.tick()
    assert h.loop.phase is Phase.IDLE and h.calls.count("stop") == 1
    assert any("no episode running" in m for m in h.operator.messages)
    assert episodes(settings) == []


@pytest.mark.parametrize("key, outcome", [("y", "success"), ("n", "fail")])
def test_episode_records_the_outcome(h, settings, key, outcome):
    h.start_episode()
    h.tick(5)
    h.operator.press(key)
    h.tick()
    e = episodes(settings)[0]
    assert e["outcome"] == outcome and e["episode_id"] == 1 and e["steps_sent"] >= 4


def test_q_ends_the_run_with_exit_0(h, settings):
    h.start_episode()
    h.tick(3)
    h.operator.press("q")
    h.tick()
    assert h.loop.phase is Phase.DONE and h.loop.run() == 0
    assert h.calls[-1] == "stop" or "stop" in h.calls
    assert episodes(settings)[0]["outcome"] == "stopped" and episodes(settings)[0]["reason"] == "operator quit key"
    assert h.loop.verdict.kind is Kind.STOP and h.operator.closed and h.listener.stopped and h.worker.stopped


def test_ctrl_c_stops_then_exits_0(h, settings):
    h.start_episode()
    h.tick(3)
    h.loop.interrupt()
    h.tick()
    assert h.loop.phase is Phase.DONE and h.loop.run() == 0
    assert "stop" in h.calls
    assert episodes(settings)[0]["outcome"] == "stopped" and episodes(settings)[0]["reason"] == "Ctrl-C"


def test_sigterm_stops_records_aborted_exits_0(h, settings):
    h.start_episode()
    h.tick(3)
    h.loop.terminate()
    h.tick()
    assert h.loop.phase is Phase.DONE and h.loop.run() == 0 and "stop" in h.calls
    e = episodes(settings)[0]
    assert e["outcome"] == "aborted" and e["reason"] == "stopped by deploy: another process exited"


def test_wrong_phase_keys_give_hints(h):
    h.operator.press("h")
    h.run_until(lambda: h.loop.phase is Phase.IDLE)
    h.start_episode()
    h.operator.press("h", "c")
    h.tick(2)
    assert "h: end the episode first (y / n / s)" in h.operator.messages
    assert any(m.startswith("c: an episode is running") for m in h.operator.messages)
    assert h.loop.phase is Phase.RUNNING


def test_c_waits_for_the_last_step(h):
    h.start_episode()
    h.worker.hold = True
    h.tick(2)
    h.operator.press("y")
    h.tick()
    h.operator.press("c")
    h.tick()
    assert "c: waiting for the last policy step to finish" in h.operator.messages and h.loop.phase is Phase.IDLE


# ---- prompt, start refused ----------------------------------------------------------------------------------------

def test_prompt_only_when_uses_prompt(settings, make_settings):
    plain = LoopHarness(settings)
    plain.start()
    plain.start_episode()
    assert not any(isinstance(c, tuple) for c in plain.calls)

    s = make_settings(task_name="raise_block", prompt_index=2)
    hp = LoopHarness(s)
    hp.policy.config_reply = policy_config(uses_prompt=True)
    hp.start()
    hp.start_episode()
    order = [c if isinstance(c, str) else c[0] for c in hp.calls if c in ("reset", "start") or isinstance(c, tuple)]
    assert order == ["reset", "prompt", "start"]
    assert ("prompt", "raise_block", 2) in hp.calls


def test_uses_prompt_without_task_name_concludes(settings):
    hp = LoopHarness(settings)
    hp.policy.config_reply = policy_config(uses_prompt=True)
    hp.start()
    assert hp.loop.phase is Phase.DONE and hp.loop.verdict.what_to_check == "check message_comms.task_name"


def test_refused_start_returns_to_idle_without_an_episode(h, settings):
    h.franka.start_ack = Ack(ok=False, reason="refused: mode is homing")
    h.operator.press("c")
    h.tick(3)
    assert h.loop.phase is Phase.IDLE and h.loop.progress().episode_id == 0 and episodes(settings) == []
    events = (settings.session_dir / "events.jsonl").read_text()
    assert "start_refused" in events
    assert any("refused: mode is homing" in m for m in h.operator.messages)


# ---- homing -------------------------------------------------------------------------------------------------------

def test_home_pauses_heartbeats_and_they_resume(h):
    h.franka.home_s = 1.0
    h.operator.press("h")
    h.tick()
    homes = [t for name, t in h.franka.timed if name == "home"]
    beats = [t for name, t in h.franka.timed if name == "heartbeat"]
    assert len(homes) == 1
    during = [t for t in beats if homes[0] < t < homes[0] + 900 * MS]
    assert during == []
    after = [t for t in beats if t > homes[0]]
    assert after and after[0] - homes[0] <= 1_000 * MS + 1          # the one right after home returns
    assert h.loop.phase is Phase.IDLE
    assert any(line.startswith("[homing] keys inactive; Ctrl-C quits") for line in h.operator.statuses)


@pytest.mark.parametrize("signal_name", ["interrupt", "terminate"])
def test_ctrl_c_or_sigterm_during_homing_exits_within_one_slice(h, signal_name):
    h.franka.home_behaviour = "hang"
    slices = {"n": 0}

    def during():
        slices["n"] += 1
        if slices["n"] == 10:
            getattr(h.loop, signal_name)()
    h.franka.during_home = during
    stops = h.calls.count("stop")
    h.operator.press("h")
    h.tick()
    assert h.loop.phase is Phase.DONE and h.loop.run() == 0
    assert slices["n"] == 10 and h.franka.home_closed
    assert h.calls.count("stop") == stops                         # no stop: franka_control is busy homing
    assert h.loop.verdict.kind is Kind.STOP


def test_homing_refused_concludes(h):
    h.franka.home_behaviour = "refuse"
    h.operator.press("h")
    h.tick()
    assert h.loop.phase is Phase.DONE and h.loop.run() == 1
    assert h.loop.verdict.problem.code is C.HOMING_FAILED and "stop" in h.calls


# ---- state checks -------------------------------------------------------------------------------------------------

def test_stopped_before_the_start_ack_does_not_conclude_after_it_does(h):
    h.start_episode()
    h.tick(3)
    h.operator.press("y")
    h.tick()
    h.tick(3)                                            # states now say "stopped"
    h.run_until(lambda: not h.worker.busy())
    h.publish = False                                    # the newest state stays the "stopped" one read before
    h.operator.press("c")
    h.run_until(lambda: h.loop.phase is Phase.RUNNING)
    assert h.history.latest().mode == "stopped" and h.history.latest().t_ns < h.loop._start_ack_ns
    h.tick(3)
    assert h.loop.phase is Phase.RUNNING                 # not taken for a stop on its own
    h.publish = True
    h.mode_override = "stopped"                          # now the arm really stops by itself
    h.tick(2)
    assert h.loop.phase is Phase.DONE and h.loop.verdict.problem.code is C.ARM_STOPPED and h.loop.run() == 1


def test_fault_with_a_broken_message_is_arm_fault(h):
    h.publish = False
    h.push_state(mode="fault", error="joint reflex")
    h.listener.set_error(ContractError("c2.state/v1", "O_T_TCP", "value", "not rigid"))
    h.loop.cycle()
    assert h.loop.verdict.problem.code is C.ARM_FAULT and "joint reflex" in h.loop.verdict.what_failed


def test_broken_message_alone_is_a_contract_violation(h):
    h.listener.set_error(ContractError("c2.state/v1", "O_T_TCP", "value", "not rigid"))
    h.tick()
    assert h.loop.verdict.problem.code is C.CONTRACT_VIOLATION and h.loop.run() == 1


def test_state_lost(h):
    h.publish = False
    h.tick(60)
    assert h.loop.verdict.problem.code is C.STATE_LOST


# ---- slow cases never conclude ------------------------------------------------------------------------------------

def test_long_skipping_never_concludes(h, settings):
    h.start_episode()
    h.tick(3)
    sent = len(h.franka.sent)
    set_cameras(h, main="stale")
    h.tick(1500)                                         # 30 s of stale camera
    assert h.loop.phase is Phase.RUNNING and len(h.franka.sent) == sent
    assert any("SKIPPING" in line and "main camera stale" in line for line in h.operator.statuses[-5:])
    set_cameras(h)
    h.tick(5)
    assert len(h.franka.sent) > sent                     # resumes by itself
    assert not any("SKIPPING" in line for line in h.operator.statuses[-1:])


def test_skips_are_retried_once_per_cycle(h):
    h.start_episode()
    set_cameras(h, main="stale")
    h.tick(3)
    jobs = len(h.worker.jobs)
    h.tick(10)
    assert len(h.worker.jobs) - jobs <= 10


def test_odd_chunk_is_skipped_counted_and_clips_summed(h, settings):
    from test_policy_step import chunk_step0
    h.start_episode()
    h.tick(2)
    h.policy.chunk = lambda obs: chunk_step0(0.05, clipped=3)
    sent = len(h.franka.sent)
    h.tick(3)
    assert len(h.franka.sent) == sent
    h.policy.chunk = lambda obs: chunk_step0(0.0, clipped=1)
    h.tick(3)
    assert len(h.franka.sent) > sent
    h.operator.press("y")
    h.tick()
    e = episodes(settings)[0]
    assert e["odd_chunks"] >= 2 and e["width_clips"] >= 3 * e["odd_chunks"] + 1
    assert e["skips"]["odd_chunk"] == e["odd_chunks"]


def test_late_policy_counted(h, settings):
    h.start_episode()
    h.tick(2)
    h.worker.delay_s = 0.5                               # slower than the 0.35 s the waypoints last
    h.tick(80)
    h.operator.press("y")
    h.tick()
    assert episodes(settings)[0]["late"] >= 1 and h.loop.phase is Phase.IDLE


# ---- concludes ----------------------------------------------------------------------------------------------------

def test_camera_down_concludes_with_stop_first(h, settings):
    h.start_episode()
    h.tick(3)
    set_cameras(h, ultrawide="down")
    h.tick(30)
    assert h.loop.run() == 1
    v = h.loop.verdict
    assert f"{v.what_failed}: {v.what_to_check}" == "ultrawide camera down: check the Pi stream and cable"
    assert "stop" in h.calls
    e = episodes(settings)[0]
    assert e["outcome"] == "aborted" and "ultrawide camera down" in e["reason"]


def test_gate_rejection_concludes(h):
    h.start_episode()
    h.franka.reply = rejecting_reply
    h.tick(3)
    assert h.loop.verdict.problem.code is C.GATE_REJECTED
    assert "gate rejected waypoint 3: outside the box by 4 cm" in h.loop.verdict.what_failed


def test_main_thread_error_concludes_naming_the_request(h):
    from contracts import TransportTimeout
    h.franka.error["heartbeat"] = TransportTimeout("franka_control", "tcp://x", "c3.heartbeat/v1", 0.2)
    h.tick()
    v = h.loop.verdict
    assert v.problem.code is C.NO_ANSWER and "c3.heartbeat/v1" in v.what_failed


def test_conclude_does_not_retry_a_failed_stop(h):
    from contracts import TransportTimeout
    h.franka.error["stop"] = TransportTimeout("franka_control", "tcp://x", "c3.stop/v1", 0.2)
    h.operator.press("q")
    h.tick()
    assert h.loop.run() == 1 and h.calls.count("stop") == 1


# ---- gate deltas, tap ---------------------------------------------------------------------------------------------

def test_gate_and_loop_late_deltas_from_the_episode_start(h, settings):
    h.gate.update(shortened=10, pulled_back=4, loop_late=7)
    h.tick(2)
    h.start_episode()
    h.gate.update(shortened=13, pulled_back=5, loop_late=9)
    h.tick(3)
    snap = h.loop._snapshot(h.clock())
    assert (snap.shortened, snap.pulled_back, snap.loop_late) == (3, 1, 2)
    h.operator.press("y")
    h.tick()
    e = episodes(settings)[0]
    assert (e["shortened"], e["pulled_back"], e["loop_late"]) == (3, 1, 2)
    snap = h.loop._snapshot(h.clock())
    assert (snap.shortened, snap.pulled_back, snap.loop_late) == (13, 5, 9)   # IDLE: session totals


@pytest.mark.parametrize("save_tap", [True, False])
def test_tap_after_the_next_step_is_submitted(make_settings, save_tap):
    s = make_settings(save_tap=save_tap)
    harness = LoopHarness(s)
    harness.start()
    seen = []
    original = harness.log.save_tap

    def spy(step, reply):
        seen.append(harness.worker.busy() and harness.worker.pending_kind() == "step")
        original(step, reply)
    harness.log.save_tap = spy
    harness.start_episode()
    harness.tick(5)
    if save_tap:
        assert seen and all(seen)
        assert list((s.session_dir / "tap" / "ep_0001").glob("*.npz"))
    else:
        assert seen == [] and not (s.session_dir / "tap").exists()


# ---- startup ------------------------------------------------------------------------------------------------------

def test_ready_file_only_after_state_config_and_cameras(settings, tmp_path):
    harness = LoopHarness(settings)
    set_cameras(harness, main="stale")
    ready = tmp_path / "ready"
    written_at = {}
    original = harness.health.get_health

    def get_health():
        report = original()
        if harness.clock() > 1_000 * 1_000_000_000 + 3 * 1_000_000_000:
            harness.health.report = health()
        written_at.setdefault("first", ready.exists())
        return report
    harness.health.get_health = get_health
    harness.start(ready)
    assert written_at["first"] is False
    assert ready.exists() and harness.loop.phase is Phase.IDLE
    assert harness.calls.index("get_state") < harness.calls.index("config") and harness.operator.started


def test_startup_times_out_naming_the_camera(make_settings, tmp_path):
    s = make_settings(timeouts={"startup_s": 2.0})
    harness = LoopHarness(s)
    set_cameras(harness, main="stale")
    ready = tmp_path / "ready"
    harness.start(ready)
    assert harness.loop.phase is Phase.DONE and not ready.exists()
    v = harness.loop.verdict
    assert f"{v.what_failed}: {v.what_to_check}" == \
        "startup: no camera ok within 2 s (main stale): check process_frames"


def test_startup_without_state_times_out(make_settings):
    s = make_settings(timeouts={"startup_s": 1.0})
    harness = LoopHarness(s)
    harness.publish = False
    harness.loop.startup(None)
    assert harness.loop.verdict.problem.code is C.STARTUP and "no fresh arm state" in harness.loop.verdict.what_failed


def test_config_mismatch_concludes(settings, tmp_path):
    harness = LoopHarness(settings)
    harness.policy.config_reply = policy_config(step_ms=100)
    harness.start(tmp_path / "ready")
    v = harness.loop.verdict
    assert harness.loop.phase is Phase.DONE and not (tmp_path / "ready").exists()
    assert f"{v.what_failed}: {v.what_to_check}" == \
        "policy config: step_ms is 100, message_comms expects 50: check the checkpoint"


def test_startup_camera_down_concludes(settings):
    harness = LoopHarness(settings)
    set_cameras(harness, main="down")
    harness.start()
    assert harness.loop.verdict.problem.code is C.CAMERA_DOWN


# ---- run() with a script ------------------------------------------------------------------------------------------

def test_run_with_a_script_two_episodes(settings):
    harness = LoopHarness(settings)
    text = ("- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 5}\n- {key: s}\n- {wait_phase: idle}\n"
            "- {wait_s: 0.2}\n- {key: h}\n- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 5}\n- {key: n}\n"
            "- {wait_s: 0.2}\n- {key: q}\n")
    harness.operator = ScriptedOperator(parse_script(text, "t"), io.StringIO(), harness.clock,
                                        settings.operator_script_wait_s)
    harness.loop._operator = harness.operator
    harness.start()
    assert harness.loop.run() == 0
    assert [e["outcome"] for e in episodes(settings)] == ["stopped", "fail"]
    assert harness.calls.count("home") == 1 and harness.calls[-1] in ("stop", "heartbeat")
    assert np.all(np.diff([msg.chunk_id for _, msg in harness.franka.sent]) > 0)


def scripted(harness, text, out=None):
    harness.operator = ScriptedOperator(parse_script(text, "t"), out or io.StringIO(), harness.clock,
                                        harness.settings.operator_script_wait_s, "t")
    harness.loop._operator = harness.operator


def test_run_with_a_script_retries_a_refused_key(settings):
    """s while a step is in flight, then h at once: h is refused (the last step is still running) and retried."""
    harness = LoopHarness(settings)
    harness.worker.delay_s = 0.2                         # a step is in flight when s is pressed
    out = io.StringIO()
    scripted(harness, "- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 5}\n- {key: s}\n- {key: h}\n"
                      "- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 2}\n- {key: y}\n- {key: q}\n", out)
    harness.start()
    assert harness.loop.run() == 0, harness.loop.verdict
    assert "h: waiting for the last policy step to finish" in out.getvalue()
    assert harness.calls.count("home") == 1
    assert [e["outcome"] for e in episodes(settings)] == ["stopped", "success"]


def test_run_with_a_script_that_waits_forever_concludes(make_settings):
    settings = make_settings(operator_script_wait_s=1.0)
    harness = LoopHarness(settings)
    scripted(harness, "- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 100000}\n- {key: y}\n- {key: q}\n")
    harness.start()
    assert harness.loop.run() == 1
    v = harness.loop.verdict
    assert v.problem.code is C.OPERATOR_SCRIPT and v.kind is Kind.CONCLUDE
    assert v.what_failed.startswith("operator script: t:3: wait_steps 100000: not reached (phase running, ")
    assert v.what_to_check == "check the script (message_comms.operator_script) or raise operator_script_wait_s"
    assert "stop" in harness.calls and [e["outcome"] for e in episodes(settings)] == ["aborted"]
