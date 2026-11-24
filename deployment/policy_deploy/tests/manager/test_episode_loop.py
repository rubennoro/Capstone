# adapted from rnoro/deployment:deployment/message_comms/tests/test_episode_loop.py @ 91b178b
# changes: the new LoopHarness (FakePcClient on a PC clock 1000 s ahead, a real PC clock map and HeartbeatSender);
# keep-alives are hellos or heartbeats; the new codes and texts; start-up order (hellos before get_state);
# + local refusal of c while the PC homes, c until the map is ready, margins in PC time, the clock-mismatch guard,
# KEY_HANDLERS; h is not a laptop key (homing and the start pose are PC keyboard keys only).
"""EpisodeLoop with fakes, FakeClock, ManualWorker and scripted keys, driving cycle() (STRUCTURE.md section 11)."""
import io
import json

import numpy as np
import pytest

from deploy_contracts import Ack, ContractError
from manager_fakes import health, policy_config, rejecting_reply
from manager_harness import LoopHarness, episodes, events, session_dir

from policy_deploy.manager.episode_loop import KEY_HANDLERS, Phase
from policy_deploy.manager.operator import Key
from policy_deploy.manager.problems import Kind
from policy_deploy.manager.problems import ProblemCode as C
from policy_deploy.manager.scripted_operator import ScriptedOperator, parse_script

MS = 1_000_000


@pytest.fixture
def h(settings):
    harness = LoopHarness(settings)
    harness.start()
    assert harness.loop.phase is Phase.IDLE, harness.loop.verdict
    return harness


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
    before = h.keepalives()
    h.tick(100)                                     # 2 s: the step is still held back
    assert h.keepalives() - before == 100           # one keep-alive (hello or heartbeat) per cycle
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
    assert h.loop.phase is Phase.IDLE
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
    assert e["outcome"] == "aborted" and e["reason"] == "stopped by the launcher: another process exited"


def test_wrong_phase_keys_give_hints(h):
    h.operator.press("y")
    h.tick()
    assert "y: no episode running (c start, q quit)" in h.operator.messages
    h.start_episode()
    h.operator.press("c")
    h.tick(2)
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
    assert hp.loop.phase is Phase.DONE and hp.loop.verdict.what_to_check == "check manager.task_name"


def test_refused_start_returns_to_idle_without_an_episode(h, settings):
    h.franka.start_ack = Ack(ok=False, reason="refused: mode is homing")
    h.operator.press("c")
    h.tick(3)
    assert h.loop.phase is Phase.IDLE and h.loop.progress().episode_id == 0 and episodes(settings) == []
    assert "start_refused" in events(settings)
    assert "c: the PC refused start: refused: mode is homing" in h.operator.messages


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
    assert h.history.latest().mode == "stopped" and h.history.latest().t_ns < h.loop._start_ack_pc_ns
    h.tick(3)
    assert h.loop.phase is Phase.RUNNING                 # not taken for a stop on its own
    h.publish = True
    h.mode_override = "stopped"                          # now the arm really stops by itself
    h.tick(2)
    assert h.loop.phase is Phase.DONE and h.loop.verdict.problem.code is C.PC_STOPPED and h.loop.run() == 1


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
    h.tick(90)
    assert h.loop.phase is not Phase.DONE                # 1.8 s: not yet (state_lost_s 2.0, by arrival)
    h.tick(30)
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


def chunk_step0(pos_m=0.0, rot_deg=0.0, clipped=0):
    from manager_fakes import make_chunk, rz
    pos = np.zeros((16, 3))
    pos[0, 0] = pos_m
    rot = np.tile(np.eye(3), (16, 1, 1))
    rot[0] = rz(rot_deg)
    return make_chunk(pos=pos, rot=rot, width_clipped=clipped)


def test_odd_chunk_is_skipped_counted_and_clips_summed(h, settings):
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
    assert f"{v.what_failed}: {v.what_to_check}" == "ultrawide camera down: check the Pi and ASUS_5g"
    assert "stop" in h.calls
    e = episodes(settings)[0]
    assert e["outcome"] == "aborted" and "ultrawide camera down" in e["reason"]


def test_gate_rejection_concludes(h):
    h.start_episode()
    h.franka.reply = rejecting_reply
    h.tick(3)
    assert h.loop.verdict.problem.code is C.GATE_REJECTED
    assert "the PC's gate rejected waypoint 3: outside the box by 4 cm" == h.loop.verdict.what_failed


def test_main_thread_error_concludes_naming_the_request(h):
    from deploy_contracts import TransportTimeout
    h.franka.error["heartbeat"] = TransportTimeout("PC", "tcp://x", "c3.heartbeat/v1", 0.3)
    h.tick(2)
    v = h.loop.verdict
    assert v.problem.code is C.PC_NOT_ANSWERING
    assert f"{v.what_failed}: {v.what_to_check}" == ("PC not answering: no reply to c3.heartbeat/v1 within 0.3 s: "
                                                     "check franka_control on the PC and the link")


def test_conclude_does_not_retry_a_failed_stop(h):
    from deploy_contracts import TransportTimeout
    h.franka.error["stop"] = TransportTimeout("PC", "tcp://x", "c3.stop/v1", 0.3)
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
        assert list((session_dir(s) / "tap" / "ep_0001").glob("*.npz"))
    else:
        assert seen == [] and not (session_dir(s) / "tap").exists()


# ---- startup ------------------------------------------------------------------------------------------------------

def test_ready_file_only_after_state_config_and_cameras(settings, tmp_path):
    """The cameras turn ok 3.5 s after the start (the clock map takes 2 s of hellos before them)."""
    harness = LoopHarness(settings)
    set_cameras(harness, main="stale")
    ready = tmp_path / "ready"
    written_at = {}
    original = harness.health.get_health

    def get_health():
        report = original()
        if harness.clock() > 1_000 * 1_000_000_000 + int(3.5 * 1_000_000_000):
            harness.health.report = health()
        written_at.setdefault("first", ready.exists())
        return report
    harness.health.get_health = get_health
    harness.start(ready)
    assert written_at["first"] is False
    assert ready.exists() and harness.loop.phase is Phase.IDLE
    calls = harness.calls
    assert calls.index("hello") < calls.index("get_state") < calls.index("config") and harness.operator.started


def test_startup_times_out_naming_the_camera(make_settings, tmp_path):
    s = make_settings(timeouts={"startup_s": 4.0})
    harness = LoopHarness(s)
    set_cameras(harness, main="stale")
    ready = tmp_path / "ready"
    harness.start(ready)
    assert harness.loop.phase is Phase.DONE and not ready.exists()
    v = harness.loop.verdict
    assert f"{v.what_failed}: {v.what_to_check}" == \
        "startup: no camera ok within 4 s (main stale): check frames"


def test_startup_without_state_times_out(make_settings):
    s = make_settings(timeouts={"startup_s": 1.0})
    harness = LoopHarness(s)
    harness.publish = False
    harness.loop.startup(None)
    v = harness.loop.verdict
    assert v.problem.code is C.PC_NOT_ANSWERING and "no C2 state from tcp://127.0.0.1:" in v.what_failed
    assert harness.calls == []                          # nothing sent on C3


def test_config_mismatch_concludes(settings, tmp_path):
    harness = LoopHarness(settings)
    harness.policy.config_reply = policy_config(step_ms=100)
    harness.start(tmp_path / "ready")
    v = harness.loop.verdict
    assert harness.loop.phase is Phase.DONE and not (tmp_path / "ready").exists()
    assert f"{v.what_failed}: {v.what_to_check}" == \
        "policy config: step_ms is 100, manager expects 50: check the checkpoint"


def test_startup_camera_down_concludes(settings):
    harness = LoopHarness(settings)
    set_cameras(harness, main="down")
    harness.start()
    assert harness.loop.verdict.problem.code is C.CAMERA_DOWN


# ---- run() with a script ------------------------------------------------------------------------------------------

def test_run_with_a_script_two_episodes(settings):
    harness = LoopHarness(settings)
    text = ("- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 5}\n- {key: s}\n- {wait_phase: idle}\n"
            "- {wait_s: 0.2}\n- {key: c}\n- {wait_steps: 5}\n- {key: n}\n"
            "- {wait_s: 0.2}\n- {key: q}\n")
    harness.operator = ScriptedOperator(parse_script(text, "t"), io.StringIO(), harness.clock,
                                        settings.operator_script_wait_s)
    harness.loop._operator = harness.operator
    harness.start()
    assert harness.loop.run() == 0
    assert [e["outcome"] for e in episodes(settings)] == ["stopped", "fail"]
    assert "home" not in harness.calls and harness.calls[-1] in ("stop", "heartbeat", "hello")
    assert np.all(np.diff([msg.chunk_id for _, msg in harness.franka.sent]) > 0)


def scripted(harness, text, out=None):
    harness.operator = ScriptedOperator(parse_script(text, "t"), out or io.StringIO(), harness.clock,
                                        harness.settings.operator_script_wait_s, "t")
    harness.loop._operator = harness.operator


def test_run_with_a_script_retries_a_refused_key(settings):
    """s while a step is in flight, then c at once: c is refused (the last step is still running) and retried."""
    harness = LoopHarness(settings)
    harness.worker.delay_s = 0.2                         # a step is in flight when s is pressed
    stop = harness.franka.stop

    def stop_then_idle():                                # the PC idle before the retried c's start: with zero latency
        stop()                                           # a `stopped` C2 stamped at the start ack would read as
        harness.franka.mode = "idle"                     # PC_STOPPED (not this test's subject)
    harness.franka.stop = stop_then_idle
    out = io.StringIO()
    scripted(harness, "- {wait_phase: idle}\n- {key: c}\n- {wait_steps: 5}\n- {key: s}\n- {key: c}\n"
                      "- {wait_steps: 2}\n- {key: y}\n- {key: q}\n", out)
    harness.start()
    assert harness.loop.run() == 0, harness.loop.verdict
    assert "c: waiting for the last policy step to finish" in out.getvalue()
    assert "home" not in harness.calls
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
    assert v.what_to_check == "check the script (manager.operator_script) or raise operator_script_wait_s"
    assert "stop" in harness.calls and [e["outcome"] for e in episodes(settings)] == ["aborted"]


# ---- added: the PC side (STRUCTURE.md 6.4, 6.5, 7.5) --------------------------------------------------------------

def test_key_handlers_cover_every_key():
    assert set(KEY_HANDLERS) == {k.value for k in Key}


def test_h_is_not_a_laptop_key():
    """No laptop key moves the arm to a pose: homing (h) and the start pose (p) are PC keyboard keys only."""
    assert {k.value for k in Key} == {"c", "y", "n", "s", "q"}
    assert "h" not in KEY_HANDLERS and "p" not in KEY_HANDLERS
    assert "homing" not in [phase.value for phase in Phase]


def test_c_refused_locally_while_the_pc_is_homing(h, settings):
    h.mode_override = "homing"                           # the PC's own h, g or p
    h.tick(2)
    h.operator.press("c")
    h.tick()
    assert h.loop.phase is Phase.IDLE and "reset" not in h.calls and "home" not in h.calls
    assert "c: the PC is homing (its h, g or p key): wait for idle" in h.operator.messages
    assert [k for k, _ in h.operator.refused] == [Key.START]
    assert h.loop.verdict is None
    h.mode_override = None                               # idle again: accepted
    h.start_episode()
    assert h.loop.phase is Phase.RUNNING


def test_c_refused_until_the_clock_map_is_ready(h, monkeypatch):
    monkeypatch.setattr(h.pc_clock, "ready", lambda now: False)
    h.operator.press("c")
    h.tick()
    assert h.loop.phase is Phase.IDLE and "c: waiting for the PC clock map" in h.operator.messages
    monkeypatch.undo()
    h.start_episode()


def test_margin_and_waypoints_on_the_pc_clock(h, settings):
    h.start_episode()
    h.tick(10)
    steps = [json.loads(x) for x in (session_dir(settings) / "steps.jsonl").read_text().splitlines()]
    sent = [s for s in steps if s["sent"]]
    margins = [s["margin_ms"] for s in sent if s["margin_ms"] is not None]
    assert margins and all(0 < m < 400 for m in margins)
    for s in sent:
        assert s["t_anchor_pc_ns"] - s["t_anchor_ns"] == pytest.approx(h.franka.offset_ns, abs=2 * MS)
        assert s["c3_rtt_ms"] is not None
    t, msg = h.franka.sent[-1]
    assert msg.t_ns[0] > h.franka.pc_ns(t) - 200 * MS              # PC times, not laptop times


def test_skipped_sets_timed_ahead_conclude_clock_mismatch(h, settings):
    h.start_episode()
    h.tick(5)
    h.franka.offset_ns += 500_000_000                    # the PC clock jumps 0.5 s; the map lags behind
    h.tick(20)
    v = h.loop.verdict
    assert v is not None and v.problem.code is C.CLOCK_MISMATCH
    assert f"{v.what_failed}: {v.what_to_check}" == ("clock mismatch: the PC skipped 3 waypoint sets the laptop "
                                                     "timed ahead: run pc_probe; check clock.estimator")
    assert "stop" in h.calls and episodes(settings)[0]["outcome"] == "aborted"


def test_a_c2_from_the_future_counts_toward_clock_mismatch(h, settings, monkeypatch):
    h.start_episode()
    h.tick(5)
    h.franka.offset_ns += 100_000_000                    # C2 now 100 ms ahead of the (held) map
    monkeypatch.setattr(h.pc_clock, "add_hello", lambda s: None)     # the laptop's refit held
    h.tick(20)
    v = h.loop.verdict
    assert v is not None and v.problem.code is C.CLOCK_MISMATCH and "ahead of the mapped now" in v.what_failed


def test_one_skipped_set_is_only_policy_late(h, settings):
    h.start_episode()
    h.tick(3)
    n = {"k": 0}
    original = h.franka.set_waypoints

    def once_skipped(msg):
        n["k"] += 1
        reply = original(msg)
        if n["k"] == 1:
            from deploy_contracts import SetWaypointsReply
            return SetWaypointsReply(chunk_id=msg.chunk_id, accepted=0, shortened=0, skipped=len(msg.t_ns),
                                     rejected=())
        return reply
    h.franka.set_waypoints = once_skipped
    h.tick(10)
    assert n["k"] > 2                                    # later sets were accepted again
    assert h.loop.phase is Phase.RUNNING and h.loop._mismatch_count == 0 and h.loop.verdict is None


def test_conclude_before_hellos_sends_nothing(make_settings):
    s = make_settings()
    harness = LoopHarness(s)
    harness.mode_override = "fault"
    harness.start()
    assert harness.loop.verdict.problem.code is C.ARM_FAULT and harness.calls == []


def test_an_entirely_skipped_set_counted_toward_clock_mismatch_is_not_also_policy_late(settings, monkeypatch):
    """Fix wave (one cause per set): a C2 from the future makes each set count toward CLOCK_MISMATCH; the PC skipping
    every waypoint of those sets (lead below mismatch_min_lead_ms here) must not also count them as POLICY_LATE."""
    import dataclasses
    s = dataclasses.replace(settings, clock=dataclasses.replace(settings.clock, mismatch_min_lead_ms=1e6,
                                                                mismatch_sets=1000))
    h = LoopHarness(s)
    h.start()
    h.start_episode()
    h.tick(3)
    h.franka.offset_ns += 100_000_000                    # C2 now 100 ms ahead of the (held) map
    monkeypatch.setattr(h.pc_clock, "add_hello", lambda sample: None)
    from deploy_contracts import SetWaypointsReply
    original = h.franka.set_waypoints

    def all_skipped(msg):
        original(msg)
        return SetWaypointsReply(chunk_id=msg.chunk_id, accepted=0, shortened=0, skipped=len(msg.t_ns), rejected=())
    h.franka.set_waypoints = all_skipped
    h.tick(10)
    assert h.loop._mismatch_count >= 3 and h.loop.verdict is None
    h.operator.press("y")
    h.tick()
    assert episodes(s)[0]["late"] == 0
