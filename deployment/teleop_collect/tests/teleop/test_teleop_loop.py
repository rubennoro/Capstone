"""TeleopLoop and EpisodeFlow on the fake-clock harness (STRUCTURE.md 6.3-6.7, 12 row E3; PLAN 5, 7, 8): start-up,
one C3 request per cycle, hello priority, the teleop step, start / stop / discard / abort / conclude, home and its
refusals, start refusals, the guards. E4's motion and episode classes are the real ones; the PC and frames are fakes."""
from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml
from teleop_helpers import FAST, NS_PER_MS, TeleopHarness, merge, rejecting_reply
from deploy_contracts import SetWaypointsReply

from teleop_collect.contracts import (CreateTask, DeleteEpisode, GetStatus, GripperCommand, HomeRequest, SelectTask,
                                      StartEpisode, StopArm, StopEpisode, TeleopStatus)
from teleop_collect.teleop.problems import ProblemCode as C
from teleop_collect.teleop.teleop_loop import HOMING_REFUSAL, Phase


@pytest.fixture
def harness(make_settings):
    made = []

    def make(**teleop):
        h = TeleopHarness(make_settings(**merge(FAST, teleop)))
        made.append(h)
        return h
    yield make
    for h in made:
        h.log.close()


def per_cycle(h, cycles: int) -> list[list[str]]:
    """The C3 requests of each of the next n cycles."""
    out = []
    for _ in range(cycles):
        n = len(h.pc.calls)
        h.tick()
        out.append(h.pc.calls[n:])
    return out


def ep_dir(h, name="episode_000"):
    return h.task_dir() / name


# ---- start-up ------------------------------------------------------------------------------------------------------

def test_startup_order_home_and_ready(harness, tmp_path):
    h = harness()
    ready = tmp_path / "ready" / "teleop.ready"
    h.start(ready)
    assert h.loop.phase is Phase.IDLE, h.verdict_text()
    calls = h.pc.calls
    first_get_state = calls.index("get_state")
    assert set(calls[:first_get_state]) == {"hello"} and first_get_state >= 5          # hellos until the map is ready
    assert calls.index("home") > first_get_state                                        # then the home (on_start)
    assert "get_health" in h.frames.calls and ready.read_text() == "ready\n"
    assert (h.session / "session.yaml").exists()
    h.tick(3)
    assert h.loop.home_reference.known and h.loop.at_home()
    kinds = [e["kind"] for e in h.events()]
    assert kinds[0] == "startup" and "ready" in kinds and "home" in kinds


def test_startup_without_home_on_start(harness):
    h = harness(home={"on_start": False})
    h.start()
    assert h.loop.phase is Phase.IDLE and "home" not in h.pc.calls
    h.tick(3)
    s = h.status()
    assert not s.home_known and s.start_refusal == "select or create a task first"


def test_startup_deletes_incomplete_episodes(harness, make_settings):
    h = harness()
    h.store.create("pick_block", "", {})
    t = h.store.open("pick_block")
    left = t.new_episode()
    h.start()
    assert not left.path.exists()
    assert any("incomplete" in text for text in h.loop.messages.texts())
    assert any(e["kind"] == "incomplete_deleted" for e in h.events())


def test_startup_cameras_never_ok(harness):
    h = harness(timeouts={"startup_s": 1.0})
    h.frames.ultrawide = "stale"
    h.start()
    assert h.code is C.STARTUP and "no camera ok within 1 s (ultrawide stale)" in h.loop.verdict.what_failed
    assert h.pc.calls[-1] == "stop"                                     # hellos had started: stop sent once


def test_startup_home_failure_concludes_and_stopped_goes_on(harness):
    h = harness()
    h.pc.home_behaviour = "refuse"
    h.start()
    assert h.code is C.HOMING_FAILED
    h2 = harness()
    h2.pc.home_behaviour = "stopped"
    h2.start()
    assert h2.loop.phase is Phase.IDLE and not h2.loop.home_reference.known
    assert any("home stopped on the PC" in t for t in h2.loop.messages.texts())


def test_hello_mismatch_concludes(harness):
    h = harness()
    h.pc.hello_change = {"mock": False}
    h.start()
    assert h.code is C.HELLO_MISMATCH and h.loop.verdict.what_failed.startswith("hello mismatch: mock")


# ---- the rhythm ----------------------------------------------------------------------------------------------------

def test_exactly_one_c3_request_per_cycle_and_hello_priority(harness):
    h = harness()
    h.ready()
    h.start_episode()
    cycles = []
    for i in range(60):
        h.keys("x+")
        n = len(h.pc.calls)
        h.tick()
        cycles.append(h.pc.calls[n:])
    assert all(len(c) == 1 for c in cycles), cycles
    flat = [c[0] for c in cycles]
    assert "hello" in flat and flat.count("set_waypoints") >= 50
    for i, name in enumerate(flat):                     # a hello takes the cycle's slot: no set beside it
        assert name in ("hello", "set_waypoints"), flat
    idle = per_cycle(h, 1)
    assert len(idle[0]) == 1


def test_idle_cycles_are_hellos_and_heartbeats(harness):
    h = harness()
    h.ready()
    names = [c for cyc in per_cycle(h, 50) for c in cyc]
    assert len(names) == 50 and set(names) == {"hello", "heartbeat"}
    assert names.count("hello") == 2                                    # every 0.5 s at 50 Hz


# ---- the teleop step -----------------------------------------------------------------------------------------------

def test_keys_move_release_once_then_heartbeats(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 10)
    moves = [msg for _, msg in h.pc.sent]
    assert len(moves) >= 8 and all(len(m.t_ns) == 2 for m in moves)
    m = moves[-1]
    assert np.all(np.diff(m.t_ns) == 50 * NS_PER_MS)                     # Delta apart
    assert m.O_T_TCP[1][0, 3] > m.O_T_TCP[0][0, 3] > h.T[0, 3]           # +x, from the PC's target
    assert m.chunk_id == moves[0].chunk_id + len(moves) - 1
    n = len(h.pc.sent)
    h.keys()                                                             # released
    h.tick(10)
    after = [msg for _, msg in h.pc.sent[n:]]
    assert len(after) <= 2 and len(after[-1].t_ns) == 1                  # one release set (or a hello first)
    tail = h.pc.calls[-8:]
    assert set(tail) <= {"hello", "heartbeat"}


def test_stale_keys_are_released(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["y+"], 3)
    n = len(h.pc.sent)
    h.tick(20)                                                           # the page went silent (400 ms > 150 ms)
    later = h.pc.sent[n:]
    assert len(later) <= 9                                               # until 150 ms passed, then one release
    assert len(later[-1][1].t_ns) == 1
    h.tick(10)
    assert len(h.pc.sent) == n + len(later)


def test_focus_lost_is_released(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["z+"], 3)
    h.keys("z+", focus=False)
    n = len(h.pc.sent)
    h.tick(5)
    assert len(h.pc.sent) - n <= 1 and len(h.pc.sent[-1][1].t_ns) == 1


def test_no_set_while_the_newest_c2_is_too_old(harness):
    h = harness()
    h.state_delay_ns = 80 * NS_PER_MS                                    # every C2 80 ms old (> 50 ms)
    h.ready()
    h.start_episode()
    h.tick(2)
    n = len(h.pc.sent)
    h.hold(["x+"], 20)
    assert len(h.pc.sent) == n
    assert h.monitor.counts().get(C.STATE_OLD, 0) > 0 and h.loop.phase is Phase.RECORDING
    h2 = harness(motion={"state_max_age_ms": 90, "v_max_m_s": 0.03, "w_max_deg_s": 8.0})   # within the guard
    h2.state_delay_ns = 80 * NS_PER_MS
    h2.ready("second_task")
    h2.start_episode()
    h2.hold(["x+"], 5)
    assert len(h2.pc.sent) >= 4


def test_fingers(harness):
    h = harness()
    h.ready()
    h.start_episode()
    assert h.u1(GripperCommand(command="close")).ok
    h.tick(3)
    m = h.pc.sent[-1][1]
    assert len(m.t_ns) == 1 and m.gripper_width_m[0] == 0.0 and np.allclose(m.O_T_TCP[0], h.T)
    h.u1(GripperCommand(command="open"))
    h.hold(["x-"], 3)
    assert h.pc.sent[-1][1].gripper_width_m.tolist() == [0.08, 0.08]
    assert h.status().commanded_width_m == 0.08


def test_gripper_outside_an_episode_is_refused(harness):
    h = harness()
    h.ready()
    assert not h.u1(GripperCommand(command="close")).ok


def test_at_limit_from_pulled_back(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(2)
    assert not h.status().at_limit
    h.gate["pulled_back"] = 3
    h.tick(2)
    assert h.status().at_limit
    h.tick(30)
    assert not h.status().at_limit


# ---- start, stop, discard, delete ----------------------------------------------------------------------------------

def test_start_stop_success_writes_the_episode(harness):
    h = harness()
    h.ready()
    h.start_episode()
    assert h.pc.calls.count("start") == 1 and h.frames.calls.count("start_recording") == 1
    assert h.frames.dirs == [str(ep_dir(h))]
    s = h.status()
    assert s.phase == "recording" and s.recording_episode == "episode_000" and not s.start_allowed
    h.hold(["x+"], 10)
    h.keys()
    h.tick(5)
    assert h.u1(StopEpisode(outcome="success")).ok
    assert h.loop.phase is Phase.IDLE and h.pc.calls[-1] == "stop"
    assert h.frames.stop_timeouts == [None]                              # a normal Stop waits timeouts.frames_s
    d = ep_dir(h)
    assert (d / "COMPLETE").exists()
    meta = yaml.safe_load((d / "meta.yaml").read_text())
    assert meta["outcome"] == "success" and meta["counts"]["frames_main"] == 30
    assert meta["times"]["stop_ns"] > meta["times"]["start_ns"] and meta["times"]["start_pc_ns"] is not None
    assert meta["pc_hello"]["mock"] is True and len(meta["pc_hello"]["F_T_TCP"]) == 4
    assert meta["counts"]["sets_sent"] >= 8 and meta["laptop"]["commit"]
    with open(d / "commands.csv") as f:
        rows = list(csv.DictReader(f))
    requests = {r["request"] for r in rows}
    assert {"c3.start/v1", "c3.set_waypoints/v1", "c3.stop/v1"} <= requests
    with open(d / "robot_states.csv") as f:
        assert len(list(csv.DictReader(f))) > 20
    with open(d / "keys.csv") as f:
        assert [r["held"] for r in csv.DictReader(f)][:2] == ["x+", ""]
    st = h.status()
    assert [e.name for e in st.episodes] == ["episode_000"] and st.episodes[0].outcome == "success"
    assert any("episode_000 saved" in t for t in h.loop.messages.texts())


def test_discard_deletes_and_numbers_are_never_reused(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(3)
    assert h.u1(StopEpisode(outcome="discard")).ok
    assert not ep_dir(h).exists() and h.status().episodes == ()
    h.tick(2)                                                            # the C2 showing stopped
    h.start_episode()
    h.tick(3)
    h.u1(StopEpisode(outcome="fail"))
    assert (ep_dir(h, "episode_001") / "COMPLETE").exists()
    assert h.u1(DeleteEpisode(episode="episode_001")).ok
    assert not ep_dir(h, "episode_001").exists()
    assert not h.u1(DeleteEpisode(episode="episode_001")).ok


def test_stop_outside_an_episode_is_refused(harness):
    h = harness()
    h.ready()
    assert h.u1(StopEpisode(outcome="success")).reason == "no episode recording"


def test_tasks(harness):
    h = harness()
    h.start()
    assert h.status().start_refusal == "select or create a task first"
    assert not h.u1(CreateTask(name="Bad Name", description="")).ok
    assert h.u1(CreateTask(name="pick_block", description="pick it")).ok
    assert not h.u1(CreateTask(name="pick_block", description="again")).ok
    assert h.u1(CreateTask(name="place", description="")).ok
    assert h.u1(SelectTask(name="pick_block")).ok and h.status().task_description == "pick it"
    assert not h.u1(SelectTask(name="missing")).ok
    s = h.status()
    assert s.tasks == ("pick_block", "place") and s.task == "pick_block"


# ---- start refusals ------------------------------------------------------------------------------------------------

def test_start_refusals(harness):
    h = harness()
    h.start()
    h.tick(3)
    assert h.u1(StartEpisode()).reason == "select or create a task first"
    h.u1(CreateTask(name="pick_block", description=""))
    h.q = np.full(7, 0.05)                                               # moved away from home
    h.tick(3)
    assert h.u1(StartEpisode()).reason == "home the arm first (Home, or h on the PC)"
    h.q = np.zeros(7)
    h.frames.main = "stale"
    h.tick(10)
    assert h.u1(StartEpisode()).reason == "main camera stale"
    h.frames.main = "down"
    h.tick(10)
    assert h.status().start_refusal == "main camera down" and h.loop.phase is Phase.IDLE    # NORMAL between
    h.frames.main = "ok"
    h.tick(10)
    h.mode_override = "homing"
    h.tick(2)
    assert h.u1(StartEpisode()).reason == "the PC is homing"
    assert not h.u1(HomeRequest()).ok
    h.mode_override = None
    h.tick(2)
    assert h.status().start_allowed
    assert not ep_dir(h).exists()


def test_start_refused_by_frames_or_the_pc_leaves_no_folder(harness):
    from deploy_contracts import Ack
    h = harness()
    h.ready()
    h.frames.start_ack = Ack(ok=False, reason="Pi clock not synced")
    r = h.u1(StartEpisode())
    assert not r.ok and "Pi clock not synced" in r.reason and not ep_dir(h).exists()
    h.frames.start_ack = Ack(ok=True, reason="")
    h.pc.start_ack = Ack(ok=False, reason="refused: mode is homing")
    r = h.u1(StartEpisode())
    assert not r.ok and "the PC refused start" in r.reason
    assert not ep_dir(h, "episode_001").exists() and h.frames.calls[-1] == "stop_recording"
    assert h.loop.phase is Phase.IDLE


# ---- home ----------------------------------------------------------------------------------------------------------

def test_home_runs_in_the_next_cycle_and_serves_u1_meanwhile(harness):
    h = harness()
    h.ready()
    seen = []

    class U1:
        def serve(self, n):
            if h.loop.phase is Phase.HOMING and not seen:
                seen.append(h.loop.dispatch(GetStatus(after_message=0)))
                seen.append(h.loop.dispatch(StartEpisode()))
            return 0
    h.loop.attach_u1(U1())
    assert h.u1(HomeRequest()).ok
    assert "home" not in h.pc.calls[-1:]
    assert not h.u1(HomeRequest()).ok                                    # already asked
    n = h.pc.calls.count("home")
    cycles = per_cycle(h, 2)
    assert cycles[0][0] == "home" and h.pc.calls.count("home") == n + 1
    assert isinstance(seen[0], TeleopStatus) and seen[0].phase == "homing"
    assert seen[1].reason == HOMING_REFUSAL
    assert h.loop.phase is Phase.IDLE


def test_home_refused_while_recording_and_stop_arm_aborts(harness):
    h = harness()
    h.ready()
    h.start_episode()
    assert not h.u1(HomeRequest()).ok
    h.hold(["x+"], 3)
    assert h.u1(StopArm()).ok
    assert h.loop.phase is Phase.IDLE and not ep_dir(h).exists() and h.pc.calls[-1] in ("stop", "stop_recording")
    assert h.frames.stop_timeouts[-1] == 1.0                             # the abort path's short wait
    assert "stop" in h.pc.calls
    assert any("episode_000 aborted and deleted: operator stop" in t for t in h.loop.messages.texts())
    assert any(e["kind"] == "episode_aborted" for e in h.events())


def test_stop_arm_between_episodes_sends_stop(harness):
    h = harness()
    h.ready()
    assert h.u1(StopArm()).ok and h.pc.calls[-1] == "stop" and h.loop.phase is Phase.IDLE


def test_the_pcs_own_home_sets_the_reference(harness):
    h = harness(home={"on_start": False})
    h.ready()
    assert not h.loop.home_reference.known
    h.q = np.full(7, 0.2)                                                # away from home
    h.tick(2)
    h.mode_override = "homing"                                           # the PC's h: the joints move home
    h.tick(2)
    h.q = np.zeros(7)
    h.tick(2)
    h.mode_override = None
    h.tick(3)
    assert h.loop.home_reference.known and h.status().start_allowed


# ---- aborts (EPISODE) ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("how, text", [
    ("camera_down", "main camera down: check the Pi and ASUS_5g"),
    ("recording_error", "recording could not keep up / disk error: queue full"),
    ("pc_stopped", "the PC stopped the arm"),
])
def test_episode_problems_abort_and_the_app_stays_up(harness, how, text):
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 3)
    if how == "camera_down":
        h.frames.main = "down"
    elif how == "recording_error":
        h.frames.recording_error = "queue full"
    else:
        h.pc.mode = "stopped"                                            # its watchdog, or s on the PC
    h.tick(30)
    assert h.loop.phase is Phase.IDLE and h.loop.verdict is None
    assert not ep_dir(h).exists()
    assert any(text in t for t in h.loop.messages.texts()), h.loop.messages.texts()
    h.frames.main, h.frames.recording_error = "ok", ""
    h.tick(10)
    h.start_episode()                                                    # ready for the next one
    assert ep_dir(h, "episode_001").exists()


def test_a_set_refused_because_the_pc_stopped_aborts(harness):
    from deploy_contracts import ErrorReply, RemoteError
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 2)
    h.publish = False                                                    # the C2 saying stopped has not arrived
    h.pc.mode = "stopped"
    h.pc.error["set_waypoints"] = RemoteError("PC", ErrorReply(request="c3.set_waypoints/v1",
                                                               error="refused: mode is stopped", field=""))
    h.hold(["x+"], 1)
    assert h.loop.phase is Phase.IDLE and h.loop.verdict is None and not ep_dir(h).exists()


def test_a_skipped_set_is_slow(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.pc.reply = lambda m: SetWaypointsReply(chunk_id=m.chunk_id, accepted=0, shortened=0, skipped=len(m.t_ns),
                                             rejected=()) if len(m.t_ns) == 1 else \
        SetWaypointsReply(chunk_id=m.chunk_id, accepted=len(m.t_ns), shortened=1, skipped=0, rejected=())
    h.hold(["x+"], 3)
    h.keys()
    h.tick(3)                                                            # the release (one waypoint) is skipped
    assert h.monitor.counts().get(C.SET_SKIPPED) == 1 and h.loop.phase is Phase.RECORDING


# ---- conclude, Ctrl-C, SIGTERM -------------------------------------------------------------------------------------

def test_fault_while_recording_concludes_and_deletes(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(3)
    h.mode_override = "fault"
    h.tick(3)
    assert h.code is C.ARM_FAULT and h.loop.phase is Phase.DONE and h.loop._exit_code == 1
    assert not ep_dir(h).exists() and "stop" in h.pc.calls and h.frames.calls[-1] == "stop_recording"


def test_gate_rejected_concludes(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.pc.reply = rejecting_reply
    h.hold(["x+"], 2)
    assert h.code is C.GATE_REJECTED and "outside the box" in h.loop.verdict.what_failed
    assert not ep_dir(h).exists()


def test_clock_mismatch_guard(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.pc.reply = lambda m: SetWaypointsReply(chunk_id=m.chunk_id, accepted=0, shortened=0, skipped=len(m.t_ns),
                                             rejected=())
    h.hold(["x+"], 6)
    assert h.code is C.CLOCK_MISMATCH


def test_state_lost_concludes(harness):
    h = harness()
    h.ready()
    h.publish = False
    h.tick(120)
    assert h.code is C.STATE_LOST


def test_ctrl_c_aborts_the_episode_and_exits_0(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(3)
    h.loop.interrupt()
    h.tick()
    assert h.loop.phase is Phase.DONE and h.loop._exit_code == 0 and h.code is C.OPERATOR_STOP
    assert not ep_dir(h).exists() and "stop" in h.pc.calls


def test_sigterm_during_home_exits_without_stop(harness):
    h = harness()
    h.ready()
    h.pc.home_behaviour = "hang"
    h.u1(HomeRequest())
    h.pc.during_home = lambda: h.loop.terminate() if h.pc.interrupted_calls > 5 else None
    n = h.pc.calls.count("stop")
    h.tick(2)
    assert h.code is C.LAUNCHER_STOP and h.loop._exit_code == 0 and h.pc.calls.count("stop") == n


# ---- status --------------------------------------------------------------------------------------------------------

def test_status_is_a_valid_contract_message_and_messages_page(harness):
    h = harness()
    h.ready()
    s = h.status()
    assert isinstance(s, TeleopStatus) and s.phase == "idle" and s.ready and s.start_allowed
    assert s.arm_mode == "idle" and s.tcp_m.shape == (3,) and s.cameras.main.status == "ok"
    assert s.c2_hz > 50 and s.clock_ok and s.pc_mock and s.home_known and s.at_home
    TeleopStatus.from_dict(s.to_dict()) if hasattr(TeleopStatus, "from_dict") else None
    last = s.messages[-1].id
    h.u1(StopArm())
    newer = h.status(after=last).messages
    assert [m.text for m in newer] == ["arm stopped: it holds where it is"]


# ---- review fixes: no rescans on the main loop, write errors, health timeouts --------------------------------------

def test_the_lists_are_never_reread_while_recording_or_by_status(harness, monkeypatch):
    from teleop_collect.teleop.episode.store import Task, TaskStore
    h = harness()
    h.ready()
    reads = {"episodes": 0, "tasks": 0}
    real_episodes, real_tasks = Task.episodes, TaskStore.tasks

    def episodes(self):
        reads["episodes"] += 1
        return real_episodes(self)

    def tasks(self):
        reads["tasks"] += 1
        return real_tasks(self)
    monkeypatch.setattr(Task, "episodes", episodes)
    monkeypatch.setattr(TaskStore, "tasks", tasks)
    for _ in range(20):
        h.status()
        h.tick()
    h.start_episode()
    for _ in range(20):
        h.status()
        h.hold(["x+"], 1)
    h.u1(StopEpisode(outcome="success"))
    h.tick(2)
    h.start_episode()
    h.tick(2)
    h.u1(StopEpisode(outcome="fail"))
    assert [e.name for e in h.status().episodes] == ["episode_000", "episode_001"]    # from what was written
    assert h.u1(DeleteEpisode(episode="episode_000")).ok
    assert [e.name for e in h.status().episodes] == ["episode_001"]
    assert reads == {"episodes": 0, "tasks": 0}
    assert [e.name for e in real_episodes(h.store.open("pick_block"))] == ["episode_001"]  # the cache equals the disk
    h.u1(SelectTask(name="pick_block"))
    assert reads["episodes"] == 1                                         # a task change reads it once


def test_a_write_error_on_the_main_thread_aborts_the_episode(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(2)

    def full(*args, **kw):
        raise OSError(28, "No space left on device")
    h.loop.flow.recording.commands.write = full
    h.hold(["x+"], 3)
    assert h.loop.phase is Phase.IDLE and h.loop.verdict is None and not ep_dir(h).exists()
    assert any("recording could not keep up / disk error" in t and "No space left" in t
               for t in h.loop.messages.texts())
    h.tick(5)
    h.start_episode()                                                    # the next episode records normally
    h.hold(["x+"], 3)
    assert h.u1(StopEpisode(outcome="success")).ok and (ep_dir(h, "episode_001") / "COMPLETE").exists()


def test_health_timeouts_are_tolerated_three_times_in_a_row(harness):
    from deploy_contracts import TransportTimeout
    h = harness()
    h.ready()
    h.start_episode()
    h.frames.error["get_health"] = TransportTimeout("frames", "tcp://x", "r1.get_health/v1", 0.05)
    warnings = lambda: sum("no reply to get_health" in t for t in h.loop.messages.texts())   # noqa: E731
    h.run_until(lambda: warnings() == 3, 100)
    assert h.loop.phase is Phase.RECORDING                               # still recording
    del h.frames.error["get_health"]
    h.tick(10)                                                           # one answer resets the count
    h.frames.error["get_health"] = TransportTimeout("frames", "tcp://x", "r1.get_health/v1", 0.05)
    h.run_until(lambda: warnings() == 6, 100)
    assert h.loop.phase is Phase.RECORDING
    h.run_until(lambda: h.loop.phase is Phase.DONE, 100)
    assert h.code is C.NO_ANSWER and not ep_dir(h).exists()


def test_the_launcher_stop_text_is_neutral(harness):
    h = harness()
    h.ready()
    h.loop.terminate()
    h.tick()
    assert h.loop.verdict.what_failed == "stopped by the launcher (Ctrl-C, or another process exited)"


# ---- final review fixes --------------------------------------------------------------------------------------------

def test_an_open_hand_wider_than_the_limit_is_clipped_at_start(harness):
    h = harness()
    h.width = 0.081                                                      # the Franka Hand can read a bit over 0.08
    h.ready()
    h.start_episode()
    assert h.loop.motion.commanded_width_m == 0.08
    assert any("start width 0.0810 m clipped to 0.0800 m" in t for t in h.loop.messages.texts())
    h.hold(["x+"], 3)
    assert all(m.gripper_width_m.max() <= 0.08 for _, m in h.pc.sent)


@pytest.mark.parametrize("error", ["timeout", "broken"])
def test_an_r1_stop_failure_aborts_instead_of_concluding(harness, error):
    from deploy_contracts import ContractError, TransportTimeout
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 3)
    h.frames.error["stop_recording"] = (TransportTimeout("frames", "tcp://x", "r1.stop_recording/v1", 4.0)
                                        if error == "timeout" else
                                        ContractError("r1.summary/v1", "packets", "type", "not an int"))
    n = h.pc.calls.count("stop")
    r = h.u1(StopEpisode(outcome="success"))
    assert not r.ok and "the episode was deleted" in r.reason
    assert h.pc.calls.count("stop") == n + 1                             # the C3 stop went out first
    assert h.loop.phase is Phase.IDLE and h.loop.verdict is None and not ep_dir(h).exists()
    assert any("recording could not keep up / disk error: r1.stop_recording" in t for t in h.loop.messages.texts())
    del h.frames.error["stop_recording"]
    h.frames.recording = False
    h.tick(3)
    h.start_episode()                                                    # the app stays up


def test_delete_is_refused_while_an_episode_records(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(2)
    h.u1(StopEpisode(outcome="success"))
    h.tick(2)
    h.start_episode()
    r = h.u1(DeleteEpisode(episode="episode_000"))
    assert not r.ok and r.reason == "stop the episode first"
    assert (ep_dir(h) / "COMPLETE").exists()
    h.u1(StopEpisode(outcome="discard"))
    assert h.u1(DeleteEpisode(episode="episode_000")).ok


def test_a_camera_down_for_longer_than_down_s_says_restart(harness):
    h = harness()
    h.ready()
    h.frames.main = "down"
    h.tick(10)                                                           # 0.2 s
    assert h.u1(StartEpisode()).reason == "main camera down"
    h.tick(110)                                                          # 2.4 s > health.down_s 2.0
    assert h.u1(StartEpisode()).reason == ("main camera down: check the Pi (after a Pi restart: Ctrl-C and run "
                                           "teleop_collect again)")
    h.frames.main = "ok"
    h.tick(10)
    assert h.status().start_allowed
