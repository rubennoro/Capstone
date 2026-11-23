"""TeleopLoop and EpisodeFlow on the fake-clock harness (STRUCTURE.md 6.3-6.7, 12 row E3; PLAN 5, 7, 8): start-up,
one C3 request per cycle, hello priority, the teleop step, start / stop / discard / abort / conclude, the start pose
(learned at start-up, confirmed or dropped by the PC's start reply, no laptop home), start refusals, the guards.
Wave 3 (no software stops): a C2 seq gap never pauses motion, an old state only beyond 250 ms; a camera down or stale
and a recording failure warn (page, meta.yaml warnings) and the episode goes on (a failed one can only be discarded).
E4's motion and episode classes are the real ones; the PC and frames are fakes."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import yaml
from teleop_helpers import FAST, NS_PER_MS, TeleopHarness, merge, rejecting_reply
from deploy_contracts import SetWaypointsReply

from teleop_collect.contracts import (CreateTask, DeleteEpisode, DeleteTask, GripperCommand, SelectTask, SetDataRoot,
                                      StartEpisode, StopArm, StopEpisode, TeleopStatus)
from teleop_collect.contracts.station import STORAGE_FILE, Station
from teleop_collect.teleop.episode.store import TaskStore
from teleop_collect.teleop.episode_flow import (FULL_PATH, NOT_AT_START_POSE, START_POSE_UNKNOWN, START_REFUSED_HERE)
from teleop_collect.teleop.problems import ProblemCode as C
from teleop_collect.teleop.teleop_loop import Phase

PC_REFUSAL = "refused: not at the start pose: press h, then p on the PC"   # control_deploy's onStart text


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

def test_startup_order_learns_the_start_pose_and_ready(harness, tmp_path):
    h = harness()
    ready = tmp_path / "ready" / "teleop.ready"
    h.start(ready)
    assert h.loop.phase is Phase.IDLE, h.verdict_text()
    calls = h.pc.calls
    first_get_state = calls.index("get_state")
    assert set(calls[:first_get_state]) == {"hello"} and first_get_state >= 5          # hellos until the map is ready
    assert "home" not in calls and set(calls) <= {"hello", "heartbeat", "get_state"}   # no C3 home: nothing moves
    assert "get_health" in h.frames.calls and ready.read_text() == "ready\n"
    assert (h.session / "session.yaml").exists()
    ref = h.loop.start_pose
    assert ref.known and not ref.confirmed and h.loop.at_start_pose()
    assert np.allclose(ref.p_ref, h.T[:3, 3])
    text = "start pose: the arm's pose now (0.450, 0.000, 0.300 m); every episode starts here (h, then p on the PC)"
    assert text in h.loop.messages.texts()
    events = h.events()
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "startup" and "ready" in kinds and "home" not in kinds
    assert kinds.index("start_pose") < kinds.index("ready")
    h.tick(3)
    s = h.status()
    assert s.start_pose_known and not s.start_pose_confirmed and s.at_start_pose
    assert s.start_refusal == "select or create a task first"


def test_startup_while_the_pc_moves_the_arm_learns_nothing(harness):
    """Step 10 with C2 showing homing (the PC's h, g or p pressed during the camera wait): no pose is learned; the
    pose after that move is the start pose."""
    from teleop_collect.teleop.startup_sequence import StartupSequence
    h = harness()
    h.mode_override = "homing"
    h._sleep(0.1)
    StartupSequence(h.loop)._learn_start_pose()
    assert not h.loop.start_pose.known
    assert ("start pose not learned: the PC is homing; the pose after its move is the start pose"
            in h.loop.messages.texts())
    h.loop._observe_state(h.clock())                                    # homing seen
    h.T = h.T.copy()
    h.T[2, 3] += 0.05
    h.mode_override = None
    h._sleep(0.05)
    h.loop._observe_state(h.clock())                                    # idle after a move: the candidate
    assert h.loop.start_pose.known and h.loop.at_start_pose()


def test_startup_skip_seeds_the_observe_so_a_move_ending_before_the_loop_counts(harness):
    """The race: step 10 sees homing, and the PC's move ends before the loop's first observe (which sees idle only).
    The skipped learn passes its homing C2 to observe, so that homing -> idle is still the candidate."""
    from teleop_collect.teleop.startup_sequence import StartupSequence
    h = harness()
    h.mode_override = "homing"
    h._sleep(0.1)
    StartupSequence(h.loop)._learn_start_pose()
    assert not h.loop.start_pose.known
    h.T = h.T.copy()
    h.T[2, 3] += 0.05
    h.mode_override = None
    h._sleep(0.05)
    h.loop._observe_state(h.clock())                                    # the loop's first observe: idle, moved
    assert h.loop.start_pose.known and h.loop.at_start_pose()
    assert np.allclose(h.loop.start_pose.p_ref, h.T[:3, 3])


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
    assert len(moves) >= 8 and all(len(m.t_ns) == 6 for m in moves)                # K 6 (wave 3)
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
    h.state_delay_ns = 300 * NS_PER_MS                                   # every C2 300 ms old (> 250 ms)
    h.ready()
    h.start_episode()
    h.tick(2)
    n = len(h.pc.sent)
    h.hold(["x+"], 20)
    assert len(h.pc.sent) == n
    assert h.monitor.counts().get(C.STATE_OLD, 0) > 0 and h.loop.phase is Phase.RECORDING
    h2 = harness()                                                       # 200 ms: within state_max_age_ms 250
    h2.state_delay_ns = 200 * NS_PER_MS
    h2.ready("second_task")
    h2.start_episode()
    h2.hold(["x+"], 5)
    assert len(h2.pc.sent) >= 4 and h2.monitor.counts().get(C.STATE_OLD, 0) == 0


def test_a_c2_seq_gap_never_pauses_motion(harness):
    """Wave 3: a seq gap is counted and logged (state_gap), the set still goes out (the planner extrapolates from
    the newest C2's own target and time)."""
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 3)
    sent = 0
    for _ in range(10):
        h.seq += 3                                                       # C2 messages lost before every cycle
        n = len(h.pc.sent)
        h.hold(["x+"], 1)
        sent += len(h.pc.sent) - n
    assert sent >= 9                                                     # (a hello may take one cycle)
    assert h.monitor.counts().get(C.STATE_GAP, 0) >= 9 and h.loop.phase is Phase.RECORDING
    assert any(e["kind"] == "state_gap" and "motion goes on" in e["detail"] for e in h.events())
    h.keys()
    h.tick(3)
    assert h.u1(StopEpisode(outcome="success")).ok
    meta = yaml.safe_load((ep_dir(h) / "meta.yaml").read_text())
    assert meta["counts"]["state_gap"] >= 9 and meta["warnings"] == []


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
    assert h.pc.sent[-1][1].gripper_width_m.tolist() == [0.08] * 6
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
    h.q = np.full(7, 0.05)                                               # moved away from the start pose
    h.tick(3)
    assert h.u1(StartEpisode()).reason == "not at the start pose: h, then p on the PC" == NOT_AT_START_POSE
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
    assert h.u1(StartEpisode()).reason == "the PC is homing"             # the PC's h, g or p: the mode rule
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


# ---- the start pose ------------------------------------------------------------------------------------------------

def test_an_ok_start_confirms_the_start_pose(harness):
    h = harness()
    h.ready()
    assert not h.loop.start_pose.confirmed and h.status().start_allowed
    h.start_episode()
    assert h.loop.start_pose.confirmed
    assert any(e["kind"] == "start_pose" and e["detail"].startswith("confirmed") for e in h.events())
    h.tick(3)
    assert h.u1(StopEpisode(outcome="success")).ok
    h.tick(3)
    s = h.status()
    assert s.start_pose_confirmed and s.start_pose_known and s.at_start_pose and s.start_allowed


def test_the_pc_refusing_away_from_its_start_pose_drops_the_learned_pose(harness):
    """Start-up learned the wrong pose (the operator forgot p); the PC refuses the start; h, then p on the PC give
    the new candidate, and the next start the PC accepts confirms it."""
    from deploy_contracts import Ack
    h = harness()
    h.ready()
    wrong = h.T.copy()
    h.pc.start_ack = Ack(ok=False, reason=PC_REFUSAL)
    r = h.u1(StartEpisode())
    assert not r.ok and r.reason == f"the PC refused start: {PC_REFUSAL}"
    assert not ep_dir(h).exists() and h.loop.phase is Phase.IDLE
    ref = h.loop.start_pose
    assert not ref.known and not ref.confirmed and ref.at_refused_pose(h.q, wrong)
    h.tick(3)
    s = h.status()
    assert not s.start_pose_known and not s.at_start_pose
    assert s.start_refusal == START_REFUSED_HERE
    assert START_REFUSED_HERE == "the PC refused the start here (not the start pose): press p on the PC"
    assert any(e["kind"] == "start_pose" and "dropped" in e["detail"] for e in h.events())
    home = wrong.copy()
    home[:3, 3] = (0.30, 0.0, 0.45)
    h.pc_move(T=home)                                                    # h on the PC: the arm goes home
    assert h.status().start_allowed                                      # home is the candidate now; the PC decides
    assert h.u1(StartEpisode()).reason == f"the PC refused start: {PC_REFUSAL}"
    h.tick(3)
    assert h.status().start_refusal == START_REFUSED_HERE                # at home: press p
    start = wrong.copy()
    start[:3, 3] = (0.50, 0.05, 0.25)
    h.pc_move(T=start)                                                   # p on the PC: the start pose
    s = h.status()
    assert s.start_pose_known and not s.start_pose_confirmed and s.at_start_pose and s.start_allowed
    h.pc.start_ack = Ack(ok=True, reason="")
    h.start_episode()
    assert h.loop.start_pose.confirmed and np.allclose(h.loop.start_pose.p_ref, start[:3, 3])


def test_other_pc_refusals_keep_the_learned_pose(harness):
    from deploy_contracts import Ack
    h = harness()
    h.ready()
    h.pc.start_ack = Ack(ok=False, reason="refused: mode is stopped")
    assert not h.u1(StartEpisode()).ok
    assert h.loop.start_pose.known and not h.loop.start_pose.confirmed


def test_not_known_and_away_from_the_refused_pose(harness):
    from deploy_contracts import Ack
    h = harness()
    h.ready()
    h.pc.start_ack = Ack(ok=False, reason=PC_REFUSAL)
    h.u1(StartEpisode())
    h.T = h.T.copy()
    h.T[0, 3] += 0.02                                                    # moved, but not by the PC's h/p
    h.tick(3)
    assert h.status().start_refusal == START_POSE_UNKNOWN == "put the arm at the start pose: h, then p on the PC"


def test_the_pcs_g_never_sets_the_start_pose(harness):
    from deploy_contracts import Ack
    h = harness()
    h.ready()
    h.pc.start_ack = Ack(ok=False, reason=PC_REFUSAL)
    h.u1(StartEpisode())
    h.pc_move()                                                          # g: homing -> idle, the arm stays
    assert not h.loop.start_pose.known
    assert h.status().start_refusal == START_REFUSED_HERE


def test_a_confirmed_start_pose_is_kept_across_an_h(harness):
    """Confirmed: the PC's h (home, joints far away) changes nothing; its p back to the start pose allows Start."""
    h = harness()
    h.ready()
    start = h.T.copy()
    h.start_episode()
    h.hold(["x+"], 10)
    h.tick(5)
    assert h.u1(StopEpisode(outcome="success")).ok
    h.T = h.loop._history.latest().O_T_TCP_target.copy()                # where the episode left the arm
    h.T[0, 3] = start[0, 3] + 0.03
    h.pc.mode = "idle"
    h.tick(3)
    assert h.status().start_refusal == NOT_AT_START_POSE
    other = start.copy()
    other[:3, 3] = (0.30, 0.0, 0.45)
    q_start = h.q.copy()
    h.pc_move(T=other, q=q_start + 0.5)                                  # h: home; not the confirmed start pose
    assert h.status().start_refusal == NOT_AT_START_POSE and h.loop.start_pose.confirmed
    h.pc_move(T=start, q=q_start)                                        # p: back at the start pose
    assert h.status().start_allowed
    h.start_episode()


def test_a_pc_move_ending_within_reanchor_tol_re_anchors_the_confirmed_pose(harness):
    """The PC's p ends within its homing.at_pose_tol_rad (0.05) but beyond teleop's 0.01 rad / 5 mm: teleop follows
    it (re-anchored, still confirmed) instead of refusing Start there for the rest of the session."""
    h = harness()
    h.ready()
    start, q_start = h.T.copy(), h.q.copy()
    h.start_episode()
    assert h.loop.start_pose.confirmed
    assert h.u1(StopEpisode(outcome="success")).ok
    h.pc.mode = "idle"
    h.tick(3)
    home = start.copy()
    home[:3, 3] = (0.30, 0.0, 0.45)
    h.pc_move(T=home, q=q_start + 0.5)                                   # h
    end_T, end_q = start.copy(), q_start + 0.04
    end_T[0, 3] += 0.008
    h.pc_move(T=end_T, q=end_q)                                          # p: within 0.05 rad of the confirmed q
    ref = h.loop.start_pose
    assert ref.confirmed and np.allclose(ref.q_ref, end_q) and np.allclose(ref.p_ref, end_T[:3, 3])
    assert h.status().start_allowed
    assert any(e["kind"] == "start_pose" and e["detail"].startswith("re-anchored") for e in h.events())


def test_start_rule_order(harness):
    """The mode rule, then the start pose, then the cameras, then the clock."""
    h = harness()
    h.ready()
    h.frames.main = "stale"
    h.tick(10)
    h.q = np.full(7, 0.05)
    h.mode_override = "homing"
    h.tick(2)
    assert h.status().start_refusal == "the PC is homing"
    h.mode_override = None
    h.tick(2)
    assert h.status().start_refusal == NOT_AT_START_POSE
    h.q = np.zeros(7)
    h.tick(2)
    assert h.status().start_refusal == "main camera stale"


def test_there_is_no_home_request(harness):
    """No laptop home: no U1 home request, no handler, no HOMING phase."""
    import teleop_collect.contracts as contracts
    from teleop_collect.contracts import PHASES
    from teleop_collect.teleop.teleop_loop import U1_HANDLERS
    from teleop_collect.teleop.u1_server import U1_REQUESTS
    assert not hasattr(contracts, "HomeRequest") and "homing" not in PHASES
    assert {cls.SCHEMA for cls in U1_HANDLERS} == {cls.SCHEMA for cls in U1_REQUESTS}
    assert not any(cls.SCHEMA == "u1.home/v1" for cls in U1_REQUESTS)
    assert not hasattr(Phase, "HOMING")


def test_stop_arm_while_recording_aborts(harness):
    h = harness()
    h.ready()
    h.start_episode()
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


# ---- aborts (EPISODE) ----------------------------------------------------------------------------------------------

def test_the_pc_stopping_the_arm_aborts_and_the_app_stays_up(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 3)
    h.pc.mode = "stopped"                                                # its watchdog, or s on the PC
    h.tick(30)
    assert h.loop.phase is Phase.IDLE and h.loop.verdict is None
    assert not ep_dir(h).exists()
    assert any("the PC stopped the arm" in t for t in h.loop.messages.texts()), h.loop.messages.texts()
    h.tick(10)
    h.start_episode()                                                    # ready for the next one
    assert ep_dir(h, "episode_001").exists()


# ---- data problems (WARN, wave 3): the arm and the episode go on ----------------------------------------------------

def _sets_and_stops(h):
    return h.pc.calls.count("set_waypoints"), h.pc.calls.count("stop")


@pytest.mark.parametrize("camera, status, text, code", [
    ("main", "down", "main camera down: check the Pi and ASUS_5g", "camera_down"),
    ("ultrawide", "stale", "ultrawide camera stale, last frame 500 ms ago: check the Pi stream and ASUS_5g",
     "camera_stale"),
])
def test_a_camera_problem_while_recording_warns_and_the_episode_goes_on(harness, camera, status, text, code):
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 3)
    sets, stops = _sets_and_stops(h)
    setattr(h.frames, camera, status)
    h.hold(["x+"], 60)                                                   # 1.2 s: two or three health polls
    assert h.loop.phase is Phase.RECORDING and h.loop.verdict is None
    new_sets, new_stops = _sets_and_stops(h)
    assert new_stops == stops and new_sets >= sets + 55                  # no C3 stop, motion never paused
    warned = [m for m in h.loop.messages.after(0, 10**6) if code.replace("_", " ") in m.text]
    assert [(m.level, m.text) for m in warned] == [                     # once per episode, not per poll
        ("warn", f"episode_000: {text}; the episode goes on (noted in its meta.yaml)")]
    assert any(e["kind"] == "episode_warning" for e in h.events())
    setattr(h.frames, camera, "ok")
    h.keys()
    h.tick(5)
    assert h.u1(StopEpisode(outcome="success")).ok                      # the operator decides: saved
    meta = yaml.safe_load((ep_dir(h) / "meta.yaml").read_text())
    [w] = meta["warnings"]
    assert w["code"] == code and w["what"] == text.split(": check")[0] and 0.0 < w["t_s"] < 1.0
    assert meta["counts"][code] >= 2
    events = [json.loads(line) for line in (ep_dir(h) / "events.jsonl").read_text().splitlines()]
    assert [e["detail"]["code"] for e in events if e["kind"] == "warning"] == [code]
    h.tick(10)
    h.start_episode()                                                    # the next episode starts clean
    h.tick(3)
    assert h.u1(StopEpisode(outcome="fail")).ok
    assert yaml.safe_load((ep_dir(h, "episode_001") / "meta.yaml").read_text())["warnings"] == []


def test_a_camera_down_still_refuses_start_between_episodes(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.frames.main = "down"
    h.tick(30)
    assert h.u1(StopEpisode(outcome="success")).ok                      # saved, with its warning
    h.tick(10)
    r = h.u1(StartEpisode())
    assert not r.ok and r.reason == "main camera down" and h.loop.phase is Phase.IDLE
    h.frames.main = "ok"
    h.tick(30)
    h.start_episode()


def test_a_recording_failure_warns_and_only_discard_ends_the_episode(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.hold(["x+"], 3)
    sets, stops = _sets_and_stops(h)
    h.frames.recording_error = "queue full"                              # frames' writer failed ...
    h.frames.summary_error = "queue full"                                # ... and its summary says so at stop
    h.hold(["x+"], 60)
    assert h.loop.phase is Phase.RECORDING and h.loop.verdict is None
    new_sets, new_stops = _sets_and_stops(h)
    assert new_stops == stops and new_sets >= sets + 55
    warned = [m for m in h.loop.messages.after(0, 10**6) if "recording could not keep up" in m.text]
    assert [(m.level, m.text) for m in warned] == [
        ("warn", "episode_000: recording could not keep up / disk error: queue full: check the free disk space under "
                 "data_root and the frames log; the arm goes on: Stop and Discard (this episode cannot be saved)")]
    for outcome in ("success", "fail"):
        r = h.u1(StopEpisode(outcome=outcome))
        assert not r.ok and r.reason == ("this episode cannot be saved (recording could not keep up / disk error: "
                                         "queue full): Discard it")
        assert h.loop.phase is Phase.RECORDING and h.pc.calls.count("stop") == stops
    h.keys()
    h.tick(3)
    assert h.u1(StopEpisode(outcome="discard")).ok                      # the summary's error: deleted anyway
    assert h.loop.phase is Phase.IDLE and not ep_dir(h).exists() and h.loop.verdict is None
    assert h.pc.calls.count("stop") == stops + 1                         # the operator's stop, the only one
    assert any(t == "episode_000 discarded (deleted)" for t in h.loop.messages.texts())
    h.frames.recording_error = h.frames.summary_error = ""
    h.tick(10)
    h.start_episode()                                                    # the next one can be saved again
    h.tick(3)
    assert h.u1(StopEpisode(outcome="success")).ok and (ep_dir(h, "episode_001") / "COMPLETE").exists()


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


# ---- status --------------------------------------------------------------------------------------------------------

def test_status_is_a_valid_contract_message_and_messages_page(harness):
    h = harness()
    h.ready()
    s = h.status()
    assert isinstance(s, TeleopStatus) and s.phase == "idle" and s.ready and s.start_allowed
    assert s.arm_mode == "idle" and s.tcp_m.shape == (3,) and s.cameras.main.status == "ok"
    assert s.c2_hz > 50 and s.clock_ok and s.pc_mock and s.start_pose_known and s.at_start_pose
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


def test_a_write_error_on_the_main_thread_can_only_be_discarded(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(2)

    def full(*args, **kw):
        raise OSError(28, "No space left on device")
    h.loop.flow.recording.commands.write = full
    stops = h.pc.calls.count("stop")
    h.hold(["x+"], 3)
    assert h.loop.phase is Phase.RECORDING and h.loop.verdict is None and h.pc.calls.count("stop") == stops
    assert sum("recording could not keep up / disk error" in t and "No space left" in t
               for t in h.loop.messages.texts()) == 1
    r = h.u1(StopEpisode(outcome="success"))
    assert not r.ok and "this episode cannot be saved" in r.reason and "No space left" in r.reason
    assert h.u1(StopEpisode(outcome="discard")).ok
    assert h.loop.phase is Phase.IDLE and not ep_dir(h).exists()
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


def test_an_r1_stop_failure_at_discard_is_ok(harness):
    """Discard deletes the folder anyway: a failure found at that stop is logged, the reply is ok."""
    from deploy_contracts import TransportTimeout
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(3)
    h.frames.error["stop_recording"] = TransportTimeout("frames", "tcp://x", "r1.stop_recording/v1", 4.0)
    assert h.u1(StopEpisode(outcome="discard")).ok
    assert h.loop.phase is Phase.IDLE and h.loop.verdict is None and not ep_dir(h).exists()
    assert any(e["kind"] == "episode_stop" and "r1.stop_recording" in e["detail"] for e in h.events())


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


# ---- the storage folder and whole tasks (2026-11-22) ---------------------------------------------------------------

def storage(h) -> dict:
    return yaml.safe_load((h.settings.station_file.parent / STORAGE_FILE).read_text())


def test_set_data_root_switches_deselects_and_is_remembered(harness, tmp_path):
    h = harness()
    h.ready()
    old = h.settings.data_root
    new = tmp_path / "elsewhere" / "teleop"
    assert h.u1(SetDataRoot(path=str(new))).ok
    assert new.is_dir() and not any(new.iterdir())                       # made; the write probe is gone
    s = h.status()
    assert s.data_root == str(new.resolve()) and s.task == "" and s.tasks == () and s.episodes == ()
    assert s.start_refusal == "select or create a task first"
    assert f"storage folder: {new.resolve()}" in h.loop.messages.texts()
    assert any(e["kind"] == "data_root" for e in h.events())
    assert storage(h) == {"mock_data_root": str(new.resolve())}          # --mock harness: the mock key
    assert Station.load(h.settings.station_file, mock=True).data_root == new.resolve()
    assert Station.load(h.settings.station_file, mock=False).data_root == h.settings.other_data_root
    assert h.u1(CreateTask(name="place", description="")).ok
    assert (new / "place" / "task.yaml").exists()
    assert h.u1(SetDataRoot(path=str(old))).ok                            # back: the old tasks are there again
    s = h.status()
    assert s.tasks == ("pick_block",) and s.task == "" and storage(h) == {"mock_data_root": str(old.resolve())}


def test_set_data_root_names_the_unsaved_episodes_there_in_one_line(harness, tmp_path):
    other = TaskStore(tmp_path / "other")
    pour = other.create("pour", "", {})
    pour.new_episode()                                                   # no COMPLETE: unsaved
    pour.new_episode()
    h = harness()
    h.ready()
    n = len(h.loop.messages.texts())
    assert h.u1(SetDataRoot(path=str(tmp_path / "other"))).ok
    new = h.loop.messages.texts()[n:]
    assert new == [f"storage folder: {other.root.resolve()}",
                   f"2 unsaved episode(s) in {other.root.resolve()}: select their task to save or delete them"]
    assert h.status().tasks == ("pour",)
    assert sum(e["kind"] == "unsaved_kept" for e in h.events()) == 2


def test_set_data_root_same_folder_changes_nothing_but_is_remembered(harness):
    h = harness()
    h.ready()
    n = len(h.loop.messages.texts())
    assert h.u1(SetDataRoot(path=str(h.settings.data_root))).ok
    s = h.status()
    assert s.task == "pick_block" and len(h.loop.messages.texts()) == n
    assert storage(h) == {"mock_data_root": str(h.settings.data_root.resolve())}


@pytest.mark.parametrize("path", ["", "  ", "data", "./data", "../x", "~nosuchuser_xyz/data"])
def test_set_data_root_wants_a_full_path(harness, path):
    h = harness()
    h.ready()
    r = h.u1(SetDataRoot(path=path))
    assert not r.ok and r.reason == FULL_PATH == "give a full path: /... or ~/..."
    assert h.status().task == "pick_block"


def test_set_data_root_tilde_is_the_home_folder(harness, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    h = harness()
    h.ready()
    assert h.u1(SetDataRoot(path="~/teleop_data")).ok
    assert h.status().data_root == str((tmp_path / "home" / "teleop_data").resolve())


def test_set_data_root_refuses_the_other_modes_folder(harness):
    h = harness()
    h.ready()
    other = h.settings.other_data_root                                   # --mock harness: the real folder
    for path in (str(other), str(other / ".." / other.name)):
        r = h.u1(SetDataRoot(path=path))
        assert not r.ok and r.reason == (f"{other.resolve()} is the storage folder of real runs: --mock episodes "
                                          "are kept apart from the real ones")
    assert h.status().data_root == str(h.settings.data_root)
    assert not (h.settings.station_file.parent / STORAGE_FILE).exists()


def test_set_data_root_refuses_a_folder_inside_or_around_the_other_modes(harness):
    h = harness()
    h.ready()
    other = h.settings.other_data_root.resolve()                         # --mock harness: the real folder
    for root in (other / "sub", other.parent):
        r = h.u1(SetDataRoot(path=str(root)))
        assert not r.ok and r.reason == (f"{root} and the storage folder of real runs ({other}) are inside one "
                                          "another: --mock episodes are kept apart from the real ones")
    assert h.status().data_root == str(h.settings.data_root) and not (other / "sub").exists()


def test_set_data_root_refuses_a_folder_it_cannot_write(harness, tmp_path):
    h = harness()
    h.ready()
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    r = h.u1(SetDataRoot(path=str(blocker / "data")))
    assert not r.ok and r.reason.startswith(f"cannot use {blocker / 'data'}: ")
    assert h.status().task == "pick_block" and not (h.settings.station_file.parent / STORAGE_FILE).exists()


def test_set_data_root_not_remembered_still_switches_and_warns(harness, tmp_path):
    h = harness()
    h.ready()
    (h.settings.station_file.parent / STORAGE_FILE).write_text("[not, a, map]\n")
    new = tmp_path / "new"
    assert h.u1(SetDataRoot(path=str(new))).ok
    assert h.status().data_root == str(new.resolve())
    warn = [m for m in h.status().messages if m.level == "warn"]
    assert warn and warn[-1].text.startswith(f"storage folder {new.resolve()} is used for this run only: it could "
                                             "not be remembered (")


def test_set_data_root_and_delete_task_are_refused_while_recording(harness, tmp_path):
    h = harness()
    h.ready()
    h.start_episode()
    for msg in (SetDataRoot(path=str(tmp_path / "x")), DeleteTask(name="pick_block")):
        r = h.u1(msg)
        assert not r.ok and r.reason == "an episode is recording: stop it first"
    assert h.status().data_root == str(h.settings.data_root) and h.task_dir().is_dir()
    assert not (tmp_path / "x").exists()
    h.u1(StopEpisode(outcome="discard"))


def test_set_data_root_and_delete_task_are_refused_while_starting_or_done(harness, tmp_path):
    h = harness()
    assert h.loop.phase is Phase.STARTING
    assert h.loop.flow.set_root(str(tmp_path / "x")).reason == "starting up: wait until teleop is ready"
    assert h.loop.flow.delete_task("pick_block").reason == "starting up: wait until teleop is ready"
    h.loop._phase = Phase.DONE
    assert h.loop.flow.set_root(str(tmp_path / "x")).reason == "teleop has stopped"
    assert h.loop.flow.delete_task("pick_block").reason == "teleop has stopped"
    assert not (tmp_path / "x").exists()


def test_delete_task_removes_the_folder_and_deselects_it(harness):
    h = harness()
    h.ready()
    h.start_episode()
    h.tick(3)
    assert h.u1(StopEpisode(outcome="success")).ok
    assert h.u1(CreateTask(name="place", description="")).ok             # selects place
    assert h.u1(SelectTask(name="pick_block")).ok
    assert h.u1(DeleteTask(name="pick_block")).ok
    assert not h.task_dir().exists() and h.task_dir("place").is_dir()
    s = h.status()
    assert s.task == "" and s.tasks == ("place",) and s.episodes == ()
    assert "task pick_block deleted (1 episodes)" in h.loop.messages.texts()
    assert any(e["kind"] == "task_deleted" and e["detail"] == "pick_block (1 episodes)" for e in h.events())


def test_delete_task_of_another_task_keeps_the_selection(harness):
    h = harness()
    h.ready()
    assert h.u1(CreateTask(name="place", description="")).ok
    assert h.u1(SelectTask(name="pick_block")).ok
    assert h.u1(DeleteTask(name="place")).ok
    s = h.status()
    assert s.task == "pick_block" and s.tasks == ("pick_block",)
    assert "task place deleted (0 episodes)" in h.loop.messages.texts()


def test_delete_task_stopped_halfway_deselects_and_refuses(harness, monkeypatch):
    import teleop_collect.teleop.episode.store as store_module
    h = harness()
    h.ready()
    assert h.u1(CreateTask(name="place", description="")).ok
    assert h.u1(SelectTask(name="pick_block")).ok

    def half(path):
        (path / "task.yaml").unlink()                                    # part of it gone
        raise PermissionError(13, "Permission denied", str(path / "episode_000"))
    monkeypatch.setattr(store_module.shutil, "rmtree", half)
    r = h.u1(DeleteTask(name="pick_block"))
    assert not r.ok and r.reason.startswith("task pick_block only partly deleted: ") and "Permission denied" in r.reason
    s = h.status()
    assert s.task == "" and s.episodes == () and s.tasks == ("place",)


@pytest.fixture
def read_only(tmp_path):
    """A folder nobody can write in (skipped as root, who can)."""
    import os
    if os.geteuid() == 0:
        pytest.skip("root writes anywhere")
    folder = tmp_path / "read_only"
    folder.mkdir()
    folder.chmod(0o555)
    yield folder
    folder.chmod(0o755)


def test_startup_step_9_names_the_storage_file_when_the_folder_comes_from_it(harness, tmp_path, read_only):
    (tmp_path / STORAGE_FILE).write_text(f"mock_data_root: {read_only}\n")
    h = harness()
    assert h.settings.data_root == read_only and h.settings.data_root_source == STORAGE_FILE
    h.start()
    assert h.code is C.STARTUP and f"data_root {read_only} is not writable" in h.loop.verdict.what_failed
    assert str(tmp_path / STORAGE_FILE) in h.loop.verdict.what_to_check


def test_startup_with_a_file_as_the_storage_folder_fails_cleanly_at_step_9(harness, tmp_path):
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    (tmp_path / STORAGE_FILE).write_text(f"mock_data_root: {blocker}\n")
    h = harness()
    h.start()
    assert h.code is C.STARTUP
    assert h.loop.verdict.what_failed == f"startup: data_root {blocker} is not writable: not a folder: {blocker}"
    assert str(tmp_path / STORAGE_FILE) in h.loop.verdict.what_to_check
    assert not any(e["kind"] == "unsaved_kept" for e in h.events())


def test_startup_step_9_without_the_storage_file_names_station_yaml(harness, read_only):
    h = harness()
    h.settings.data_root.mkdir(parents=True, exist_ok=True)
    h.settings.data_root.chmod(0o555)
    try:
        h.start()
    finally:
        h.settings.data_root.chmod(0o755)
    assert h.code is C.STARTUP and "station.yaml data_root (or teleop_collect run --data-root PATH)" in \
        h.loop.verdict.what_to_check


@pytest.mark.parametrize("name,reason", [("missing", "no task missing under {root}"),
                                         ("Bad Name", "'Bad Name' is not a task name"),
                                         ("../data", "'../data' is not a task name")])
def test_delete_task_refuses_unknown_or_bad_names(harness, name, reason):
    h = harness()
    h.ready()
    r = h.u1(DeleteTask(name=name))
    assert not r.ok and r.reason == reason.format(root=h.settings.data_root)
    assert h.task_dir().is_dir() and h.status().task == "pick_block"

