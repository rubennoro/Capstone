"""EpisodeRecording: the columns of PLAN 6.2, exact values, counts, thread safety, close, errors."""
from __future__ import annotations

import csv
import json
import threading

import numpy as np
import pytest
from core_helpers import MS, T0_PC, arm_state, motion_settings, tilted_pose
from deploy_contracts import Ack, ErrorReply, SetWaypoints, SetWaypointsReply, WaypointIssue

from teleop_collect.teleop.episode import (CLOCK_PC_FILE, COMMANDS_FILE, EVENTS_FILE, KEYS_FILE, STATES_FILE,
                                           EpisodeRecording, TaskStore, command_columns)
from teleop_collect.teleop.motion import HeldKeys, MotionController, BaseState, Twist


@pytest.fixture
def rec(tmp_path):
    episode = TaskStore(tmp_path).create("t", "", {}).new_episode()
    r = EpisodeRecording(episode, waypoints=2)
    yield r
    r.close()


def rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def mat_cols(p):
    return [f"{p}_{r}{c}" for r in range(4) for c in range(4)]


def test_state_columns_and_values(rec):
    T = tilted_pose()
    st = arm_state(7, t_ns=T0_PC, T=T, width=0.0321, pulled_back=2, loop_late=3, error="")
    rec.states.write(st, 1_000, 999)
    rec.states.write(arm_state(8), 2_000, None)
    rec.close()
    head, r1, r2 = rows(rec.episode.path / STATES_FILE)
    assert head == ["t_pc_ns", "t_ns", "arrival_ns", "seq", "mode", *mat_cols("O_T_TCP"), *mat_cols("O_T_TCP_target"),
                    *[f"q_{i}" for i in range(7)], *[f"dq_{i}" for i in range(7)], "gripper_width_m", "gate_accepted",
                    "gate_shortened", "gate_pulled_back", "gate_rejected", "loop_late", "error"]
    d = dict(zip(head, r1))
    assert (d["t_pc_ns"], d["t_ns"], d["arrival_ns"], d["seq"], d["mode"]) == (str(T0_PC), "999", "1000", "7",
                                                                              "holding")
    got = np.array([float(d[c]) for c in mat_cols("O_T_TCP")]).reshape(4, 4)
    assert np.array_equal(got, T)                                      # row-major, exact (repr round trip)
    assert float(d["gripper_width_m"]) == 0.0321 and d["gate_pulled_back"] == "2" and d["loop_late"] == "3"
    assert dict(zip(head, r2))["t_ns"] == ""                           # not mapped: empty


def waypoints_msg(n, chunk_id=5):
    return SetWaypoints(chunk_id=chunk_id, t_ns=np.array([T0_PC + (k + 1) * 50 * MS for k in range(n)],
                                                         dtype=np.int64),
                        O_T_TCP=np.stack([tilted_pose()] * n), gripper_width_m=np.full(n, 0.04))


def test_command_rows(rec):
    tw = Twist(np.array([0.05, 0.0, 0.0]), 0.1, np.array([0.0, -0.2]))
    rec.commands.write(1, 2, "c3.set_waypoints/v1", waypoints_msg(2),
                       SetWaypointsReply(chunk_id=5, accepted=2, shortened=1, skipped=0, rejected=()), tw, True)
    rec.commands.write(3, 4, "c3.set_waypoints/v1", waypoints_msg(1, 6),
                       SetWaypointsReply(chunk_id=6, accepted=0, shortened=0, skipped=1, rejected=()), tw, False)
    rec.commands.write(5, 6, "c3.heartbeat/v1", None, Ack(ok=True, reason=""), None, False)
    rec.commands.write(7, None, "c3.heartbeat/v1", None, None, None, False)
    rec.commands.write(8, 9, "c3.set_waypoints/v1", waypoints_msg(1, 7),
                       ErrorReply(request="c3.set_waypoints/v1", error="mode is stopped", field=""), None, False)
    rec.commands.write(10, 11, "c3.set_waypoints/v1", waypoints_msg(1, 8),
                       SetWaypointsReply(chunk_id=8, accepted=0, shortened=0, skipped=0,
                                         rejected=(WaypointIssue(index=0, reason="outside box"),)), None, False)
    with pytest.raises(ValueError, match="slots"):
        rec.commands.write(1, 2, "c3.set_waypoints/v1", waypoints_msg(3), None, None, False)
    counts = rec.counts()
    rec.close()
    head, *body = rows(rec.episode.path / COMMANDS_FILE)
    assert tuple(head) == command_columns(2)
    assert head[:5] == ["t_send_ns", "t_reply_ns", "request", "chunk_id", "n"]
    assert head[5:23] == ["wp0_t_pc_ns", *mat_cols("wp0"), "wp0_width_m"]
    assert head[-14:] == ["twist_vx_m_s", "twist_vy_m_s", "twist_vz_m_s", "twist_w_twist_rad_s",
                          "twist_w_tilt_x_rad_s", "twist_w_tilt_y_rad_s", "slow", "accepted", "shortened", "skipped",
                          "rejected", "rejected_reasons", "ack_ok", "ack_reason"]
    d = [dict(zip(head, r)) for r in body]
    assert len(d) == 6
    assert d[0]["chunk_id"] == "5" and d[0]["n"] == "2" and d[0]["wp1_t_pc_ns"] == str(T0_PC + 100 * MS)
    assert np.array_equal(np.array([float(d[0][c]) for c in mat_cols("wp1")]).reshape(4, 4), tilted_pose())
    assert d[0]["twist_vx_m_s"] == "0.05" and d[0]["slow"] == "1" and d[0]["shortened"] == "1"
    assert d[1]["wp1_t_pc_ns"] == "" and d[1]["wp1_width_m"] == ""              # a 1-waypoint set: padded
    assert d[2]["chunk_id"] == "" and d[2]["n"] == "" and d[2]["ack_ok"] == "1" and d[2]["twist_vx_m_s"] == ""
    assert d[3]["t_reply_ns"] == "" and d[3]["ack_ok"] == ""
    assert d[4]["ack_ok"] == "0" and d[4]["ack_reason"] == "error: mode is stopped"
    assert d[5]["rejected"] == "1" and d[5]["rejected_reasons"] == "0: outside box"
    assert counts["requests"] == 6 and counts["sets_sent"] == 4 and counts["sets_all_skipped"] == 1
    assert counts["waypoints_sent"] == 5 and counts["waypoints_shortened"] == 1 and counts["no_reply"] == 1
    assert counts["waypoints_rejected"] == 1 and counts["waypoints_skipped"] == 1


def test_controller_commands_fit_the_columns(tmp_path):
    """A real move and release, as E3 sends them, fit commands.csv with K = MotionSettings.waypoints."""
    s = motion_settings()
    episode = TaskStore(tmp_path).create("t", "", {}).new_episode()
    rec = EpisodeRecording(episode, waypoints=s.waypoints)
    c = MotionController(s, 0.08)
    c.begin(0.08)
    for keys in (HeldKeys(frozenset({"x+"}), False), HeldKeys(frozenset(), False)):
        cmd = c.step(T0_PC, BaseState(tilted_pose(), T0_PC, 0.08, "holding"), keys)
        msg = SetWaypoints(chunk_id=1, t_ns=np.array([w.t_pc_ns for w in cmd.waypoints], dtype=np.int64),
                           O_T_TCP=np.stack([w.O_T_TCP for w in cmd.waypoints]),
                           gripper_width_m=np.array([w.width_m for w in cmd.waypoints]))
        rec.commands.write(1, 2, msg.SCHEMA, msg, None, cmd.twist, cmd.slow)
    rec.close()
    assert len(rows(episode.path / COMMANDS_FILE)) == 3


def test_keys_clock_events(rec):
    rec.keys.write(10, HeldKeys(frozenset({"tilt_left", "x+"}), True), True)
    rec.keys.write(20, HeldKeys(frozenset(), False), False)
    rec.pc_clock.hello(100, 104, 5_000)
    rec.pc_clock.fit(200, -123, 1.5, 20, 6, 0.8)
    rec.pc_clock.fit(300, None, None, 0, 0, None)
    rec.events.write(1, "start", {"episode": "episode_000", "arr": np.arange(2)})
    rec.events.write(2, "hint", "at workspace limit")
    rec.events.write(3, "stop", None)
    counts = rec.counts()
    rec.close()
    assert rows(rec.episode.path / KEYS_FILE) == [["t_ns", "held", "slow", "focus"], ["10", "x+ tilt_left", "1", "1"],
                                                  ["20", "", "0", "0"]]
    assert rows(rec.episode.path / CLOCK_PC_FILE) == [
        ["kind", "t_send_ns", "t_recv_ns", "t_pc_ns", "t_ns", "offset_ns", "drift_ppm", "samples", "used",
         "rtt_min_ms"],
        ["hello", "100", "104", "5000", "", "", "", "", "", ""],
        ["fit", "", "", "", "200", "-123", "1.5", "20", "6", "0.8"],
        ["fit", "", "", "", "300", "", "", "0", "0", ""]]
    lines = [json.loads(x) for x in (rec.episode.path / EVENTS_FILE).read_text().splitlines()]
    assert lines == [{"t_ns": 1, "kind": "start", "detail": {"episode": "episode_000", "arr": [0, 1]}},
                     {"t_ns": 2, "kind": "hint", "detail": "at workspace limit"},
                     {"t_ns": 3, "kind": "stop", "detail": None}]
    assert counts["key_changes"] == 2 and counts["hellos"] == 1


def test_state_counts(rec):
    for seq, pb, late in ((10, 5, 1), (11, 5, 1), (14, 6, 2), (15, 8, 4)):
        rec.states.write(arm_state(seq, pulled_back=pb, loop_late=late), 0, 0)
    c = rec.counts()
    assert (c["c2_messages"], c["c2_seq_gaps"], c["c2_seq_missing"]) == (4, 1, 2)
    assert c["pulled_back"] == 3 and c["loop_late"] == 3


def test_empty_counts(rec):
    c = rec.counts()
    assert c["c2_messages"] == 0 and c["pulled_back"] == 0 and c["sets_sent"] == 0


def test_rows_are_flushed_before_close(rec):
    rec.states.write(arm_state(1), 0, 0)
    rec.keys.write(1, HeldKeys(frozenset(), False), True)
    rec.events.write(1, "start")
    assert len(rows(rec.episode.path / STATES_FILE)) == 2
    assert len(rows(rec.episode.path / KEYS_FILE)) == 2
    assert (rec.episode.path / EVENTS_FILE).read_text().count("\n") == 1


def test_close_is_idempotent_and_later_writes_ignored(rec):
    rec.close()
    rec.close()
    assert rec.closed
    rec.states.write(arm_state(1), 0, 0)
    rec.commands.write(1, 2, "c3.heartbeat/v1", None, None, None, False)
    rec.keys.write(1, HeldKeys(frozenset(), False), True)
    rec.pc_clock.hello(1, 2, 3)
    rec.events.write(1, "late")
    assert len(rows(rec.episode.path / STATES_FILE)) == 1


def test_files_never_overwritten(rec):
    with pytest.raises(FileExistsError):
        EpisodeRecording(rec.episode)


def test_state_writes_from_a_thread_while_closing(rec):
    stop = threading.Event()
    errors = []

    def listener():
        seq = 0
        try:
            while not stop.is_set():
                seq += 1
                rec.states.write(arm_state(seq), seq, seq)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    th = threading.Thread(target=listener)
    th.start()
    while rec.counts()["c2_messages"] < 50:
        pass
    rec.close()
    stop.set()
    th.join(5)
    assert errors == []
    body = rows(rec.episode.path / STATES_FILE)[1:]
    assert len(body) == rec.counts()["c2_messages"] and all(len(r) == len(body[0]) for r in body)


def test_state_write_error_is_kept_not_raised(rec, monkeypatch):
    def broken(values):
        raise OSError("disk full")

    monkeypatch.setattr(rec.states._csv, "row", broken)
    rec.states.write(arm_state(1), 0, 0)                               # never raises into the listener
    assert "disk full" in rec.error
    rec.states.write(arm_state(2), 0, 0)
    assert rec.counts()["c2_messages"] == 0
