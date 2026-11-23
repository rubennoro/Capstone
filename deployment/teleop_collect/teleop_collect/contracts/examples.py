# new, except _in (with its inner wrap/apply): adapted from
# rnoro/final-deployment:deployment/policy_deploy/policy_deploy/contracts/examples.py @ 2ffe641
# changes: _in's docstring example names an R1 field (main.frames) instead of a C0 one; the code is unchanged.
# + start pose (2026-11-20): no u1/home example; the status examples carry start_pose_known / start_pose_confirmed /
# at_start_pose (an idle one at the start pose, one with the start pose not known yet). + 2026-11-22: u1/label_episode,
# u1/delete_task, u1/set_data_root; the status carries data_root.
"""The laptop-only examples of R1, U1 and U2, good and deliberately broken (STRUCTURE.md 4.5).

Good ones are built from the message classes; broken ones take a good message's dict and change one thing, then pack
it (deploy_contracts.broken_example). Value paths as in deploy_contracts.examples (dotted, [i] for an index). Built in
memory, never written to deployment/contracts/examples/.

Example names: "<contract>/<name>", contract r1, u1 or u2 (e.g. "r1/get_health", "u1/status_recording").
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from deploy_contracts import Ack, Example, broken_example, drop_key, good_example, set_key

from teleop_collect.contracts.health import CameraHealthReport, HealthReport
from teleop_collect.contracts.r1_recorder import (CameraSummary, FramesHealth, GetFramesHealth, RecordingSummary,
                                                  StartRecording, StopRecording)
from teleop_collect.contracts.u1_teleop import (CreateTask, DeleteEpisode, DeleteTask, EpisodeInfo, GetStatus,
                                                GripperCommand, KeysUpdate, LabelEpisode, SelectTask, SetDataRoot,
                                                StartEpisode, StopArm, StopEpisode, TeleopStatus, UiMessage)
from teleop_collect.contracts.u2_preview import GetPreview, Preview

_T0 = 123_456_789_000_000          # a laptop monotonic time, ns
_EPISODE_DIR = "/home/operator/CapstoneTeleop/pick_block/episode_003"
#: A few bytes standing in for a JPEG (U2 carries the bytes as they are; nothing decodes them).
JPEG = b"\xff\xd8\xff\xe0" + bytes(range(32)) + b"\xff\xd9"


def health(main: str = "ok", ultrawide: str = "ok") -> HealthReport:
    age = {"ok": 20.0, "stale": 500.0, "down": 2500.0}

    def cam(status: str) -> CameraHealthReport:
        return CameraHealthReport(status=status, last_frame_age_ms=age[status], fps=30.0 if status == "ok" else 0.0,
                                  dropped=2)
    return HealthReport(main=cam(main), ultrawide=cam(ultrawide))


def _in(*path: Any) -> Callable[[Callable[[dict], Any]], Callable[[dict], None]]:
    """Apply a change to a sub-map: _in("main")(set_key("frames", -1))."""
    def wrap(change: Callable[[dict], Any]) -> Callable[[dict], None]:
        def apply(d: dict) -> None:
            node: Any = d
            for key in path:
                node = node[key]
            change(node)
        return apply
    return wrap


# ----------------------------------------------------------------------------------------------------------------
# R1: teleop -> frames


def recording_summary() -> RecordingSummary:
    main = CameraSummary(frames=900, frame_no_gaps=1, bytes=43_200_000, first_t_mid_ns=_T0,
                         last_t_mid_ns=_T0 + 29_966_000_000)
    uw = CameraSummary(frames=897, frame_no_gaps=0, bytes=67_275_000, first_t_mid_ns=_T0 + 4_000_000,
                       last_t_mid_ns=_T0 + 29_970_000_000)
    return RecordingSummary(main=main, ultrawide=uw, packets=900, packet_no_gaps=1, uw_reused=3, pair_refused=0,
                            error="", pi_clock_offset_ns=1_000_000_123_456, pi_clock_drift_ppm=20.5)


def _r1() -> list[Example]:
    start = StartRecording(episode_dir=_EPISODE_DIR)
    summary = recording_summary()
    empty = CameraSummary(frames=0, frame_no_gaps=0, bytes=0, first_t_mid_ns=None, last_t_mid_ns=None)
    failed = RecordingSummary(main=empty, ultrawide=empty, packets=0, packet_no_gaps=0, uw_reused=0, pair_refused=0,
                              error="recording could not keep up: write queue full (90 packets)",
                              pi_clock_offset_ns=None, pi_clock_drift_ppm=None)
    fh = FramesHealth(health=health(), synced=True, recording=True, recording_error="", frames_main=120,
                      frames_ultrawide=119)
    fh_down = FramesHealth(health=health("down", "ok"), synced=False, recording=False, recording_error="",
                           frames_main=0, frames_ultrawide=0)
    good = [
        good_example("r1/start_recording", start, {"episode_dir": _EPISODE_DIR}),
        good_example("r1/start_refused", Ack(ok=False, reason="main camera stale"), {"ok": False}),
        good_example("r1/stop_recording", StopRecording()),
        good_example("r1/get_health", GetFramesHealth()),
        good_example("r1/summary", summary, {"main.frames": 900, "ultrawide.frame_no_gaps": 0, "uw_reused": 3,
                                             "pi_clock_drift_ppm": 20.5}),
        good_example("r1/summary_writer_failed", failed, {"main.first_t_mid_ns": None, "error": failed.error}),
        good_example("r1/health_recording", fh, {"health.main.status": "ok", "recording": True, "frames_main": 120}),
        good_example("r1/health_main_down", fh_down, {"health.main.status": "down", "synced": False}),
    ]
    broken = [
        broken_example("r1/start_relative_dir", start, set_key("episode_dir", "pick_block/episode_003"),
                       "episode_dir", "value", "the folder must be an absolute path"),
        broken_example("r1/start_dir_not_str", start, set_key("episode_dir", 3), "episode_dir", "type",
                       "a number instead of a path"),
        broken_example("r1/start_missing_dir", start, drop_key("episode_dir"), "episode_dir", "missing",
                       "no folder"),
        broken_example("r1/summary_negative_frames", summary, _in("main")(set_key("frames", -1)), "main.frames",
                       "value", "uint64"),
        broken_example("r1/summary_times_without_frames", summary, _in("ultrawide")(set_key("frames", 0)),
                       "ultrawide.first_t_mid_ns", "value", "first/last are set exactly when frames > 0"),
        broken_example("r1/summary_last_before_first", summary,
                       _in("main")(set_key("last_t_mid_ns", _T0 - 1)), "main.last_t_mid_ns", "value",
                       "the last frame before the first"),
        broken_example("r1/summary_missing_error", summary, drop_key("error"), "error", "missing", "error left out"),
        broken_example("r1/summary_drift_int", summary, set_key("pi_clock_drift_ppm", 20), "pi_clock_drift_ppm",
                       "type", "an integer is not a float64"),
        broken_example("r1/health_bad_status", fh, _in("health", "ultrawide")(set_key("status", "dead")),
                       "health.ultrawide.status", "value", "unknown camera status"),
        broken_example("r1/health_unknown_field", fh, set_key("queue", 3), "queue", "unknown_field",
                       "a field the contract doesn't have"),
        broken_example("r1/health_schema_version", fh, set_key("schema", "r1.health/v2"), "", "schema_version",
                       "a version this side doesn't know", schema="r1.health/v2"),
    ]
    return good + broken


# ----------------------------------------------------------------------------------------------------------------
# U2: ui -> frames


def _u2() -> list[Example]:
    first = GetPreview(main_after=-1, ultrawide_after=-1)
    later = GetPreview(main_after=1200, ultrawide_after=1199)
    both = Preview(main_jpeg=JPEG, ultrawide_jpeg=JPEG, main_frame_no=1201, ultrawide_frame_no=1200, health=health())
    none_new = Preview(main_jpeg=None, ultrawide_jpeg=None, main_frame_no=1200, ultrawide_frame_no=1199,
                       health=health())
    nothing_yet = Preview(main_jpeg=None, ultrawide_jpeg=None, main_frame_no=-1, ultrawide_frame_no=-1,
                          health=health("down", "down"))
    good = [
        good_example("u2/get_preview_first", first, {"main_after": -1}),
        good_example("u2/get_preview", later, {"main_after": 1200, "ultrawide_after": 1199}),
        good_example("u2/preview_both", both, {"main_jpeg": JPEG, "ultrawide_frame_no": 1200}),
        good_example("u2/preview_nothing_new", none_new, {"main_jpeg": None, "main_frame_no": 1200}),
        good_example("u2/preview_no_frame_yet", nothing_yet, {"main_frame_no": -1, "health.main.status": "down"}),
    ]
    broken = [
        broken_example("u2/get_preview_main_after_below_minus_one", later, set_key("main_after", -2), "main_after",
                       "value", "-1 means none; nothing is below it"),
        broken_example("u2/get_preview_float", later, set_key("ultrawide_after", 3.0), "ultrawide_after", "type",
                       "a float frame_no"),
        broken_example("u2/preview_empty_jpeg", both, set_key("main_jpeg", b""), "main_jpeg", "value",
                       "an empty JPEG"),
        broken_example("u2/preview_jpeg_without_frame", both, set_key("ultrawide_frame_no", -1), "ultrawide_jpeg",
                       "value", "a JPEG while no frame is known"),
        broken_example("u2/preview_jpeg_as_str", both, set_key("main_jpeg", "jpeg"), "main_jpeg", "type",
                       "str instead of bytes"),
        broken_example("u2/preview_missing_health", both, drop_key("health"), "health", "missing", "no health"),
        broken_example("u2/preview_unknown_field", both, set_key("age_ms", 3.0), "age_ms", "unknown_field",
                       "a field the contract doesn't have"),
        broken_example("u2/preview_schema_version", both, set_key("schema", "u2.preview/v2"), "", "schema_version",
                       "a version this side doesn't know", schema="u2.preview/v2"),
    ]
    return good + broken


# ----------------------------------------------------------------------------------------------------------------
# U1: ui -> teleop


def teleop_status(**change: Any) -> TeleopStatus:
    """A status while recording episode_003 of pick_block; keyword arguments replace fields."""
    d: dict[str, Any] = dict(
        phase="recording", ready=True, data_root="/home/operator/CapstoneTeleop", task="pick_block",
        task_description="pick the red block and lift it",
        tasks=("pick_block", "place_cup"),
        episodes=(EpisodeInfo(name="episode_000", outcome="success", duration_s=24.1, frames_main=723),
                  EpisodeInfo(name="episode_002", outcome="fail", duration_s=31.0, frames_main=930)),
        recording_episode="episode_003", episode_s=4.5, arm_mode="running",
        tcp_m=np.array([0.452, 0.003, 0.301]), gripper_width_m=0.08, commanded_width_m=0.08,
        start_pose_known=True, start_pose_confirmed=True, at_start_pose=False, start_allowed=False,
        start_refusal="recording", at_limit=False,
        moving=True, slow=False, cameras=health(), c2_hz=100.0, rtt_ms=4.2, clock_ok=True, pc_mock=True,
        messages=(UiMessage(id=7, level="info", text="episode_003 started"),
                  UiMessage(id=8, level="warn", text="at workspace limit")),
        final_line="")
    d.update(change)
    return TeleopStatus(**d)


def _u1() -> list[Example]:
    keys = KeysUpdate(held=("x+", "tilt_left"), slow=True, focus=True)
    status = teleop_status()
    idle = teleop_status(phase="idle", recording_episode="", episode_s=0.0, arm_mode="idle", at_start_pose=True,
                         start_allowed=True, start_refusal="", moving=False, messages=())
    no_start_pose = teleop_status(phase="idle", recording_episode="", episode_s=0.0, arm_mode="idle",
                                  start_pose_known=False, start_pose_confirmed=False, at_start_pose=False,
                                  start_allowed=False,
                                  start_refusal="put the arm at the start pose: h, then p on the PC", moving=False,
                                  messages=())
    starting = teleop_status(phase="starting", ready=False, task="", task_description="", tasks=(), episodes=(),
                             recording_episode="", episode_s=0.0, arm_mode="", tcp_m=None, gripper_width_m=None,
                             start_pose_known=False, start_pose_confirmed=False, at_start_pose=False,
                             start_refusal="starting up", moving=False,
                             cameras=None, c2_hz=0.0, rtt_ms=None, clock_ok=False, messages=())
    done = teleop_status(phase="done", recording_episode="", episode_s=0.0, arm_mode="fault",
                         start_refusal="starting up", moving=False,
                         final_line="FATAL teleop: the PC reports fault: on the PC Ctrl-C and control_deploy run again")
    delete = DeleteEpisode(episode="episode_002")
    good = [
        good_example("u1/get_status", GetStatus(after_message=6), {"after_message": 6}),
        good_example("u1/keys", keys, {"held[1]": "tilt_left", "slow": True}),
        good_example("u1/keys_released", KeysUpdate(held=(), slow=False, focus=False), {"focus": False}),
        good_example("u1/gripper_close", GripperCommand(command="close"), {"command": "close"}),
        good_example("u1/select_task", SelectTask(name="pick_block"), {"name": "pick_block"}),
        good_example("u1/create_task", CreateTask(name="place_cup", description="put the cup on the plate"),
                     {"name": "place_cup"}),
        good_example("u1/start_episode", StartEpisode()),
        good_example("u1/stop_episode_success", StopEpisode(outcome="success"), {"outcome": "success"}),
        good_example("u1/stop_episode_discard", StopEpisode(outcome="discard"), {"outcome": "discard"}),
        good_example("u1/delete_episode", delete, {"episode": "episode_002"}),
        good_example("u1/label_episode", LabelEpisode(episode="episode_002", outcome="fail"),
                     {"episode": "episode_002", "outcome": "fail"}),
        good_example("u1/delete_task", DeleteTask(name="place_cup"), {"name": "place_cup"}),
        good_example("u1/set_data_root", SetDataRoot(path="~/CapstoneTeleop"), {"path": "~/CapstoneTeleop"}),
        good_example("u1/stop_arm", StopArm()),
        good_example("u1/refused", Ack(ok=False, reason="not at the start pose: h, then p on the PC"), {"ok": False}),
        good_example("u1/status_recording", status, {"phase": "recording", "tcp_m[2]": 0.301,
                                                      "episodes[1].name": "episode_002", "messages[1].id": 8,
                                                      "cameras.main.status": "ok"}),
        good_example("u1/status_idle_at_start_pose", idle, {"start_allowed": True, "start_refusal": "",
                                                             "at_start_pose": True, "start_pose_confirmed": True}),
        good_example("u1/status_start_pose_unknown", no_start_pose, {"start_allowed": False,
                                                                     "start_pose_known": False}),
        good_example("u1/status_starting", starting, {"ready": False, "tcp_m": None, "cameras": None}),
        good_example("u1/status_done", done, {"phase": "done", "final_line": done.final_line}),
    ]
    broken = [
        broken_example("u1/keys_unknown_key", keys, set_key("held", ["x+", "w"]), "held[1]", "value",
                       "a keyboard letter instead of a motion key name"),
        broken_example("u1/keys_twice", keys, set_key("held", ["x+", "x+"]), "held", "value", "a key listed twice"),
        broken_example("u1/keys_held_not_a_list", keys, set_key("held", "x+"), "held", "type", "a str, not a list"),
        broken_example("u1/keys_missing_focus", keys, drop_key("focus"), "focus", "missing", "no focus flag"),
        broken_example("u1/gripper_toggle", GripperCommand(command="close"), set_key("command", "toggle"),
                       "command", "value", "there is no toggle (PLAN 13.10)"),
        broken_example("u1/stop_episode_unknown_outcome", StopEpisode(outcome="success"), set_key("outcome", "ok"),
                       "outcome", "value", "an unknown outcome"),
        broken_example("u1/delete_bad_episode_name", delete, set_key("episode", "../episode_002"), "episode",
                       "value", "not an episode name (and a path)"),
        broken_example("u1/delete_short_episode_name", delete, set_key("episode", "episode_2"), "episode", "value",
                       "at least three digits"),
        broken_example("u1/get_status_negative", GetStatus(after_message=6), set_key("after_message", -1),
                       "after_message", "value", "uint64"),
        broken_example("u1/status_bad_phase", status, set_key("phase", "paused"), "phase", "value",
                       "an unknown phase"),
        broken_example("u1/status_negative_episode_s", status, set_key("episode_s", -0.1), "episode_s", "value",
                       "time since Start is never negative"),
        broken_example("u1/status_message_ids_not_increasing", status,
                       lambda d: d["messages"][1].update(id=7), "messages[1].id", "value",
                       "message ids must strictly increase"),
        broken_example("u1/status_refusal_while_allowed", idle, set_key("start_refusal", "recording"),
                       "start_refusal", "value", "a refusal text while start is allowed"),
        broken_example("u1/status_no_refusal_while_refused", status, set_key("start_refusal", ""), "start_refusal",
                       "value", "start refused without a reason"),
        broken_example("u1/status_confirmed_unknown_start_pose", no_start_pose,
                       set_key("start_pose_confirmed", True), "start_pose_confirmed", "value",
                       "a start pose cannot be confirmed without being known"),
        broken_example("u1/status_at_unknown_start_pose", no_start_pose, set_key("at_start_pose", True),
                       "at_start_pose", "value", "at a start pose that is not known"),
        broken_example("u1/status_episode_bad_outcome", status, lambda d: d["episodes"][0].update(outcome="discard"),
                       "episodes[0].outcome", "value", "a discarded episode is never listed"),
        broken_example("u1/status_episode_bad_name", status, lambda d: d["episodes"][0].update(name="ep0"),
                       "episodes[0].name", "value", "not an episode name"),
        broken_example("u1/status_message_bad_level", status, lambda d: d["messages"][0].update(level="debug"),
                       "messages[0].level", "value", "an unknown level"),
        broken_example("u1/status_bad_arm_mode", status, set_key("arm_mode", "moving"), "arm_mode", "value",
                       "not a C2 mode"),
        broken_example("u1/status_tcp_wrong_shape", status, lambda d: d.update(tcp_m=np.zeros(4)), "tcp_m", "shape",
                       "a position has 3 values"),
        broken_example("u1/status_tcp_wrong_dtype", status, lambda d: d.update(tcp_m=d["tcp_m"].astype(np.float32)),
                       "tcp_m", "dtype", "float32 instead of float64"),
        broken_example("u1/status_too_many_messages", idle,
                       set_key("messages", [{"id": i + 1, "level": "info", "text": "x"} for i in range(51)]),
                       "messages", "value", "at most 50 messages per status"),
        broken_example("u1/status_unknown_field", status, set_key("operator", "someone"), "operator", "unknown_field",
                       "no operator names (PLAN 13.12)"),
        broken_example("u1/status_schema_version", status, set_key("schema", "u1.status/v2"), "", "schema_version",
                       "a version this side doesn't know", schema="u1.status/v2"),
    ]
    return good + broken


def build_examples() -> list[Example]:
    """Every in-memory example of R1, U2 and U1, good and broken, in a fixed order."""
    out = _r1() + _u2() + _u1()
    names = [e.name for e in out]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate example names")
    return out
