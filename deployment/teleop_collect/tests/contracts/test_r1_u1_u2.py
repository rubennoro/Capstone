"""R1, U1 and U2 (STRUCTURE.md 4.1-4.4): the names, round trips, the per-message rules, and one request over a real
REQ/REP pair on a free port (the registry decodes them on the wire)."""
from __future__ import annotations

import threading

import numpy as np
import pytest
from deploy_contracts import Ack, RemoteError, RpcClient, RpcServer, decode, encode
from station_helpers import contract_error

import teleop_collect.contracts as c
from teleop_collect.contracts.examples import health, recording_summary, teleop_status


def test_names():
    assert c.MOTION_KEYS == ("x+", "x-", "y+", "y-", "z+", "z-", "twist+", "twist-", "tilt_fwd", "tilt_back",
                             "tilt_left", "tilt_right")
    assert c.GRIPPER_COMMANDS == ("close", "open") and c.OUTCOMES == ("success", "fail", "discard")
    assert c.TASK_NAME_RE == r"^[a-z0-9_]{1,64}$" and c.EPISODE_NAME_RE == r"^episode_[0-9]{3,}$"
    assert c.PHASES == ("starting", "idle", "recording", "stopping", "done")            # no homing: no laptop home


@pytest.mark.parametrize("name,ok", [("pick_block", True), ("a", True), ("x" * 64, True), ("x" * 65, False),
                                     ("", False), ("Pick", False), ("pick-block", False), ("pick block", False),
                                     ("pick\n", False), ("../x", False), (3, False)])
def test_task_names(name, ok):
    assert c.is_task_name(name) is ok


@pytest.mark.parametrize("name,ok", [("episode_000", True), ("episode_1000", True), ("episode_00", False),
                                     ("episode_abc", False), ("episode_001/", False), ("Episode_001", False)])
def test_episode_names(name, ok):
    assert c.is_episode_name(name) is ok


@pytest.mark.parametrize("msg", [
    c.StartRecording(episode_dir="/tmp/x/episode_000"), c.StopRecording(), c.GetFramesHealth(), recording_summary(),
    c.FramesHealth(health=health(), synced=True, recording=False, recording_error="", frames_main=0,
                   frames_ultrawide=0),
    c.GetPreview(main_after=-1, ultrawide_after=5),
    c.Preview(main_jpeg=b"\xff\xd8x\xff\xd9", ultrawide_jpeg=None, main_frame_no=4, ultrawide_frame_no=-1,
              health=health("ok", "down")),
    c.GetStatus(after_message=0), c.KeysUpdate(held=["z-"], slow=False, focus=True), c.GripperCommand(command="open"),
    c.SelectTask(name="pick_block"), c.CreateTask(name="pick_block", description=""),
    c.StartEpisode(), c.StopEpisode(outcome="fail"), c.DeleteEpisode(episode="episode_012"), c.StopArm(),
    c.DeleteTask(name="pick_block"), c.SetDataRoot(path="~/CapstoneTeleop"),
    teleop_status(),
], ids=lambda m: m.SCHEMA)
def test_round_trip(msg):
    assert decode(encode(msg)) == msg


def test_start_recording_wants_an_absolute_path():
    contract_error(lambda: c.StartRecording(episode_dir=""), "episode_dir", "value")
    contract_error(lambda: c.StartRecording(episode_dir="episode_000"), "episode_dir", "value")


def test_camera_summary_times_exactly_with_frames():
    ok = c.CameraSummary(frames=1, frame_no_gaps=0, bytes=10, first_t_mid_ns=5, last_t_mid_ns=5)
    assert ok.last_t_mid_ns == 5
    contract_error(lambda: c.CameraSummary(frames=1, frame_no_gaps=0, bytes=10, first_t_mid_ns=None,
                                           last_t_mid_ns=5), "first_t_mid_ns", "value")
    contract_error(lambda: c.CameraSummary(frames=0, frame_no_gaps=0, bytes=0, first_t_mid_ns=None,
                                           last_t_mid_ns=5), "last_t_mid_ns", "value")
    contract_error(lambda: c.CameraSummary(frames=2, frame_no_gaps=0, bytes=10, first_t_mid_ns=6,
                                           last_t_mid_ns=5), "last_t_mid_ns", "value")


def test_preview_jpeg_rules():
    contract_error(lambda: c.Preview(main_jpeg=b"", ultrawide_jpeg=None, main_frame_no=1, ultrawide_frame_no=1,
                                     health=health()), "main_jpeg", "value")
    contract_error(lambda: c.Preview(main_jpeg=None, ultrawide_jpeg=b"x", main_frame_no=1, ultrawide_frame_no=-1,
                                     health=health()), "ultrawide_jpeg", "value")
    contract_error(lambda: c.GetPreview(main_after=0, ultrawide_after=-2), "ultrawide_after", "value")


def test_keys_rules():
    assert c.KeysUpdate(held=[], slow=False, focus=False).held == ()
    contract_error(lambda: c.KeysUpdate(held=["x+", "q"], slow=False, focus=True), "held[1]", "value")
    contract_error(lambda: c.KeysUpdate(held=["x+", "x+"], slow=False, focus=True), "held", "value")
    contract_error(lambda: c.KeysUpdate(held=["slow"], slow=False, focus=True), "held[0]", "value")


def test_status_rules():
    contract_error(lambda: teleop_status(episode_s=-1.0), "episode_s", "value")
    contract_error(lambda: teleop_status(start_allowed=True), "start_refusal", "value")
    contract_error(lambda: teleop_status(start_refusal=""), "start_refusal", "value")
    msgs = tuple(c.UiMessage(id=i, level="info", text="x") for i in (3, 5, 5))
    contract_error(lambda: teleop_status(messages=msgs), "messages[2].id", "value")
    msgs = tuple(c.UiMessage(id=i + 1, level="info", text="x") for i in range(51))
    contract_error(lambda: teleop_status(messages=msgs), "messages", "value")
    contract_error(lambda: teleop_status(arm_mode="driving"), "arm_mode", "value")
    contract_error(lambda: teleop_status(tcp_m=np.zeros(3, np.float32)), "tcp_m", "dtype")
    contract_error(lambda: teleop_status(commanded_width_m=0), "commanded_width_m", "type")
    contract_error(lambda: teleop_status(start_pose_known=False), "start_pose_confirmed", "value")
    contract_error(lambda: teleop_status(start_pose_known=False, start_pose_confirmed=False, at_start_pose=True),
                   "at_start_pose", "value")
    assert not teleop_status(start_pose_known=False, start_pose_confirmed=False).start_pose_known
    contract_error(lambda: c.EpisodeInfo(name="episode_001", outcome="discard", duration_s=1.0, frames_main=1),
                   "outcome", "value")
    assert teleop_status(tcp_m=None, cameras=None, rtt_ms=None, gripper_width_m=None).cameras is None


def test_a_request_over_rep():
    seen = []

    def on_keys(msg):
        seen.append(msg)
        return Ack(ok=True, reason="")

    server = RpcServer("tcp://127.0.0.1:*", {c.KeysUpdate: on_keys, c.GetStatus: lambda m: teleop_status()})
    stop = threading.Event()
    thread = threading.Thread(target=server.serve_forever, args=(stop, 0.01), daemon=True)
    thread.start()
    client = RpcClient("teleop", server.endpoint)
    try:
        ack = client.request(c.KeysUpdate(held=["x+"], slow=True, focus=True), Ack, timeout_s=2.0)
        assert ack.ok and seen[0].held == ("x+",)
        status = client.request(c.GetStatus(after_message=0), c.TeleopStatus, timeout_s=2.0)
        assert status.recording_episode == "episode_003"
        with pytest.raises(RemoteError):
            client.request(c.StopArm(), Ack, timeout_s=2.0)                 # no handler: error/v1
    finally:
        stop.set()
        thread.join(2.0)
        client.close()
        server.close()
