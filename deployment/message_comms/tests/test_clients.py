import time

import numpy as np
import pytest

from contracts import (Ack, ArmState, ContractError, ErrorReply, FramesReply,
                       HealthReply, HealthReport, PolicyConfig, RemoteError, SetWaypoints, SetWaypointsReply,
                       TransportTimeout, decode, pack_obj)
from fakes import health, make_chunk, make_frames, make_state, policy_config, pose
from message_comms.errors import Refused, RequestCancelled
from message_comms.frames_client import FramesClient
from message_comms.franka_client import FrankaClient
from message_comms.policy_client import PolicyClient
from raw_server import RawServer

T = (9_950_000_000, 10_000_000_000)


@pytest.fixture
def server():
    servers = []

    def make(answer):
        s = RawServer(answer)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.close()


def by_request(table):
    """answer from {schema: reply or callable(request)}."""
    def answer(data):
        req = decode(data)
        reply = table[req.SCHEMA]
        return reply(req) if callable(reply) else reply
    return answer


def raw(msg, **changes):
    d = msg.to_dict()
    for key, value in changes.items():
        d[key] = value
    return pack_obj(d)


def frames_reply():
    return make_frames(T)


# ---- FramesClient ---------------------------------------------------------------------------------------------------

def test_frames_good_replies(server, settings):
    s = server(by_request({"c1.get_frames/v1": frames_reply(), "c1.get_health/v1": HealthReply(health=health())}))
    c = FramesClient(s.endpoint, 1.0)
    assert isinstance(c.get_frames(), FramesReply)
    h = c.get_health()
    assert isinstance(h, HealthReport) and h.main.status == "ok"
    c.close()


@pytest.mark.parametrize("bad, field, problem", [
    (raw(frames_reply(), schema="c1.frames/v2"), "", "schema_version"),
    (raw(frames_reply(), main=np.zeros((2, 224, 3), np.uint8)), "main", "shape"),
    (Ack(ok=True, reason=""), "", "unexpected_message"),
])
def test_frames_rejects_broken_replies(server, bad, field, problem):
    c = FramesClient(server(lambda data: bad).endpoint, 1.0)
    with pytest.raises(ContractError) as e:
        c.get_frames()
    assert e.value.problem == problem and (not field or e.value.field == field)
    c.close()


def test_frames_rejects_images_while_a_camera_is_down(server):
    d = frames_reply().to_dict()
    d["health"]["main"]["status"] = "down"
    c = FramesClient(server(lambda data: pack_obj(d)).endpoint, 1.0)
    with pytest.raises(ContractError) as e:
        c.get_frames()
    assert e.value.schema == "c1.frames/v1"
    c.close()


def test_error_reply_is_remote_error(server):
    c = FramesClient(server(lambda data: ErrorReply(request="c1.get_frames/v1", error="boom", field="")).endpoint, 1.0)
    with pytest.raises(RemoteError) as e:
        c.get_frames()
    assert e.value.process == "process_frames" and e.value.reply.error == "boom"
    c.close()


def test_no_answer_is_a_timeout_sent_once(server):
    s = server(lambda data: None)
    c = FramesClient(s.endpoint, 0.2)
    t = time.monotonic()
    with pytest.raises(TransportTimeout) as e:
        c.get_health()
    assert 0.15 < time.monotonic() - t < 1.0
    assert e.value.request == "c1.get_health/v1" and e.value.process == "process_frames"
    assert len(s.received) == 1
    c.close()


# ---- PolicyClient ---------------------------------------------------------------------------------------------------

def policy_table(**over):
    table = {"c4.config/v1": policy_config(), "c4.reset/v1": Ack(ok=True, reason=""),
             "c4.prompt/v1": Ack(ok=True, reason=""), "c4.infer/v1": make_chunk()}
    table.update(over)
    return table


def observation():
    from message_comms.pose_converter import PoseConverter
    obs, _ = PoseConverter(8, 0).to_observation(np.stack([pose(), pose()]), np.array([0.05, 0.05]), pose(),
                                                frames_reply(), T[1], "target")
    return obs


def test_policy_good_replies(server, settings):
    c = PolicyClient(server(by_request(policy_table())).endpoint, settings.timeouts)
    assert isinstance(c.config(), PolicyConfig)
    c.reset()
    c.prompt("raise", 0)
    chunk = c.infer(observation())
    assert chunk.eef_pos.shape == (16, 3)
    c.close()


def test_policy_refusals(server, settings):
    c = PolicyClient(server(by_request(policy_table(**{
        "c4.reset/v1": Ack(ok=False, reason="busy"), "c4.prompt/v1": Ack(ok=False, reason="no prompts")}))).endpoint,
        settings.timeouts)
    with pytest.raises(Refused) as e:
        c.reset()
    assert e.value.process == "policy" and e.value.request == "c4.reset/v1" and e.value.reason == "busy"
    with pytest.raises(Refused):
        c.prompt("raise", 0)
    c.close()


def test_policy_rejects_a_broken_chunk(server, settings):
    bad = raw(make_chunk(), eef_rot6d=np.full((16, 6), 3.0, np.float32))
    c = PolicyClient(server(by_request(policy_table(**{"c4.infer/v1": bad}))).endpoint, settings.timeouts)
    with pytest.raises(ContractError) as e:
        c.infer(observation())
    assert e.value.schema == "c4.chunk/v1" and e.value.field == "eef_rot6d"
    c.close()


def test_policy_rejects_wrong_reply_type_and_error(server, settings):
    c = PolicyClient(server(by_request(policy_table(**{
        "c4.config/v1": Ack(ok=True, reason=""),
        "c4.infer/v1": ErrorReply(request="c4.infer/v1", error="CUDA error", field="")}))).endpoint,
        settings.timeouts)
    with pytest.raises(ContractError) as e:
        c.config()
    assert e.value.problem == "unexpected_message"
    with pytest.raises(RemoteError):
        c.infer(observation())
    c.close()


def test_infer_hang_limit(server, make_settings):
    s = make_settings(timeouts={"policy_hang_s": 0.2})
    srv = server(lambda data: None)
    c = PolicyClient(srv.endpoint, s.timeouts)
    with pytest.raises(TransportTimeout) as e:
        c.infer(observation())
    assert e.value.request == "c4.infer/v1" and e.value.timeout_s == 0.2 and len(srv.received) == 1
    c.close()


# ---- FrankaClient ---------------------------------------------------------------------------------------------------

def waypoints(n=8, chunk_id=5):
    return SetWaypoints(chunk_id=chunk_id, t_ns=np.arange(n, dtype=np.int64) * 50_000_000 + 1,
                        O_T_TCP=np.tile(pose(), (n, 1, 1)), gripper_width_m=np.full(n, 0.04))


def franka_table(**over):
    table = {"c3.heartbeat/v1": Ack(ok=True, reason=""), "c3.start/v1": Ack(ok=False, reason="mode is fault"),
             "c3.stop/v1": Ack(ok=True, reason=""), "c3.home/v1": Ack(ok=True, reason=""),
             "c3.get_state/v1": make_state(3, 5, mode="idle"),
             "c3.set_waypoints/v1": lambda req: SetWaypointsReply(chunk_id=req.chunk_id, accepted=5, shortened=1,
                                                                   skipped=3, rejected=())}
    table.update(over)
    return table


def test_franka_good_replies(server, settings):
    c = FrankaClient(server(by_request(franka_table())).endpoint, settings.timeouts, 0.02)
    c.heartbeat()
    c.stop()
    c.home(lambda: False)
    ack = c.start()
    assert ack.ok is False and ack.reason == "mode is fault"      # returned, not raised
    assert isinstance(c.get_state(), ArmState)
    reply = c.set_waypoints(waypoints())
    assert reply.accepted == 5 and reply.skipped == 3
    c.close()


def test_franka_refusals(server, settings):
    c = FrankaClient(server(by_request(franka_table(**{
        "c3.heartbeat/v1": Ack(ok=False, reason="?"), "c3.home/v1": Ack(ok=False, reason="too far")}))).endpoint,
        settings.timeouts, 0.02)
    with pytest.raises(Refused):
        c.heartbeat()
    with pytest.raises(Refused) as e:
        c.home(lambda: False)
    assert e.value.request == "c3.home/v1" and e.value.reason == "too far"
    c.close()


def test_set_waypoints_checks_echo_and_counts(server, settings):
    replies = iter([SetWaypointsReply(chunk_id=6, accepted=8, shortened=0, skipped=0, rejected=()),
                    SetWaypointsReply(chunk_id=5, accepted=4, shortened=0, skipped=3, rejected=())])
    c = FrankaClient(server(by_request(franka_table(**{"c3.set_waypoints/v1": lambda req: next(replies)}))).endpoint,
                     settings.timeouts, 0.02)
    with pytest.raises(ContractError) as e:
        c.set_waypoints(waypoints())
    assert e.value.field == "chunk_id"
    with pytest.raises(ContractError) as e:
        c.set_waypoints(waypoints())
    assert "7" in e.value.detail and "8" in e.value.detail
    c.close()


def test_set_waypoints_rejects_broken_reply(server, settings):
    bad = raw(SetWaypointsReply(chunk_id=5, accepted=8, shortened=0, skipped=0, rejected=()), shortened=9)
    c = FrankaClient(server(lambda data: bad).endpoint, settings.timeouts, 0.02)
    with pytest.raises(ContractError):
        c.set_waypoints(waypoints())
    c.close()


def test_home_cancelled_drops_the_socket(server, settings):
    srv = server(lambda data: None if decode(data).SCHEMA == "c3.home/v1" else Ack(ok=True, reason=""))
    c = FrankaClient(srv.endpoint, settings.timeouts, 0.02)
    flag = {"set": False}
    t = time.monotonic()

    def interrupted():
        if time.monotonic() - t > 0.1:
            flag["set"] = True
        return flag["set"]
    with pytest.raises(RequestCancelled):
        c.home(interrupted)
    assert time.monotonic() - t < 0.5
    c.heartbeat()                 # a new socket: the next request works
    c.close()


def test_command_timeout(server, make_settings):
    s = make_settings(timeouts={"commands_s": 0.1})
    srv = server(lambda data: None)
    c = FrankaClient(srv.endpoint, s.timeouts, 0.02)
    with pytest.raises(TransportTimeout) as e:
        c.heartbeat()
    assert e.value.timeout_s == 0.1 and len(srv.received) == 1
    c.close()
