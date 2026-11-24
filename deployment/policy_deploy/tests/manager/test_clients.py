# adapted from rnoro/deployment:deployment/message_comms/tests/test_clients.py @ 91b178b
# changes: PcClient (was FrankaClient): process "PC", link.yaml c3_timeout_s, hello() with its stamps and sample,
# home() returns its Ack (home_stopped, home_refused_homing), round trips to LinkStats; FramesClient's process name is
# frames; PolicyClient expects c4.config.reply/v2; imports.
import time

import numpy as np
import pytest
from deploy_contracts import (Ack, ArmState, ContractError, ErrorReply, RemoteError, SetWaypoints, SetWaypointsReply,
                              TransportTimeout, decode, pack_obj)
from manager_fakes import health, hello_reply, make_chunk, make_frames, make_state, policy_config, pose
from manager_raw_server import RawServer

from policy_deploy.contracts import FramesReply, HealthReply, HealthReport, PolicyConfig
from policy_deploy.manager.errors import Refused, RequestCancelled
from policy_deploy.manager.frames_client import FramesClient
from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.pc_client import PcClient
from policy_deploy.manager.policy_client import PolicyClient

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

def test_frames_good_replies(server):
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
    assert e.value.process == "frames" and e.value.reply.error == "boom"
    c.close()


def test_no_answer_is_a_timeout_sent_once(server):
    s = server(lambda data: None)
    c = FramesClient(s.endpoint, 0.2)
    t = time.monotonic()
    with pytest.raises(TransportTimeout) as e:
        c.get_health()
    assert 0.15 < time.monotonic() - t < 1.0
    assert e.value.request == "c1.get_health/v1" and e.value.process == "frames"
    assert len(s.received) == 1
    c.close()


# ---- PolicyClient ---------------------------------------------------------------------------------------------------

def policy_table(**over):
    table = {"c4.config/v1": policy_config(), "c4.reset/v1": Ack(ok=True, reason=""),
             "c4.prompt/v1": Ack(ok=True, reason=""), "c4.infer/v1": make_chunk()}
    table.update(over)
    return table


def observation():
    from policy_deploy.manager.pose_converter import PoseConverter
    obs, _ = PoseConverter(8, 0).to_observation(np.stack([pose(), pose()]), np.array([0.05, 0.05]), pose(),
                                                frames_reply(), T[1], T[1] + 1000, "target")
    return obs


def test_policy_good_replies(server, settings):
    c = PolicyClient(server(by_request(policy_table())).endpoint, settings.timeouts)
    config = c.config()
    assert isinstance(config, PolicyConfig) and config.kind == "hold_still"
    c.reset()
    c.prompt("raise", 0)
    chunk = c.infer(observation())
    assert chunk.eef_pos.shape == (16, 3)
    c.close()


def test_policy_refuses_a_v1_config_reply(server, settings):
    d = policy_config().to_dict()
    del d["kind"]
    d["schema"] = "c4.config.reply/v1"
    c = PolicyClient(server(by_request(policy_table(**{"c4.config/v1": pack_obj(d)}))).endpoint, settings.timeouts)
    with pytest.raises(ContractError) as e:
        c.config()
    assert e.value.problem == "schema_version"
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


# ---- PcClient -------------------------------------------------------------------------------------------------------

def waypoints(n=8, chunk_id=5):
    return SetWaypoints(chunk_id=chunk_id, t_ns=np.arange(n, dtype=np.int64) * 50_000_000 + 1,
                        O_T_TCP=np.tile(pose(), (n, 1, 1)), gripper_width_m=np.full(n, 0.04))


def pc_table(**over):
    table = {"c3.heartbeat/v1": Ack(ok=True, reason=""), "c3.start/v1": Ack(ok=False, reason="refused: mode is fault"),
             "c3.stop/v1": Ack(ok=True, reason=""), "c3.home/v1": Ack(ok=True, reason=""),
             "c3.get_state/v1": make_state(3, 5, mode="idle"),
             "c3.hello/v1": lambda req: hello_reply(time.monotonic_ns() + 10**12),
             "c3.set_waypoints/v1": lambda req: SetWaypointsReply(chunk_id=req.chunk_id, accepted=5, shortened=1,
                                                                   skipped=3, rejected=())}
    table.update(over)
    return table


def pc(srv, timeout=0.3, home_s=2.0, stats=None):
    return PcClient(srv.endpoint, timeout, home_s, 0.02, stats)


def test_pc_good_replies(server):
    stats = LinkStats(10.0)
    c = pc(server(by_request(pc_table())), stats=stats)
    c.heartbeat()
    c.stop()
    ack = c.home(lambda: False)
    assert ack.ok
    ack = c.start()
    assert ack.ok is False and ack.reason == "refused: mode is fault"      # returned, not raised
    assert isinstance(c.get_state(), ArmState)
    reply = c.set_waypoints(waypoints())
    assert reply.accepted == 5 and reply.skipped == 3
    assert stats.snapshot().rtt_n == 5                                     # every request but home
    c.close()


def test_hello_stamps_around_the_request(server):
    c = pc(server(by_request(pc_table())))
    before = time.monotonic_ns()
    reply, sample = c.hello()
    after = time.monotonic_ns()
    assert before <= sample.t_send_ns <= sample.t_recv_ns <= after
    assert sample.t_pc_ns == reply.t_pc_ns and sample.rtt_ns == c.last_rtt_ns
    assert sample.t_send_ns < sample.mid_ns < sample.t_recv_ns
    c.close()


def test_hello_uses_no_poll_or_cancel(server, monkeypatch):
    c = pc(server(by_request(pc_table())))
    seen = {}
    original = c._rpc.request

    def spy(msg, expect, timeout_s, poll_s=None, cancel=None):
        seen[msg.SCHEMA] = (poll_s, cancel, timeout_s)
        return original(msg, expect, timeout_s, poll_s, cancel)
    monkeypatch.setattr(c._rpc, "request", spy)
    c.hello()
    assert seen["c3.hello/v1"] == (None, None, 0.3)
    c.close()


def test_pc_refusals_and_home_acks(server):
    c = pc(server(by_request(pc_table(**{"c3.heartbeat/v1": Ack(ok=False, reason="?"),
                                         "c3.home/v1": Ack(ok=False, reason="too far")}))))
    with pytest.raises(Refused) as e:
        c.heartbeat()
    assert e.value.process == "PC"
    ack = c.home(lambda: False)                         # returned: the loop decides
    assert not ack.ok and not PcClient.home_stopped(ack) and not PcClient.home_refused_homing(ack)
    c.close()


@pytest.mark.parametrize("reason, stopped, refused_homing", [
    ("stopped: homing stopped by the PC keyboard", True, False),
    ("refused: mode is homing", False, True),
    ("refused: mode is running", False, False),
    ("homing failed: timeout", False, False),
])
def test_home_reply_kinds(reason, stopped, refused_homing):
    ack = Ack(ok=False, reason=reason)
    assert PcClient.home_stopped(ack) is stopped and PcClient.home_refused_homing(ack) is refused_homing
    assert not PcClient.home_stopped(Ack(ok=True, reason=""))


def test_set_waypoints_checks_echo_and_counts(server):
    replies = iter([SetWaypointsReply(chunk_id=6, accepted=8, shortened=0, skipped=0, rejected=()),
                    SetWaypointsReply(chunk_id=5, accepted=4, shortened=0, skipped=3, rejected=())])
    c = pc(server(by_request(pc_table(**{"c3.set_waypoints/v1": lambda req: next(replies)}))))
    with pytest.raises(ContractError) as e:
        c.set_waypoints(waypoints())
    assert e.value.field == "chunk_id"
    with pytest.raises(ContractError) as e:
        c.set_waypoints(waypoints())
    assert "7" in e.value.detail and "8" in e.value.detail
    c.close()


def test_set_waypoints_rejects_broken_reply(server):
    bad = raw(SetWaypointsReply(chunk_id=5, accepted=8, shortened=0, skipped=0, rejected=()), shortened=9)
    c = pc(server(lambda data: bad))
    with pytest.raises(ContractError):
        c.set_waypoints(waypoints())
    c.close()


def test_home_cancelled_drops_the_socket(server):
    srv = server(lambda data: None if decode(data).SCHEMA == "c3.home/v1" else Ack(ok=True, reason=""))
    c = pc(srv)
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


def test_command_timeout_is_c3_timeout(server):
    srv = server(lambda data: None)
    c = pc(srv, timeout=0.1)
    with pytest.raises(TransportTimeout) as e:
        c.heartbeat()
    assert e.value.timeout_s == 0.1 and len(srv.received) == 1 and e.value.process == "PC"
    c.close()


def test_home_timeout_is_home_s(server):
    srv = server(lambda data: None)
    c = pc(srv, home_s=0.15)
    with pytest.raises(TransportTimeout) as e:
        c.home(lambda: False)
    assert e.value.timeout_s == 0.15 and e.value.request == "c3.home/v1"
    c.close()
