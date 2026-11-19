# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/manager/test_clients.py @ 2ffe641
# changes: the PcClient tests unchanged (imports); the C1 FramesClient and PolicyClient tests replaced by the R1
# FramesClient (start / stop recording, health, its two timeouts, refusals returned, error/v1 raised).
import time

import numpy as np
import pytest
from deploy_contracts import (Ack, ArmState, ContractError, ErrorReply, RemoteError, SetWaypoints, SetWaypointsReply,
                              TransportTimeout, decode, pack_obj)
from teleop_helpers import RawServer, frames_health, hello_reply, make_state, pose, summary

from teleop_collect.contracts import FramesHealth, RecordingSummary
from teleop_collect.teleop.errors import Refused, RequestCancelled
from teleop_collect.teleop.frames_client import FramesClient
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.pc_client import PcClient


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


# ---- FramesClient (R1) ----------------------------------------------------------------------------------------------

def r1_table(**over):
    table = {"r1.start_recording/v1": Ack(ok=True, reason=""), "r1.stop_recording/v1": summary(30),
             "r1.get_health/v1": frames_health()}
    table.update(over)
    return table


def test_r1_good_replies(server):
    srv = server(by_request(r1_table()))
    c = FramesClient(srv.endpoint, 0.05, 1.0)
    assert c.start_recording("/tmp/x/episode_000").ok
    assert decode(srv.received[0]).episode_dir == "/tmp/x/episode_000"
    s = c.stop_recording()
    assert isinstance(s, RecordingSummary) and s.main.frames == 30
    h = c.get_health()
    assert isinstance(h, FramesHealth) and h.health.main.status == "ok"
    c.close()


def test_r1_refusal_is_returned_and_not_recording_is_remote_error(server):
    srv = server(by_request(r1_table(**{
        "r1.start_recording/v1": Ack(ok=False, reason="main camera stale"),
        "r1.stop_recording/v1": ErrorReply(request="r1.stop_recording/v1", error="not recording", field="")})))
    c = FramesClient(srv.endpoint, 0.05, 1.0)
    ack = c.start_recording("/tmp/x/episode_000")
    assert not ack.ok and ack.reason == "main camera stale"
    with pytest.raises(RemoteError) as e:
        c.stop_recording()
    assert e.value.process == "frames"
    c.close()


def test_r1_broken_reply_is_a_contract_error(server):
    bad = raw(frames_health(), recording="yes")
    c = FramesClient(server(lambda data: bad).endpoint, 0.05, 1.0)
    with pytest.raises(ContractError):
        c.get_health()
    c.close()


@pytest.mark.parametrize("call, timeout", [("get_health", 0.1), ("stop_recording", 0.25)])
def test_r1_timeouts(server, call, timeout):
    srv = server(lambda data: None)
    c = FramesClient(srv.endpoint, 0.1, 0.25)
    t = time.monotonic()
    with pytest.raises(TransportTimeout) as e:
        getattr(c, call)()
    assert e.value.timeout_s == timeout and e.value.process == "frames" and len(srv.received) == 1
    assert time.monotonic() - t < timeout + 0.5
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
