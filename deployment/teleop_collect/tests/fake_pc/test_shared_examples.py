# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/fake_pc/test_shared_examples.py @ 2ffe641
# changes: imports (launcher_helpers, teleop_collect.contracts); the laptop-only schemas are R1/U1/U2 (not C1/C4);
# the fake PC's clock offset 1e6 s, so the stored example times are past on any machine.
"""The fake PC against the stored shared examples (STRUCTURE.md 12.2; TOP 3.4, 3.11): it decodes and answers what
franka_control does. Raw bytes over a ZeroMQ REQ socket."""
from __future__ import annotations

import pytest
import zmq
from deploy_contracts import ErrorReply, decode, load_index
from launcher_helpers import make_fake_pc

import teleop_collect.contracts  # noqa: F401  (the laptop schemas are registered in this test process)
from teleop_collect.contracts.examples import build_examples

REPLY_OF = {"c3.hello/v1": "c3.hello.reply/v1", "c3.get_state/v1": "c2.state/v1", "c3.heartbeat/v1": "ack/v1",
            "c3.start/v1": "ack/v1", "c3.stop/v1": "ack/v1", "c3.home/v1": "ack/v1",
            "c3.set_waypoints/v1": "c3.set_waypoints.reply/v1"}
STORED = [e for e in load_index() if e.contract == "c3"]
GOOD_REQUESTS = [e for e in STORED if e.valid and e.schema in REPLY_OF]
BROKEN = [e for e in STORED if not e.valid]
LAPTOP = {e.name: e for e in build_examples()}


@pytest.fixture(scope="module")
def pc(tmp_path_factory):
    # clock_offset_s 1e6: the stored set_waypoints examples (t_ns about 1.2e5 s) are always in the past, whatever
    # the laptop's uptime (the source's 1000 s offset needs an uptime above about 34 h)
    fake = make_fake_pc(tmp_path_factory.mktemp("fake_pc"), home_duration_s=0.05, clock_offset_s=1e6)
    fake.start()
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(fake.commands_endpoint)

    def ask(data: bytes):
        sock.send(data)
        assert sock.poll(5000), "no reply"
        return decode(sock.recv())
    yield fake, ask
    sock.close()
    fake.close()


def test_there_are_examples():
    assert len(GOOD_REQUESTS) >= 7 and len(BROKEN) > 20


@pytest.mark.parametrize("example", GOOD_REQUESTS, ids=lambda e: e.name)
def test_good_requests_get_a_valid_reply_of_the_right_schema(pc, example):
    fake, ask = pc
    if example.schema == "c3.set_waypoints/v1":
        fake.set_mode("holding")                           # the gate answers (the example times are long past)
    elif example.schema in ("c3.start/v1", "c3.home/v1"):
        fake.set_mode("idle")
    reply = ask(example.data)
    assert not isinstance(reply, ErrorReply), reply
    assert reply.SCHEMA == REPLY_OF[example.schema]
    if example.schema == "c3.set_waypoints/v1":
        assert reply.skipped == len(decode(example.data).t_ns) and reply.accepted == 0


@pytest.mark.parametrize("example", BROKEN, ids=lambda e: e.name)
def test_broken_requests_get_error_with_their_field(pc, example):
    fake, ask = pc
    before = len(fake.request_names())
    reply = ask(example.data)
    assert isinstance(reply, ErrorReply)
    assert reply.field == example.field, reply.error
    assert len(fake.request_names()) == before            # a broken request never reaches a handler


@pytest.mark.parametrize("name", ["r1/start_recording", "r1/health_recording", "u1/keys", "u1/status_recording",
                                  "u2/get_preview", "u2/preview_both"])
def test_laptop_only_schemas_are_unknown(pc, name):
    _, ask = pc
    reply = ask(LAPTOP[name].data)
    assert isinstance(reply, ErrorReply) and reply.field == ""
    assert "unknown message name" in reply.error, reply.error         # unknown_schema, not unexpected_message
    assert reply.request in ("", LAPTOP[name].schema)
