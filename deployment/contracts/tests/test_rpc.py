# adapted from rnoro/deployment:deployment/contracts/tests/test_rpc.py @ 91b178b
# changes: imports renamed; the C4 RemoteError text uses a C3 request; + the RpcServer schema filter
# (STRUCTURE.md 3.11)
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest
import zmq

from deploy_contracts import (SHARED_SCHEMAS, Ack, ArmState, ContractError, ErrorReply, GetState, Heartbeat, Hello,
                              Home, Message, RemoteError, RpcClient, RpcServer, Start, Stop, TransportCancelled,
                              TransportTimeout, decode, encode, load_index, pack_obj, register)
from deploy_contracts import registry
from deploy_contracts.fields import scalar

ENDPOINT = "tcp://127.0.0.1:*"


class Server:
    """An RpcServer in its own thread. `hold` makes the Home handler wait until `release` is set."""

    def __init__(self, handlers=None, schemas=None):
        self.calls: list[str] = []
        self.hold = threading.Event()
        self.release = threading.Event()
        self.handling = threading.Event()
        handlers = handlers or {Heartbeat: self.heartbeat, Home: self.home, Stop: self.boom, GetState: self.wrong}
        self.server = RpcServer(ENDPOINT, handlers, schemas=schemas)
        self.endpoint = self.server.endpoint
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.server.serve_forever, args=(self.stop, 0.01), daemon=True)
        self.thread.start()

    def heartbeat(self, msg):
        self.calls.append(msg.SCHEMA)
        return Ack(ok=True, reason="")

    def home(self, msg):
        self.calls.append(msg.SCHEMA)
        if self.hold.is_set():
            self.handling.set()
            self.release.wait(5.0)
            return Ack(ok=True, reason="late")
        return Ack(ok=True, reason="fresh")

    def boom(self, msg):
        raise RuntimeError("motor on fire")

    def wrong(self, msg):
        return Ack(ok=True, reason="")  # not the ArmState the client expects

    def close(self):
        self.release.set()
        self.stop.set()
        self.thread.join(5.0)
        self.server.close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture
def client(server):
    c = RpcClient("franka_control", server.endpoint)
    yield c
    c.close()


def test_request_reply(server, client):
    assert server.endpoint.startswith("tcp://127.0.0.1:") and not server.endpoint.endswith("*")
    assert client.request(Heartbeat(), Ack, timeout_s=2.0) == Ack(ok=True, reason="")
    assert client.request(Home(), Ack, timeout_s=2.0).reason == "fresh"


def test_timeout_sent_once_new_socket_late_reply_dropped(server, client):
    server.hold.set()
    t0 = time.monotonic()
    with pytest.raises(TransportTimeout) as info:
        client.request(Home(), Ack, timeout_s=0.3)
    elapsed = time.monotonic() - t0
    assert 0.29 <= elapsed < 1.5
    assert str(info.value) == (f"franka_control ({server.endpoint}): c3.home/v1: no reply within 0.300 s "
                               "(sent once, not resent)")
    assert (info.value.process, info.value.request, info.value.timeout_s) == ("franka_control", "c3.home/v1", 0.3)
    assert server.handling.wait(2.0)
    server.hold.clear()
    server.release.set()  # the late reply goes to the closed socket
    reply = client.request(Home(), Ack, timeout_s=2.0)
    assert reply.reason == "fresh"  # not the late one
    assert server.calls == ["c3.home/v1", "c3.home/v1"]  # the first request was sent once


def test_cancel_in_slices(server, client):
    server.hold.set()
    t0 = time.monotonic()
    cancel_at = t0 + 0.2
    polls = []

    def cancel():
        polls.append(time.monotonic())
        return time.monotonic() >= cancel_at

    with pytest.raises(TransportCancelled) as info:
        client.request(Home(), Ack, timeout_s=10.0, poll_s=0.05, cancel=cancel)
    elapsed = time.monotonic() - t0
    assert 0.2 <= elapsed < 0.2 + 0.05 + 0.5  # within about one poll_s of the cancel
    assert len(polls) >= 3
    assert info.value.waited_s == pytest.approx(elapsed, abs=0.1) and info.value.request == "c3.home/v1"
    assert server.handling.wait(2.0)
    server.hold.clear()
    server.release.set()
    assert client.request(Home(), Ack, timeout_s=2.0).reason == "fresh"  # new socket, late reply dropped
    assert server.calls == ["c3.home/v1", "c3.home/v1"]


def test_sliced_wait_returns_a_normal_reply(server, client):
    assert client.request(Home(), Ack, timeout_s=2.0, poll_s=0.01, cancel=lambda: False).reason == "fresh"


def test_sliced_wait_still_times_out(server, client):
    server.hold.set()
    t0 = time.monotonic()
    with pytest.raises(TransportTimeout):
        client.request(Home(), Ack, timeout_s=0.25, poll_s=0.05, cancel=lambda: False)
    assert 0.24 <= time.monotonic() - t0 < 1.5


def test_poll_and_cancel_go_together(client):
    with pytest.raises(ValueError):
        client.request(Heartbeat(), Ack, timeout_s=1.0, poll_s=0.1)
    with pytest.raises(ValueError):
        client.request(Heartbeat(), Ack, timeout_s=1.0, cancel=lambda: False)


def test_broken_request_gets_error_reply_naming_the_field(server):
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(server.endpoint)
    try:
        sock.send(pack_obj({"schema": "c3.heartbeat/v1", "extra": 1}))
        assert sock.poll(2000)
        reply = decode(sock.recv())
        assert isinstance(reply, ErrorReply)
        assert (reply.request, reply.field) == ("c3.heartbeat/v1", "extra")
        assert "unknown field" in reply.error
        sock.send(b"\xc1")
        assert sock.poll(2000)
        reply = decode(sock.recv())
        assert isinstance(reply, ErrorReply) and reply.request == "" and "not msgpack" in reply.error
    finally:
        sock.close()


def test_unknown_request_type_gets_error_reply(server, client):
    with pytest.raises(RemoteError) as info:
        client.request(Start(), Ack, timeout_s=2.0)
    assert info.value.reply.request == "c3.start/v1"
    assert "does not handle" in info.value.reply.error


def test_raising_handler_gets_error_reply_and_server_keeps_serving(server, client):
    with pytest.raises(RemoteError) as info:
        client.request(Stop(), Ack, timeout_s=2.0)
    assert "motor on fire" in str(info.value)
    assert info.value.process == "franka_control"
    assert "RuntimeError: motor on fire" in info.value.reply.error
    assert client.request(Heartbeat(), Ack, timeout_s=2.0).ok


def test_wrong_reply_type_raises_contract_error(server, client):
    with pytest.raises(ContractError) as info:
        client.request(GetState(), ArmState, timeout_s=2.0)
    assert info.value.problem == "unexpected_message"
    assert client.request(Heartbeat(), Ack, timeout_s=2.0).ok  # the socket is still usable


def test_broken_reply_raises_contract_error():
    rep = zmq.Context.instance().socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(ENDPOINT)
    endpoint = rep.getsockopt_string(zmq.LAST_ENDPOINT)

    def answer():
        if rep.poll(2000):
            rep.recv()
            rep.send(pack_obj({"schema": "ack/v1", "ok": "yes", "reason": ""}))

    t = threading.Thread(target=answer)
    t.start()
    client = RpcClient("policy", endpoint)
    try:
        with pytest.raises(ContractError) as info:
            client.request(Heartbeat(), Ack, timeout_s=2.0)
        assert (info.value.field, info.value.problem) == ("ok", "type")
    finally:
        t.join(5.0)
        client.close()
        rep.close()


def test_remote_error_carries_the_text():
    reply = ErrorReply(request="c3.set_waypoints/v1", error="set_waypoints refused: mode is idle", field="")
    e = RemoteError("PC", reply)
    assert str(e) == "PC: error reply to c3.set_waypoints/v1: set_waypoints refused: mode is idle"
    assert e.reply is reply


def test_serve_once_without_request_returns_false():
    s = RpcServer(ENDPOINT, {Heartbeat: lambda m: Ack(ok=True, reason="")})
    try:
        assert s.serve_once(0.01) is False
    finally:
        s.close()


def test_handler_returning_a_non_message_gets_error_reply():
    def bad(msg):
        return {"ok": True}

    s = Server({Heartbeat: bad})
    c = RpcClient("x", s.endpoint)
    try:
        with pytest.raises(RemoteError) as info:
            c.request(Heartbeat(), Ack, timeout_s=2.0)
        assert "not a Message" in info.value.reply.error
    finally:
        c.close()
        s.close()


def test_encode_used_on_the_wire():
    assert encode(Heartbeat()) == pack_obj({"schema": "c3.heartbeat/v1"})


# ------------------------------------------------------------------------------------------------ schema filter


@dataclass(frozen=True, eq=False)
class LaptopOnly(Message):
    """Stands in for a laptop-only message (C1/C4) that policy_deploy registers in the same process."""

    SCHEMA = "c9.laptop_only/v1"
    n: int = scalar("uint64")


@pytest.fixture
def laptop_only_registered(monkeypatch):
    monkeypatch.setattr(registry, "_BY_NAME", dict(registry._BY_NAME))  # restored after the test
    register(LaptopOnly)
    return LaptopOnly


def _ask(endpoint: str, data: bytes) -> Message:
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(endpoint)
    try:
        sock.send(data)
        assert sock.poll(2000)
        return decode(sock.recv())
    finally:
        sock.close()


HANDLERS = {Heartbeat: lambda m: Ack(ok=True, reason=""), Hello: lambda m: Ack(ok=True, reason="hello")}


def test_decode_filter_unknown_schema_for_a_registered_non_shared_name(laptop_only_registered):
    data = encode(LaptopOnly(n=1))
    assert isinstance(decode(data), LaptopOnly)  # registered in this process
    with pytest.raises(ContractError) as info:
        decode(data, schemas=SHARED_SCHEMAS)
    assert (info.value.problem, info.value.field, info.value.schema) == ("unknown_schema", "", "c9.laptop_only/v1")
    assert str(info.value) == "c9.laptop_only/v1: unknown message name 'c9.laptop_only'"


def test_filtered_server_answers_unknown_schema_like_franka_control(laptop_only_registered):
    filtered, plain = Server(HANDLERS, SHARED_SCHEMAS), Server(HANDLERS)
    try:
        data = encode(LaptopOnly(n=1))
        reply = _ask(filtered.endpoint, data)
        assert isinstance(reply, ErrorReply)
        assert (reply.request, reply.field) == ("c9.laptop_only/v1", "")
        assert reply.error == "c9.laptop_only/v1: unknown message name 'c9.laptop_only'"
        # the same text as for a name nobody registered
        unknown = _ask(filtered.endpoint, pack_obj({"schema": "c9.nobody/v1"}))
        assert unknown.error == "c9.nobody/v1: unknown message name 'c9.nobody'"
        # without the filter: as before (registered, no handler -> unexpected_message "does not handle")
        old = _ask(plain.endpoint, data)
        assert isinstance(old, ErrorReply) and "does not handle" in old.error
        assert _ask(filtered.endpoint, encode(Hello())) == Ack(ok=True, reason="hello")  # shared ones still served
    finally:
        filtered.close()
        plain.close()


def test_filter_leaves_every_other_problem_code_unchanged(laptop_only_registered):
    for example in load_index():
        try:
            want = decode(example.data)
        except ContractError as e:
            want = (e.schema, e.field, e.problem, str(e))
        try:
            got = decode(example.data, schemas=SHARED_SCHEMAS)
        except ContractError as e:
            got = (e.schema, e.field, e.problem, str(e))
        assert got == want, example.name
    # a known name with another version stays schema_version (franka_control compares names, then versions)
    with pytest.raises(ContractError) as info:
        decode(pack_obj({"schema": "c3.hello/v2"}), schemas=SHARED_SCHEMAS)
    assert info.value.problem == "schema_version"
    with pytest.raises(ContractError) as info:
        decode(pack_obj({"schema": "garbage"}), schemas=SHARED_SCHEMAS)
    assert info.value.problem == "unknown_schema"


def test_filtered_server_replies_equal_the_plain_server_for_every_stored_example(laptop_only_registered):
    filtered, plain = Server(HANDLERS, SHARED_SCHEMAS), Server(HANDLERS)
    try:
        for example in load_index():
            assert _ask(filtered.endpoint, example.data) == _ask(plain.endpoint, example.data), example.name
    finally:
        filtered.close()
        plain.close()


def test_filter_arguments_checked_at_construction():
    with pytest.raises(ValueError):
        RpcServer(ENDPOINT, {}, schemas=["garbage"])
    with pytest.raises(TypeError):
        RpcServer(ENDPOINT, {}, schemas="ack/v1")
