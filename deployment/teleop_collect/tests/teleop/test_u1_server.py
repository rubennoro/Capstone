"""U1Server and MessageLog (STRUCTURE.md 4.4, 6.6): a real REP socket on tcp://127.0.0.1:*, served without waiting."""
from __future__ import annotations

import threading
import time

import zmq
from deploy_contracts import Ack, RpcClient, decode, pack_obj

from teleop_collect.contracts import STATUS_MAX_MESSAGES, KeysUpdate, StartEpisode, StopArm
from teleop_collect.teleop.u1_server import U1_REQUESTS, MessageLog, U1Server


def test_serves_up_to_n_without_waiting_and_every_request_reaches_dispatch():
    seen = []

    def dispatch(msg):
        seen.append(msg.SCHEMA)
        return Ack(ok=True, reason="")
    server = U1Server("tcp://127.0.0.1:*", dispatch)
    clients = [RpcClient("teleop", server.endpoint) for _ in range(3)]
    try:
        assert server.serve(4) == 0                                  # nothing waiting: returns at once
        results = []
        threads = [threading.Thread(target=lambda c=c, m=m: results.append(c.request(m, Ack, 2.0)))
                   for c, m in zip(clients, (StopArm(), StartEpisode(),
                                             KeysUpdate(held=("x+",), slow=False, focus=True)))]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 2.0
        served = 0
        while served < 3 and time.monotonic() < deadline:
            served += server.serve(2)
            time.sleep(0.01)
        for t in threads:
            t.join(2.0)
        assert served == 3 and len(results) == 3 and all(r.ok for r in results)
        assert sorted(seen) == sorted(["u1.stop_arm/v1", "u1.start_episode/v1", "u1.keys/v1"])
    finally:
        for c in clients:
            c.close()
        server.close()
    assert len(U1_REQUESTS) == 12       # no u1.home/v1 (no laptop home); + label, delete_task, set_data_root


def test_a_home_request_is_unknown_and_never_reaches_dispatch():
    """There is no laptop home: u1.home/v1 is not a U1 request (homing and the start pose are the PC's h and p)."""
    seen = []
    server = U1Server("tcp://127.0.0.1:*", lambda m: seen.append(m) or Ack(ok=True, reason=""))
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(server.endpoint)
    try:
        sock.send(pack_obj({"schema": "u1.home/v1"}))
        deadline = time.monotonic() + 2.0
        while not server.serve(1) and time.monotonic() < deadline:
            time.sleep(0.01)
        reply = decode(sock.recv())
        assert reply.SCHEMA == "error/v1" and seen == []
    finally:
        sock.close()
        server.close()


def test_a_broken_request_never_reaches_dispatch():
    seen = []
    server = U1Server("tcp://127.0.0.1:*", lambda m: seen.append(m) or Ack(ok=True, reason=""))
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(server.endpoint)
    try:
        sock.send(pack_obj({"schema": "u1.keys/v1", "held": ["warp"], "slow": False, "focus": True}))
        deadline = time.monotonic() + 2.0
        while not server.serve(1) and time.monotonic() < deadline:
            time.sleep(0.01)
        reply = decode(sock.recv())
        assert reply.SCHEMA == "error/v1" and reply.field.startswith("held") and seen == []
    finally:
        sock.close()
        server.close()


def test_message_log_ids_and_limits():
    log = MessageLog(kept=5)
    for i in range(8):
        log.add("info", f"m{i}")
    msgs = log.after(0)
    assert [m.id for m in msgs] == [4, 5, 6, 7, 8]                 # the newest 5 kept
    assert [m.id for m in log.after(6)] == [7, 8]
    big = MessageLog(kept=200)
    for i in range(120):
        big.add("warn", "  two   spaces\nnewline ")
    assert len(big.after(0)) == STATUS_MAX_MESSAGES and big.after(0)[0].id == 1
    assert big.after(0)[0].text == "two spaces newline"
