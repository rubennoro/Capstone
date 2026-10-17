"""F1: requests are sent exactly once and time out on schedule (robotmq's default would resend)."""
import multiprocessing as mp
import socket
import time

import pytest

robotmq = pytest.importorskip("robotmq")

from franka_deploy.rmq_rpc import RpcError, RpcTimeout, request


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def slow_server(endpoint, delay_s, count, stop_after_s):
    s = robotmq.RMQServer("slow_server", endpoint, robotmq.RMQLogLevel.ERROR)
    s.add_topic("infer", message_remaining_time_s=10)
    t0 = time.monotonic()
    while time.monotonic() - t0 < stop_after_s:
        data, topic = s.wait_for_request(timeout_s=0.05)
        if not topic:  # robotmq returns topic "" (not None) when wait_for_request times out
            continue
        with count.get_lock():
            count.value += 1
        req = robotmq.deserialize(data)
        if req == "error":
            s.reply_request(topic=topic, data=robotmq.serialize("ERROR: boom"))
            continue
        time.sleep(delay_s if req == "slow" else 0.0)
        s.reply_request(topic=topic, data=robotmq.serialize({"echo": req}))


@pytest.fixture
def server():
    procs = []

    def start(delay_s, stop_after_s=6.0):
        ep = f"tcp://127.0.0.1:{free_port()}"
        count = mp.Value("i", 0)
        p = mp.get_context("fork").Process(target=slow_server, args=(ep, delay_s, count, stop_after_s), daemon=True)
        p.start()
        procs.append(p)
        time.sleep(0.5)
        return ep, count

    yield start
    for p in procs:
        p.terminate()
        p.join(2)


def test_ok_and_error_reply(server):
    ep, count = server(0.0)
    c = robotmq.RMQClient("c", ep, robotmq.RMQLogLevel.ERROR)
    assert request(c, "infer", "fast", timeout_s=2.0) == {"echo": "fast"}
    with pytest.raises(RpcError, match="boom"):
        request(c, "infer", "error", timeout_s=2.0)
    assert count.value == 2


def test_timeout_not_resent(server):
    ep, count = server(delay_s=1.5)
    c = robotmq.RMQClient("c", ep, robotmq.RMQLogLevel.ERROR)
    timeout = 0.4
    t0 = time.monotonic()
    with pytest.raises(RpcTimeout):
        request(c, "infer", "slow", timeout_s=timeout)
    dt = time.monotonic() - t0
    assert dt < timeout + 0.1, dt
    time.sleep(2.0)  # long enough for robotmq to have resent several times if it were going to
    assert count.value == 1, f"server saw {count.value} requests"
    # the client is usable again afterwards (the stale reply is dropped, not delivered to the next call)
    assert request(c, "infer", "fast", timeout_s=2.0) == {"echo": "fast"}


def test_upstream_default_timeout_is_not_a_timeout(server):
    """Documents the hazard behind F1: with robotmq's default automatic_resend=True, timeout_s does not bound the
    call. The client silently resends and returns whatever reply arrives later (here after 1.0 s with timeout 0.4 s);
    upstream PolicyAgent's 5 s "timeout" behaves the same way."""
    ep, count = server(delay_s=1.0)
    c = robotmq.RMQClient("c", ep, robotmq.RMQLogLevel.OFF)
    t0 = time.monotonic()
    reply = robotmq.deserialize(c.request_with_data("infer", robotmq.serialize("slow"), timeout_s=0.4))
    dt = time.monotonic() - t0
    time.sleep(1.5)
    print(f"upstream default: reply after {dt:.2f} s (timeout_s=0.4), server executed {count.value} time(s)")
    assert reply == {"echo": "slow"}
    assert dt >= 1.0


def test_dead_server_times_out(server):
    ep = f"tcp://127.0.0.1:{free_port()}"  # nothing listening
    c = robotmq.RMQClient("c", ep, robotmq.RMQLogLevel.ERROR)
    t0 = time.monotonic()
    with pytest.raises(RpcTimeout):
        request(c, "infer", "x", timeout_s=0.3)
    dt = time.monotonic() - t0
    print(f"dead server: raised after {dt:.3f} s")
    # measured ~0.30 s extra with nothing listening (socket close linger + reconnect inside robotmq); still bounded
    assert dt < 0.3 + 0.5, dt
