import time

import pytest
from contracts import Ack, ContractError, Publisher
from fakes import make_state
from message_comms.state_history import StateHistory
from message_comms.state_listener import StateListener

MS = 1_000_000


def wait_for(condition, timeout_s=5.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def pub():
    p = Publisher("tcp://127.0.0.1:*", 1000)
    yield p
    p.close()


def connect(listener, pub):
    """PUB/SUB joins asynchronously: publish (increasing seq) until the first message arrives. Returns the next seq."""
    seq = [0]

    def once():
        seq[0] += 1
        pub.publish(make_state(seq[0], 1000 * MS + seq[0]))
        return listener.received() > 0
    assert wait_for(once)
    return seq[0] + 1


def test_states_arrive_in_order(pub):
    h = StateHistory(10.0)
    listener = StateListener(pub.endpoint, h, 1000, 0.02)
    listener.start()
    try:
        first = connect(listener, pub)
        assert wait_for(lambda: h.latest().seq == first - 1)          # the last connect message is in
        base = listener.received()
        for i in range(50):
            pub.publish(make_state(first + 100 + i, 2000 * MS + i * 10 * MS))
        assert wait_for(lambda: listener.received() >= base + 50)
        assert h.latest().seq == first + 149 and listener.error() is None
        assert h.gap_count() == 1                                    # the jump of 100, once
    finally:
        listener.stop()


def test_broken_message_sets_error_and_ends_thread(pub):
    h = StateHistory(10.0)
    listener = StateListener(pub.endpoint, h, 1000, 0.02)
    listener.start()
    try:
        connect(listener, pub)
        pub.publish(Ack(ok=True, reason=""))          # not a c2.state/v1 message: ContractError in receive
        assert wait_for(lambda: listener.error() is not None)
        assert isinstance(listener.error(), ContractError)
        assert wait_for(lambda: not listener._thread.is_alive())
    finally:
        listener.stop()


def test_non_increasing_seq_sets_error(pub):
    listener = StateListener(pub.endpoint, StateHistory(10.0), 1000, 0.02)
    listener.start()
    try:
        connect(listener, pub)
        pub.publish(make_state(1, 3000 * MS))
        assert wait_for(lambda: listener.error() is not None)
        assert listener.error().field == "seq"
    finally:
        listener.stop()


def test_stop_returns_within_poll_s(pub):
    listener = StateListener(pub.endpoint, StateHistory(1.0), 1000, 0.05)
    listener.start()
    t = time.monotonic()
    listener.stop()
    assert time.monotonic() - t < 0.2
    assert not listener._thread.is_alive()
