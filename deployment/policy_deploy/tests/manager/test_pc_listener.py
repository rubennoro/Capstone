# adapted from rnoro/deployment:deployment/message_comms/tests/test_state_listener.py @ 91b178b
# changes: PcListener (was StateListener): arrival stamps, PcClock.add_state only when t_ns advanced, LinkStats; a seq
# going back is PcRestarted; imports.
import time

import pytest
from deploy_contracts import Ack, ContractError, Publisher
from manager_fakes import make_state

from policy_deploy.manager.errors import PcRestarted
from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.pc_listener import PcListener
from policy_deploy.manager.state_history import StateHistory

MS = 1_000_000


def wait_for(condition, timeout_s=5.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


class RecordingClock:
    """A PcClock stand-in that records add_state calls."""

    def __init__(self):
        self.states = []

    def add_state(self, t_pc_ns, arrival_ns):
        self.states.append((t_pc_ns, arrival_ns))


@pytest.fixture
def pub():
    p = Publisher("tcp://127.0.0.1:*", 1000)
    yield p
    p.close()


def make_listener(pub, history=None, poll_s=0.02):
    clock = RecordingClock()
    stats = LinkStats(10.0)
    listener = PcListener(pub.endpoint, history or StateHistory(10.0), clock, stats, 1000, poll_s)
    return listener, clock, stats


def connect(listener, pub):
    """PUB/SUB joins asynchronously: publish (increasing seq) until the first message arrives. Returns the next seq."""
    seq = [0]

    def once():
        seq[0] += 1
        pub.publish(make_state(seq[0], 1000 * MS + seq[0]))
        return listener.received() > 0
    assert wait_for(once)
    last = seq[0]
    assert wait_for(lambda: listener._history.latest() is not None and listener._history.latest().seq == last)
    return last + 1


def test_states_arrive_in_order_with_arrival_stamps_and_stats(pub):
    h = StateHistory(10.0)
    listener, clock, stats = make_listener(pub, h)
    listener.start()
    try:
        first = connect(listener, pub)
        assert wait_for(lambda: h.latest().seq == first - 1)
        base = listener.received()
        before = time.monotonic_ns()
        for i in range(50):
            pub.publish(make_state(first + 100 + i, 2000 * MS + i * 10 * MS))
        assert wait_for(lambda: listener.received() >= base + 50)
        assert h.latest().seq == first + 149 and listener.error() is None
        assert h.gap_count() == 1                                    # the jump of 100, once
        assert listener.last_arrival_ns() >= before
        snap = stats.snapshot()
        assert snap.seq_gaps == 1 and snap.c2_received == listener.received()
        assert clock.states[-1][0] == 2000 * MS + 49 * 10 * MS and clock.states[-1][1] >= before
    finally:
        listener.stop()


def test_repeated_t_ns_is_not_a_clock_sample(pub):
    listener, clock, _ = make_listener(pub)
    listener.start()
    try:
        first = connect(listener, pub)
        n = len(clock.states)
        for i in range(5):
            pub.publish(make_state(first + i, 5000 * MS))           # the same t_ns five times
        pub.publish(make_state(first + 5, 5010 * MS))
        assert wait_for(lambda: listener._history.latest().seq == first + 5)
        assert [t for t, _ in clock.states[n:]] == [5000 * MS, 5010 * MS]
    finally:
        listener.stop()


def test_broken_message_sets_error_and_ends_thread(pub):
    listener, _, _ = make_listener(pub)
    listener.start()
    try:
        connect(listener, pub)
        pub.publish(Ack(ok=True, reason=""))          # not a c2.state/v1 message: ContractError in receive
        assert wait_for(lambda: listener.error() is not None)
        assert isinstance(listener.error(), ContractError)
        assert wait_for(lambda: not listener._thread.is_alive())
    finally:
        listener.stop()


def test_equal_seq_is_a_contract_error_and_seq_back_a_restart(pub):
    listener, _, _ = make_listener(pub)
    listener.start()
    try:
        nxt = connect(listener, pub)
        pub.publish(make_state(nxt - 1, 3000 * MS))
        assert wait_for(lambda: listener.error() is not None)
        assert isinstance(listener.error(), ContractError) and listener.error().field == "seq"
    finally:
        listener.stop()
    listener, _, _ = make_listener(pub)
    listener.start()
    try:
        nxt = connect(listener, pub)
        for i in range(3):
            pub.publish(make_state(nxt + i, 3000 * MS + i))
        assert wait_for(lambda: listener._history.latest().seq == nxt + 2)
        pub.publish(make_state(1, 4000 * MS))
        assert wait_for(lambda: listener.error() is not None)
        assert isinstance(listener.error(), PcRestarted)
    finally:
        listener.stop()


def test_stop_returns_within_poll_s(pub):
    listener, _, _ = make_listener(pub, poll_s=0.05)
    listener.start()
    t = time.monotonic()
    listener.stop()
    assert time.monotonic() - t < 0.3
    assert not listener._thread.is_alive()
