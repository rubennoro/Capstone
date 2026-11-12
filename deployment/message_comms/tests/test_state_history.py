import threading

import numpy as np
import pytest

from contracts import ContractError
from fakes import make_state, pose, rz
from message_comms.state_history import LookupMiss, PoseSample, StateHistory

MS = 1_000_000
T0 = 1_000 * MS


def filled(n=5, step_ms=10, **kw):
    h = StateHistory(1.0)
    for i in range(n):
        h.add(make_state(i + 1, T0 + i * step_ms * MS, **kw))
    return h


def test_interpolation_between_two_states():
    h = StateHistory(1.0)
    h.add(make_state(1, T0, T=pose((0.4950, 0, 0.30), rz(89.0)), width=0.04))
    h.add(make_state(2, T0 + 10 * MS, T=pose((0.5000, 0, 0.30), rz(90.0)), width=0.05))
    s = h.pose_at(T0 + 4 * MS, "measured")
    assert isinstance(s, PoseSample)
    assert np.allclose(s.T[:3, 3], [0.4970, 0, 0.30], atol=1e-12)
    assert np.allclose(s.T[:3, :3], rz(89.4), atol=1e-12)
    assert s.width_m == pytest.approx(0.044)


def test_exact_hit():
    h = StateHistory(1.0)
    h.add(make_state(1, T0, T=pose((0.1, 0, 0))))
    h.add(make_state(2, T0 + 10 * MS, T=pose((0.2, 0, 0))))
    assert np.array_equal(h.pose_at(T0 + 10 * MS, "measured").T, pose((0.2, 0, 0)))
    assert np.array_equal(h.pose_at(T0, "measured").T, pose((0.1, 0, 0)))


def test_misses():
    assert StateHistory(1.0).pose_at(T0, "target") is LookupMiss.EMPTY
    h = filled()
    assert h.pose_at(T0 - 1, "target") is LookupMiss.TOO_OLD
    assert h.pose_at(T0 + 40 * MS + 1, "target") is LookupMiss.TOO_NEW


def test_seq_gap_around_the_image_time_only():
    h = StateHistory(1.0)
    h.add(make_state(1, T0))
    h.add(make_state(2, T0 + 10 * MS))
    h.add(make_state(4, T0 + 20 * MS))        # seq 3 missed
    h.add(make_state(5, T0 + 30 * MS))
    assert h.pose_at(T0 + 15 * MS, "target") is LookupMiss.SEQ_GAP
    assert isinstance(h.pose_at(T0 + 5 * MS, "target"), PoseSample)
    assert isinstance(h.pose_at(T0 + 25 * MS, "target"), PoseSample)
    assert h.gap_count() == 1


def test_trimmed_to_history_s():
    h = StateHistory(0.1)
    for i in range(30):
        h.add(make_state(i + 1, T0 + i * 10 * MS))
    newest = T0 + 29 * 10 * MS
    assert h.pose_at(newest - 100 * MS, "target") != LookupMiss.TOO_OLD
    assert h.pose_at(newest - 110 * MS, "target") is LookupMiss.TOO_OLD


def test_target_vs_measured_fields():
    h = StateHistory(1.0)
    h.add(make_state(1, T0, T=pose((0.1, 0, 0)), target=pose((0.2, 0, 0))))
    assert h.pose_at(T0, "measured").T[0, 3] == pytest.approx(0.1)
    assert h.pose_at(T0, "target").T[0, 3] == pytest.approx(0.2)


def test_seq_and_t_ns_must_increase():
    h = filled(2)
    with pytest.raises(ContractError) as e:
        h.add(make_state(2, T0 + 50 * MS))
    assert e.value.field == "seq" and e.value.schema == "c2.state/v1"
    with pytest.raises(ContractError) as e:
        h.add(make_state(3, T0))
    assert e.value.field == "t_ns"


def test_equal_t_ns_updates_latest_but_adds_no_sample():
    h = StateHistory(1.0)
    h.add(make_state(1, T0, T=pose((0.1, 0, 0))))
    h.add(make_state(2, T0 + 10 * MS, T=pose((0.2, 0, 0))))
    h.add(make_state(3, T0 + 10 * MS, T=pose((0.2, 0, 0)), mode="fault", error="reflex"))
    assert h.latest().mode == "fault" and h.latest().error == "reflex"
    assert h.gap_count() == 0
    h.add(make_state(4, T0 + 20 * MS, T=pose((0.3, 0, 0))))
    assert h.gap_count() == 0
    s = h.pose_at(T0 + 15 * MS, "measured")
    assert isinstance(s, PoseSample) and s.T[0, 3] == pytest.approx(0.25)


def test_latest_and_gap_count():
    h = StateHistory(1.0)
    assert h.latest() is None and h.gap_count() == 0
    h.add(make_state(1, T0))
    h.add(make_state(3, T0 + MS))
    h.add(make_state(6, T0 + 2 * MS))
    assert h.latest().seq == 6 and h.gap_count() == 2


def test_one_thread_adds_while_another_reads():
    h = StateHistory(0.2)
    errors = []

    def writer():
        try:
            for i in range(2000):
                h.add(make_state(i + 1, T0 + i * MS, T=pose((i * 1e-4, 0, 0))))
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    while t.is_alive():
        latest = h.latest()
        if latest is not None:
            s = h.pose_at(latest.t_ns, "measured")
            if isinstance(s, PoseSample):
                assert s.T[0, 3] == pytest.approx((latest.seq - 1) * 1e-4)
    t.join()
    assert not errors
