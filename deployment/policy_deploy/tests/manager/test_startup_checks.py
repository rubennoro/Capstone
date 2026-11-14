"""StartupChecks (STRUCTURE.md 8.4, 12.2): the hello and budget checks of start-up step 5 and the settled rule."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from deploy_contracts import SHARED_SCHEMAS, TcpFrame
from manager_fakes import COMMIT, hello_reply

from policy_deploy.manager.pc_clock import HelloSample, RoundTripPcClock
from policy_deploy.manager.startup_checks import StartupChecks, shortest_hello

S = 1_000_000_000
MS = 1_000_000
OFFSET = 1000 * S
T0 = 100 * S


def checks(settings, mock=True, commit=COMMIT):
    return StartupChecks(settings, mock=mock, commit=commit)


def warm_clock(settings):
    c = RoundTripPcClock(settings.clock, lambda: T0 + 3 * S)
    for i in range(101):
        t = T0 + i * 20 * MS
        c.add_hello(HelloSample(t, t + MS + OFFSET, t + 2 * MS))
    return c


def test_an_equal_hello_passes(settings):
    c = warm_clock(settings)
    assert checks(settings).hello_failures(hello_reply(0), c, c.hellos()) == []


@pytest.mark.parametrize("field, change", [
    ("schemas", dict(schemas=tuple(sorted(SHARED_SCHEMAS[:-1] + ("zz.extra/v1",))))),
    ("mock", dict(mock=False)),
    ("F_T_TCP", dict(F_T_TCP=np.eye(4))),
    ("rotation_tol", dict(rotation_tol=2.0e-6)),
    ("gripper_width_max_m", dict(gripper_width_max_m=0.1)),
    ("max_waypoints", dict(max_waypoints=32)),
    ("watchdog_limit_s", dict(watchdog_limit_s=2.0)),
])
def test_each_compare_hello_field_refuses_naming_it(settings, field, change):
    failures = checks(settings).equality(hello_reply(0, **change))
    assert [f.name for f in failures] == ["equality"]
    assert failures[0].what.startswith(f"hello mismatch: {field}: PC ")


def test_worst_gap(settings):
    ck = checks(settings)
    assert settings.worst_gap_s == pytest.approx(0.57)
    assert ck.worst_gap(hello_reply(0)) is None                          # vs 1.0
    f = ck.worst_gap(hello_reply(0, watchdog_limit_s=0.5))
    assert f is not None and "0.57 s" in f.what and "0.50 s" in f.what
    assert f.check == "raise watchdog_limit_s in link.yaml or lower the laptop timeouts"


def test_home_timeout(settings):
    ck = checks(settings)
    assert ck.home_timeout(hello_reply(0, homing_timeout_s=30.0)) is None
    assert ck.home_timeout(hello_reply(0, homing_timeout_s=90.0)) is not None     # home_s == homing_timeout_s
    assert ck.home_timeout(hello_reply(0, homing_timeout_s=120.0)) is not None


def test_waypoint_lead(make_settings):
    s = make_settings(exec_h=16)
    f = checks(s).waypoint_lead(hello_reply(0, max_lead_s=0.5))
    assert f is not None and "750 ms" in f.what
    assert checks(s).waypoint_lead(hello_reply(0, max_lead_s=1.5)) is None


@pytest.mark.parametrize("agree_ms, ok", [(4.9, True), (5.1, False)])
def test_clock_agreement_uses_the_shortest_rtt_hello(settings, agree_ms, ok):
    c = warm_clock(settings)
    t = T0 + 3 * S
    good = HelloSample(t, t + MS + OFFSET + int(agree_ms * MS), t + 2 * MS)      # the shortest round trip
    slow_but_fine = HelloSample(t + S, t + S + 50 * MS + OFFSET, t + S + 100 * MS)
    hellos = [slow_but_fine, good]
    assert shortest_hello(hellos) is good
    f = checks(settings).clock_agreement(c, hellos)
    assert (f is None) is ok
    if not ok:
        assert f.what == "clock mismatch at start-up: map and hello differ by 5.1 ms"
        assert f.check == "run pc_probe; try clock.estimator round_trip"


def test_commit_warning_only(settings):
    ck = checks(settings, commit=COMMIT)
    assert ck.commit_warning(hello_reply(0)) is None
    other = "f" * 40
    assert "differs" in ck.commit_warning(hello_reply(0, commit=other))
    assert "differs" in ck.commit_warning(hello_reply(0, commit="unknown"))
    assert "uncommitted" in checks(settings, commit=COMMIT + "+dirty").commit_warning(hello_reply(0))
    c = warm_clock(settings)
    assert checks(settings, commit="unknown").hello_failures(hello_reply(0, commit=other), c, c.hellos()) == []


def test_settled_rule(settings):
    unsettled = dataclasses.replace(settings, frame=TcpFrame(F_T_TCP=settings.frame.F_T_TCP, settled=False))
    settled = dataclasses.replace(settings, frame=TcpFrame(F_T_TCP=settings.frame.F_T_TCP, settled=True))
    f = checks(unsettled).settled("diffusion", pc_mock=False)
    assert f is not None and f.what == ("trained policy refused on the real arm: contracts/frames.yaml settled is "
                                        "false (PLAN §9)")
    assert checks(unsettled).settled("diffusion", pc_mock=True) is None        # the fake arm
    assert checks(unsettled).settled("hold_still", pc_mock=False) is None
    assert checks(settled).settled("diffusion", pc_mock=False) is None


def test_rows_for_pc_probe(settings):
    c = warm_clock(settings)
    rows = checks(settings).rows(hello_reply(0, commit="f" * 40), c, c.hellos())
    by = {r.name: r for r in rows}
    assert by["hello equality"].status == "pass" and by["worst gap"].status == "pass"
    assert by["clock agreement"].status == "pass" and by["PC commit"].status == "warn"
    assert by["PC mock"].status == "info"
    rows = checks(settings, mock=False).rows(hello_reply(0), c, c.hellos())
    assert {r.name: r.status for r in rows}["hello equality"] == "fail"
