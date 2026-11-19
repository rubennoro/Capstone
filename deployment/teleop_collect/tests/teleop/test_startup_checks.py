# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/manager/test_startup_checks.py @ 2ffe641
# changes: imports; the waypoint lead is K x Delta; no settled rule (+ split_commit).
"""StartupChecks (STRUCTURE.md 6.2): the hello and budget checks of start-up step 5."""
from __future__ import annotations

import numpy as np
import pytest
from deploy_contracts import SHARED_SCHEMAS
from teleop_helpers import COMMIT, hello_reply

from teleop_collect.teleop.pc_clock import HelloSample, RoundTripPcClock
from teleop_collect.teleop.startup_checks import StartupChecks, shortest_hello, split_commit

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


def test_waypoint_lead_is_k_times_delta(make_settings):
    s = make_settings(motion={"waypoints": 4, "spacing_ms": 25})
    assert checks(s).waypoint_lead_ms() == 100.0
    f = checks(s).waypoint_lead(hello_reply(0, max_lead_s=0.1))
    assert f is not None and "100 ms" in f.what and f.check == "lower teleop.motion.waypoints or spacing_ms"
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


def test_split_commit():
    assert split_commit(COMMIT + "+dirty") == (COMMIT, True)
    assert split_commit(COMMIT) == (COMMIT, False)
    assert split_commit("unknown") == ("unknown", False)


def test_rows_for_pc_probe(settings):
    c = warm_clock(settings)
    rows = checks(settings).rows(hello_reply(0, commit="f" * 40), c, c.hellos())
    by = {r.name: r for r in rows}
    assert by["hello equality"].status == "pass" and by["worst gap"].status == "pass"
    assert by["clock agreement"].status == "pass" and by["PC commit"].status == "warn"
    assert by["PC mock"].status == "info"
    rows = checks(settings, mock=False).rows(hello_reply(0), c, c.hellos())
    assert {r.name: r.status for r in rows}["hello equality"] == "fail"


def test_laptop_commit_counts_untracked_files_as_dirty(tmp_path, monkeypatch):
    import subprocess
    from teleop_collect.teleop import startup_checks
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(git + ["add", "a.py"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "a"], check=True)
    monkeypatch.setattr(startup_checks.paths, "deployment_dir", lambda: repo)
    clean = startup_checks.laptop_commit()
    assert len(clean) == 40
    (repo / "new_module.py").write_text("y = 2\n")                    # untracked code
    assert startup_checks.laptop_commit() == clean + "+dirty"
