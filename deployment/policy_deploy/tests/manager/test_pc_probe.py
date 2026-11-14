"""pc_probe against an in-process FakePc (STRUCTURE.md 8.6, 12.2): reads C2, sends only hello/heartbeat, prints both
estimators, refuses while holding past the H2 limit; --check --json lines."""
from __future__ import annotations

import io
import json

import pytest
from manager_pc_fakepc import make_fake_pc
from manager_pc_harness import FAST, merge, small_link
from station_helpers import write_station

from policy_deploy.contracts import Station
from policy_deploy.manager.pc_probe import PcCheck, PcProbe, emit
from policy_deploy.manager.settings import Settings

FAST_LINK = small_link(watchdog_limit_s=0.5, c3_timeout_s=0.15)


@pytest.fixture
def probe_for(tmp_path):
    made = []

    def make(link=None, manager=None, **fake_overrides):
        fake = make_fake_pc(tmp_path / "fake", link=link, **fake_overrides)
        fake.start()
        station = Station.load(write_station(tmp_path, manager=merge(FAST, manager or {})), mock=True)
        settings = Settings.from_station(station, link=link, state_endpoint=fake.state_endpoint,
                                         commands_endpoint=fake.commands_endpoint)
        out = io.StringIO()
        probe = PcProbe(settings, out)
        made.append((fake, probe))
        return fake, probe, out
    yield make
    for fake, probe in made:
        probe.close()
        fake.close()


def test_reads_c2_sends_only_hello_and_heartbeat_and_prints_both_estimators(probe_for):
    fake, probe, out = probe_for()
    rows = probe.run(2.2, report_every_s=0.5)
    by = {r.name: r for r in rows}
    assert by["H2"].status == "pass" and by["hello"].status == "pass" and by["C2 rate"].status == "pass"
    assert set(fake.request_names()) == {"c3.hello/v1", "c3.heartbeat/v1"}
    assert set(probe.sent) == {"c3.hello/v1", "c3.heartbeat/v1"} and probe.sent[0] == "c3.hello/v1"
    lines = out.getvalue().splitlines()
    assert lines and "round_trip +" in lines[-1] and "one_way +" in lines[-1] and "diff" in lines[-1]
    import time
    now = time.monotonic_ns()
    true_offset_s = (fake.clock.at(now) - now) / 1e9                    # +1000 s and 30 ppm of the laptop clock
    for name in ("round_trip", "one_way"):
        printed = float(lines[-1].split(f"{name} ")[1].split(" s")[0])
        assert printed == pytest.approx(true_offset_s, abs=0.005)
    assert probe.reply is not None and probe.reply.mock is True
    assert probe.history.latest() is not None and probe.stats.snapshot().c2_received > 100


def test_refuses_while_holding_past_the_h2_limit(probe_for):
    fake, probe, _ = probe_for(link=FAST_LINK, manager={"link_jitter_allowance_s": 0.05}, start_mode="holding")
    fake.hold_forever()
    rows = probe.run(1.0)
    assert rows[-1].name == "H2" and rows[-1].status == "fail"
    assert "another C3 client" in rows[-1].detail
    assert fake.request_names() == [] and probe.sent == []


def test_waits_out_an_orphaned_holding(probe_for):
    fake, probe, _ = probe_for(link=FAST_LINK, manager={"link_jitter_allowance_s": 0.05}, start_mode="holding")
    rows = probe.run(0.3, report_every_s=None)
    assert rows[0].name == "H2" and rows[0].status == "pass"
    assert fake.request_names()[0] == "c3.hello/v1"


def test_check_rows_and_json_lines(probe_for):
    fake, probe, _ = probe_for()
    rows = probe.check_rows(mock=True)
    names = [r.name for r in rows]
    for want in ("H2", "hello", "C2 rate", "hello equality", "worst gap", "home timeout", "waypoint lead",
                 "clock agreement", "PC commit", "PC mock"):
        assert want in names, names
    by = {r.name: r for r in rows}
    for name in ("hello equality", "worst gap", "home timeout", "waypoint lead", "clock agreement"):
        assert by[name].status == "pass", by[name]
    assert set(fake.request_names()) <= {"c3.hello/v1", "c3.heartbeat/v1"}
    buf = io.StringIO()
    emit(rows, True, buf)
    parsed = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert len(parsed) == len(rows)
    assert all(set(p) == {"name", "status", "detail"} and p["status"] in ("pass", "warn", "fail", "info")
               for p in parsed)


def test_check_with_mock_false_against_a_mock_pc_fails_equality(probe_for):
    fake, probe, _ = probe_for()
    rows = {r.name: r for r in probe.check_rows(mock=False)}
    assert rows["hello equality"].status == "fail" and "mock" in rows["hello equality"].detail


def test_no_c2_row(tmp_path):
    from station_helpers import free_port
    station = Station.load(write_station(tmp_path, manager=merge(FAST, {"timeouts": {"startup_s": 0.3}})), mock=True)
    port = free_port()
    settings = Settings.from_station(station, state_endpoint=f"tcp://127.0.0.1:{port}",
                                     commands_endpoint=f"tcp://127.0.0.1:{free_port()}")
    probe = PcProbe(settings, io.StringIO())
    try:
        rows = probe.run(0.5)
    finally:
        probe.close()
    assert rows == [PcCheck("C2", "fail", rows[0].detail)] and "no C2 state" in rows[0].detail
