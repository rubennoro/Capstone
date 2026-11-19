"""SessionLog (STRUCTURE.md 6.2): session.yaml and the three jsonl files; nothing else."""
from __future__ import annotations

import json
from datetime import datetime

import yaml
from teleop_helpers import COMMIT, FakeClock, hello_reply

from teleop_collect.teleop.session_log import SessionLog


def test_files_and_rows(tmp_path, settings):
    clock = FakeClock()
    log = SessionLog(tmp_path / "teleop", datetime.now, clock)
    log.write_session(settings, hello_reply(5), COMMIT, {"offset_ns": 7})
    log.log_event("start", "x")
    log.log_clock({"kind": "fit", "offset_ns": 7})
    log.log_link({"rtt_n": 3})
    log.close()
    log.log_event("after close", "ignored")
    names = sorted(p.name for p in (tmp_path / "teleop").iterdir())
    assert names == ["clock.jsonl", "events.jsonl", "link.jsonl", "session.yaml"]
    doc = yaml.safe_load((tmp_path / "teleop" / "session.yaml").read_text())
    assert doc["hello"]["commit"] == COMMIT and len(doc["hello"]["F_T_TCP"]) == 4 and "schema" not in doc["hello"]
    assert doc["settings"]["motion"]["waypoints"] == 2 and doc["laptop_commit"] == COMMIT
    events = [json.loads(x) for x in (tmp_path / "teleop" / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["start"] and events[0]["t_ns"] == clock.now
