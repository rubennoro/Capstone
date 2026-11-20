"""`policy_deploy run --mock` end to end (STRUCTURE.md 12.3): the launcher starts the fake PC, frames with the fake
Pi playing station_helpers.RECORDED_EPISODE, policy hold_still and the manager with a scripted operator."""
from __future__ import annotations

from integration_helpers import (jsonl, need_episode, need_pc_ports_free, run_policy_deploy, session_dir,
                                 write_run_station)

SCRIPT = [{"key": "c"}, {"wait_steps": 20}, {"key": "y"}, {"key": "c"}, {"wait_steps": 5}, {"key": "s"},
          {"key": "h"}, {"key": "q"}]
MAX_LEAD_NS = 1_500_000_000          # config/fake_pc.yaml max_lead_s


def _fail_text(done) -> str:
    return f"exit {done.returncode}\nstdout:\n{done.stdout[-3000:]}\nstderr:\n{done.stderr[-3000:]}"


def test_run_mock_two_episodes(tmp_path):
    need_episode()
    need_pc_ports_free()
    done = run_policy_deploy(write_run_station(tmp_path, SCRIPT))
    assert done.returncode == 0, _fail_text(done)
    assert "policy_deploy: done" in done.stdout + done.stderr
    session = session_dir(tmp_path)
    for name in ("station.yaml", "contracts/link.yaml", "contracts/frames.yaml", "contracts/limits.yaml",
                 "logs/launcher.log", "logs/fake_pc.log", "logs/frames.log", "logs/policy.log", "logs/manager.log",
                 "fake_pc/commands.jsonl", "frames/stats.jsonl", "policy/startup.json", "manager/session.yaml",
                 "manager/episodes.jsonl", "manager/steps.jsonl", "manager/events.jsonl", "manager/clock.jsonl",
                 "manager/link.jsonl"):
        assert (session / name).is_file(), name
    episodes = jsonl(session / "manager/episodes.jsonl")
    assert [e["outcome"] for e in episodes] == ["success", "stopped"]
    assert episodes[0]["steps_sent"] >= 20 and episodes[1]["steps_sent"] >= 5

    commands = jsonl(session / "fake_pc/commands.jsonl")
    sets = [c for c in commands if c["request"] == "c3.set_waypoints/v1"]
    assert len(sets) >= 25
    ids = [c["chunk_id"] for c in sets]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert all(c["rejected"] == 0 and c["reply"] == "c3.set_waypoints.reply/v1" for c in sets)
    assert sum(c["accepted"] for c in sets) > 0
    assert all(c["t_last_pc_ns"] - c["t_pc_ns"] <= MAX_LEAD_NS for c in sets)      # on the fake PC's own clock
    assert commands[-1]["request"] == "c3.stop/v1"
    assert any(c["request"] == "c3.home/v1" and c["ok"] for c in commands)

    # the clock map: offset (PC - laptop) as the fake PC's drifting clock says, about +1000 s
    fits = [r for r in jsonl(session / "manager/clock.jsonl") if r.get("kind") == "fit" and r["offset_ns"]]
    assert fits
    true_offsets = [c["t_pc_ns"] - c["t_ns"] for c in commands]
    offset = fits[-1]["offset_ns"]
    assert offset > 999e9
    assert min(abs(offset - t) for t in true_offsets) < 5e6        # within 5 ms (drift 30 ppm moves it slowly)

    steps = [s for s in jsonl(session / "manager/steps.jsonl") if s["sent"]]
    assert steps and all(s["t_anchor_pc_ns"] - s["t_anchor_ns"] > 999e9 for s in steps)
    assert all(s["c3_rtt_ms"] is not None for s in steps)


def test_run_mock_h2_wait_refuses_a_pc_driven_by_another_client(tmp_path):
    """The fake PC starts in holding and another C3 client keeps it there (start_hold_s): rule H2, the manager sends
    nothing on C3 for 2 x watchdog, then refuses (PC_BUSY); the launcher stops everything, exit 1. (The orphaned
    holding that the PC's watchdog stops is covered in-process by tests/manager/test_startup_sequence.py: under the
    launcher the fake PC's watchdog would trip before the manager even starts.)"""
    need_episode()
    need_pc_ports_free()
    script = [{"key": "c"}, {"wait_steps": 3}, {"key": "y"}, {"key": "q"}]
    done = run_policy_deploy(write_run_station(tmp_path, script,
                                               fake_pc={"start_mode": "holding", "start_hold_s": 1000.0}))
    out = done.stdout + done.stderr
    assert done.returncode == 1, _fail_text(done)
    assert "policy_deploy: FAILED manager" in out
    assert "FATAL manager: the PC is driving an episode for another C3 client" in out
    session = session_dir(tmp_path)
    events = jsonl(session / "manager/events.jsonl")
    assert any(e["detail"] == "C2 mode holding" for e in events)
    log = session / "fake_pc/commands.jsonl"
    assert not log.exists() or jsonl(log) == []                    # nothing at all on C3 (rule H2)
