"""build_meta: PLAN 6.3 content, plain YAML types, validation, and the round trip through finish / episodes."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from core_helpers import T0_PC, hello_reply, motion_settings, recording_summary

from teleop_collect.teleop.episode import EpisodeInfo, EpisodeRecording, TaskStore, build_meta

TIMES = {"start_ns": 1_000_000_000, "stop_ns": 25_100_000_000, "start_pc_ns": T0_PC, "stop_pc_ns": T0_PC + 24 * 10**9}


@pytest.fixture
def task_ep(tmp_path):
    task = TaskStore(tmp_path).create("pick_block", "pick the red block", {})
    return task, task.new_episode()


def build(task, ep, **changes):
    args = dict(task=task, episode=ep, outcome="success", times=TIMES, frames=recording_summary(),
                counts={"c2_messages": 2400, "sets_sent": 500}, hello=hello_reply(), laptop_commit="a" * 40,
                laptop_dirty=True, rig_file=Path("/x/rigs/rig_pi_v2_640.yaml"), motion=motion_settings(),
                pi_fit={"offset_ns": 5, "drift_ppm": 1.0}, pc_fit={"offset_ns": -7, "drift_ppm": None, "used": 6})
    args.update(changes)
    return build_meta(**args)


def test_content(task_ep):
    task, ep = task_ep
    m = build(task, ep)
    assert m["task"] == "pick_block" and m["task_description"] == "pick the red block"
    assert m["episode"] == "episode_000" and m["episode_number"] == 0 and m["outcome"] == "success"
    assert m["times"] == TIMES and m["duration_s"] == pytest.approx(24.1)
    assert m["counts"]["frames_main"] == 300 and m["counts"]["frames_ultrawide"] == 290
    assert m["counts"]["uw_reused"] == 10 and m["counts"]["c2_messages"] == 2400 and m["counts"]["sets_sent"] == 500
    assert m["frames"]["main"]["frames"] == 300 and "schema" not in m["frames"]
    h = m["pc_hello"]
    assert h["F_T_TCP"] == hello_reply().F_T_TCP.tolist() and len(h["F_T_TCP"]) == 4
    assert h["mock"] is True and h["max_lead_s"] == 1.5 and h["commit"] == "unknown" and isinstance(h["schemas"], list)
    assert m["laptop"] == {"commit": "a" * 40, "dirty": True}
    assert m["rig_file"] == "/x/rigs/rig_pi_v2_640.yaml"
    assert m["teleop"]["v_max_m_s"] == 0.05 and m["teleop"]["waypoints"] == 2 and m["teleop"]["lead_ms"] == 100.0
    assert m["teleop"]["worst_step_m"] == pytest.approx(0.0075)
    assert m["clock"] == {"pi": {"offset_ns": 5, "drift_ppm": 1.0}, "pc": {"offset_ns": -7, "drift_ppm": None,
                                                                           "used": 6}}
    assert yaml.safe_load(yaml.safe_dump(m)) == m                    # plain types only


def test_validation(task_ep):
    task, ep = task_ep
    with pytest.raises(ValueError, match="only success, fail"):
        build(task, ep, outcome="discard")
    with pytest.raises(ValueError, match="stop_ns"):
        build(task, ep, times={**TIMES, "stop_ns": 0})
    with pytest.raises(ValueError, match="unknown"):
        build(task, ep, times={**TIMES, "extra_ns": 1})
    with pytest.raises(ValueError, match="start_ns"):
        build(task, ep, times={**TIMES, "start_ns": 1.5})
    with pytest.raises(ValueError, match="frames summary"):
        build(task, ep, counts={"frames_main": 1})
    m = build(task, ep, times={**TIMES, "start_pc_ns": None, "stop_pc_ns": None})
    assert m["times"]["start_pc_ns"] is None


def test_round_trip_through_finish(task_ep):
    task, ep = task_ep
    rec = EpisodeRecording(ep)
    rec.close()
    ep.finish(build(task, ep, outcome="fail", counts=rec.counts()))
    assert task.episodes() == [EpisodeInfo("episode_000", "fail", pytest.approx(24.1), 300)]
    on_disk = yaml.safe_load((ep.path / "meta.yaml").read_text())
    assert on_disk["counts"]["c2_messages"] == 0 and on_disk["pc_hello"]["F_T_TCP"][2][3] == pytest.approx(0.15)


def test_rig_embedded(task_ep):
    from teleop_collect.contracts import paths
    rig_file = paths.rigs_dir() / "rig_pi_v2_640.yaml"
    m = build(*task_ep, rig_file=rig_file)
    assert m["rig_file"] == str(rig_file) and m["rig_note"] == ""
    assert m["rig"] == yaml.safe_load(rig_file.read_text()) and isinstance(m["rig"], dict)
    assert yaml.safe_load(yaml.safe_dump(m)) == m


def test_rig_fails_soft(task_ep, tmp_path):
    m = build(*task_ep, rig_file=tmp_path / "missing.yaml")
    assert m["rig"] is None and "missing.yaml" in m["rig_note"] and "FileNotFoundError" in m["rig_note"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n")
    m = build(*task_ep, rig_file=bad)
    assert m["rig"] is None and "bad.yaml" in m["rig_note"] and "\n" not in m["rig_note"]
