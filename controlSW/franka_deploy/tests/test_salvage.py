"""Abnormal-end logging (F2): 1 s trace parts + incremental events are completed by the salvage and the validator
accepts the aborted episode."""
import json
import os

import numpy as np

from franka_deploy.controllers.safe_controller import merge_trace_parts
from franka_deploy.eval.salvage import salvage_episode
from franka_deploy.tools.validate_episode import validate_episode


def make_episode(tmp_path):
    d = tmp_path / "episode_000003"
    (d / "tap").mkdir(parents=True)
    (d / "config").mkdir()
    for f in ("deploy.yaml", "bridge.yaml", "franka_cartesian.yaml", "git.json", "versions.json"):
        (d / "config" / f).write_text("{}")
    (d / "config" / "checksums.json").write_text(json.dumps({"checkpoint_sha256": "a" * 64, "prompt_index": 0}))
    parts = d / "controller_trace_parts"
    parts.mkdir()
    for k in range(3):  # three 1 s parts at 200 Hz
        t = 100.0 + k + np.arange(200) / 200.0
        np.savez(parts / f"part_{k:05d}.npz", t=t, fk=np.zeros((200, 7)), commanded=np.zeros((200, 7)),
                 width_measured=np.zeros(200), width_commanded=np.zeros(200), held=np.zeros(200), fault=np.zeros(200))
    with open(d / "events.jsonl", "w") as f:
        f.write(json.dumps({"t": 100.0, "kind": "episode_start"}) + "\n")
        f.write(json.dumps({"t": 102.9, "kind": "prompted"}) + "\n")
    (d / "label.json").write_text(json.dumps({"label": "aborted:unfinalized", "notes": []}))
    return d


def test_merge_parts(tmp_path):
    d = make_episode(tmp_path)
    assert merge_trace_parts(str(d / "controller_trace_parts"), str(d / "tr.npz"))
    z = np.load(d / "tr.npz")
    assert len(z["t"]) == 600 and np.all(np.diff(z["t"]) > 0) and not (d / "controller_trace_parts").exists()


def test_salvaged_aborted_episode_validates(tmp_path):
    d = make_episode(tmp_path)
    assert validate_episode(str(d))  # incomplete before the salvage (no summary)
    done = salvage_episode(str(d), "controller_died")
    assert done.get("trace_merged") and done.get("summary_written")
    lab = json.load(open(d / "label.json"))
    assert lab["label"] == "aborted:controller_died" and lab["salvaged"]
    s = json.load(open(d / "summary.json"))
    assert s["policy_calls"] == 0 and abs(s["duration_s"] - 2.995) < 0.01
    assert validate_episode(str(d)) == []
    assert salvage_episode(str(d), "again") == {}  # idempotent: already salvaged
