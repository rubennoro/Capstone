"""launcher.task_replay_args (B8): `task.history_source: replay_gt` + `task.replay_npz` -> task flags, with the clock
start and loop span taken from the sender ready file; `task.extra_args` verbatim."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from franka_deploy.launch import launcher as lm
from franka_deploy.launch.launcher import Abort, Launcher, load_config

PKG = Path(__file__).resolve().parents[1]


def _launcher(tmp_path, profile="sim-iphone-fake", **task):
    cfg = load_config(PKG / "configs" / "deploy.yaml", profile)
    cfg.update(work_root=str(tmp_path / "runs"), python={"bp": sys.executable, "realenv": sys.executable})
    cfg["task"] = {**cfg["task"], **task}
    L = Launcher(cfg, interactive=False)
    (L.session / "config").mkdir(parents=True)
    return L


def _npz(p):
    np.savez(p, t_ns=np.arange(10, dtype=np.int64), xyz_wxyz=np.zeros((10, 7)), width=np.zeros(10), origin_ns=np.array(0))
    return str(p)


def test_no_replay_keys_means_no_flags(tmp_path):
    assert _launcher(tmp_path).task_replay_args() == []
    assert _launcher(tmp_path / "b", extra_args=["--no-settle"]).task_replay_args() == ["--no-settle"]


def test_replay_gt_flags_span_and_start_from_the_ready_file(tmp_path):
    L = _launcher(tmp_path, history_source="replay_gt", replay_npz=_npz(tmp_path / "base.npz"))
    (L.session / "sender_ready.json").write_text(json.dumps({"start_now_ns": 111, "first_tick_ns": 222, "loop_span_ns": 5_000_000_000}))
    a = L.task_replay_args()
    i = a.index("--replay-start-now-ns")
    assert a[:2] == ["--history-source", "replay_gt"] and a[i + 1] == "111"          # video_replay: slot 0 = start_now_ns
    z = np.load(a[a.index("--replay-npz") + 1])
    assert int(z["span_ns"]) == 5_000_000_000 and int(z["origin_ns"]) == 0 and len(z["t_ns"]) == 10
    assert L.summary["replay_gt"]["start_now_ns"] == 111 and L.summary["replay_gt"]["source_key"] == "start_now_ns"
    R = _launcher(tmp_path / "r", "sim-rig-fake", history_source="replay_gt", replay_npz=_npz(tmp_path / "base2.npz"))
    (R.session / "sender_ready.json").write_text(json.dumps({"start_now_ns": 111, "first_tick_ns": 222, "loop_span_ns": 7}))
    ra = R.task_replay_args()
    assert ra[ra.index("--replay-start-now-ns") + 1] == "222"                           # rig: slot 0 = first_tick_ns


def test_replay_gt_needs_npz_and_a_sender(tmp_path):
    with pytest.raises(Abort, match="replay_npz"):
        _launcher(tmp_path, history_source="replay_gt").task_replay_args()
    L = _launcher(tmp_path / "m", "mock", history_source="replay_gt", replay_npz="x.npz")
    with pytest.raises(Abort, match="sim sender"):
        L.task_replay_args()
