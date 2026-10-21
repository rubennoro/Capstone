"""Stage 2b pieces that need no launcher (B9 follow-ups to the B8 review): sender_died -> INCOMPLETE, build_replay_npz reports its
time source instead of swallowing errors, the render profile's replay source, the loop-boundary start, the latency note once.
cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_eval_stage2b.py"""

import argparse
import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from franka_sim.eval import runner as rn  # noqa: E402
from franka_sim.eval import stage2b as s2b  # noqa: E402


def _args(tmp_path, profile="sim-iphone-fake", policy="fake", **kw):
    d = dict(deploy_config=None, profile=profile, policy=policy, set=[], out=str(tmp_path / "out"), seed=0, stage="2b", demos="0",
             episodes=None, port_offset=0, sim_stub=None, stage1=None, quick=False, jobs=1)
    d.update(kw)
    return argparse.Namespace(**d)


def _fake_demo_npz(tmp_path):
    n = 30
    np.savez(tmp_path / "d.npz", pos=np.zeros((n, 3)), rotvec=np.zeros((n, 3)), width=np.full(n, 0.05), ends=np.array([n]))
    return str(tmp_path / "d.npz")


def test_sender_died_makes_the_stage_incomplete(tmp_path, monkeypatch):
    R = rn.resolve(_args(tmp_path))
    info = {"name": "ep0", "n": 100, "t_ns": np.arange(100, dtype=np.int64) * 16_666_667, "time_source": "uniform_60hz", "time_source_error": None}
    monkeypatch.setattr(s2b, "build_replay_npz", lambda R_, did, path: info)
    monkeypatch.setattr(s2b, "derive_deploy", lambda *a, **k: (tmp_path / "dep.yaml", {}))

    def fake_launch(R_, dep, script, cwd, timeout_s, driver=None):
        sess = cwd / "work" / "session_x"
        sess.mkdir(parents=True)
        (sess / "launcher_summary.json").write_text(json.dumps({"exit_code": 0, "sender_died": {"returncode": -9, "t": 12.5}}))
        return 0

    monkeypatch.setattr(s2b, "launch", fake_launch)
    results = {}
    rep = rn.run_one_stage(R, "2b", results)
    assert rep["verdict"] == "INCOMPLETE"
    assert any(n.startswith("INCOMPLETE") and "sender_died" in n and "episode aborted" in n for n in rep["notes"]), rep["notes"]
    assert all(g["pass"] is None for g in rep["gates"].values())


def test_build_replay_npz_reports_its_time_source_and_the_zarr_rows(tmp_path):
    R = rn.resolve(_args(tmp_path))
    R.policy["prompt_zarr"] = _fake_demo_npz(tmp_path)
    info = s2b.build_replay_npz(R, 0, tmp_path / "r.npz")
    assert info["time_source"] == "uniform_60hz" and info["time_source_error"]          # no raw episode here: reported, not silent
    z = np.load(tmp_path / "r.npz")
    assert int(z["zarr_start"]) == 0 and list(z["zarr_local"]) == list(range(30)) and len(z["t_ns"]) == 30


def test_render_profile_uses_the_replay_source_of_its_protocol(tmp_path):
    R = rn.resolve(_args(tmp_path, profile="sim-iphone"))
    assert R.image_source == "render" and R.protocol == "v1"
    R2 = s2b.replay_run(R)
    assert R2.image_source == "video_replay" and R2.run_ids["sim"] == "video_replay"
    assert any("replay source video_replay" in n for n in R.notes)
    Rr = rn.resolve(_args(tmp_path / "r", profile="sim-rig"))
    assert Rr.image_source == "render" and Rr.protocol == "v2"
    assert s2b.replay_run(Rr).image_source == "rig_raw_replay"
    # a replay profile is used as is
    R3 = rn.resolve(_args(tmp_path / "x"))
    assert s2b.replay_run(R3) is R3
    # forcing the image source on a replay profile (what the B8 test did) swaps the same way
    R4 = rn.resolve(_args(tmp_path / "y", set=["sim.image_source=render"]))
    assert R4.image_source == "render" and s2b.replay_run(R4).image_source == "video_replay"


def test_plan_start_is_the_next_loop_boundary():
    s, span, now = 1_000_000_000, 16_000_000_000, 5_000_000_000
    assert s2b.plan_start(s, span, now, 3.0) == s + span                      # first boundary >= now + 3 s
    assert s2b.plan_start(s, span, now, 0.0) == s + span
    assert s2b.plan_start(s, span, s - 5, 0.0) == s                           # before the first lap: the first boundary
    t = s2b.plan_start(s, span, 40_000_000_000, 12.0)
    assert (t - s) % span == 0 and t >= 52_000_000_000 and t - 52_000_000_000 < span


def test_latency_note_is_reported_once_not_per_demo(tmp_path, monkeypatch):
    R = rn.resolve(_args(tmp_path, demos="0,1"))
    calls = []

    def fake_demo(R_, d, did, po, results):
        calls.append(did)
        return {"demo": did, "session": f"s{did}", "time_source": "iphone_slot_times", "time_source_error": None, "first_call_frame": 3.0}

    monkeypatch.setattr(s2b, "run_demo", fake_demo)
    monkeypatch.setattr(s2b, "aggregate", lambda parts, R_: ({"n_demos": len(parts)}, []))
    R.notes.clear()
    s2b.run(R, tmp_path, {})
    assert calls == [0, 1]
    assert sum("capture latency" in n for n in R.notes) == 1, R.notes
    assert sum("replay time source" in n for n in R.notes) == 2
