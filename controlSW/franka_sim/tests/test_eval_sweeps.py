"""Sweeps (B9): expansion, max_runs, resume after a kill, run-dir identity, the comparison summary. Pure: the stage function is
a fake that writes a report.json, so no launcher or simulator runs.
cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_eval_sweeps.py"""

import argparse
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mujoco")

from franka_sim.eval import criteria as crit  # noqa: E402
from franka_sim.eval import report as rpt  # noqa: E402
from franka_sim.eval import runner as rn  # noqa: E402
from franka_sim.eval import sweeps as sw  # noqa: E402

SPEC = {"id": "t", "stage": "3", "seed": 10, "max_runs": None, "resume": True, "n_episodes": 1, "n_poses": 1,
        "axes": [{"group": "link", "ids": ["ideal", "wifi_good"]}, {"group": "policy", "ids": ["fake", "demo_replay"]},
                 {"set": "criteria.limits.v_max_m_s", "values": [0.25, 0.5]}]}


def _args(tmp_path, spec_path, **kw):
    d = dict(deploy_config=None, profile="sim-iphone-fake", policy=None, set=[], out=str(tmp_path / "out"), seed=0, stage="all",
             sweep=str(spec_path), max_runs=None, episodes_per_pose=None, poses=None, episodes=None, demos=None, sim_stub=None)
    d.update(kw)
    return argparse.Namespace(**d)


def _spec(tmp_path, **kw):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump({**SPEC, **kw}))
    return p


class Fake:
    """Stage function: writes report.json (verdict from `verdicts` by call count) and records the calls."""

    def __init__(self, verdicts=None, die_after=None):
        self.calls, self.verdicts, self.die_after = [], verdicts or {}, die_after

    def __call__(self, R, stage, results):
        if self.die_after is not None and len(self.calls) >= self.die_after:
            raise SystemExit("killed")                       # the process dies between points
        d = R.run_dir(stage)
        d.mkdir(parents=True, exist_ok=True)
        v = self.verdicts.get(len(self.calls), "PASS")
        m = {"success_rate": 1.0 if v == "PASS" else 0.0, "success_k": 1 if v == "PASS" else 0, "success_n": 1, "success_ci95_lo": 0.2,
             "success_ci95_hi": 1.0, "collisions": 0 if v == "PASS" else 2}
        ev = {"criteria_version": "x", "verdict": v, "gates": {"stage3.collisions_max": {"value": m["collisions"], "threshold": 0, "op": "<=",
                                                                                         "pass": v == "PASS", "metric": "collisions"}}}
        rep = rpt.make_report(stage=stage, policy_id=R.policy.get("id"), policy_kind="fake", profile=R.profile, sensor=R.sensor, robot="panda",
                              station_fingerprint=R.station.fingerprint(), image_source=R.image_source, seed=R.seed, started="t",
                              duration_s=0.1, metrics=m, evaluation=ev, notes=[])
        rpt.write_report(d, rep)
        self.calls.append((R.run_dir(stage).name, R.seed, stage))
        return rep


def test_expansion_is_row_major_with_seed_plus_index():
    pts = sw.expand(SPEC)
    assert len(pts) == 8
    assert [p["i"] for p in pts] == list(range(8)) and [p["seed"] for p in pts] == list(range(10, 18))
    assert pts[0]["axes"] == {"link": "ideal", "policy": "fake", "criteria.limits.v_max_m_s": 0.25}
    assert pts[1]["axes"]["criteria.limits.v_max_m_s"] == 0.5 and pts[1]["axes"]["policy"] == "fake"      # last axis fastest
    assert pts[2]["axes"]["policy"] == "demo_replay" and pts[4]["axes"]["link"] == "wifi_good"
    assert pts[1]["run_overrides"] == {"link": "ideal", "policy": "fake"} and pts[1]["extra_sets"] == ["criteria.limits.v_max_m_s=0.5"]
    assert sw.expand({"axes": []}) == []


def test_real_sweep_files_expand():
    d = sw.load_spec("block_raise_default")
    assert d["stage"] == 3 or str(d["stage"]) == "3"
    labels = [sw.axis_label(a) for a in d["axes"]]
    assert labels == ["link", "policy", "criteria.limits.v_max_m_s", "sensor.task.camera_latency_s"]
    assert len(sw.expand(d)) == 4 * 2 * 2 * 3
    s = sw.load_spec("block_raise_smoke")
    assert len(sw.expand(s)) == 4 and s["n_episodes"] == 1 and s["n_poses"] == 1


def test_all_points_get_distinct_run_dirs_and_set_axes_change_the_hash(tmp_path):
    f = Fake()
    rc = sw.run_sweep(_args(tmp_path, _spec(tmp_path)), stage_fn=f)
    assert rc == 0 and len(f.calls) == 8 and len({c[0] for c in f.calls}) == 8
    names = [c[0] for c in f.calls]
    a, b = names[0], names[1]                              # same link and policy, only the v_max set differs
    assert a.rsplit("_", 1)[1] != b.rsplit("_", 1)[1]      # the user-set hash segment moves with the set axis
    assert [c[1] for c in f.calls] == list(range(10, 18))     # each point ran with its own seed


def test_max_runs_resume_and_kill(tmp_path):
    spec = _spec(tmp_path)
    out = tmp_path / "out"
    k = Fake(die_after=2)
    with pytest.raises(SystemExit):
        sw.run_sweep(_args(tmp_path, spec, max_runs=4), stage_fn=k)       # killed between points 2 and 3
    assert len(k.calls) == 2
    idx = json.loads((out / "sweep_index.json").read_text())
    assert [p["stages"]["3"]["verdict"] for p in idx["points"]] == ["PASS", "PASS"]           # the index survived the kill
    f = Fake()
    rc_pending = sw.run_sweep(_args(tmp_path, spec, max_runs=2), stage_fn=f)                 # max_runs caps NEW runs
    assert rc_pending == rn.EXIT_INCOMPLETE                                                  # 4 points still unrun: not finished, not PASS
    assert len(f.calls) == 2 and {c[0] for c in f.calls}.isdisjoint({c[0] for c in k.calls})   # only the unfinished points ran
    f2 = Fake()
    assert sw.run_sweep(_args(tmp_path, spec), stage_fn=f2) == 0
    assert len(f2.calls) == 4                                                                # 8 - 2 - 2
    f3 = Fake()
    assert sw.run_sweep(_args(tmp_path, spec), stage_fn=f3) == 0 and f3.calls == []          # everything is done: nothing runs
    idx = json.loads((out / "sweep_index.json").read_text())
    assert len(idx["points"]) == 8 and all(p["stages"]["3"]["skipped"] for p in idx["points"])


def test_incomplete_points_rerun_and_exit_codes(tmp_path):
    spec = _spec(tmp_path, axes=[{"group": "link", "ids": ["ideal", "wifi_good"]}])
    f = Fake({0: "INCOMPLETE", 1: "FAIL"})
    rc = sw.run_sweep(_args(tmp_path, spec), stage_fn=f)
    assert rc == rn.EXIT_FAIL == 3 and rn.EXIT_INCOMPLETE == 1 != rn.EXIT_FAIL
    f2 = Fake()
    rc2 = sw.run_sweep(_args(tmp_path, spec), stage_fn=f2)            # INCOMPLETE reruns, FAIL is a finished verdict
    assert len(f2.calls) == 1 and rc2 == rn.EXIT_FAIL
    assert rn.exit_code(["PASS", "INCOMPLETE"]) == 1 and rn.exit_code(["PASS"]) == 0 and rn.exit_code(["INCOMPLETE", "FAIL"]) == 3


def test_comparison_summary_one_column_per_point(tmp_path):
    spec = _spec(tmp_path, axes=[{"group": "link", "ids": ["ideal", "wifi_good"]}, {"group": "policy", "ids": ["fake", "demo_replay"]}])
    f = Fake({3: "FAIL"})
    sw.run_sweep(_args(tmp_path, spec), stage_fn=f)
    out = tmp_path / "out"
    s = json.loads((out / "summary.json").read_text())
    assert s["n_runs"] == 4 and len(s["comparison"]["columns"]) == 4
    assert [c["axes"] for c in s["comparison"]["columns"]][3] == {"link": "wifi_good", "policy": "demo_replay"}
    rows = {r["name"]: r for r in s["comparison"]["rows"]}
    assert [c["text"] for c in rows["verdict"]["cells"]] == ["PASS", "PASS", "PASS", "FAIL"]
    assert rows["stage3.collisions_max"]["cells"][3]["pass"] is False and rows["stage3.collisions_max"]["cells"][0]["pass"] is True
    assert "(1/1)" in rows["success_rate (95% CI)"]["cells"][0]["text"]
    h = (out / "summary.html").read_text()
    assert "Sweep comparison" in h and "wifi_good" in h and "demo_replay" in h and "class='bad'" in h and "class='ok'" in h
    assert "<script" not in h and "http" not in h.replace("http://www.w3.org", "")
    assert h.count("report.html") >= 8                                  # the run table and the comparison header both link the reports


def test_usage_errors_exit_2_not_1(tmp_path):
    with pytest.raises(SystemExit) as e:
        sw.run_sweep(_args(tmp_path, "no_such_sweep_id"), stage_fn=Fake())
    assert e.value.code == rn.EXIT_USAGE == 2 != rn.EXIT_INCOMPLETE
    with pytest.raises(SystemExit) as e2:
        rn.resolve(_args(tmp_path, tmp_path / "x", profile="mock"))        # a profile without a station
    assert e2.value.code == 2
    with pytest.raises(SystemExit) as e3:
        rn.resolve(_args(tmp_path, tmp_path / "x", set=["criteria.nope.x=1"]))   # a bad --set
    assert e3.value.code == 2
