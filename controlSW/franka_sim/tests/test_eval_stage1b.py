"""Stage 1b wrapper (B9 Part 2 addendum 3): ``eval/stage1b.py`` maps A8's ``run_campaign`` result into criteria.stage1b gates.
``run_campaign`` is monkeypatched (A8 builds it in parallel); pure, no MuJoCo run.
cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_eval_stage1b.py"""

import argparse
import json
import sys
import types

import pytest

from franka_sim.eval import criteria as crit
from franka_sim.eval import report as rpt
from franka_sim.eval import runner as rn
from franka_sim.eval import stage1b as s1b


def _rows():
    rows = []
    for d in range(10):
        rows.append({"demo": d, "speed": 0.5, "track_ok": d != 3, "track_fraction": 0.95, "stage1_infeasible": d == 9,
                     "system_fault": None, "reflexes": 0, "pad_table": 1 if d == 2 else 0, "session": f"/s/{d}"})
    rows.append({"demo": 0, "speed": 1.0, "track_ok": False, "stage1_infeasible": False, "system_fault": None, "reflexes": 2,
                 "pad_table": 0})
    return rows


def test_metrics_per_speed_wilson_and_infeasible_excluded():
    m, table, notes = s1b.metrics_from({"schema": "sim_campaign/v1", "rows": _rows()}, [0.5, 1.0])
    assert m["track_ok_n_s0.5"] == 9 and m["track_ok_k_s0.5"] == 8          # demo 9 infeasible: out of the denominator
    assert m["track_ok_fraction"] == pytest.approx(8 / 9) and m["track_ok_ci95_lo_s0.5"] < 8 / 9 < m["track_ok_ci95_hi_s0.5"]
    assert m["reflex"] == 0 and m["reflex_s1"] == 2 and m["pad_table_contacts"] == 1 and m["system_faults"] == 0
    assert "incomplete_reason" not in m and len(table) == 11
    assert any("demos 3" in n for n in notes) and any("stage1_infeasible" in n and "9" in n for n in notes)


def test_gates_come_from_criteria_stage1b_only_the_min_max_keys():
    crits = {"stage1b": {"track_pos_m": 0.01, "track_rot_deg": 3, "track_ok_fraction_min": 0.9, "reflex_max": 0}}
    m, _, _ = s1b.metrics_from({"rows": _rows()}, [0.5, 1.0])
    ev = crit.evaluate(crits, m, "1b")
    assert set(ev["gates"]) == {"stage1b.track_ok_fraction_min", "stage1b.reflex_max"}
    assert ev["verdict"] == "FAIL" and ev["gates"]["stage1b.track_ok_fraction_min"]["pass"] is False      # 8/9 < 0.9
    crits["stage1b"]["track_ok_fraction_min"] = 0.85                         # a config value, not a constant
    assert crit.evaluate(crits, m, "1b")["verdict"] == "PASS"


def test_system_fault_is_incomplete():
    rows = _rows()
    rows[0]["system_fault"] = "sender_died"
    m, _, notes = s1b.metrics_from({"rows": rows}, [0.5])
    assert m["system_faults"] == 1 and "system fault" in m["incomplete_reason"] and m["track_ok_n_s0.5"] == 8
    assert s1b.metrics_from({"rows": []}, [0.5])[0]["incomplete_reason"] == "campaign.json has no runs"


def test_stage_1b_through_the_runner_with_a_monkeypatched_campaign(tmp_path, monkeypatch):
    calls = []

    def fake_run_campaign(config, profile, stage1_json, demos, speeds, out_dir, sets=(), jobs=1):
        calls.append(dict(config=config, profile=profile, stage1_json=stage1_json, demos=demos, speeds=speeds, out_dir=out_dir,
                          sets=sets, jobs=jobs))
        return {"schema": "sim_campaign/v1", "fingerprint": "abc", "rows": [r for r in _rows() if r["speed"] in speeds]}

    monkeypatch.setitem(sys.modules, "franka_sim.campaign", types.SimpleNamespace(run_campaign=fake_run_campaign))
    s1 = tmp_path / "s1"
    s1.mkdir()
    (s1 / "stage1.json").write_text(json.dumps({"recommendation": {}}))
    a = argparse.Namespace(deploy_config=None, profile="sim-iphone", policy="demo_replay", out=str(tmp_path / "out"), seed=0,
                           stage="1b", set=["tooling.finger_adapter.extension_m=0.10", "criteria.stage1b.track_ok_fraction_min=0.85"],
                           stage1=str(s1), demos="0,1,2", episodes=None, speeds="0.5", jobs=2, sweep=None)
    assert rn.run(a) == rn.EXIT_PASS
    c = calls[0]
    assert c["profile"] == "sim-iphone" and c["demos"] == [0, 1, 2] and c["speeds"] == [0.5] and c["jobs"] == 2
    assert c["stage1_json"].endswith("stage1.json") and "tooling.finger_adapter.extension_m=0.10" in c["sets"]
    rep = json.loads(next((tmp_path / "out").glob("1b_*/report.json")).read_text())
    assert rpt.validate(rep) == [] and rep["stage"] == "1b" and rep["verdict"] == "PASS"
    assert rep["metrics"]["campaign_fingerprint"] == "abc"
    assert any("extension_m=0.10" in n for n in rep["notes"])


def test_stage1b_is_in_stage_all_after_1():
    assert rn.STAGES[:2] == ("1", "1b") and rn.STAGE_FN["1b"] == "stage1b"


def test_missing_stage1_is_incomplete(tmp_path):
    R = types.SimpleNamespace(args=argparse.Namespace(stage1=None))
    with pytest.raises(rn.StageIncomplete):
        s1b._stage1_json(R, {})
