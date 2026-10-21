"""eval.criteria (B8): pure gate evaluation. Run with the real-env interpreter:
cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_eval_criteria.py tests/test_eval_stage2a.py"""

import json
from pathlib import Path

import pytest
import yaml

from franka_sim.eval import criteria as C
from franka_sim.eval import report as rpt

CRIT = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "run" / "criteria" / "block_raise.yaml").read_text())


def test_min_max_bool_and_names():
    ev = C.evaluate({"stage1": {"reachable_fraction_min": 0.95, "collisions_max": 0, "width_ok": True}},
                    {"reachable_fraction": 0.95, "collisions": 0, "width_ok": True}, "1")
    assert ev["verdict"] == "PASS"
    assert set(ev["gates"]) == {"stage1.reachable_fraction_min", "stage1.collisions_max", "stage1.width_ok"}
    g = ev["gates"]["stage1.reachable_fraction_min"]
    assert g == {"value": 0.95, "threshold": 0.95, "op": ">=", "pass": True, "metric": "reachable_fraction"}
    assert ev["gates"]["stage1.collisions_max"]["op"] == "<="
    bad = C.evaluate({"stage1": {"reachable_fraction_min": 0.95}}, {"reachable_fraction": 0.9499}, "1")
    assert bad["verdict"] == "FAIL" and bad["gates"]["stage1.reachable_fraction_min"]["pass"] is False
    over = C.evaluate({"stage1": {"collisions_max": 0}}, {"collisions": 1}, "1")
    assert over["verdict"] == "FAIL"
    b = C.evaluate({"stage1": {"width_ok": True}}, {"width_ok": False}, "1")
    assert b["verdict"] == "FAIL" and b["gates"]["stage1.width_ok"]["op"] == "is"


def test_missing_metric_is_incomplete_never_pass():
    ev = C.evaluate({"stage1": {"reachable_fraction_min": 0.95, "collisions_max": 0}}, {"collisions": 0}, "1")
    assert ev["gates"]["stage1.reachable_fraction_min"]["pass"] is None and ev["verdict"] == "INCOMPLETE"
    nan = C.evaluate({"stage1": {"collisions_max": 0}}, {"collisions": float("nan")}, "1")
    assert nan["verdict"] == "INCOMPLETE"
    # FAIL wins over INCOMPLETE
    mix = C.evaluate({"stage1": {"reachable_fraction_min": 0.95, "collisions_max": 0}}, {"collisions": 3}, "1")
    assert mix["verdict"] == "FAIL"
    assert C.evaluate({"stage1": {}}, {}, "1")["verdict"] == "INCOMPLETE"      # no gates at all is not a pass
    assert C.evaluate({}, {}, "3")["verdict"] == "INCOMPLETE"


def test_per_protocol_threshold_dicts():
    crit = {"stage2": {"cam_stamp_err_ms_p95_max": {"v1": 30, "v2": 5}}}
    m = {"cam_stamp_err_ms_p95": 12.0}
    assert C.evaluate(crit, {**m, "protocol": "v1"}, "2b")["verdict"] == "PASS"
    e2 = C.evaluate(crit, {**m, "protocol": "v2"}, "2b")
    assert e2["verdict"] == "FAIL" and e2["gates"]["stage2.cam_stamp_err_ms_p95_max"]["threshold"] == 5
    assert C.evaluate(crit, m, "2b")["verdict"] == "INCOMPLETE"                      # no protocol
    assert C.evaluate(crit, {**m, "protocol": "v3"}, "2b")["verdict"] == "INCOMPLETE"


def test_k_threshold_dicts_fan_out_and_substage_keys():
    ev = C.evaluate(CRIT, {"chunk_reachable_fraction": 1.0, "chunk_joint_margin_deg": 20,
                           "pred_vs_demo_pos_err_mm_p95_k4": 5, "pred_vs_demo_pos_err_mm_p95_k8": 30,
                           "pred_vs_demo_pos_err_mm_p95_k15": 10}, "2a")
    assert ev["verdict"] == "FAIL"
    assert [k for k, g in ev["gates"].items() if g["pass"] is False] == ["stage2.pred_vs_demo_pos_err_mm_p95_k8_max"]
    assert set(ev["gates"]) == {"stage2.chunk_reachable_fraction_min", "stage2.chunk_joint_margin_deg_min",
                                "stage2.pred_vs_demo_pos_err_mm_p95_k4_max", "stage2.pred_vs_demo_pos_err_mm_p95_k8_max",
                                "stage2.pred_vs_demo_pos_err_mm_p95_k15_max"}
    m2b = {"live_vs_offline_pos_err_mm_p50_k15": 4.0, "live_vs_offline_pos_err_mm_p95_k15": 20.0, "n_pairs": 60,
           "frame_selection_errors": 0, "cam_stamp_err_ms_p95": 10, "protocol": "v1"}
    b = C.evaluate(CRIT, m2b, "2b")
    assert b["verdict"] == "PASS" and len(b["gates"]) == 5, b["gates"].keys()
    assert "stage2.live_vs_offline_pos_err_mm_p50_k15_max" in b["gates"] and "stage2.n_pairs_at_least" in b["gates"]
    # p95 beyond the bound fails; p50 beyond fails; too few pairs is INCOMPLETE, never FAIL and never PASS
    assert C.evaluate(CRIT, {**m2b, "live_vs_offline_pos_err_mm_p95_k15": 26.0}, "2b")["verdict"] == "FAIL"
    assert C.evaluate(CRIT, {**m2b, "live_vs_offline_pos_err_mm_p50_k15": 5.5}, "2b")["verdict"] == "FAIL"
    few = C.evaluate(CRIT, {**m2b, "n_pairs": 11}, "2b")
    assert few["verdict"] == "INCOMPLETE" and few["gates"]["stage2.n_pairs_at_least"]["pass"] is None
    assert C.evaluate(CRIT, {**m2b, "n_pairs": 11, "live_vs_offline_pos_err_mm_p95_k15": 99}, "2b")["verdict"] == "FAIL"


def test_stage3_gate_names_and_lag_gate_semantics():
    m3 = {"collisions": 0, "reflex": 0, "tracking_err_mm_p95": 1.0, "gripper_width_lag_steps_p95": 0,
          "inference_latency_s_p95": 0.1, "receiver_age_s_p95": 0.1, "stale_image_errors": 0, "obs_motion_mm_p95": 1.0}
    ev = C.evaluate(CRIT, m3, "3")
    assert ev["verdict"] == "PASS", ev["gates"]          # a lag of 0 used to FAIL (key without a suffix meant ==)
    assert set(ev["gates"]) == {f"stage3.{k}_max" for k in ("collisions", "reflex", "tracking_err_mm_p95", "inference_latency_s_p95",
                                "receiver_age_s_p95", "stale_image_errors", "obs_motion_mm_p95", "gripper_width_lag_steps_p95")}
    assert C.evaluate(CRIT, {**m3, "reflex": 1}, "3")["verdict"] == "FAIL"


def test_criteria_placeholders_are_marked():
    from franka_deploy.station import load_station
    st = load_station({"station": {"robot": "panda", "tooling": "franka_hand_iphumi_iphone16pro", "sensor": "iphone16pro_wifi",
                                   "scene": "block_raise_table", "host": "pc_laptop"},
                       "run": {"policy": "fake", "link": "ideal", "criteria": "block_raise", "sim": "video_replay"}})
    ph = {p for p, _ in st.placeholders() if p.startswith("criteria.")}
    assert {"criteria.limits.v_max_m_s", "criteria.success.lift_min_m", "criteria.success.timeout_s",
            "criteria.stage2.n_pairs_at_least", "criteria.stage2.live_vs_offline_pos_err_mm_p95_max.k15"} <= ph


def test_real_criteria_stage1_and_unknown_stage():
    ev = C.evaluate(CRIT, {"reachable_fraction": 1.0, "joint_margin_deg": 12, "collisions": 0, "width_p99_mapped_le_stroke": True}, "1")
    assert ev["verdict"] == "PASS" and len(ev["gates"]) == 4
    assert ev["criteria_version"].startswith("run.criteria/v1")
    with pytest.raises(ValueError):
        C.evaluate(CRIT, {}, "9")


def test_report_schema_and_html(tmp_path):
    ev = C.evaluate(CRIT, {"reachable_fraction": 1.0, "joint_margin_deg": 12, "collisions": 0, "width_p99_mapped_le_stroke": True}, "1")
    rep = rpt.make_report(stage="1", policy_id="p", policy_kind="fake", profile="x", sensor="s", robot="panda",
                          station_fingerprint="abc", image_source="render", seed=0, started="t", duration_s=1.0,
                          metrics={"a": 1.5, "b": float("nan")}, evaluation=ev, notes=["n1"], rows=[{"demo": "d", "x": 1}])
    assert rpt.validate(rep) == []
    rpt.write_report(tmp_path, rep)
    back = json.loads((tmp_path / "report.json").read_text())
    assert rpt.validate(back) == [] and back["metrics"]["b"] is None
    h = (tmp_path / "report.html").read_text()
    for key in ("PASS", "stage1.collisions_max", "not a real-world estimate", "panda", "abc", "<style>"):
        assert key in h
    assert "http" not in h.replace("http://www.w3.org", "") and "<script" not in h
    rep2 = dict(rep, image_source="video_replay")
    assert "not a real-world estimate" not in rpt.render_html(rep2)
    assert any("verdict" in e for e in rpt.validate({**rep, "verdict": "MAYBE"}))
