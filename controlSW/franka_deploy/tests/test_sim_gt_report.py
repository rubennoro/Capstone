"""A6: franka_deploy eval report's additive ``sim_gt`` (franka_sim.ground_truth on the episode's bridge.jsonl sim lines).
A human label ``n`` stays the outcome; the sim truth is reported separately (``sim_success_rate``, HTML column).

    cd controlSW/franka_deploy && env -u PYTHONPATH PYTHONPATH=.:../franka_sim $BP -m pytest -q tests/test_sim_gt_report.py
"""
import json
import math

import yaml

from franka_deploy.eval import report as R

SUCCESS = {"lift_min_m": 0.05, "hold_s": 2.0, "timeout_s": 60, "slip_tol_m": 0.002}


def make_episode(root, name, label, sim=True):
    d = root / "task" / "episodes" / "p0" / "s0" / "c0" / name
    (d / "config").mkdir(parents=True)
    (d / "config" / "station_resolved.yaml").write_text(yaml.safe_dump({"ids": {}, "groups": {"criteria": {"success": SUCCESS}}}))
    (d / "summary.json").write_text(json.dumps({"duration_s": 8.0}))
    (d / "label.json").write_text(json.dumps({"label": label, "is_successful": label == "y", "notes": []}))
    (d / "events.jsonl").write_text(json.dumps({"t": 50.0, "kind": "episode_start"}) + "\n")
    with open(d / "bridge.jsonl", "w") as f:
        f.write(json.dumps({"t": 50.0, "kind": "stats", "state": "running"}) + "\n")
        for k in range(8 * 25):
            t = k / 25
            z = 0.006 + (0.06 * min(1.0, max(0.0, t - 2.0)) if sim else 0.0)
            f.write(json.dumps({"t": 50.0 + t, "kind": "sim", "obj": [0.45, 0.0, z, 1.0, 0.0, 0.0, 0.0], "gap": 0.012,
                                "grasped": t >= 1.0, "hand_mode": "holding", "contacts": [], "F_ext_max": 0.0,
                                "reflex": None}) + "\n")
    return d


def test_sim_gt_added_label_wins_rate_separate(tmp_path):
    d1 = make_episode(tmp_path, "episode_0001", "n")                 # physics says lifted; the human says n
    d2 = make_episode(tmp_path, "episode_0002", "y", sim=False)      # the human says y; physics never lifted
    eps = [R.episode_metrics(str(d), None, None, float("nan"), None, None, None) for d in (d1, d2)]
    assert eps[0]["sim_gt"]["success"] is True and eps[0]["outcome"] == "failure"
    assert abs(eps[0]["sim_gt"]["t_success_s"] - (2.0 + 0.05 / 0.06 + 2.0)) <= 0.05      # reach + hold_s
    assert eps[1]["sim_gt"]["success"] is False and eps[1]["outcome"] == "success"
    rep = R.summarize(eps)
    assert rep["success_rate"]["k"] == 1 and rep["success_rate"]["n"] == 2          # label-based, unchanged
    s = rep["sim_success_rate"]
    assert (s["k"], s["n"]) == (1, 2) and math.isclose(s["p"], 0.5) and len(s["ci95"]) == 2
    out = tmp_path / "report.html"
    R.write_html(str(out), rep, eps, 0.01)
    html = out.read_text()
    assert "<th>sim_gt</th>" in html and "Sim ground truth" in html


def test_real_episode_gets_no_sim_gt(tmp_path):
    d = make_episode(tmp_path, "episode_0003", "y")
    (d / "bridge.jsonl").write_text(json.dumps({"t": 50.0, "kind": "stats"}) + "\n")
    ep = R.episode_metrics(str(d), None, None, float("nan"), None, None, None)
    assert "sim_gt" not in ep
    rep = R.summarize([ep])
    assert "sim_success_rate" not in rep
    R.write_html(str(tmp_path / "r.html"), rep, [ep], 0.01)
    assert "<td>-</td>" in (tmp_path / "r.html").read_text()
