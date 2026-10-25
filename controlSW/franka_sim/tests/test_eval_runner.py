"""Eval runner end to end (B8): process tests (launcher + sender + receiver + policy child), run under the suite lock:
cd controlSW/franka_sim && flock $LOCK env -u PYTHONPATH FD_TEST_PORT_OFFSET=2000 MUJOCO_GL=egl $RE -m pytest -q -s tests/test_eval_runner.py
Needs data/raw/2026-10-01, the firstreal zarr and the BP interpreter (skipped otherwise)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from franka_sim.eval import report as rpt  # noqa: E402

SIM = Path(__file__).resolve().parents[1]
DEPLOY = SIM.parent / "franka_deploy"
S = Path(os.environ.get("S", "/tmp/claude-1000/-home-ruben-Capstone/299e5dab-9afb-41ab-aa6c-ac52f8fb5552/scratchpad"))
ZARR = Path(os.environ.get("ZARR") or S / "deploy_work" / "prompt_zarr" / "firstreal-single.zarr.zip")
RAW = SIM.parents[1] / "data" / "raw" / "2026-10-01"
pytestmark = pytest.mark.skipif(not (ZARR.is_file() and RAW.is_dir() and (S / "env_bp" / "bin" / "python").is_file()),
                                reason="needs the firstreal zarr, data/raw/2026-10-01 and the BP interpreter")


def _eval(out, *args, timeout=1500):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=f"{SIM}:{DEPLOY}", MUJOCO_GL="egl")
    r = subprocess.run([sys.executable, "-m", "franka_sim", "eval", "--out", str(out), *args], capture_output=True, text=True,
                       env=env, cwd=str(SIM), timeout=timeout)
    print(r.stdout[-2500:])
    return r


def _only(out, prefix):
    ds = sorted(p for p in Path(out).iterdir() if p.is_dir() and p.name.startswith(prefix))
    assert len(ds) == 1, ds
    return ds[0]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    r1 = _eval(out, "--profile", "sim-iphone-fake", "--stage", "1", "--policy", "fake", "--demos", "0,1", "--quick", "--jobs", "4")
    assert r1.returncode in (0, 3), r1.stderr[-2000:]          # a verdict was produced: PASS 0, FAIL 3 (B9: FAIL exits non-zero)
    # the live-vs-offline evidence gate (n_pairs_at_least, a placeholder) needs far more pairs than one fake demo gives: lower it here
    r2 = _eval(out, "--profile", "sim-iphone-fake", "--stage", "2b", "--policy", "fake", "--demos", "0",
               "--set", "criteria.stage2.n_pairs_at_least=5")
    assert r2.returncode == 0, r2.stderr[-2000:] + r2.stdout[-2000:]
    return out


def test_help_lists_eval():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = f"{SIM}:{DEPLOY}"
    r = subprocess.run([sys.executable, "-m", "franka_sim", "eval", "--help"], capture_output=True, text=True, env=env, cwd=str(SIM))
    assert r.returncode == 0 and "--profile" in r.stdout and "--stage" in r.stdout and "--demos" in r.stdout


def test_stage2b_fake_report_and_stamp_metrics(runs):
    d = _only(runs, "2b_fake_")
    rep = json.loads((d / "report.json").read_text())
    assert rpt.validate(rep) == [], rpt.validate(rep)
    assert rep["stage"] == "2b" and rep["policy_id"] == "fake" and rep["policy_kind"] == "fake"
    assert rep["profile"] == "sim-iphone-fake" and rep["sensor"] == "iphone16pro_wifi" and rep["image_source"] == "video_replay"
    m = rep["metrics"]
    assert m["stamp_err_ms_p50_main"] is not None and m["stamp_err_ms_p95_main"] is not None
    assert m["stamp_err_ms_p95_main"] < 30                                   # the v1 gate
    assert m["n_calls"] > 10 and m["n_pairs"] > 5 and m["contract_ok"] is True
    assert m["live_vs_offline_pos_err_mm_p95"] == pytest.approx(0.0, abs=1e-6)   # fake: identity chunks on both sides
    assert m["time_source"] and m["first_call_demo_frame_max"] < 120, m["first_call_demo_frame_max"]   # c typed at the demo's start
    names = set(rep["gates"])
    assert names == {"stage2.live_vs_offline_pos_err_mm_p50_k15_max", "stage2.live_vs_offline_pos_err_mm_p95_k15_max", "stage2.n_pairs_at_least",
                     "stage2.frame_selection_errors_max", "stage2.cam_stamp_err_ms_p95_max"}
    assert sum("capture latency" in n for n in rep["notes"]) == 1
    assert "image_mae_224_main_p50" in m and "uw_frame_offset_hist" in m
    assert all(g["pass"] is True for g in rep["gates"].values()), rep["gates"]
    assert rep["verdict"] == "PASS"
    assert (d / "demo_00" / "work").is_dir() and (d / "demo_00" / "launcher.log").is_file()
    html = (d / "report.html").read_text()
    for key in ("PASS", "stage2.cam_stamp_err_ms_p95_max", "iphone16pro_wifi", "fake"):
        assert key in html
    assert "not a real-world estimate" not in html                           # a banner for render runs only
    assert (d / "station_resolved.yaml").is_file() and (d / "criteria.yaml").is_file() and (d / "metrics.json").is_file()


def test_stage1_quick_grid_and_summary(runs):
    d = _only(runs, "1_fake_")
    rep = json.loads((d / "report.json").read_text())
    assert rpt.validate(rep) == [] and rep["stage"] == "1"
    assert rep["metrics"]["n_demos"] == 2 and rep["metrics"]["n_feasible"] in (0, 1, 2)
    assert set(rep["gates"]) == {"stage1.reachable_fraction_min", "stage1.joint_margin_deg_min", "stage1.collisions_max",
                                 "stage1.width_p99_mapped_le_stroke"}
    assert rep["verdict"] in ("PASS", "FAIL")
    assert (d / "stage1" / "panda_ext0_py" / "stage1.json").is_file() or list((d / "stage1").glob("*/stage1.json"))
    # summary across the two runs
    s = json.loads((runs / "summary.json").read_text())
    assert s["n_runs"] == 2 and {r["stage"] for r in s["rows"]} == {"1", "2b"}
    assert all(r["verdict"] in ("PASS", "FAIL", "INCOMPLETE") for r in s["rows"])
    sh = (runs / "summary.html").read_text()
    assert "report.html" in sh and "sim-iphone-fake" in sh


def test_render_profile_stage2b_uses_the_replay_source_and_keeps_the_banner(tmp_path):
    """image_source render (the closed-loop source) has no replay of its own: Stage 2b swaps in the protocol's replay source
    (video_replay for v1) with a note instead of going INCOMPLETE (B8 raised StageIncomplete here); the HTML keeps the render banner."""
    r = _eval(tmp_path, "--profile", "sim-iphone-fake", "--stage", "2b", "--policy", "fake", "--demos", "0",
              "--set", "sim.image_source=render", "--set", "criteria.stage2.n_pairs_at_least=5")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    d = _only(tmp_path, "2b_fake_")
    rep = json.loads((d / "report.json").read_text())
    assert rpt.validate(rep) == [] and rep["verdict"] == "PASS" and rep["image_source"] == "render"
    assert any("replay source video_replay" in n for n in rep["notes"]), rep["notes"]
    assert "not a real-world estimate" in (d / "report.html").read_text()
