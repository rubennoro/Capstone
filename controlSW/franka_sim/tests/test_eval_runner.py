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
S = Path(os.environ.get("S", "/tmp/claude-1000/-home-ruben-Capstone/b5f58f43-1b87-4846-beab-dc85de4129a4/scratchpad"))
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
    assert r1.returncode == 0, r1.stderr[-2000:]
    r2 = _eval(out, "--profile", "sim-iphone-fake", "--stage", "2b", "--policy", "fake", "--demos", "0")
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
    names = set(rep["gates"])
    assert names == {"stage2.live_vs_offline_pos_err_mm_p95_max", "stage2.frame_selection_errors_max", "stage2.cam_stamp_err_ms_p95_max"}
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


def test_incomplete_with_reason_and_render_banner(tmp_path):
    """No replay sender for image_source render (A7/B9): Stage 2b is INCOMPLETE with the reason, never a PASS, and the HTML
    carries the render banner."""
    r = _eval(tmp_path, "--profile", "sim-iphone-fake", "--stage", "2b", "--policy", "fake", "--demos", "0",
              "--set", "sim.image_source=render")
    assert r.returncode == 1, r.stdout[-1500:] + r.stderr[-1500:]
    d = _only(tmp_path, "2b_fake_")
    rep = json.loads((d / "report.json").read_text())
    assert rpt.validate(rep) == [] and rep["verdict"] == "INCOMPLETE" and rep["image_source"] == "render"
    assert any(n.startswith("INCOMPLETE") and "replay image source" in n for n in rep["notes"])
    assert all(g["pass"] is None for g in rep["gates"].values())
    assert "not a real-world estimate" in (d / "report.html").read_text()
