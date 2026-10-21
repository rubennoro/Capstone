"""Stage 2a (B8): demo_replay predictions through the real predict_demos (BP subprocess) + Stage 1 placement + IK.
Needs the firstreal zarr and both scratch interpreters (skipped otherwise). Real-env interpreter:
cd controlSW/franka_sim && env -u PYTHONPATH $RE -m pytest -q tests/test_eval_criteria.py tests/test_eval_stage2a.py"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from franka_sim.eval import runner as rn  # noqa: E402
from franka_sim.eval import stage2a as s2a  # noqa: E402

S = Path(os.environ.get("S", "/tmp/claude-1000/-home-ruben-Capstone/b5f58f43-1b87-4846-beab-dc85de4129a4/scratchpad"))
ZARR = Path(os.environ.get("ZARR") or S / "deploy_work" / "prompt_zarr" / "firstreal-single.zarr.zip")
pytestmark = pytest.mark.skipif(not ZARR.is_file() or not (S / "env_bp" / "bin" / "python").is_file(),
                                reason="needs the firstreal zarr and the BP interpreter")


def _args(out, policy, demos="0,1", **kw):
    d = dict(deploy_config=None, profile="sim-iphone-fake", stage="2a", policy=policy, episodes=None, demos=demos, set=[],
             seed=0, quick=True, jobs=4, stage1=None, port_offset=None, out=str(out))
    return argparse.Namespace(**{**d, **kw})


@pytest.fixture(scope="module")
def stage1_dir(tmp_path_factory):
    """One small (quick grid) Stage 1 run on demos 0,1, shared by the 2a tests."""
    from franka_sim.eval import stage1 as s1w

    out = tmp_path_factory.mktemp("s1")
    R = rn.resolve(_args(out, "demo_replay"))
    d = out / "s1"
    d.mkdir()
    p = s1w.run_stage1_cli(R, [0, 1], d / "stage1", d / "stage1.log")
    assert p.is_file()
    return d


def _run(tmp_path, policy, stage1_dir):
    a = _args(tmp_path, policy, stage1=str(stage1_dir))
    R = rn.resolve(a)
    rep = rn.run_one_stage(R, "2a", {})
    return R, rep, R.run_dir("2a")


def test_demo_replay_zero_error_and_full_reachability(tmp_path, stage1_dir):
    R, rep, d = _run(tmp_path, "demo_replay", stage1_dir)
    m = rep["metrics"]
    assert rep["verdict"] == "PASS", rep["gates"]
    assert m["n_demos"] == 2 and m["n_chunks"] > 40
    for k in (4, 8, 15):
        assert m[f"pred_vs_demo_pos_err_mm_p95_k{k}"] <= 1e-3          # <= 1e-6 m
        assert m[f"pred_vs_demo_width_err_mm_p95_k{k}"] <= 1e-3
    assert m["chunk_reachable_fraction"] == 1.0 and m["demo_chunk_reachable_fraction"] == 1.0
    assert m["chunk_joint_margin_deg"] >= 10
    assert (d / "pred.npz").is_file() and (d / "report.html").is_file() and (d / "metrics.json").is_file()
    assert (d / "station_resolved.yaml").is_file() and (d / "criteria.yaml").is_file()
    assert m["inference_s_p50"] is None and "inference_s_p50" in m["timing"]   # a replay has no inference time


def test_fake_policy_error_is_the_demos_own_motion(tmp_path, stage1_dir):
    R, rep, d = _run(tmp_path, "fake", stage1_dir)
    m = rep["metrics"]
    P = np.load(d / "pred.npz")
    ok = P["gt_valid"]
    for k in (4, 8, 15):
        disp = np.linalg.norm(P["gt_chunks"][ok][:, k, :3], axis=-1) * 1000      # hold chunk: error = |demo displacement|
        assert m[f"pred_vs_demo_pos_err_mm_p95_k{k}"] == pytest.approx(np.percentile(disp, 95), rel=1e-9)
    assert m["pred_vs_demo_pos_err_mm_p95_k15"] > m["pred_vs_demo_pos_err_mm_p95_k4"] > 0
    # the identity chunk stays at the anchor, which the demo reaches: reachability is that of the demo
    assert m["chunk_reachable_fraction"] >= 0.95
    assert rep["verdict"] in ("PASS", "FAIL")
    assert json.loads(str(P["meta"]))["policy_kind"] == "fake"


def test_reproducible_same_seed(tmp_path, stage1_dir):
    _, a, _ = _run(tmp_path / "a", "demo_replay", stage1_dir)
    _, b, _ = _run(tmp_path / "b", "demo_replay", stage1_dir)
    timing = set(a["metrics"]["timing"])
    ka = {k: v for k, v in a["metrics"].items() if k not in timing and k != "timing"}
    kb = {k: v for k, v in b["metrics"].items() if k not in timing and k != "timing"}
    assert ka == kb
    assert {k: g["pass"] for k, g in a["gates"].items()} == {k: g["pass"] for k, g in b["gates"].items()}


def test_chunk_errors_helper():
    rng = np.random.default_rng(0)
    gt = np.zeros((5, 16, 10))
    gt[..., 3:9] = [1, 0, 0, 0, 1, 0]
    rel = gt.copy()
    rel[:, 8, :3] = [0.003, 0.004, 0.0]
    rel[:, 8, 9] = 0.002
    e = s2a.chunk_errors(rel, gt)
    assert np.allclose(e["pos_8"], 5.0) and np.allclose(e["pos_4"], 0) and np.allclose(e["width_8"], 2.0)
    assert np.allclose(e["rot_8"], 0, atol=1e-5)
