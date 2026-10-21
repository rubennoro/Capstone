"""predict_demos (B8): run in env_bp:  cd controlSW/franka_deploy && env -u PYTHONPATH PYTHONPATH=. $BP -m pytest -q tests/test_predict_demos.py
Needs the firstreal zarr (FRANKA_DEPLOY_PROMPT_ZARR or the sim_plan scratch copy). The real bp_server part is opt-in:
FRANKA_DEPLOY_CKPT=<.ckpt> (record the result in the B8 WORKLOG)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("zarr")
from franka_deploy.tools import predict_demos as pdm  # noqa: E402

ZARR = os.environ.get("FRANKA_DEPLOY_PROMPT_ZARR") or "/tmp/claude-1000/-home-ruben-Capstone/b5f58f43-1b87-4846-beab-dc85de4129a4/scratchpad/deploy_work/prompt_zarr/firstreal-single.zarr.zip"
CKPT = os.environ.get("FRANKA_DEPLOY_CKPT")
need_zarr = pytest.mark.skipif(not Path(ZARR).is_file(), reason="firstreal zarr missing")
PKG = Path(__file__).resolve().parents[1]


def _codecs_ok() -> bool:
    """The zarr images need the imagecodecs JPEG-XL codec + behavior_prompting (env_bp only; the RE suite skips this file)."""
    try:
        pdm.DemoZarr(ZARR, images=True).main_image(0, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


if Path(ZARR).is_file() and not _codecs_ok():
    pytest.skip("imagecodecs_jpegxl / behavior_prompting missing: run in env_bp", allow_module_level=True)


def _run(tmp_path, kind, extra=(), **pol):
    p = tmp_path / f"{kind}.yaml"
    p.write_text(yaml.safe_dump({"id": kind, "kind": kind, "contract": {"K": 16, "exec_h": 8, "uses_prompt": False}, **pol}))
    out = tmp_path / f"{kind}.npz"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(PKG)
    r = subprocess.run([sys.executable, "-m", "franka_deploy.tools.predict_demos", "--policy", str(p), "--zarr", ZARR,
                        "--demos", "0,1", "--out", str(out), *extra], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    return dict(np.load(out, allow_pickle=False))


@need_zarr
def test_replay_and_fake_shapes_and_values(tmp_path):
    rp = _run(tmp_path, "replay")
    n = len(rp["frame"])
    assert rp["rel_chunks"].shape == (n, 16, 10) and rp["gt_chunks"].shape == (n, 16, 10)
    assert set(rp["demo"].tolist()) == {0, 1} and rp["frame"][0] == 0 and n > 20
    assert np.abs(rp["rel_chunks"] - rp["gt_chunks"]).max() < 1e-6
    assert rp["gt_valid"].all()
    assert np.allclose(rp["gt_chunks"][:, 0, :3], 0, atol=1e-9)                    # action 0 = identity
    assert np.allclose(rp["gt_chunks"][:, 0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-9)
    assert np.diff(rp["frame"][rp["demo"] == 0]).tolist()[0] == 24               # every exec_h agent steps (3 frames each)
    meta = json.loads(str(rp["meta"]))
    assert meta["policy_kind"] == "replay" and meta["demos"] == [0, 1] and len(meta["demo_names"]) == 2
    fk = _run(tmp_path, "fake")
    assert fk["rel_chunks"].shape == (n, 16, 10)
    assert np.allclose(fk["rel_chunks"][..., :3], 0) and np.allclose(fk["rel_chunks"][..., 3:9], [1, 0, 0, 0, 1, 0])
    # fake error = the demo's own motion
    disp = np.linalg.norm(fk["gt_chunks"][..., :3] - fk["rel_chunks"][..., :3], axis=-1)
    assert disp.max() > 0.05 and np.allclose(disp[:, 0], 0)


@need_zarr
def test_gt_chunk_matches_replay_server_and_obs_convention():
    from franka_deploy.tools.replay_server import ReplayServer

    z = pdm.DemoZarr(ZARR, images=True)
    d = z.demo(0)
    rs = ReplayServer(d["T"], d["w"], start=30.0)
    ref = rs.chunk()
    g, ok = pdm.gt_chunk(d["T"], d["w"], 30.0)
    assert ok
    assert np.abs(g[:, :3] - ref["action0_eef_pos"]).max() < 1e-6
    assert np.abs(g[:, 3:9] - ref["action0_eef_rot_axis_angle"]).max() < 1e-6
    assert np.abs(g[:, 9] - ref["action0_gripper_width"][:, 0]).max() < 1e-6
    # a chunk near the end of the demo is flagged
    _, ok_end = pdm.gt_chunk(d["T"], d["w"], len(d["T"]) - 10.0)
    assert not ok_end
    # observation: proprio history (2 steps), last = identity, shapes/dtypes as analyze_tap.CONTRACT_OBS
    obs = pdm.deploy_obs(z, d, 100.0, 0.0, {"proprio_length": 1, "img_obs_horizon": 2, "obs_down_sample_steps": 3,
                                            "ultrawide_down_sample_steps": 1})
    from franka_deploy.tools.analyze_tap import CONTRACT_OBS

    for k, (shape, dtype) in CONTRACT_OBS.items():
        if k in ("robot0_main_camera", "robot0_ultrawide_camera"):
            continue
        assert tuple(np.asarray(obs[k]).shape) == shape, k
    assert np.allclose(obs["robot0_eef_pos"][-1], 0) and np.allclose(obs["robot0_eef_rot_axis_angle"][-1], [1, 0, 0, 0, 1, 0])
    assert obs["robot0_eef_pos"].dtype == np.float32 and obs["robot0_main_camera"].shape[0] == 2


@need_zarr
def test_queries_file_orders_and_fractional_frames(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"demo": 1, "frame": 40.5, "start_frame": 0.0}, {"demo": 0, "frame": 10.25, "start_frame": 7.0}]))
    r = _run(tmp_path, "replay", extra=["--queries", str(q)])
    assert r["demo"].tolist() == [1, 0] and r["frame"].tolist() == [40.5, 10.25] and r["start_frame"].tolist() == [0.0, 7.0]
    assert np.abs(r["rel_chunks"] - r["gt_chunks"]).max() < 1e-6


@pytest.mark.skipif(not (CKPT and Path(ZARR).is_file()), reason="opt-in: FRANKA_DEPLOY_CKPT")
def test_real_bp_server(tmp_path):
    r = _run(tmp_path, "bp_server", extra=["--limit", "1", "--demos", "0", "--every", "24", "--device", "cuda"],
             python=sys.executable, checkpoint=CKPT, prompt_zarr=ZARR, prompt_index=1, ready_timeout_s=600,
             contract={"K": 16, "exec_h": 8, "uses_prompt": True})
    assert r["rel_chunks"].shape[1:] == (16, 10) and np.isfinite(r["rel_chunks"]).all()
    assert (r["infer_s"] > 0).all() and (r["noise_seed"] >= 0).all()
