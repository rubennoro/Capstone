"""Opt-in: the real inference server (upstream node + light workspace) on a real checkpoint and prompt zarr.

Run (ideally without network: `unshare -rn sh -c 'ip link set lo up && <pytest cmd>'`):
  FRANKA_DEPLOY_CKPT=<.ckpt> FRANKA_DEPLOY_PROMPT_ZARR=<.zarr.zip> FRANKA_DEPLOY_WORK=<scratch dir> pytest tests/test_server_ckpt.py
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import CKPT, PROMPT_ZARR, WORK, PKG_ROOT

pytestmark = pytest.mark.skipif(not (CKPT and PROMPT_ZARR and WORK), reason="needs FRANKA_DEPLOY_CKPT/PROMPT_ZARR/WORK")

robotmq = pytest.importorskip("robotmq")
from franka_deploy.paths import REPO_ROOT  # noqa: E402
from franka_deploy.rmq_rpc import RpcError, RpcTimeout, request  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def network_reachable() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1.0).close()
        return True
    except OSError:
        return False


def server_env():
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(PKG_ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
    return env


@pytest.fixture(scope="module")
def server():
    work = Path(WORK) / "test_server"
    ready = work / "ready.json"
    if ready.exists():
        ready.unlink()
    work.mkdir(parents=True, exist_ok=True)
    port = free_port()
    ep = f"tcp://127.0.0.1:{port}"
    log = open(work / "server.log", "w")
    results_ckpt_dir = Path(CKPT).resolve().parent
    before = sorted(os.listdir(results_ckpt_dir))
    p = subprocess.Popen([sys.executable, "-u", "-m", "franka_deploy.inference.server", "--ckpt", CKPT,
                          "--work-dir", str(work), "--dataset-path", PROMPT_ZARR, "--endpoint", ep,
                          "--ready-file", str(ready)], stdout=log, stderr=subprocess.STDOUT, env=server_env(),
                         cwd=str(work.parent))
    t0 = time.monotonic()
    while not ready.exists():
        if p.poll() is not None:
            raise RuntimeError(f"server exited rc={p.returncode}: " + (work / "server.log").read_text()[-3000:])
        if time.monotonic() - t0 > 300:
            p.kill()
            raise TimeoutError("server not ready in 300 s")
        time.sleep(0.5)
    report = json.loads(ready.read_text())
    client = robotmq.RMQClient("test_client", ep, robotmq.RMQLogLevel.ERROR)
    yield {"proc": p, "report": report, "client": client, "work": work, "log": work / "server.log",
           "results_before": before, "results_dir": results_ckpt_dir}
    p.terminate()
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()
    log.close()


def deploy_obs(h=720, w=960, seed=0):
    rng = np.random.default_rng(seed)
    rot = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (2, 1))
    return {
        "robot0_main_camera": rng.integers(0, 255, (2, h, w, 3), dtype=np.uint8),
        "robot0_ultrawide_camera": rng.integers(0, 255, (2, h, w, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros((2, 3), np.float32),
        "robot0_eef_rot_axis_angle": rot.copy(),
        "robot0_gripper_width": np.full((2, 1), 0.07, np.float32),
        "robot0_eef_rot_axis_angle_wrt_start": rot.copy(),
        "episode_idx": 0,
    }


def n_inferences(server) -> int:
    return server["log"].read_text().count("[policy] inference time")


def test_network_status_recorded():
    print("network reachable:", network_reachable())


def test_startup_report(server):
    r = server["report"]
    assert r["load_report"] == {"model": {"tensors": 961, "missing": 0, "unexpected": 0}}
    assert r["clip_normalization_checked"] and r["heavy_modules_loaded"] == []
    assert r["prompt_tasks"] == {"block_raise": 43}
    work = server["work"]
    assert r["cwd"] == str(work) and Path(r["staged_ckpt"]).parent == work / "ckpt"
    assert Path(r["staged_ckpt"].replace(".ckpt", ".yaml")).is_file()  # upstream's yaml export lands in the work dir
    assert sorted(os.listdir(server["results_dir"])) == server["results_before"]
    assert not any(n.endswith(".yaml") for n in os.listdir(server["results_dir"]))
    assert not (REPO_ROOT / "tmp_vis").exists() and not Path(REPO_ROOT / "behavior_prompting" / "tmp_vis").exists()
    assert len(r["warmup_infer_s"]) >= 1
    assert not (Path.home() / ".libero").exists()


def test_prompt_lifecycle(server):
    c = server["client"]
    cfg = request(c, "policy_config", True, timeout_s=10)
    assert cfg["use_prompting"] and cfg["workspace"]["obs_down_sample_steps"] == 3 and cfg["epoch"] == 50
    with pytest.raises(RpcError, match="is_prompted|prompt"):
        request(c, "policy_inference", deploy_obs(), timeout_s=30)
    assert request(c, "policy_prompt", {"task_name": "block_raise", "prompt_index": 0}, timeout_s=60) == "OK"
    out = request(c, "policy_inference", deploy_obs(), timeout_s=30)
    assert out["action0_eef_pos"].shape == (16, 3) and out["action0_eef_rot_axis_angle"].shape == (16, 6)
    assert out["action0_gripper_width"].shape == (16, 1)
    out224 = request(c, "policy_inference", deploy_obs(224, 224), timeout_s=30)  # pre-resized path also accepted
    assert out224["action0_eef_pos"].shape == (16, 3)
    assert request(c, "policy_reset", "RESET", timeout_s=10) == "OK"
    with pytest.raises(RpcError):
        request(c, "policy_inference", deploy_obs(), timeout_s=30)
    with pytest.raises(RpcError, match="out of range"):
        request(c, "policy_prompt", {"task_name": "block_raise", "prompt_index": 43}, timeout_s=60)
    with pytest.raises(RpcError, match="not in dataset"):
        request(c, "policy_prompt", {"task_name": "nope", "prompt_index": 0}, timeout_s=60)
    assert request(c, "policy_prompt", {"task_name": "block_raise", "prompt_index": 42}, timeout_s=60) == "OK"
    # upstream: a second prompt without a reset is refused (the client must reset, then prompt, per episode)
    with pytest.raises(RpcError, match="Call `reset`"):
        request(c, "policy_prompt", {"task_name": "block_raise", "prompt_index": 0}, timeout_s=60)
    assert request(c, "policy_reset", "RESET", timeout_s=10) == "OK"
    assert server["proc"].poll() is None


def test_slow_inference_times_out_without_resend(server):
    c = server["client"]
    request(c, "policy_reset", "RESET", timeout_s=10)
    request(c, "policy_prompt", {"task_name": "block_raise", "prompt_index": 0}, timeout_s=60)
    request(c, "policy_inference", deploy_obs(), timeout_s=30)  # warm
    before = n_inferences(server)
    timeout = 0.05  # far below the ~0.25 s inference time
    t0 = time.monotonic()
    with pytest.raises(RpcTimeout):
        request(c, "policy_inference", deploy_obs(seed=1), timeout_s=timeout)
    dt = time.monotonic() - t0
    assert dt < timeout + 0.1, dt
    time.sleep(3.0)
    assert n_inferences(server) - before == 1, "the timed-out request was executed more than once"
    out = request(c, "policy_inference", deploy_obs(seed=2), timeout_s=30)  # server and client still healthy
    assert out["action0_eef_pos"].shape == (16, 3)
    assert n_inferences(server) - before == 2


def test_work_dir_under_results_refused(tmp_path):
    bad = REPO_ROOT / "results" / "block_raise" / "_franka_deploy_should_not_exist"
    r = subprocess.run([sys.executable, "-m", "franka_deploy.inference.server", "--ckpt", CKPT, "--work-dir", str(bad),
                        "--dataset-path", PROMPT_ZARR], env=server_env(), capture_output=True, text=True, timeout=120,
                       cwd=str(tmp_path))
    assert r.returncode != 0 and "ForbiddenPathError" in r.stderr
    assert not bad.exists()


@pytest.fixture(scope="module")
def cpu_node():
    from franka_deploy.inference.server import attach_prompt_dataset, build_node

    node, rep = build_node(CKPT, str(Path(WORK) / "test_cpu_node"), None, f"tcp://127.0.0.1:{free_port()}", "cpu")
    attach_prompt_dataset(node, os.path.abspath(PROMPT_ZARR))
    return node, rep


def _leaves(t, p=""):
    if isinstance(t, dict):
        for k, v in t.items():
            yield from _leaves(v, f"{p}/{k}")
    else:
        yield p, np.asarray(t)


def test_prompt_tensors_deterministic(cpu_node):
    """Fix 2: the prompt for (block_raise, i) is bitwise identical every time (no RNG in the prompt path)."""
    import torch

    node, rep = cpu_node
    assert rep["imagecodecs"] == "2023.9.18"
    ds = node.prompt_dataset
    idx = ds.get_unique_task_name_to_dataset_indices()["block_raise"]
    for i in (0, 7, 42):
        a = dict(_leaves(ds[idx[i]]["obs"]["prompt"]))
        np.random.seed(1234)
        torch.manual_seed(99)
        b = dict(_leaves(ds[idx[i]]["obs"]["prompt"]))
        assert sorted(a) == sorted(b)
        for k in a:
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), (i, k)
