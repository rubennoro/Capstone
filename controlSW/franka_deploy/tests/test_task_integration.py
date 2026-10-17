"""FrankaPromptTask end to end with a fake policy server and real processes for the mock controller,
mock spacemouse and RigCamera (synthetic mock). Checks: exactly one prompt per episode on c / C / after R / after
a stop (and never a double prompt), MainLogger with empty logger_endpoints, stop-file -> HOLD latency, release.
Runs in the real-env env; ports 2xxxx-offset so it does not collide with a running harness."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("real_env")
pytest.importorskip("sensor_rig")
robotmq = pytest.importorskip("robotmq")

from conftest import PKG_ROOT  # noqa: E402
from franka_deploy.harness.dryrun import cleanup_shm  # noqa: E402
from franka_deploy.rmq_rpc import NoResendClient  # noqa: E402

EP = {"policy": "tcp://127.0.0.1:28765", "camera": "tcp://localhost:26792", "logger": "tcp://localhost:25556",
      "controller": "tcp://localhost:26910", "spacemouse": "tcp://localhost:25557"}


def env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"PYTHONPATH": str(PKG_ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return e


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    work = tmp_path_factory.mktemp("integ")
    procs = []

    def start(name, cmd):
        log = open(work / f"{name}.log", "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=str(work), start_new_session=True)
        procs.append((p, log))
        return p

    start("policy", [sys.executable, "-u", "-m", "franka_deploy.mocks.fake_policy_server", "--endpoint", EP["policy"],
                     "--stats", str(work / "policy_stats.json")])
    start("controller", [sys.executable, "-u", "-m", "franka_deploy.controllers.mock_franka", "--endpoint", EP["controller"]])
    start("spacemouse", [sys.executable, "-u", "-m", "franka_deploy.mocks.spacemouse_server", "--endpoint", EP["spacemouse"]])
    start("camera", [sys.executable, "-u", "-m", "real_env.peripherals.rig_camera", "wrist_mock",
                     f"server_endpoint={EP['camera']}", f"logger_endpoint={EP['logger']}", "stats_period_s=0"])
    time.sleep(4.0)
    yield {"work": work, "procs": procs}
    for p, log in procs:
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(5)
        log.close()
        cleanup_shm(p.pid)


def test_prompt_once_per_episode_and_stop_hold(stack):
    """The task runs as its own process (as deployed); this test plays the operator and polls the controller."""
    work = stack["work"]
    cmd, stop = work / "commands.txt", work / "STOP"
    cmd.write_text("")
    task_dir = work / "task"
    log = open(work / "task.log", "w")
    p = subprocess.Popen([sys.executable, "-u", "-m", "franka_deploy.tasks.run_task", "--work-dir", str(task_dir),
                          "--command-file", str(cmd), "--stop-file", str(stop), "--no-stdin", "--duration", "60",
                          "--policy-endpoint", EP["policy"], "--camera-endpoint", EP["camera"],
                          "--controller-endpoint", EP["controller"], "--spacemouse-endpoint", EP["spacemouse"]],
                         stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=str(work))
    status = NoResendClient("test_status", EP["controller"], default_timeout_s=0.2)
    out = {}

    def key(k):
        with open(cmd, "a") as f:
            f.write(k + "\n")

    def st():
        raw, _ = status.peek_data(topic="status", n=-1)
        return robotmq.deserialize(raw[0])

    def wait_prompts(n, timeout=30):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            f = work / "policy_stats.json"
            if f.exists() and json.loads(f.read_text())["counts"]["policy_prompt"] >= n:
                return
            time.sleep(0.05)

    try:
        wait_prompts(0)
        time.sleep(12.0)                                # task startup (agent, camera, controller clients)
        key("c"); wait_prompts(1); time.sleep(2.0)      # prompt 1 (episode start)
        n0 = st()["counters"]["holds"]
        t0 = time.monotonic(); stop.write_text("s")     # stop via stop file -> hold by the watcher process
        while st()["counters"]["holds"] <= n0:
            assert time.monotonic() - t0 < 2.0, "no hold"
            time.sleep(0.002)
        out["hold_status"] = st()
        out["key_to_hold_s"] = out["hold_status"]["hold_applied_t"] - t0
        time.sleep(1.5)
        out["released"] = not st()["held"]
        key("C"); time.sleep(2.0)                       # prompt 2 (lazily, after the stop)
        key("C"); time.sleep(2.0)                       # same single-step session: no new prompt
        key("R"); time.sleep(0.5)
        key("C"); time.sleep(2.0)                       # prompt 3 (after R)
        key("c"); time.sleep(2.5)                       # prompt 4 (new episode)
        n1 = st()["counters"]["holds"]
        t1 = time.monotonic(); key("s")                 # stop via command-file key -> hold by the watcher
        while st()["counters"]["holds"] <= n1:
            assert time.monotonic() - t1 < 2.0, "no hold on key"
            time.sleep(0.002)
        out["key2_to_hold_s"] = st()["hold_applied_t"] - t1
        time.sleep(1.5)
        key("q")
        p.wait(30)
    finally:
        if p.poll() is None:
            p.kill()
        log.close()
    summ = json.loads((task_dir / "task_summary.json").read_text())
    stats = json.loads((work / "policy_stats.json").read_text())
    assert summ["prompt_count"] == 4, summ["events"]
    assert len(stats["prompts"]) == 4 and stats["counts"]["policy_prompt"] == 4, stats
    assert stats["counts"]["errors"] == 0, stats  # never a double prompt / unprompted inference
    assert summ["counters"]["policy_errors"] == 0 and summ["counters"]["idle_errors"] == 0, summ["events"]
    assert out["key_to_hold_s"] <= 0.1 and out["key2_to_hold_s"] <= 0.1, out
    lat = out["hold_status"]["hold_applied_t"] - out["hold_status"]["hold_request_t"]
    assert 0 <= lat <= 0.05, lat
    assert out["released"]
    eps = sorted(os.listdir(task_dir / "episodes" / "franka_deploy" / "block_raise" / "run"))
    assert eps == ["episode_000000", "episode_000001"], eps  # MainLogger with logger_endpoints={} records episodes
    res = {"key_to_hold_s": out["key_to_hold_s"], "key2_to_hold_s": out["key2_to_hold_s"],
           "controller_hold_latency_s": lat, "counts": stats["counts"]}
    (work / "result.json").write_text(json.dumps(res))
    print("INTEGRATION", json.dumps(res))
