"""B3: tools/policy_contract.py against real server processes (fake, fake without prompt rules, replay on a synthetic
zarr) and deliberately broken fakes. Ports: 3xxxx + FD_TEST_PORT_OFFSET (Track B: 2000)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("robotmq")

from conftest import PKG_ROOT  # noqa: E402
from franka_deploy.__main__ import main as cli_main  # noqa: E402
from franka_deploy.tools import policy_contract as pc  # noqa: E402
from franka_deploy.tools.analyze_tap import CONTRACT_OBS  # noqa: E402

OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))
_next = [30300 + OFFSET]


def port():
    _next[0] += 1
    return _next[0]


def env():
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"PYTHONPATH": str(PKG_ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return e


@pytest.fixture
def serve(tmp_path):
    procs = []

    def start(module, *args):
        p = port()
        ready = tmp_path / f"ready_{p}.json"
        log = open(tmp_path / f"srv_{p}.log", "w")
        ep = f"tcp://127.0.0.1:{p}"
        proc = subprocess.Popen([sys.executable, "-u", "-m", module, "--endpoint", ep, "--ready-file", str(ready), *args],
                                stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=str(tmp_path), start_new_session=True)
        procs.append((proc, log))
        t = time.monotonic()
        while not ready.exists():
            assert proc.poll() is None, (tmp_path / f"srv_{p}.log").read_text()
            assert time.monotonic() - t < 30, "server not ready"
            time.sleep(0.05)
        return ep, json.loads(ready.read_text())

    yield start
    for proc, log in procs:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


FAKE = "franka_deploy.mocks.fake_policy_server"


def status(res):
    return {c["name"]: c["status"] for c in res["checks"]}


def test_fake_ready_file_has_endpoint_pid_kind(serve):
    ep, ready = serve(FAKE)
    assert ready["endpoint"] == ep and ready["kind"] == "fake" and ready["pid"] > 0 and ready["ready_monotonic"] > 0


def test_contract_passes_on_fake(serve, tmp_path):
    ep, _ = serve(FAKE)
    res = pc.run_contract(ep)
    assert res["passed"], res["table"]
    s = status(res)
    assert s["inference_before_prompt"] == s["second_prompt_rejected"] == s["prompt_ok"] == "pass"
    assert s["reply_shape"] == s["rot6d_norm"] == s["width_range"] == s["export_topic"] == "pass"
    assert res["use_prompting"] is True and res["K"] == 16
    out = tmp_path / "contract.json"
    assert cli_main(["policy-contract", "--endpoint", ep, "--out", str(out)]) == 0   # passthrough + exit code + JSON
    assert json.loads(out.read_text())["passed"] is True


def test_contract_passes_on_fake_without_prompt_rules(serve):
    ep, _ = serve(FAKE, "--no-prompt-rules")
    res = pc.run_contract(ep)
    assert res["passed"], res["table"]
    assert res["use_prompting"] is False
    s = status(res)
    assert s["inference_before_prompt"] == s["second_prompt_rejected"] == "skip" and s["inference_ok"] == "pass"


def _synthetic_zarr(path, n=200):
    import zarr

    t = np.arange(n) / 60.0
    root = zarr.open(zarr.ZipStore(str(path), mode="w"), mode="w")
    root.create_dataset("meta/episode_names", data=np.array(["ep0"]))
    root.create_dataset("meta/episode_ends", data=np.array([n]))
    root.create_dataset("data/gripper_right_eef_pos", data=np.c_[0.4 + 0.1 * np.sin(t), 0.05 * t, 0.2 + 0 * t])
    root.create_dataset("data/gripper_right_eef_rot_axis_angle", data=np.c_[0.2 * t, 0 * t, 0.1 + 0 * t])
    root.create_dataset("data/gripper_right_gripper_width", data=(0.02 + 0.05 * t / t[-1])[:, None])
    root.store.close()


def test_contract_passes_on_replay_server(serve, tmp_path):
    pytest.importorskip("zarr")
    z = tmp_path / "demo.zarr.zip"
    _synthetic_zarr(z)
    ep, _ = serve("franka_deploy.tools.replay_server", "--zarr", str(z), "--episode", "0")
    res = pc.run_contract(ep)
    assert res["passed"], res["table"]
    s = status(res)    # the replay server does not gate on prompts: warning, second-prompt rule skipped
    assert s["inference_before_prompt"] == "warn" and s["second_prompt_rejected"] == "skip" and s["inference_ok"] == "pass"


@pytest.mark.parametrize("args,failing", [(["--k", "8"], "reply_shape"), (["--reset-delay", "0.7"], "reset_latency"),
                                          (["--accept-second-prompt"], "second_prompt_rejected")])
def test_contract_fails_on_broken_fake_naming_the_check(serve, args, failing):
    ep, _ = serve(FAKE, *args)
    res = pc.run_contract(ep)
    assert not res["passed"] and res["failed"] == [failing], res["table"]
    assert cli_main(["policy-contract", "--endpoint", ep]) == 1


def test_expect_yaml_checks_uses_prompt_and_k(serve):
    ep, _ = serve(FAKE, "--no-prompt-rules")
    assert pc.run_contract(ep, expect=pc.load_expect("run/policy/fake.yaml"))["passed"]   # uses_prompt false, K 16
    res = pc.run_contract(ep, expect=pc.load_expect("bp_run001"))                          # uses_prompt true
    assert res["failed"] == ["expect_uses_prompt"], res["table"]
    ep2, _ = serve(FAKE, "--k", "8")
    assert "expect_K" in pc.run_contract(ep2, expect=pc.load_expect("fake"))["failed"]


def test_obs_from_tap_dir(serve, tmp_path):
    ep, _ = serve(FAKE)
    tap = tmp_path / "tap"
    tap.mkdir()
    np.savez(tap / "call_0000.npz", **{f"obs_{k}": v for k, v in pc.dummy_obs().items()})
    assert pc.run_contract(ep, obs=f"tap:{tap}")["passed"]
    empty = tmp_path / "empty"
    empty.mkdir()
    res = pc.run_contract(ep, obs=f"tap:{empty}")
    assert res["failed"] == ["obs"]


def test_no_server_fails_config_keys_quickly():
    res = pc.run_contract(f"tcp://127.0.0.1:{port()}")
    assert res["failed"] == ["config_keys"] and "timeout" in res["checks"][0]["detail"]


def test_dummy_obs_matches_contract_obs():
    o = pc.dummy_obs()
    for k, (shape, dtype) in CONTRACT_OBS.items():
        assert o[k].shape == shape and o[k].dtype == np.dtype(dtype), k


def _good_reply(K=16):
    rot = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (K, 1))
    return {"action0_eef_pos": np.zeros((K, 3), np.float32), "action0_eef_rot_axis_angle": rot,
            "action0_gripper_width": np.full((K, 1), 0.05, np.float32)}


def _check(reply, K=16):
    rep = pc.Report()
    pc.check_reply(rep, reply, K)
    return {c["name"]: c["status"] for c in rep.checks}


def test_check_reply_unit_failures():
    assert set(_check(_good_reply()).values()) == {"pass"}
    r = _good_reply()
    del r["action0_gripper_width"]
    assert _check(r)["reply_keys"] == "fail"
    r = _good_reply()
    r["action0_eef_pos"][3, 1] = np.nan
    assert _check(r)["reply_finite"] == "fail"
    r = _good_reply()
    r["action0_eef_rot_axis_angle"][2] *= 2
    assert _check(r)["rot6d_norm"] == "fail"
    r = _good_reply()
    r["action0_eef_rot_axis_angle"][2] = [1, 0, 0, 1, 0, 0]
    s = _check(r)
    assert s["rot6d_orthogonal"] == "fail" and s["rot6d_norm"] == "pass"
    r = _good_reply()
    r["action0_gripper_width"][5] = 0.2
    assert _check(r)["width_range"] == "fail"
    r = _good_reply()
    r["action0_eef_pos"][0] = [0.05, 0, 0]
    assert _check(r)["action0_identity"] == "warn"                  # warning only
    assert _check({"x": 1})["reply_keys"] == "fail"


def test_doc_example_policy_yaml_validates(tmp_path, capsys):
    """The umi_dp.yaml example in docs/adding_a_policy.md must be a valid run/policy file (and load with a station)."""
    import re
    import shutil

    from franka_deploy.station import STATION_ROOT, _validate_all, load_station

    doc = (PKG_ROOT.parent / "docs" / "adding_a_policy.md").read_text()
    blocks = [b for b in re.findall(r"```yaml\n(.*?)```", doc, re.S) if "id: umi_dp" in b]
    assert len(blocks) == 1
    root = tmp_path / "configs"
    shutil.copytree(STATION_ROOT, root)
    (root / "run" / "policy" / "umi_dp.yaml").write_text(blocks[0])
    assert _validate_all(root) == []
    st = load_station({"station": {"robot": "panda", "tooling": "franka_hand_iphumi_iphone16pro",
                                   "sensor": "iphone16pro_wifi", "scene": "block_raise_table", "host": "pc_laptop"},
                       "run": {"policy": "umi_dp", "criteria": "block_raise", "sim": "video_replay"}}, root)
    assert st.get("policy.kind") == "command"
