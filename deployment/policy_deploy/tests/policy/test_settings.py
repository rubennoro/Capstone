# adapted from rnoro/deployment:deployment/policy/tests/test_settings.py @ 91b178b
# changes: the new station schema (Station.load + load_config("policy", station)), so the merge and the relative
# paths are policy_deploy.contracts' (their own tests are in tests/contracts, tests/launcher); the endpoint is
# ports.policy; the session folder is ProcessArgs'; + precision and mock_kind; + prompt_dataset and prompt_split.
# copied from rnoro/final-deployment:deployment/policy_deploy/tests/policy/test_settings.py @ f632576
# changes: the kind-not-set message lists replay; the unknown kind is 'bogus' (replay is a kind here)
# (sim_deploy, rnoro/main-sim)
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from policy_deploy.contracts import paths
from policy_deploy.policy.errors import SettingsError
from policy_deploy.policy.settings import Settings


def _problems(fn) -> list[str]:
    with pytest.raises(SettingsError) as e:
        fn()
    return e.value.problems


def test_config_loads_with_all_defaults(policy_station):
    s = Settings.load(policy_station({"kind": "hold_still"}), mock=False)
    defaults = yaml.safe_load(paths.config_file("policy").read_text())
    assert s.kind == "hold_still" and s.active_kind == "hold_still" and not s.mock
    assert s.checkpoint is None and defaults["checkpoint"] is None
    assert s.device == defaults["device"] == "cuda"
    assert s.diffusion_steps is None
    assert s.precision == defaults["precision"] == "fp32"
    assert s.prompt_dataset is None and defaults["prompt_dataset"] is None
    assert s.prompt_split == defaults["prompt_split"] == "train"
    assert s.mock_kind == defaults["mock_kind"] == "hold_still"
    assert s.warmup_inferences == defaults["warmup_inferences"] == 2
    assert s.warmup_gripper_width_m == defaults["warmup_gripper_width_m"]
    assert s.reset_max_s == 0.5 and s.poll_s == defaults["poll_s"] and s.self_check_timeout_s == 20.0
    assert s.training_timing == {"obs_down_sample_steps": 3, "ultrawide_down_sample_steps": 1}
    assert s.key_map == defaults["key_map"]
    assert s.key_map["obs"]["main"] == "camera_right_main_rgb"


def test_station_overrides_merge_maps_key_by_key(policy_station):
    s = Settings.load(policy_station({"kind": "hold_still", "device": "cpu", "diffusion_steps": 8, "precision": "bf16",
                                      "key_map": {"obs": {"main": "camera_main"}},
                                      "training_timing": {"obs_down_sample_steps": 2}}), mock=False)
    assert s.device == "cpu" and s.diffusion_steps == 8 and s.precision == "bf16"
    assert s.key_map["obs"]["main"] == "camera_main"
    assert s.key_map["obs"]["ultrawide"] == "camera_right_ultrawide_rgb"       # untouched sibling kept
    assert s.key_map["action"]["eef_pos"] == "right_pos"
    assert s.training_timing == {"obs_down_sample_steps": 2, "ultrawide_down_sample_steps": 1}


def test_unknown_key_refused_with_its_path(policy_station):
    problems = _problems(lambda: Settings.load(policy_station({"kind": "hold_still", "chekpoint": "x.ckpt"}),
                                               mock=False))
    assert problems == ["station.yaml: policy.chekpoint: not a key of config/policy.yaml"]
    problems = _problems(lambda: Settings.load(policy_station({"kind": "hold_still",
                                                               "key_map": {"obs": {"foo": "x"}}}), mock=False))
    assert problems == ["station.yaml: policy.key_map.obs.foo: not a key of config/policy.yaml"]


def test_kind_unset_is_an_error_naming_policy_kind_but_fine_with_mock(policy_station):
    problems = _problems(lambda: Settings.load(policy_station({}), mock=False))
    assert problems == ["policy.kind is not set: choose diffusion or hold_still or replay in station.yaml "
                        "(policy.kind)"]
    s = Settings.load(policy_station(None), mock=True)
    assert s.kind is None and s.active_kind == "hold_still" and s.mock


def test_mock_runs_mock_kind(policy_station):
    s = Settings.load(policy_station({"kind": "diffusion"}), mock=True)            # no checkpoint needed
    assert s.kind == "diffusion" and s.active_kind == "hold_still"
    station = policy_station({"kind": "hold_still", "mock_kind": "diffusion", "checkpoint": "m/b.ckpt"},
                             python={"light": sys.executable, "torch": sys.executable})
    s = Settings.load(station, mock=True)
    assert s.active_kind == "diffusion" and s.mock_kind == "diffusion"
    s = Settings.load(station, mock=False)
    assert s.active_kind == "hold_still"


def test_mock_kind_diffusion_needs_a_checkpoint(policy_station):
    station = policy_station({"mock_kind": "diffusion"}, python={"light": sys.executable, "torch": sys.executable})
    problems = _problems(lambda: Settings.load(station, mock=True))
    assert len(problems) == 1 and "policy.checkpoint: required for kind diffusion" in problems[0]


def test_unknown_kind_and_mock_kind(policy_station):
    problems = _problems(lambda: Settings.load(policy_station({"kind": "bogus"}), mock=False))
    assert len(problems) == 1 and "policy.kind: unknown kind 'bogus'" in problems[0]
    problems = _problems(lambda: Settings.load(policy_station({"kind": "hold_still", "mock_kind": "x"}), mock=True))
    assert len(problems) == 1 and "policy.mock_kind: unknown kind 'x'" in problems[0]


def test_diffusion_without_checkpoint_refused(policy_station):
    station = policy_station({"kind": "diffusion"}, python={"light": sys.executable, "torch": sys.executable})
    problems = _problems(lambda: Settings.load(station, mock=False))
    assert len(problems) == 1 and "policy.checkpoint: required for kind diffusion" in problems[0]


def test_diffusion_needs_python_torch(policy_station):
    problems = _problems(lambda: Settings.load(policy_station({"kind": "diffusion", "checkpoint": "b.ckpt"},
                                                              python={"light": sys.executable}), mock=False))
    assert any("python.torch: required for policy kind diffusion" in p for p in problems)


@pytest.mark.parametrize("key,value", [("diffusion_steps", 0), ("diffusion_steps", 2.5), ("diffusion_steps", True),
                                       ("warmup_inferences", 0), ("warmup_gripper_width_m", -0.01),
                                       ("reset_max_s", 0), ("poll_s", -1), ("self_check_timeout_s", "x"),
                                       ("device", "tpu"), ("precision", "fp16"), ("precision", None),
                                       ("prompt_split", "test"), ("prompt_split", None), ("prompt_dataset", 3),
                                       ("prompt_dataset", "")])
def test_ranges(policy_station, key, value):
    problems = _problems(lambda: Settings.load(policy_station({"kind": "hold_still", key: value}), mock=False))
    assert len(problems) == 1 and f"policy.{key}: expected" in problems[0]


def test_every_problem_listed_at_once(policy_station):
    problems = _problems(lambda: Settings.load(policy_station({"diffusion_steps": 0, "device": "tpu",
                                                               "precision": "x"}), mock=False))
    assert len(problems) == 4


def test_relative_checkpoint_from_the_station_folder(policy_station, tmp_path):
    torch_py = {"light": sys.executable, "torch": sys.executable}
    s = Settings.load(policy_station({"kind": "diffusion", "checkpoint": "models/best.ckpt"}, python=torch_py),
                      mock=False)
    assert s.checkpoint == (tmp_path / "models" / "best.ckpt").resolve()
    s = Settings.load(policy_station({"kind": "diffusion", "checkpoint": "~/m/best.ckpt"}, python=torch_py),
                      mock=False)
    assert s.checkpoint == (Path.home() / "m" / "best.ckpt").resolve()


def test_endpoint_from_the_station_ports(policy_station):
    station = policy_station({"kind": "hold_still"})
    port = yaml.safe_load(station.read_text())["ports"]["policy"]
    assert Settings.load(station, mock=False).endpoint == f"tcp://127.0.0.1:{port}"


def test_session_root_from_the_station(policy_station, tmp_path):
    assert Settings.load(policy_station({"kind": "hold_still"}), mock=False).session_root == tmp_path / "sessions"


def test_station_file_missing(tmp_path):
    problems = _problems(lambda: Settings.load(tmp_path / "nope.yaml", mock=True))
    assert "not found" in problems[0]


def test_default_station_loads():
    s = Settings.load(None, mock=True)
    assert s.station_file == paths.default_station() and s.active_kind == "hold_still"


def test_as_dict_is_json_ready(policy_station, tmp_path):
    s = Settings.load(policy_station({"kind": "diffusion", "checkpoint": "b.ckpt"},
                                     python={"light": sys.executable, "torch": sys.executable}), mock=False)
    d = json.loads(json.dumps(s.as_dict()))
    assert d["checkpoint"] == str((tmp_path / "b.ckpt").resolve()) and d["active_kind"] == "diffusion"
    assert d["precision"] == "fp32"


def test_prompt_dataset_resolved_like_the_checkpoint(policy_station, tmp_path):
    s = Settings.load(policy_station({"kind": "diffusion", "checkpoint": "b.ckpt", "prompt_dataset": "data/d.zarr.zip",
                                      "prompt_split": "all"}, python={"light": sys.executable,
                                                                      "torch": sys.executable}), mock=False)
    assert s.prompt_dataset == (tmp_path / "data" / "d.zarr.zip").resolve() and s.prompt_split == "all"
    assert json.loads(json.dumps(s.as_dict()))["prompt_dataset"] == str(s.prompt_dataset)
    s = Settings.load(policy_station({"kind": "hold_still", "prompt_dataset": "~/d.zarr.zip"}), mock=False)
    assert s.prompt_dataset == (Path.home() / "d.zarr.zip").resolve()            # not checked to exist here
