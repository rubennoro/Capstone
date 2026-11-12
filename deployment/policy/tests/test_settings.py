from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from policy.errors import SettingsError
from policy.settings import CONFIG_FILE, POLICY_DIR, Settings, merge, resolve_path


def _problems(fn) -> list[str]:
    with pytest.raises(SettingsError) as e:
        fn()
    return e.value.problems


def test_config_yaml_loads_with_all_defaults(station_file, tmp_path):
    s = Settings.load(station_file({"kind": "hold_still"}), tmp_path / "sess", mock=False)
    defaults = yaml.safe_load(CONFIG_FILE.read_text())
    assert s.kind == "hold_still" and s.active_kind == "hold_still" and not s.mock
    assert s.checkpoint is None and defaults["checkpoint"] is None
    assert s.device == defaults["device"] == "cuda"
    assert s.diffusion_steps is None
    assert s.warmup_inferences == defaults["warmup_inferences"] == 2
    assert s.warmup_gripper_width_m == defaults["warmup_gripper_width_m"]
    assert s.reset_max_s == 0.5 and s.poll_s == defaults["poll_s"] and s.self_check_timeout_s == 20.0
    assert s.training_timing == {"obs_down_sample_steps": 3, "ultrawide_down_sample_steps": 1}
    assert s.key_map == defaults["key_map"]
    assert s.key_map["obs"]["main"] == "camera_right_main_rgb"


def test_station_overrides_merge_maps_key_by_key(station_file, tmp_path):
    s = Settings.load(station_file({"kind": "hold_still", "device": "cpu", "diffusion_steps": 8,
                                    "key_map": {"obs": {"main": "camera_main"}},
                                    "training_timing": {"obs_down_sample_steps": 2}}), tmp_path, mock=False)
    assert s.device == "cpu" and s.diffusion_steps == 8
    assert s.key_map["obs"]["main"] == "camera_main"
    assert s.key_map["obs"]["ultrawide"] == "camera_right_ultrawide_rgb"       # untouched sibling kept
    assert s.key_map["action"]["eef_pos"] == "right_pos"
    assert s.training_timing == {"obs_down_sample_steps": 2, "ultrawide_down_sample_steps": 1}


def test_unknown_key_refused_with_its_path(station_file, tmp_path):
    problems = _problems(lambda: Settings.load(station_file({"kind": "hold_still", "chekpoint": "x.ckpt"}),
                                               tmp_path, mock=False))
    assert problems == ["station.yaml: policy.chekpoint: not a key of policy/config.yaml"]
    problems = _problems(lambda: Settings.load(station_file({"kind": "hold_still", "key_map": {"obs": {"foo": "x"}}}),
                                               tmp_path, mock=False))
    assert problems == ["station.yaml: policy.key_map.obs.foo: not a key of policy/config.yaml"]


def test_kind_unset_is_an_error_naming_policy_kind_but_fine_with_mock(station_file, tmp_path):
    problems = _problems(lambda: Settings.load(station_file({}), tmp_path, mock=False))
    assert problems == ["policy.kind is not set: choose diffusion or hold_still in station.yaml (policy.kind)"]
    s = Settings.load(station_file(None), tmp_path, mock=True)
    assert s.kind is None and s.active_kind == "hold_still" and s.mock


def test_mock_overrides_kind(station_file, tmp_path):
    s = Settings.load(station_file({"kind": "diffusion"}), tmp_path, mock=True)   # no checkpoint needed
    assert s.kind == "diffusion" and s.active_kind == "hold_still"


def test_unknown_kind(station_file, tmp_path):
    problems = _problems(lambda: Settings.load(station_file({"kind": "replay"}), tmp_path, mock=False))
    assert len(problems) == 1 and "policy.kind: unknown kind 'replay'" in problems[0]


def test_diffusion_without_checkpoint_refused(station_file, tmp_path):
    problems = _problems(lambda: Settings.load(station_file({"kind": "diffusion"}), tmp_path, mock=False))
    assert len(problems) == 1 and "policy.checkpoint: required for kind diffusion" in problems[0]


@pytest.mark.parametrize("key,value", [("diffusion_steps", 0), ("diffusion_steps", 2.5), ("diffusion_steps", True),
                                       ("warmup_inferences", 0), ("warmup_gripper_width_m", -0.01),
                                       ("reset_max_s", 0), ("poll_s", -1), ("self_check_timeout_s", "x"),
                                       ("device", "tpu")])
def test_ranges(station_file, tmp_path, key, value):
    problems = _problems(lambda: Settings.load(station_file({"kind": "hold_still", key: value}), tmp_path,
                                               mock=False))
    assert len(problems) == 1 and f"policy.{key}: expected" in problems[0]


def test_every_problem_listed_at_once(station_file, tmp_path):
    problems = _problems(lambda: Settings.load(station_file({"diffusion_steps": 0, "device": "tpu", "chekpoint": 1}),
                                               tmp_path, mock=False))
    assert len(problems) == 4


def test_relative_checkpoint_from_the_station_folder(station_file, tmp_path):
    s = Settings.load(station_file({"kind": "diffusion", "checkpoint": "models/best.ckpt"}), tmp_path, mock=False)
    assert s.checkpoint == (tmp_path / "models" / "best.ckpt").resolve()
    s = Settings.load(station_file({"kind": "diffusion", "checkpoint": "~/m/best.ckpt"}), tmp_path, mock=False)
    assert s.checkpoint == (Path.home() / "m" / "best.ckpt").resolve()


def test_resolve_path_defaults_from_policy_folder(tmp_path):
    assert resolve_path("data/x.ckpt", from_station=False, station_dir=tmp_path) == POLICY_DIR / "data" / "x.ckpt"
    assert resolve_path("data/x.ckpt", from_station=True, station_dir=tmp_path) == tmp_path / "data" / "x.ckpt"
    assert POLICY_DIR == CONFIG_FILE.parent and (POLICY_DIR / "launch.py").exists()


def test_merge_replaces_non_maps():
    problems: list[str] = []
    out = merge({"a": {"b": 1, "c": [1, 2]}, "d": 3}, {"a": {"c": [9]}, "d": {"x": 1}}, "policy", problems, "s")
    assert out == {"a": {"b": 1, "c": [9]}, "d": {"x": 1}} and problems == []


def test_endpoint_from_the_station_and_missing_is_an_error(station_file, tmp_path):
    s = Settings.load(station_file({"kind": "hold_still"}, endpoint="tcp://127.0.0.1:6000"), tmp_path, mock=False)
    assert s.endpoint == "tcp://127.0.0.1:6000"
    problems = _problems(lambda: Settings.load(station_file({"kind": "hold_still"}, endpoint=None), tmp_path,
                                               mock=False))
    assert problems == ["station.yaml: endpoints.policy: missing"]


def test_session_root_read_only_without_session(station_file, tmp_path):
    path = station_file({"kind": "hold_still"}, session_root=None)
    s = Settings.load(path, tmp_path / "run1", mock=False)                 # --session: no session_root needed
    assert s.session_dir == (tmp_path / "run1" / "policy").resolve()
    problems = _problems(lambda: Settings.load(path, None, mock=False))
    assert len(problems) == 1 and problems[0].startswith("station.yaml: session_root: missing")
    s = Settings.load(station_file({"kind": "hold_still"}, session_root="sessions"), None, mock=False)
    assert s.session_dir.parent.parent == (tmp_path / "sessions").resolve()
    assert s.session_dir.name == "policy" and s.session_dir.parent.name.endswith("_policy_alone")


def test_station_file_missing(tmp_path):
    problems = _problems(lambda: Settings.load(tmp_path / "nope.yaml", None, mock=True))
    assert "file not found" in problems[0]


def test_as_dict_is_json_ready(station_file, tmp_path):
    import json
    s = Settings.load(station_file({"kind": "diffusion", "checkpoint": "b.ckpt"}), tmp_path, mock=False)
    d = json.loads(json.dumps(s.as_dict()))
    assert d["checkpoint"] == str((tmp_path / "b.ckpt").resolve()) and d["active_kind"] == "diffusion"
