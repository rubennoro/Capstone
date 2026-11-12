from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch tests run in the policy environment")

from contracts import ActionChunk, ContractLimits, Observation, decode, encode  # noqa: E402

from conftest import plain_diffusion_shape_meta  # noqa: E402
from fakes_torch import TinyDiffusionModel  # noqa: E402
from policy.diffusion_policy import DiffusionPolicy  # noqa: E402
from policy.errors import PolicyLoadError  # noqa: E402
from policy.key_map import KeyMap  # noqa: E402
from policy.model_runner import ModelRunner  # noqa: E402
from policy.neutral_observation import neutral_observation  # noqa: E402
from policy.settings import CONFIG_FILE  # noqa: E402

EXPECTED = {
    "gripper_right_eef_pos": (1, 2, 3),
    "gripper_right_eef_rot_axis_angle": (1, 2, 6),
    "gripper_right_gripper_width": (1, 2, 1),
    "camera_right_main_rgb": (1, 2, 3, 224, 224),
    "camera_right_ultrawide_rgb": (1, 2, 3, 224, 224),
}


def _key_map() -> KeyMap:
    import yaml
    return KeyMap.from_config(yaml.safe_load(CONFIG_FILE.read_text())["key_map"])


def _policy(shape_meta=None, horizon: int = 16, width: float = 0.05):
    shape_meta = shape_meta or plain_diffusion_shape_meta()
    shape_meta["action"]["horizon"] = horizon
    model = TinyDiffusionModel(EXPECTED, horizon=horizon, width=width)
    runner = ModelRunner(model, shape_meta, _key_map(), torch.device("cpu"))
    return DiffusionPolicy(runner, model.reset, ContractLimits.load().gripper_width_max_m), model


def _obs_with_pixel(y: int, x: int, c: int, t: int, value: int) -> Observation:
    obs = neutral_observation(0.05)
    fields = {k: v for k, v in obs.to_dict().items() if k != "schema"}
    main = np.array(fields["main"])
    main[t, y, x, c] = value
    fields["main"] = main
    return Observation(**fields)


def test_chunk_shapes_dtypes_contiguous_and_valid():
    policy, _ = _policy()
    chunk = policy.infer(neutral_observation(0.05))
    assert isinstance(chunk, ActionChunk)
    for array, shape in ((chunk.eef_pos, (16, 3)), (chunk.eef_rot6d, (16, 6)), (chunk.gripper_width_m, (16, 1))):
        assert array.shape == shape and array.dtype == np.float32 and array.flags["C_CONTIGUOUS"]
    decode(encode(chunk), ActionChunk)
    assert chunk.width_clipped == 0


def test_model_gets_exactly_the_used_keys():
    policy, model = _policy()
    policy.infer(neutral_observation(0.05))
    assert set(model.received[0]) == set(EXPECTED)                    # no wrt_start: ignore_by_policy
    assert policy.inputs() == ("eef_pos", "eef_rot6d", "gripper_width_m", "main", "ultrawide")
    assert list(policy.config().inputs) == list(policy.inputs())


def test_known_pixel_arrives_at_the_right_place_bit_identical():
    policy, model = _policy()
    y, x, c, t, v = 17, 201, 2, 1, 173
    obs = _obs_with_pixel(y, x, c, t, v)
    policy.infer(obs)
    got = model.received[0]["camera_right_main_rgb"].numpy()
    assert got[0, t, c, y, x] == np.float32(v) / np.float32(255.0)
    reference = np.moveaxis(np.array(obs.main), -1, 1).astype(np.float32) / 255.0     # detached_inference
    np.testing.assert_array_equal(got[0], reference)
    assert np.count_nonzero(got) == 1


def test_horizon_20_keeps_the_first_16_rows():
    policy, _ = _policy(horizon=20)
    chunk = policy.infer(neutral_observation(0.05))
    np.testing.assert_allclose(chunk.eef_pos[:, 0], np.arange(16, dtype=np.float32) * np.float32(0.001))


def test_horizon_12_refused_at_construction():
    with pytest.raises(PolicyLoadError) as e:
        _policy(horizon=12)
    assert "model action horizon 12 < 16 (C4 action_steps)" in e.value.what


def test_out_of_range_width_clipped_and_counted():
    policy, _ = _policy(width=0.095)
    chunk = policy.infer(neutral_observation(0.05))
    assert chunk.width_clipped == 16 and (chunk.gripper_width_m == policy.width_max).all()


def test_reset_calls_the_model_and_is_fast():
    policy, model = _policy()
    t0 = time.monotonic()
    policy.reset()
    assert time.monotonic() - t0 < 0.5 and model.reset_calls == 1


def test_runner_rejects_112_pixel_images():
    meta = plain_diffusion_shape_meta()
    meta["obs"]["camera_right_main_rgb"]["shape"] = [3, 112, 112]
    with pytest.raises(PolicyLoadError) as e:
        ModelRunner(TinyDiffusionModel(EXPECTED), meta, _key_map(), torch.device("cpu"))
    assert "camera_right_main_rgb" in e.value.what and "never resized" in e.value.what


def test_runner_rejects_wrong_obs_horizon():
    meta = plain_diffusion_shape_meta()
    meta["obs"]["gripper_right_eef_pos"]["horizon"] = 3
    with pytest.raises(PolicyLoadError):
        ModelRunner(TinyDiffusionModel(EXPECTED), meta, _key_map(), torch.device("cpu"))


def test_describe_has_key_map_and_inputs():
    policy, _ = _policy()
    d = policy.describe()
    assert d["kind"] == "diffusion" and d["inputs"][0] == "eef_pos" and d["key_map"]["obs"]["main"]
