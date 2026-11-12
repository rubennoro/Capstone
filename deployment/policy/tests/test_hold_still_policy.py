from __future__ import annotations

import time

import numpy as np
from contracts import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS, ActionChunk, ContractLimits, decode, encode

from policy.hold_still_policy import HoldStillPolicy
from policy.neutral_observation import neutral_observation


def _policy() -> HoldStillPolicy:
    return HoldStillPolicy(ContractLimits.load().gripper_width_max_m)


def test_reply_passes_the_validator_and_survives_encode_decode():
    chunk = _policy().infer(neutral_observation(0.05))
    assert isinstance(chunk, ActionChunk)
    again = decode(encode(chunk), ActionChunk)
    np.testing.assert_array_equal(again.eef_pos, chunk.eef_pos)
    assert again.width_clipped == 0


def test_identity_chunk_with_the_newest_width():
    obs = neutral_observation(0.05)
    width = obs.gripper_width_m.copy()
    width[0, 0], width[1, 0] = 0.01, 0.034
    obs = type(obs)(**{**{k: v for k, v in obs.to_dict().items() if k != "schema"}, "gripper_width_m": width})
    chunk = _policy().infer(obs)
    np.testing.assert_array_equal(chunk.eef_pos, np.zeros((ACTION_STEPS, 3), np.float32))
    np.testing.assert_array_equal(chunk.eef_rot6d, np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32),
                                                           (ACTION_STEPS, 1)))
    np.testing.assert_array_equal(chunk.gripper_width_m, np.full((ACTION_STEPS, 1), 0.034, np.float32))
    for a in (chunk.eef_pos, chunk.eef_rot6d, chunk.gripper_width_m):
        assert a.dtype == np.float32


def test_width_above_max_clipped_and_counted():
    policy = _policy()
    chunk = policy.infer(neutral_observation(ContractLimits.load().gripper_width_max_m + 0.005))
    assert chunk.width_clipped == ACTION_STEPS
    assert (chunk.gripper_width_m == policy.width_max).all()


def test_config():
    config = _policy().config()
    assert (config.obs_steps, config.action_steps, config.step_ms, config.image_size) == (
        OBS_STEPS, ACTION_STEPS, STEP_MS, IMAGE_SIZE)
    assert config.uses_prompt is False
    assert list(config.inputs) == ["gripper_width_m"]


def test_reset_is_fast():
    policy = _policy()
    t0 = time.monotonic()
    policy.reset()
    assert time.monotonic() - t0 < 0.5
    assert policy.describe() == {"kind": "hold_still"}
