from __future__ import annotations

import numpy as np
from contracts import IMAGE_SIZE, OBS_STEPS, Observation, decode, encode

from policy.key_map import OBS_FIELDS
from policy.neutral_observation import neutral_observation


def test_passes_the_c4_validator():
    obs = neutral_observation(0.05)
    assert isinstance(obs, Observation)
    assert isinstance(decode(encode(obs), Observation), Observation)


def test_anchor_rows_exact_and_images_black():
    obs = neutral_observation(0.042)
    np.testing.assert_array_equal(obs.eef_pos, np.zeros((OBS_STEPS, 3), np.float32))
    np.testing.assert_array_equal(obs.eef_rot6d, np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (OBS_STEPS, 1)))
    np.testing.assert_array_equal(obs.eef_rot6d_wrt_start, obs.eef_rot6d)
    np.testing.assert_array_equal(obs.gripper_width_m, np.full((OBS_STEPS, 1), 0.042, np.float32))
    for image in (obs.main, obs.ultrawide):
        assert image.shape == (OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3) and image.dtype == np.uint8
        assert not image.any()


def test_key_map_field_names_match_the_contract():
    keys = [k for k in neutral_observation(0.05).to_dict() if k != "schema"]
    assert tuple(keys) == OBS_FIELDS
