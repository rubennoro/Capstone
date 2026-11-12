from __future__ import annotations

import numpy as np
from contracts import ActionChunk, ContractLimits, decode, encode

from fakes import identity_arrays, odd_arrays
from policy.base_policy import float32_max_at_most
from policy.hold_still_policy import HoldStillPolicy


def _policy(max_m: float | None = None) -> HoldStillPolicy:
    return HoldStillPolicy(ContractLimits.load().gripper_width_max_m if max_m is None else max_m)


def test_widths_in_range_unchanged():
    pos, rot, width = identity_arrays(0.03)
    width[3, 0] = 0.0
    chunk = _policy().make_chunk(pos, rot, width)
    assert chunk.width_clipped == 0
    np.testing.assert_array_equal(chunk.gripper_width_m, width)


def test_out_of_range_widths_clipped_and_counted():
    policy = _policy()
    pos, rot, width = identity_arrays(0.03)
    width[0, 0], width[5, 0], width[6, 0] = -0.001, 0.085, 0.085
    chunk = policy.make_chunk(pos, rot, width)
    assert chunk.width_clipped == 3
    assert chunk.gripper_width_m[0, 0] == 0
    assert chunk.gripper_width_m[5, 0] == chunk.gripper_width_m[6, 0] == policy.width_max
    assert chunk.gripper_width_m.dtype == np.float32
    np.testing.assert_array_equal(np.delete(chunk.gripper_width_m, [0, 5, 6]), np.float32(0.03))
    assert isinstance(chunk, ActionChunk)
    assert decode(encode(chunk), ActionChunk).width_clipped == 3          # passes the C4 validator both ways


def test_max_comes_from_contract_limits():
    limit = ContractLimits.load().gripper_width_max_m
    assert _policy().width_max == float32_max_at_most(limit)
    assert float(_policy().width_max) <= limit


def test_max_of_007_steps_down_below_the_float32_rounding():
    assert float(np.float32(0.07)) > 0.07                                 # why the step down is needed
    policy = _policy(0.07)
    pos, rot, width = identity_arrays(0.05)
    width[2, 0] = 0.09
    chunk = policy.make_chunk(pos, rot, width)
    expected = np.nextafter(np.float32(0.07), np.float32(0))
    assert chunk.gripper_width_m[2, 0] == expected and float(chunk.gripper_width_m[2, 0]) <= 0.07
    assert chunk.width_clipped == 1


def test_max_of_008_needs_no_step_down():
    assert float(np.float32(0.08)) <= 0.08
    assert _policy(0.08).width_max == np.float32(0.08)


def test_odd_chunk_returned_unchanged_and_valid():
    pos, rot, width = odd_arrays(0.05, 10.0)
    chunk = _policy().make_chunk(pos, rot, width)
    np.testing.assert_array_equal(chunk.eef_pos, pos)
    np.testing.assert_array_equal(chunk.eef_rot6d, rot)
    assert chunk.width_clipped == 0
