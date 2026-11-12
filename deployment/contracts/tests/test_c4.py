import dataclasses

import numpy as np
import pytest

from contracts import (ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS, ActionChunk, ContractLimits, Observation,
                       PolicyConfig, PolicyConfigRequest, Prompt, Reset, decode, encode)
from contracts.c4_policy import OBSERVATION_INPUTS
from contracts.examples import action_chunk, observation
from helpers import contract_error


def chunk(**change):
    return dataclasses.replace(action_chunk(), **change)


def obs(**change):
    return dataclasses.replace(observation(), **change)


def test_sizes():
    assert (OBS_STEPS, ACTION_STEPS, STEP_MS, IMAGE_SIZE) == (2, 16, 50, 224)


def test_shapes_use_the_size_constants():
    assert chunk().eef_pos.shape == (ACTION_STEPS, 3)
    assert chunk().gripper_width_m.shape == (ACTION_STEPS, 1)
    assert obs().main.shape == (OBS_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert obs().eef_rot6d.shape == (OBS_STEPS, 6)
    contract_error(lambda: chunk(eef_pos=np.zeros((ACTION_STEPS - 1, 3), np.float32)), "eef_pos", "shape")
    contract_error(lambda: obs(gripper_width_m=np.zeros((OBS_STEPS, 2), np.float32)), "gripper_width_m", "shape")


def test_requests_and_config():
    for msg in (PolicyConfigRequest(), Reset(), Prompt(task_name="t", prompt_index=2)):
        assert decode(encode(msg)) == msg
    c = PolicyConfig(obs_steps=2, action_steps=16, step_ms=50, image_size=224, inputs=["main", "eef_pos"],
                     uses_prompt=False)
    assert decode(encode(c)).inputs == ("main", "eef_pos")
    assert set(OBSERVATION_INPUTS) == {f.name for f in dataclasses.fields(Observation)}
    contract_error(lambda: dataclasses.replace(c, inputs=("depth",)), "inputs[0]", "value")
    contract_error(lambda: dataclasses.replace(c, step_ms=0), "step_ms", "value")
    contract_error(lambda: dataclasses.replace(c, uses_prompt=0), "uses_prompt", "type")
    contract_error(lambda: Prompt(task_name="t", prompt_index=-1), "prompt_index", "value")


def test_observation_round_trip_and_anchor_rows():
    o = obs()
    assert decode(encode(o)) == o
    pos = o.eef_pos.copy()
    pos[-1, 0] = 2e-5
    contract_error(lambda: obs(eef_pos=pos), "eef_pos", "value")
    pos[-1, 0] = 5e-6  # within anchor_row_tol
    obs(eef_pos=pos)
    rot = o.eef_rot6d.copy()
    rot[-1, 4] = 0.999
    contract_error(lambda: obs(eef_rot6d=rot), "eef_rot6d", "value")


def test_observation_rot6d_and_widths():
    wrt = observation().eef_rot6d_wrt_start.copy()
    wrt[0, :3] = [0.5, 0.0, 0.0]
    contract_error(lambda: obs(eef_rot6d_wrt_start=wrt), "eef_rot6d_wrt_start", "value")
    w = np.array([[-0.001], [0.05]], np.float32)
    contract_error(lambda: obs(gripper_width_m=w), "gripper_width_m", "value")
    nan = observation().eef_pos.copy()
    nan[0, 0] = np.nan
    contract_error(lambda: obs(eef_pos=nan), "eef_pos", "value")


def test_chunk_rot6d_norm_and_perpendicular():
    rot = action_chunk().eef_rot6d.copy()
    rot[3, 3:] = [0.0, 1.09, 0.0]  # within 1 +- 0.1
    chunk(eef_rot6d=rot)
    rot[3, 3:] = [0.0, 1.11, 0.0]
    contract_error(lambda: chunk(eef_rot6d=rot), "eef_rot6d", "value")
    rot = action_chunk().eef_rot6d.copy()
    rot[0] = [1.0, 0.0, 0.0, 0.2, 0.98, 0.0]  # |cos| ~ 0.2
    contract_error(lambda: chunk(eef_rot6d=rot), "eef_rot6d", "value")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_chunk_non_finite(value):
    pos = action_chunk().eef_pos.copy()
    pos[7, 2] = value
    contract_error(lambda: chunk(eef_pos=pos), "eef_pos", "value")


def test_odd_step0_is_accepted():
    far = action_chunk(step0_m=0.05)  # 5 cm from the anchor: odd for message_comms, not a contract rule
    assert decode(encode(far)).eef_pos[0, 0] == np.float32(0.05)


def test_width_range_not_clipped():
    top = ContractLimits.load().gripper_width_max_m
    w = np.full((ACTION_STEPS, 1), 0.04, np.float32)
    w[5, 0] = top + 0.001
    contract_error(lambda: chunk(gripper_width_m=w), "gripper_width_m", "value")
    w[5, 0] = -0.0005
    contract_error(lambda: chunk(gripper_width_m=w), "gripper_width_m", "value")
    w[5, 0] = top  # float32(0.08) is just below 0.08
    w[6, 0] = 0.0
    chunk(gripper_width_m=w)


def test_width_compared_in_float64(monkeypatch):
    """float32(0.07) is 0.0700000003: in float32 it would equal a 0.07 maximum, in float64 it is above it."""
    limits = dataclasses.replace(ContractLimits.load(), gripper_width_max_m=0.07)
    monkeypatch.setattr(ContractLimits, "_cached", limits)
    w = np.full((ACTION_STEPS, 1), 0.04, np.float32)
    w[2, 0] = 0.07
    assert float(w[2, 0]) > 0.07  # the premise
    contract_error(lambda: chunk(gripper_width_m=w), "gripper_width_m", "value")
    w[2, 0] = np.nextafter(np.float32(0.07), np.float32(0.0))  # the largest float32 below 0.07 is accepted
    chunk(gripper_width_m=w)


def test_width_clipped_range():
    assert chunk(width_clipped=ACTION_STEPS).width_clipped == ACTION_STEPS
    contract_error(lambda: chunk(width_clipped=ACTION_STEPS + 1), "width_clipped", "value")
    contract_error(lambda: chunk(width_clipped=-1), "width_clipped", "value")
    d = chunk().to_dict()
    del d["width_clipped"]
    contract_error(lambda: ActionChunk.from_dict(d), "width_clipped", "missing")


def test_chunk_dtype_not_cast():
    contract_error(lambda: chunk(eef_pos=action_chunk().eef_pos.astype(np.float64)), "eef_pos", "dtype")
