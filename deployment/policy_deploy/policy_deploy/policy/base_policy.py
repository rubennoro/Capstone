# adapted from rnoro/deployment:deployment/policy/policy/base_policy.py @ 91b178b
# changes: config() fills kind = KIND (c4.config.reply/v2); memory_stats() for startup.json (empty here, the torch
# policy reports the CUDA allocator's peaks); imports
# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/policy/base_policy.py @ f632576
# changes: make_chunk orthonormalizes eef_rot6d (Gram-Schmidt, the manager's own pose_math.matrix_from_rot6d =
#          behavior_prompting/UMI rot6d_to_mat) before the C4 check, and logs chunks the old check would have refused
#          (sim_deploy, 2026-11-22; handoff: sim_deploy/ports/rot6d_gram_schmidt.md); the Gram-Schmidt is a local
#          copy of pose_math's (policy may not import manager: tests/test_import_rules.py), 2026-11-23
"""Policy: the interface every policy implements, and the shared chunk builder that clips widths."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from policy_deploy.contracts import ACTION_STEPS, IMAGE_SIZE, OBS_STEPS, STEP_MS, ActionChunk, Observation, PolicyConfig

log = logging.getLogger(__name__)

# C4's rot6d_dot_max (contracts/c4_limits.yaml): only used to label the log line of a chunk the raw check would refuse
_RAW_DOT_MAX = 0.1
_EPS = 1e-12      # manager/pose_math.py _EPS (the reference normalize()'s eps: only guards a division by zero)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def _gram_schmidt_rows(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 6): the rows (b1, b2) of manager/pose_math.matrix_from_rot6d, same math (kept in step by
    hand: the policy process may not import the manager's)."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    return np.concatenate((b1, b2), axis=-1)


def orthonormalize_rot6d(rot6d: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(N, 6) model rot6d -> (N, 6) float32 rows of an exact rotation, as the manager reads them.

    Gram-Schmidt exactly as behavior_prompting's rot6d_to_mat / UMI (the manager's matrix_from_rot6d; here
    _gram_schmidt_rows): b1 = a1/|a1|, b2 = normalized(a2 - (b1.a2) b1); the rows (b1, b2) are returned.
    matrix_from_rot6d of the result equals matrix_from_rot6d of the input, so the robot's motion is unchanged for
    every chunk the raw check accepted.
    Degenerate rows (a zero vector, parallel vectors, NaN/inf) are NOT repaired: they come out with a zero-length or
    non-finite vector, which the C4 validator still refuses (PolicyFatal), as before.
    Also returns the raw rows' worst |cos angle| and worst | |v| - 1 |, for the log."""
    raw = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    a1, a2 = raw[:, :3], raw[:, 3:]
    n1, n2 = np.linalg.norm(a1, axis=1), np.linalg.norm(a2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.abs(np.sum(a1 * a2, axis=1)) / (n1 * n2)
    finite = np.isfinite(cos)
    worst_cos = float(cos[finite].max()) if finite.any() else float("nan")
    worst_len = float(np.abs(np.concatenate([n1, n2]) - 1.0).max()) if raw.size else 0.0
    fixed = _gram_schmidt_rows(raw)
    return np.ascontiguousarray(fixed, dtype=np.float32).reshape(np.shape(rot6d)), worst_cos, worst_len

if TYPE_CHECKING:
    from policy_deploy.policy.settings import Settings


def float32_max_at_most(limit: float) -> np.float32:
    """The largest float32 value <= `limit`.

    The width is checked again in float64 downstream (the PC's gate, and the C4 validator), so a clipped float32 width
    must not round above the limit: np.float32(0.07) is 0.0700000003 > 0.07.
    """
    value = np.float32(limit)
    if float(value) > limit:
        value = np.nextafter(value, np.float32(0))
    return value


class Policy(ABC):
    KIND: ClassVar[str]
    USES_PROMPT: ClassVar[bool] = False

    def __init__(self, gripper_width_max_m: float) -> None:
        self._width_max = float32_max_at_most(gripper_width_max_m)

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: "Settings") -> "Policy":
        """Build it (and load what it needs). Raises PolicyLoadError."""

    @abstractmethod
    def inputs(self) -> tuple[str, ...]:
        """The Observation field names it actually reads."""

    @abstractmethod
    def reset(self) -> None:
        """Clear episode state; must take < reset_max_s."""

    @abstractmethod
    def infer(self, obs: Observation) -> ActionChunk:
        """One chunk. Raises BadRequest for a problem caused by the request; anything else is its own failure."""

    def config(self) -> PolicyConfig:
        return PolicyConfig(obs_steps=OBS_STEPS, action_steps=ACTION_STEPS, step_ms=STEP_MS, image_size=IMAGE_SIZE,
                            inputs=tuple(self.inputs()), uses_prompt=self.USES_PROMPT, kind=self.KIND)

    def describe(self) -> dict[str, Any]:
        """Details for startup.json."""
        return {"kind": self.KIND}

    def memory_stats(self) -> dict[str, Any]:
        """Device memory after the warm-up, for startup.json ({} without a GPU model)."""
        return {}

    @property
    def width_max(self) -> np.float32:
        """The clip value: the largest float32 <= the max opening."""
        return self._width_max

    def make_chunk(self, eef_pos: np.ndarray, eef_rot6d: np.ndarray, gripper_width_m: np.ndarray) -> ActionChunk:
        """Orthonormalize the rot6d rows (Gram-Schmidt, as the manager and UMI read them), clip the widths into
        [0, max opening], count the clipped steps, build (and so validate) the chunk."""
        rot6d, worst_cos, worst_len = orthonormalize_rot6d(eef_rot6d)
        if not (worst_cos <= _RAW_DOT_MAX and worst_len <= 0.1):      # also logs a NaN (refused just below)
            log.info("rot6d orthonormalized: raw max |cos angle| %.4f, max ||v|-1| %.4f (the raw C4 check would "
                     "have refused this chunk)", worst_cos, worst_len)
        clipped = np.clip(gripper_width_m, np.float32(0), self._width_max)
        n = int(np.count_nonzero(clipped != gripper_width_m))
        return ActionChunk(eef_pos=eef_pos, eef_rot6d=rot6d, gripper_width_m=clipped, width_clipped=n)
