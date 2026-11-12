"""Test-only policies, a fake clock, a serve thread and the torch import blocker (no torch here)."""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from contracts import ACTION_STEPS, ActionChunk, ContractLimits, Observation

from policy.base_policy import Policy
from policy.promptable_policy import PromptablePolicy

ROT6D_IDENTITY = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
BLOCKED_MODULES = ("torch", "behavior_prompting", "hydra", "dill", "timm", "diffusers")

# Python source run first in a subprocess: every import of a blocked module raises ImportError.
BLOCKER_PRELUDE = f"""
import importlib.abc, sys
BLOCKED = {BLOCKED_MODULES!r}
class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked for this test: {{name}}", name=name)
        return None
sys.meta_path.insert(0, _Blocker())
"""


def width_max() -> float:
    return ContractLimits.load().gripper_width_max_m


def identity_arrays(width: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((ACTION_STEPS, 3), np.float32), np.tile(ROT6D_IDENTITY, (ACTION_STEPS, 1)),
            np.full((ACTION_STEPS, 1), width, np.float32))


class FakeClock:
    """A monotonic_ns stand-in: advances only when told (or by `step_ns` per call)."""

    def __init__(self, step_ns: int = 0) -> None:
        self.t_ns = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.t_ns += self.step_ns
        return self.t_ns


class _TestPolicy(Policy):
    KIND = "test"

    def __init__(self) -> None:
        super().__init__(width_max())
        self.resets = 0
        self.infers = 0

    @classmethod
    def from_settings(cls, settings):
        return cls()

    def inputs(self) -> tuple[str, ...]:
        return ("gripper_width_m",)

    def reset(self) -> None:
        self.resets += 1

    def infer(self, obs: Observation) -> ActionChunk:
        self.infers += 1
        return self.make_chunk(*identity_arrays(float(obs.gripper_width_m[-1, 0])))


class FakePromptablePolicy(PromptablePolicy):
    """Records every prompt and reset call; replies like HoldStillPolicy."""
    KIND = "fake_promptable"

    def __init__(self) -> None:
        super().__init__(width_max())
        self.calls: list[tuple] = []

    @classmethod
    def from_settings(cls, settings):
        return cls()

    def inputs(self) -> tuple[str, ...]:
        return ("gripper_width_m", "main")

    def reset(self) -> None:
        self.calls.append(("reset",))

    def prompt(self, task_name: str, prompt_index: int) -> None:
        self.calls.append(("prompt", task_name, prompt_index))

    def infer(self, obs: Observation) -> ActionChunk:
        self.calls.append(("infer",))
        return self.make_chunk(*identity_arrays(float(obs.gripper_width_m[-1, 0])))


class RaisingPolicy(_TestPolicy):
    """infer raises `make_error()` (default RuntimeError: an own failure)."""

    def __init__(self, make_error: Callable[[], Exception] = lambda: RuntimeError("model exploded")) -> None:
        super().__init__()
        self._make_error = make_error

    def infer(self, obs: Observation) -> ActionChunk:
        self.infers += 1
        raise self._make_error()


class BadChunkPolicy(_TestPolicy):
    """Returns NaN positions: its own reply breaks C4 (the ActionChunk validator refuses it)."""

    def infer(self, obs: Observation) -> ActionChunk:
        self.infers += 1
        pos, rot, width = identity_arrays()
        pos[0, 0] = np.nan
        return self.make_chunk(pos, rot, width)


class NonUnitRot6dPolicy(_TestPolicy):
    """A chunk with non-unit rot6d rows (a C4 value failure)."""

    def infer(self, obs: Observation) -> ActionChunk:
        self.infers += 1
        pos, rot, width = identity_arrays()
        return self.make_chunk(pos, rot * np.float32(2.0), width)


class WrongShapePolicy(_TestPolicy):
    """A chunk with 15 steps instead of 16 (a C4 shape failure)."""

    def infer(self, obs: Observation) -> ActionChunk:
        self.infers += 1
        pos, rot, width = identity_arrays()
        return self.make_chunk(pos[:-1], rot[:-1], width[:-1])


def odd_arrays(distance_m: float = 0.05, angle_deg: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A valid chunk whose step 0 is `distance_m` / `angle_deg` (about z) from the anchor."""
    pos, _, width = identity_arrays()
    pos[:, 0] = np.float32(distance_m)
    a = np.deg2rad(angle_deg)
    rot = np.tile(np.array([np.cos(a), -np.sin(a), 0, np.sin(a), np.cos(a), 0], np.float32), (ACTION_STEPS, 1))
    return pos, rot, width


class OddChunkPolicy(_TestPolicy):
    """A valid chunk whose step 0 is 5 cm / 10 degrees from the anchor: not the policy's failure."""

    def infer(self, obs: Observation) -> ActionChunk:
        self.infers += 1
        return self.make_chunk(*odd_arrays())


class ServeThread:
    """Runs PolicyServer.serve in a background thread and keeps what it raised."""

    def __init__(self, server, poll_s: float = 0.01) -> None:
        self.server = server
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(poll_s,), daemon=True)

    def _run(self, poll_s: float) -> None:
        try:
            self.server.serve(self.stop, poll_s)
        except BaseException as e:      # kept for the test to inspect
            self.error = e

    def __enter__(self) -> "ServeThread":
        self._thread.start()
        return self

    def join(self, timeout_s: float) -> bool:
        self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self._thread.join(5.0)
        self.server.close()
