"""StateHistory: the last history_s of C2 states, and the pose at any time inside it (STRUCTURE.md 4.3, 6.1).

Interpolation as in robot_utils interpolate_xyz_wxyz: linear position, scipy Slerp rotation.
"""
from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np

from contracts import ArmState, ContractError
from message_comms import pose_math

SCHEMA = "c2.state/v1"
NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class PoseSample:
    t_ns: int
    T: np.ndarray              # (4, 4) float64, base frame
    width_m: float


class LookupMiss(Enum):
    EMPTY = "no state yet"
    TOO_OLD = "image older than the state history"
    TOO_NEW = "image newer than the newest state"
    SEQ_GAP = "missed state message around the image time"


@dataclass(frozen=True)
class _Sample:
    t_ns: int
    target: np.ndarray
    measured: np.ndarray
    width_m: float
    gap_before: bool           # a C2 seq gap between the previous sample and this one


class StateHistory:
    """Written by the listener thread, read by the worker and the main thread; one lock, copies out."""

    def __init__(self, history_s: float) -> None:
        self._history_ns = int(history_s * NS_PER_S)
        self._lock = threading.Lock()
        self._samples: list[_Sample] = []
        self._times: list[int] = []
        self._latest: ArmState | None = None
        self._gaps = 0
        self._gap_pending = False

    def add(self, state: ArmState) -> None:
        with self._lock:
            last = self._latest
            if last is not None:
                if state.seq <= last.seq:
                    raise ContractError(SCHEMA, "seq", "value", f"seq {state.seq} after {last.seq}: must increase")
                if state.t_ns < last.t_ns:
                    raise ContractError(SCHEMA, "t_ns", "value",
                                        f"t_ns {state.t_ns} after {last.t_ns}: must not go back")
                if state.seq > last.seq + 1:
                    self._gaps += 1
                    self._gap_pending = True
            self._latest = state
            if self._times and state.t_ns == self._times[-1]:
                return                      # a repeated reading: seen in latest(), no interpolation sample
            self._samples.append(_Sample(int(state.t_ns), np.array(state.O_T_TCP_target, dtype=np.float64),
                                         np.array(state.O_T_TCP, dtype=np.float64), float(state.gripper_width_m),
                                         self._gap_pending))
            self._times.append(int(state.t_ns))
            self._gap_pending = False
            oldest_kept = bisect.bisect_left(self._times, state.t_ns - self._history_ns)
            if oldest_kept:
                del self._samples[:oldest_kept]
                del self._times[:oldest_kept]

    def latest(self) -> ArmState | None:
        with self._lock:
            return self._latest

    def gap_count(self) -> int:
        with self._lock:
            return self._gaps

    def pose_at(self, t_ns: int, source: str) -> PoseSample | LookupMiss:
        """source: "target" (O_T_TCP_target) or "measured" (O_T_TCP). Never extrapolates."""
        if source not in ("target", "measured"):
            raise ValueError(f"pose source {source!r}")
        with self._lock:
            if not self._samples:
                return LookupMiss.EMPTY
            if t_ns < self._times[0]:
                return LookupMiss.TOO_OLD
            if t_ns > self._times[-1]:
                return LookupMiss.TOO_NEW
            i = bisect.bisect_right(self._times, t_ns) - 1
            s0 = self._samples[i]
            if s0.t_ns == t_ns:
                return PoseSample(int(t_ns), getattr(s0, source).copy(), s0.width_m)
            s1 = self._samples[i + 1]
            if s1.gap_before:
                return LookupMiss.SEQ_GAP
        a = (t_ns - s0.t_ns) / (s1.t_ns - s0.t_ns)
        T = pose_math.interpolate(getattr(s0, source), getattr(s1, source), a)
        return PoseSample(int(t_ns), T, (1.0 - a) * s0.width_m + a * s1.width_m)
