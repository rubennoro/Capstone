"""Gripper driver interface for the motorized iPhUMI gripper.

Width units are the TRAINING units: metres, 0 = closed (AR-tag distance minus the calibration minimum). The driver
maps them to actuator positions with a monotonic calibration table (measured on hardware with the same tag
detector, B4) and its inverse.

Squeeze offset delta: the demos' grasp width equals the object width (~0.045 m), so position control to exactly that
barely touches the object. When the commanded width is below `squeeze_below_m`, the driver targets `w - delta` with a
current limit. This happens INSIDE the driver only: `commanded_width()` returns the policy's width, which is what the
controller publishes as its joint target and what the task stores as action history. The offset never enters the
history (0.040 m would be below the training minimum 0.0447 m, i.e. out of distribution).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class WidthCalibration:
    """Monotonic table width_m <-> actuator position (any units). Linear interpolation, clamped at the ends."""

    width_m: np.ndarray
    position: np.ndarray

    def __post_init__(self):
        self.width_m = np.asarray(self.width_m, float)
        self.position = np.asarray(self.position, float)
        if self.width_m.shape != self.position.shape or len(self.width_m) < 2:
            raise ValueError("calibration table needs >= 2 matching points")
        if not np.all(np.diff(self.width_m) > 0):
            raise ValueError("width_m must be strictly increasing")
        d = np.diff(self.position)
        if not (np.all(d > 0) or np.all(d < 0)):
            raise ValueError("position must be strictly monotonic in width")
        self._inc = bool(d[0] > 0)

    def to_position(self, w: float) -> float:
        return float(np.interp(w, self.width_m, self.position))

    def to_width(self, pos: float) -> float:
        if self._inc:
            return float(np.interp(pos, self.position, self.width_m))
        return float(np.interp(pos, self.position[::-1], self.width_m[::-1]))

    @classmethod
    def from_config(cls, cfg: dict) -> "WidthCalibration":
        return cls(np.array(cfg["width_m"]), np.array(cfg["position"]))


class GripperDriver:
    """Base class: width in training units, delta + current limit applied internally."""

    def __init__(self, calibration: WidthCalibration, squeeze_delta_m: float = 0.005, squeeze_below_m: float = 0.06,
                 current_limit: float | None = None, min_width_m: float = 0.0, max_width_m: float = 0.11):
        self.cal = calibration
        self.delta = float(squeeze_delta_m)
        self.squeeze_below = float(squeeze_below_m)
        self.min_w, self.max_w = float(min_width_m), float(max_width_m)
        self._commanded = None
        if current_limit is not None:
            self.set_current_limit(current_limit)

    # --- public API
    def set_width(self, width_m: float) -> None:
        w = float(np.clip(width_m, self.min_w, self.max_w))
        self._commanded = w
        target = w - self.delta if w < self.squeeze_below else w
        self._write_position(self.cal.to_position(max(self.min_w, target)))

    def commanded_width(self) -> float | None:
        """The policy's width (never includes delta)."""
        return self._commanded

    def get_width(self) -> float:
        return self.cal.to_width(self._read_position())

    def set_current_limit(self, limit: float) -> None:
        self._write_current_limit(float(limit))

    def close(self) -> None:
        pass

    # --- hardware hooks
    def _write_position(self, pos: float) -> None:
        raise NotImplementedError

    def _read_position(self) -> float:
        raise NotImplementedError

    def _write_current_limit(self, limit: float) -> None:
        raise NotImplementedError


class FakeGripper(GripperDriver):
    """Simulated servo: moves toward the target at `speed` (position units/s); stalls at `object_width_m` when closing
    past it (the fingers are blocked by the object), as a current-limited servo would."""

    def __init__(self, calibration: WidthCalibration, object_width_m: float | None = None, speed: float = 2000.0,
                 start_width_m: float = 0.08, **kw):
        self._pos = calibration.to_position(start_width_m)
        self._target = self._pos
        self._limit = None
        self.speed = float(speed)
        self.object_width_m = object_width_m
        self._t = time.monotonic()
        self._lock = threading.Lock()
        super().__init__(calibration, **kw)

    def _advance(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        step = self.speed * dt
        self._pos += float(np.clip(self._target - self._pos, -step, step))
        if self.object_width_m is not None and self.cal.to_width(self._pos) < self.object_width_m:
            self._pos = self.cal.to_position(self.object_width_m)  # blocked by the object

    def _write_position(self, pos):
        with self._lock:
            self._advance()
            self._target = pos

    def _read_position(self):
        with self._lock:
            self._advance()
            return self._pos

    def _write_current_limit(self, limit):
        self._limit = limit

    def target_position(self) -> float:
        return self._target


class FeetechGripper(GripperDriver):  # pragma: no cover - hardware
    """Feetech STS3215 on a USB serial bus (scservo_sdk). Hardware only: position = raw 0..4095 ticks,
    current limit via the torque-limit register. Not exercised without hardware."""

    def __init__(self, calibration, port="/dev/ttyUSB0", servo_id=1, baudrate=1_000_000, **kw):
        try:
            import scservo_sdk  # noqa: F401
        except ImportError as e:
            raise RuntimeError("FeetechGripper needs scservo_sdk (pip install feetech-servo-sdk)") from e
        raise NotImplementedError("FeetechGripper: implement on hardware (B4); see docs/panda_py_checklist.md")


class DynamixelGripper(GripperDriver):  # pragma: no cover - hardware
    """Dynamixel XM430 in current-based position control (dynamixel_sdk). Hardware only."""

    def __init__(self, calibration, port="/dev/ttyUSB0", servo_id=1, baudrate=57600, **kw):
        try:
            import dynamixel_sdk  # noqa: F401
        except ImportError as e:
            raise RuntimeError("DynamixelGripper needs dynamixel_sdk") from e
        raise NotImplementedError("DynamixelGripper: implement on hardware (B4)")


def make_gripper(cfg: dict) -> GripperDriver:
    if cfg.get("type") == "franka_hand":   # Franka Hand (A5): non-blocking I-HAND client; no servo calibration
        from franka_deploy.gripper.franka_hand import FrankaHandDriver

        return FrankaHandDriver.from_config(cfg)
    cal = WidthCalibration.from_config(cfg["calibration"])
    kind = cfg.get("type", "fake")
    kw = {k: cfg[k] for k in ("squeeze_delta_m", "squeeze_below_m", "current_limit", "min_width_m", "max_width_m")
          if k in cfg}
    if kind == "fake":
        return FakeGripper(cal, object_width_m=cfg.get("object_width_m"), start_width_m=cfg.get("start_width_m", 0.08),
                           speed=cfg.get("speed", 2000.0), **kw)
    if kind == "feetech":
        return FeetechGripper(cal, **cfg.get("port_opts", {}), **kw)
    if kind == "dynamixel":
        return DynamixelGripper(cal, **cfg.get("port_opts", {}), **kw)
    raise ValueError(f"unknown gripper type {kind!r}")
