"""Franka Hand command logic (pure: stdlib + numpy; no zmq/mujoco/robotmq, so it imports in env_bp too).

Why: the policy streams a gripper width in TRAINING units (iPhUMI tag separation, 0.0447-0.0844 m) at 50 Hz, but the
Franka Hand (libfranka ``Gripper``) takes blocking ``move(width, speed)`` / ``grasp(width, speed, force, eps_inner,
eps_outer)`` calls that last 0.3-1.5 s. This module turns the stream into a few discrete hand commands:

- ``WidthMap``: train = a * gap + b; gap = clip((train - b) / a, gap_min_m, gap_max_m). gap = libfranka width =
  distance between the fingers (the sim: finger_joint1 + finger_joint2 = 2 x the ``split`` tendon length).
- ``HandCommandPolicy``: states OPEN / GRASPING with hysteresis:
    OPEN and w < close_below_train_m        -> grasp(gap(w), force, eps) and GRASPING (not rate limited)
    OPEN, |gap(w) - last sent| > deadband, min_cmd_interval_s elapsed, hand not busy -> move(gap(w))
    GRASPING and w > open_above_train_m     -> move(gap(w)) and OPEN
  Latest wins: only the newest width is ever looked at. No command while the hand state is unknown (None).
  Limits: force_n <= force_max_n <= 70 N, speed_m_s <= 0.1 m/s, widths clipped to the gap range (``clips``).

  GRASPING, hand idle and not holding (grasp_failed, hand_server restarted mid-hold), min_cmd_interval_s elapsed
                                          -> re-grasp, at most max_regrasps times in a row (``regrasps``); then
                                             it waits for the release (w > open_above) like a normal GRASPING.

Defaults (HandPolicyConfig) are derived from the firstreal-single training widths (A5 WORKLOG): hold widths
0.046-0.0618, re-open >= 0.073 -> close below 0.064, open above 0.070 gives exactly one grasp and one release in
45/45 demos. With b = 0.0431 the grasp triggers at gap 0.018-0.021 and the 20 mm eraser stops the pads at gap 0.012
(pad offset 0.008), so eps_inner >= 0.009; 0.015 also keeps the no-object close >= 3 mm above 0. The station/config
values win.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

SPEED_MAX_M_S = 0.1          # Franka Hand datasheet max finger speed (per libfranka width)
FORCE_MAX_N = 70.0           # Franka Hand continuous grasp force


@dataclass
class WidthMap:
    a: float = 1.0
    b: float = 0.0431             # = train_closed_m (tag separation at gap 0); A5 review. Measure with fdcal width
    gap_min_m: float = 0.0
    gap_max_m: float = 0.08

    def gap_raw(self, w_train: float) -> float:
        return (float(w_train) - self.b) / self.a

    def gap(self, w_train: float) -> float:
        return float(np.clip(self.gap_raw(w_train), self.gap_min_m, self.gap_max_m))

    def train(self, gap: float) -> float:
        return self.a * float(gap) + self.b

    @classmethod
    def from_config(cls, d: dict | None) -> "WidthMap":
        d = dict(d or {})
        wm = cls(**{f.name: float(d[f.name]) for f in fields(cls) if f.name in d})
        if wm.a <= 0 or wm.gap_max_m <= wm.gap_min_m:
            raise ValueError(f"width_map: need a > 0 and gap_max_m > gap_min_m, got {wm}")
        return wm


@dataclass
class HandCmd:
    kind: str                     # move | grasp | stop | homing
    width_m: float = 0.0          # libfranka width (gap), m
    speed_m_s: float = 0.1
    force_n: float = 0.0
    eps_inner_m: float = 0.0
    eps_outer_m: float = 0.0

    def to_msg(self, seq: int) -> dict:
        """The I-HAND ``hand_cmd`` request dict."""
        return {"kind": self.kind, "width_m": float(self.width_m), "speed_m_s": float(self.speed_m_s),
                "force_n": float(self.force_n), "eps_inner_m": float(self.eps_inner_m),
                "eps_outer_m": float(self.eps_outer_m), "seq": int(seq)}


@dataclass
class HandPolicyConfig:
    close_below_train_m: float = 0.064
    open_above_train_m: float = 0.070
    move_deadband_m: float = 0.004
    min_cmd_interval_s: float = 0.25
    speed_m_s: float = 0.1
    force_n: float = 30.0
    force_max_n: float = 70.0
    eps_inner_m: float = 0.015
    eps_outer_m: float = 0.08
    max_regrasps: int = 2         # GRASPING but the hand is idle and not holding (grasp_failed, server restart)

    def __post_init__(self):
        if not self.open_above_train_m > self.close_below_train_m:
            raise ValueError(f"hand policy: open_above_train_m ({self.open_above_train_m}) must be > "
                             f"close_below_train_m ({self.close_below_train_m})")

    @classmethod
    def from_config(cls, cfg: dict | None) -> "HandPolicyConfig":
        """From the merged controller ``gripper`` dict: speed_m_s, force_n, force_max_n, grasp_epsilon_inner_m,
        grasp_epsilon_outer_m at the top level, plus an optional ``policy`` dict (any field name)."""
        cfg = dict(cfg or {})
        kw = {}
        for src, dst in (("speed_m_s", "speed_m_s"), ("force_n", "force_n"), ("force_max_n", "force_max_n"),
                         ("grasp_epsilon_inner_m", "eps_inner_m"), ("grasp_epsilon_outer_m", "eps_outer_m")):
            if cfg.get(src) is not None:
                kw[dst] = float(cfg[src])
        names = {f.name for f in fields(cls)}
        for k, v in dict(cfg.get("policy") or {}).items():
            if k not in names:
                raise ValueError(f"gripper.policy: unknown key {k!r} (known: {sorted(names)})")
            kw[k] = int(v) if k == "max_regrasps" else float(v)
        return cls(**kw)


class HandCommandPolicy:
    def __init__(self, cfg: HandPolicyConfig, wm: WidthMap):
        self.cfg, self.wm = cfg, wm
        self.state = "OPEN"
        self.last_sent: float | None = None       # gap of the last command
        self.last_t = -1e9
        self.clips = 0
        self.force_n = float(min(cfg.force_n, cfg.force_max_n, FORCE_MAX_N))
        self.speed_m_s = float(min(cfg.speed_m_s, SPEED_MAX_M_S))
        if self.force_n <= 0 or self.speed_m_s <= 0:
            raise ValueError(f"hand policy: force_n and speed_m_s must be > 0 (got {cfg.force_n}, {cfg.speed_m_s})")
        self._undo = None
        self.regrasps = 0              # total re-grasps
        self._tries = 0                # re-grasps since the last holding / release

    def _gap(self, w: float) -> float:
        raw = self.wm.gap_raw(w)
        g = self.wm.gap(w)
        if g != raw:
            self.clips += 1
        return g

    def _emit(self, cmd: HandCmd, now: float, state: str) -> HandCmd:
        self._undo = (self.state, self.last_sent, self.last_t)
        self.state, self.last_sent, self.last_t = state, cmd.width_m, now
        return cmd

    def update(self, w_train: float, now: float, hand_state: dict | None) -> HandCmd | None:
        if hand_state is None or w_train is None or not np.isfinite(w_train):
            return None
        c = self.cfg
        if self.state == "OPEN":
            if w_train < c.close_below_train_m:
                g = self._gap(w_train)
                return self._emit(HandCmd("grasp", g, self.speed_m_s, self.force_n, c.eps_inner_m, c.eps_outer_m),
                                  now, "GRASPING")
            g = self.wm.gap(w_train)
            if ((self.last_sent is None or abs(g - self.last_sent) > c.move_deadband_m)
                    and now - self.last_t >= c.min_cmd_interval_s and not hand_state.get("busy", False)):
                return self._emit(HandCmd("move", self._gap(w_train), self.speed_m_s), now, "OPEN")
            return None
        if w_train > c.open_above_train_m:                       # GRASPING: release (not rate limited)
            self._tries = 0
            return self._emit(HandCmd("move", self._gap(w_train), self.speed_m_s), now, "OPEN")
        if hand_state.get("mode") == "holding":
            self._tries = 0
        elif (not hand_state.get("busy", False) and self._tries < int(c.max_regrasps)
              and now - self.last_t >= c.min_cmd_interval_s):
            self._tries += 1
            self.regrasps += 1
            g = self._gap(w_train)
            return self._emit(HandCmd("grasp", g, self.speed_m_s, self.force_n, c.eps_inner_m, c.eps_outer_m),
                              now, "GRASPING")
        return None

    def send_failed(self) -> None:
        """The last command did not reach the hand: restore the state before it, so it is re-issued."""
        if self._undo is not None:
            self.state, self.last_sent, self.last_t = self._undo
            self._undo = None
            if self.state == "GRASPING" and self._tries > 0:     # an undelivered re-grasp does not count
                self._tries -= 1
                self.regrasps -= 1
