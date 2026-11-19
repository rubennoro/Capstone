# copied from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/fake_pc/settings.py @ 2ffe641
# changes: imports only
"""Settings of the fake PC: config/fake_pc.yaml + station.yaml fake_pc: section, plus the contract files
(STRUCTURE.md 10). The ports, the watchdog limit, the limits and F_T_TCP come only from deployment/contracts/."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from deploy_contracts import MODES, ContractLimits, LinkSettings, TcpFrame, rigid_problem

from teleop_collect.contracts import Station, StationError, load_config

START_MODES = ("idle", "holding", "running", "stopped", "homing", "fault")


@dataclass(frozen=True, eq=False)
class FakePcSettings:
    clock_offset_s: float
    clock_drift_ppm: float
    state_rate_hz: float
    state_send_hwm: int
    homing_timeout_s: float
    max_lead_s: float
    home_duration_s: float
    home_pose: np.ndarray
    start_pose: np.ndarray
    start_width_m: float
    width_speed_m_s: float
    measured_offset_m: float
    start_mode: str
    poll_s: float
    start_hold_s: float
    state_endpoint: str          # tcp://127.0.0.1:<link ports.state>
    commands_endpoint: str       # tcp://127.0.0.1:<link ports.commands>
    watchdog_limit_s: float      # link.yaml
    limits: ContractLimits
    frame: TcpFrame
    session_root: Path

    @classmethod
    def from_config(cls, config: dict[str, Any], session_root: Path, link: LinkSettings | None = None,
                    limits: ContractLimits | None = None, frame: TcpFrame | None = None,
                    state_endpoint: str | None = None, commands_endpoint: str | None = None) -> FakePcSettings:
        """Validated settings; the endpoints default to 127.0.0.1 with the link.yaml ports (tests pass tcp://...:*)."""
        link = LinkSettings.load() if link is None else link
        limits = ContractLimits.load() if limits is None else limits
        frame = TcpFrame.load() if frame is None else frame
        problems: list[str] = []

        def num(key: str, positive: bool = True) -> float:
            v = config.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or (positive and v <= 0):
                problems.append(f"fake_pc.{key}: expected a number{' > 0' if positive else ''}, got {v!r}")
                return 1.0
            return float(v)

        def pose(key: str) -> np.ndarray:
            try:
                T = np.array(config.get(key), dtype=np.float64)
            except (TypeError, ValueError):
                T = np.zeros(0)
            if T.shape != (4, 4) or rigid_problem(T, limits.rotation_tol) is not None:
                problems.append(f"fake_pc.{key}: expected a rigid 4x4 pose, got {config.get(key)!r}")
                return np.eye(4)
            return T

        values = dict(
            clock_offset_s=num("clock_offset_s", positive=False), clock_drift_ppm=num("clock_drift_ppm", False),
            state_rate_hz=num("state_rate_hz"), homing_timeout_s=num("homing_timeout_s"),
            max_lead_s=num("max_lead_s"), home_duration_s=num("home_duration_s"),
            start_width_m=num("start_width_m", positive=False), width_speed_m_s=num("width_speed_m_s"),
            measured_offset_m=num("measured_offset_m", positive=False), poll_s=num("poll_s"),
            start_hold_s=num("start_hold_s", positive=False))
        hwm = config.get("state_send_hwm")
        if not isinstance(hwm, int) or isinstance(hwm, bool) or hwm < 1:
            problems.append(f"fake_pc.state_send_hwm: expected an integer >= 1, got {hwm!r}")
            hwm = 100
        mode = config.get("start_mode")
        if mode not in START_MODES or mode not in MODES:
            problems.append(f"fake_pc.start_mode: expected one of {', '.join(START_MODES)}, got {mode!r}")
            mode = "idle"
        home, start = pose("home_pose"), pose("start_pose")
        known = set(values) | {"state_send_hwm", "start_mode", "home_pose", "start_pose"}
        for key in config:
            if key not in known:
                problems.append(f"fake_pc.{key}: unknown key")
        if problems:
            raise StationError(problems)
        return cls(**values, state_send_hwm=hwm, start_mode=mode, home_pose=home, start_pose=start,
                   state_endpoint=state_endpoint or link.state_endpoint(local=True),
                   commands_endpoint=commands_endpoint or link.commands_endpoint(local=True),
                   watchdog_limit_s=link.watchdog_limit_s, limits=limits, frame=frame, session_root=session_root)

    @classmethod
    def load(cls, station_path: Path | None, mock: bool = True) -> FakePcSettings:
        station = Station.load(station_path, mock=mock)
        return cls.from_config(load_config("fake_pc", station), station.session_root)
