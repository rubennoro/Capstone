# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/settings.py @ 2ffe641
# changes: config/teleop.yaml + station.yaml teleop: (teleop_collect STRUCTURE.md 3.3); the policy keys (exec_h,
# lookahead_ms, pose_source, odd chunk, anchor, image age, prompt, operator script, skips, steps, tap) and status_ms
# are gone; + motion: and home: built into E4's MotionSettings / HomeSettings (their ValueError, e.g. the margin rule,
# becomes a StationError); + u1_max_per_cycle, at_limit_hold_ms, messages_kept; timeouts frames_s (R1 start/stop)
# instead of the C1/C4 ones; the station's R1 and U1 endpoints, data_root and the rig file (config/frames_process.yaml
# rig_file) instead of C1/C4; worst_gap_s as before; waypoint_lead_ms = K x Delta; + gripper_width_max_m (limits.yaml,
# the width M opens to); + task_settings() (task.yaml settings); to_dict() through _plain (dataclasses, paths,
# tuples); + camera_down_s (frames' health.down_s); the problems name teleop.*; no "path", "str", "optional",
# "exec_h" or "pose_source" rules. + start pose (final-deployment's start-pose work, 927a5ba,
# 2026-11-20): start_pose: (StartPoseSettings) instead of home: (no laptop home); timeouts.home_s only builds PcClient.
# The source's own header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/settings.py @ 91b178b
# changes: config/manager.yaml + station.yaml manager: (policy_deploy.contracts.load_config) + LinkSettings + TcpFrame;
# the link values (heartbeat period, C3 timeout, state stale/lost, the clock estimator, the PC's endpoints) come only
# from deployment/contracts/link.yaml; loop_ms, max_state_age_ms, state_lost_s and timeouts.commands_s are gone; adds
# the keys of STRUCTURE.md 6.6 (health/home/pc_homing_wait timeouts, link_jitter_allowance_s, listener_poll_s, clock.*,
# link_stats.window_s); the session folder comes from ProcessArgs.
"""Settings: everything teleop runs with (STRUCTURE.md 3.3, 6.2)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deploy_contracts import ContractLimits, LinkSettings, TcpFrame

from teleop_collect.contracts import Station, StationError, load_config, resolve_path

if TYPE_CHECKING:
    from teleop_collect.teleop.motion.planner import MotionSettings
    from teleop_collect.teleop.motion.start_pose import StartPoseSettings

NAME = "teleop"
#: the config sections built and checked by their own classes (MotionSettings, StartPoseSettings)
DELEGATED = ("motion", "start_pose")


@dataclass(frozen=True)
class Timeouts:
    health_s: float           # R1 get_health on the main thread (a term of the worst C3 gap)
    frames_s: float           # R1 start_recording / stop_recording
    home_s: float             # PcClient's C3 home timeout (teleop sends no C3 home; pc_probe builds a PcClient too)
    startup_s: float          # start-up steps 2-9, not counting the waits while C2 shows homing
    pc_homing_wait_s: float   # start-up step 3: longest wait while C2 shows homing


@dataclass(frozen=True)
class ClockSettings:
    estimator: str                    # link.yaml clock.estimator: round_trip | one_way
    hello_period_s: float
    startup_hello_period_s: float
    round_trip_window_s: float
    round_trip_best_fraction: float
    one_way_window_s: float
    one_way_best_fraction: float
    drift_min_span_s: float
    max_drift_ppm: float
    min_samples_s: float
    min_samples: int
    max_sample_age_s: float
    startup_agree_ms: float
    jump_log_ms: float
    mismatch_sets: int
    mismatch_min_lead_ms: float
    future_state_ms: float
    log_every_s: float


@dataclass(frozen=True, eq=False)
class Settings:
    motion: MotionSettings
    start_pose: StartPoseSettings
    history_s: float
    state_recv_hwm: int
    listener_poll_s: float
    link_jitter_allowance_s: float
    health_poll_ms: int
    u1_max_per_cycle: int
    at_limit_hold_ms: int
    messages_kept: int
    timeouts: Timeouts
    clock: ClockSettings
    link_stats_window_s: float
    # from link.yaml and limits.yaml (never from a station or config file)
    heartbeat_period_s: float
    c3_timeout_s: float
    state_stale_s: float
    state_lost_s: float
    watchdog_limit_s: float
    gripper_width_max_m: float        # limits.yaml: M opens the fingers to this
    state_endpoint: str               # C2: tcp://<pc.address>:5620, or 127.0.0.1 under --mock
    commands_endpoint: str            # C3
    # from the station
    r1_endpoint: str                  # frames' R1
    u1_endpoint: str                  # teleop's U1 (the ui connects)
    data_root: Path
    session_root: Path
    station_file: Path
    rig_file: Path                    # config/frames_process.yaml rig_file (recorded in meta.yaml)
    camera_down_s: float              # config/frames_process.yaml health.down_s (the start refusal's Pi hint)
    mock: bool
    link: LinkSettings
    frame: TcpFrame

    @property
    def worst_gap_s(self) -> float:
        """The longest teleop can go between two C3 requests (PLAN 5.1)."""
        return self.heartbeat_period_s + self.c3_timeout_s + self.timeouts.health_s + self.link_jitter_allowance_s

    @property
    def waypoint_lead_ms(self) -> float:
        """The last waypoint's lead over the PC's now: K x Delta (PLAN 5.2)."""
        return float(self.motion.waypoints * self.motion.spacing_ms)

    def task_settings(self) -> dict[str, Any]:
        """What task.yaml keeps as `settings` (the speeds, K, Delta, the start pose tolerances)."""
        return {"motion": _plain(self.motion), "start_pose": _plain(self.start_pose),
                "heartbeat_period_s": self.heartbeat_period_s, "rig_file": str(self.rig_file)}

    def to_dict(self) -> dict[str, Any]:
        """Plain values (for session.yaml)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, LinkSettings):
                out[f.name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in value.__dict__.items()}
            elif isinstance(value, TcpFrame):
                out[f.name] = {"F_T_TCP": value.F_T_TCP.tolist(), "settled": bool(value.settled)}
            else:
                out[f.name] = _plain(value)
        return out

    @classmethod
    def load(cls, station_path: Path | None, mock: bool, link: LinkSettings | None = None,
             frame: TcpFrame | None = None) -> Settings:
        """Raises StationError listing every problem."""
        station = Station.load(station_path, mock=mock)
        return cls.from_station(station, link=link, frame=frame)

    @classmethod
    def from_station(cls, station: Station, link: LinkSettings | None = None, frame: TcpFrame | None = None,
                     state_endpoint: str | None = None, commands_endpoint: str | None = None) -> Settings:
        """The endpoints default to link.yaml's (127.0.0.1 under --mock); tests pass their own."""
        from teleop_collect.teleop.motion.planner import MotionSettings
        from teleop_collect.teleop.motion.start_pose import StartPoseSettings
        link = LinkSettings.load() if link is None else link
        frame = TcpFrame.load() if frame is None else frame
        config = load_config(NAME, station)
        problems: list[str] = []
        flat = _flatten({k: v for k, v in config.items() if k not in DELEGATED})
        for key, (kind, rule) in _RULES.items():
            _check(key, flat.get(key), kind, rule, problems)
        for key in flat:
            if key not in _RULES:
                problems.append(f"config/teleop.yaml: {key}: unknown key")
        motion = start_pose = None
        for name, build in (("motion", MotionSettings.from_config), ("start_pose", StartPoseSettings.from_config)):
            section = config.get(name)
            if not isinstance(section, dict):
                problems.append(f"{NAME}.{name}: expected a map, got {section!r}")
                continue
            try:
                built = build(section)
            except (ValueError, TypeError, KeyError) as exc:
                problems.append(f"{NAME}.{name}: {exc}")
                continue
            if name == "motion":
                motion = built
            else:
                start_pose = built
        rig_file = None
        camera_down_s = 0.0
        try:
            frames_config = load_config("frames", station)
            rig_file = resolve_path(station, "frames", frames_config, "rig_file")
            camera_down_s = float(frames_config["health"]["down_s"])
        except (StationError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"frames.rig_file / health.down_s: {exc}")
        if problems:
            raise StationError(problems)
        t = config["timeouts"]
        c = config["clock"]
        clock = ClockSettings(
            estimator=link.clock_estimator, hello_period_s=float(c["hello_period_s"]),
            startup_hello_period_s=float(c["startup_hello_period_s"]),
            round_trip_window_s=float(c["round_trip"]["window_s"]),
            round_trip_best_fraction=float(c["round_trip"]["best_fraction"]),
            one_way_window_s=float(c["one_way"]["window_s"]),
            one_way_best_fraction=float(c["one_way"]["best_fraction"]),
            drift_min_span_s=float(c["drift_min_span_s"]), max_drift_ppm=float(c["max_drift_ppm"]),
            min_samples_s=float(c["min_samples_s"]), min_samples=int(c["min_samples"]),
            max_sample_age_s=float(c["max_sample_age_s"]), startup_agree_ms=float(c["startup_agree_ms"]),
            jump_log_ms=float(c["jump_log_ms"]), mismatch_sets=int(c["mismatch_sets"]),
            mismatch_min_lead_ms=float(c["mismatch_min_lead_ms"]), future_state_ms=float(c["future_state_ms"]),
            log_every_s=float(c["log_every_s"]))
        mock = station.mock
        return cls(
            motion=motion, start_pose=start_pose, history_s=float(config["history_s"]),
            state_recv_hwm=config["state_recv_hwm"],
            listener_poll_s=float(config["listener_poll_s"]),
            link_jitter_allowance_s=float(config["link_jitter_allowance_s"]),
            health_poll_ms=config["health_poll_ms"], u1_max_per_cycle=config["u1_max_per_cycle"],
            at_limit_hold_ms=config["at_limit_hold_ms"], messages_kept=config["messages_kept"],
            timeouts=Timeouts(**{k: float(t[k]) for k in Timeouts.__dataclass_fields__}),
            clock=clock, link_stats_window_s=float(config["link_stats"]["window_s"]),
            heartbeat_period_s=link.heartbeat_period_s, c3_timeout_s=link.c3_timeout_s,
            state_stale_s=link.state_stale_s, state_lost_s=link.state_lost_s, watchdog_limit_s=link.watchdog_limit_s,
            gripper_width_max_m=float(ContractLimits.load().gripper_width_max_m),
            state_endpoint=state_endpoint or link.state_endpoint(local=mock),
            commands_endpoint=commands_endpoint or link.commands_endpoint(local=mock),
            r1_endpoint=station.r1_endpoint(), u1_endpoint=station.u1_endpoint(), data_root=station.data_root,
            session_root=station.session_root, station_file=station.path, rig_file=rig_file,
            camera_down_s=camera_down_s, mock=mock, link=link,
            frame=frame)


# key -> (kind, rule). kinds: int, float, bool. rules: "pos" (> 0), "nonneg" (>= 0), "ge1", "fraction" (0 < x <= 1).
_RULES: dict[str, tuple[str, str]] = {
    "history_s": ("float", "pos"),
    "state_recv_hwm": ("int", "pos"),
    "listener_poll_s": ("float", "pos"),
    "link_jitter_allowance_s": ("float", "nonneg"),
    "health_poll_ms": ("int", "pos"),
    "u1_max_per_cycle": ("int", "ge1"),
    "at_limit_hold_ms": ("int", "nonneg"),
    "messages_kept": ("int", "ge1"),
    **{f"timeouts.{k}": ("float", "pos") for k in Timeouts.__dataclass_fields__},
    "clock.hello_period_s": ("float", "pos"),
    "clock.startup_hello_period_s": ("float", "pos"),
    "clock.round_trip.window_s": ("float", "pos"),
    "clock.round_trip.best_fraction": ("float", "fraction"),
    "clock.one_way.window_s": ("float", "pos"),
    "clock.one_way.best_fraction": ("float", "fraction"),
    "clock.drift_min_span_s": ("float", "pos"),
    "clock.max_drift_ppm": ("float", "pos"),
    "clock.min_samples_s": ("float", "nonneg"),
    "clock.min_samples": ("int", "ge1"),
    "clock.max_sample_age_s": ("float", "pos"),
    "clock.startup_agree_ms": ("float", "pos"),
    "clock.jump_log_ms": ("float", "pos"),
    "clock.mismatch_sets": ("int", "ge1"),
    "clock.mismatch_min_lead_ms": ("float", "pos"),
    "clock.future_state_ms": ("float", "pos"),
    "clock.log_every_s": ("float", "pos"),
    "link_stats.window_s": ("float", "pos"),
}


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _check(key: str, value: Any, kind: str, rule: str, problems: list[str]) -> None:
    where = f"{NAME}.{key}"
    if value is None:
        problems.append(f"{where}: must be set")
        return
    if kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    else:                                   # bool
        ok = isinstance(value, bool)
    if not ok:
        problems.append(f"{where}: expected {kind}, got {value!r}")
        return
    if rule == "pos" and not value > 0:
        problems.append(f"{where}: must be > 0, got {value!r}")
    elif rule == "nonneg" and not value >= 0:
        problems.append(f"{where}: must be >= 0, got {value!r}")
    elif rule == "ge1" and not value >= 1:
        problems.append(f"{where}: must be >= 1, got {value!r}")
    elif rule == "fraction" and not 0 < value <= 1:
        problems.append(f"{where}: must be in (0, 1], got {value!r}")
