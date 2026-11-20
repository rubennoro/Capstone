# adapted from rnoro/deployment:deployment/message_comms/message_comms/settings.py @ 91b178b
# changes: config/manager.yaml + station.yaml manager: (policy_deploy.contracts.load_config) + LinkSettings + TcpFrame;
# the link values (heartbeat period, C3 timeout, state stale/lost, the clock estimator, the PC's endpoints) come only
# from deployment/contracts/link.yaml; loop_ms, max_state_age_ms, state_lost_s and timeouts.commands_s are gone; adds
# the keys of STRUCTURE.md 6.6 (health/home/pc_homing_wait timeouts, link_jitter_allowance_s, listener_poll_s, clock.*,
# link_stats.window_s); the session folder comes from ProcessArgs.
"""Settings: everything the manager runs with (STRUCTURE.md 6.2, 6.6)."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from deploy_contracts import LinkSettings, TcpFrame

from policy_deploy.contracts import ACTION_STEPS, Station, StationError, load_config, resolve_path

NAME = "manager"
PC_CONFIG = Path(__file__).resolve().parents[3] / "control_deploy" / "franka_control" / "config.yaml"
POSE_SOURCES = ("target", "measured")
EXEC_H_RANGE = (1, ACTION_STEPS)


@dataclass(frozen=True)
class Timeouts:
    health_s: float           # C1 get_health on the main thread
    frames_s: float           # C1 get_frames (worker)
    policy_hang_s: float      # C4 infer
    config_s: float
    reset_s: float
    prompt_s: float
    home_s: float             # C3 home
    startup_s: float          # start-up steps 2-9, not counting the homing waits
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
    exec_h: int
    lookahead_ms: int
    pose_source: str
    odd_chunk_max_pos_m: float
    odd_chunk_max_rot_deg: float
    anchor_fallback_m: float
    status_ms: int
    health_poll_ms: int
    history_s: float
    max_image_age_ms: float
    state_recv_hwm: int
    link_jitter_allowance_s: float
    listener_poll_s: float
    timeouts: Timeouts
    clock: ClockSettings
    link_stats_window_s: float
    task_name: str | None
    prompt_index: int
    operator_script: Path | None
    operator_script_wait_s: float
    skip_warn_s: float
    log_steps: bool
    save_tap: bool
    tap_images_every: int
    workspace_box: list[list[float]] | None   # [min xyz, max xyz] the waypoints are clamped into (the PC's box, inset)
    # from link.yaml (never from a station or config file)
    heartbeat_period_s: float
    c3_timeout_s: float
    state_stale_s: float
    state_lost_s: float
    watchdog_limit_s: float
    state_endpoint: str               # C2: tcp://<pc.address>:5620, or 127.0.0.1 under --mock
    commands_endpoint: str            # C3
    # from the station
    frames_endpoint: str              # C1
    policy_endpoint: str              # C4
    session_root: Path
    station_file: Path
    mock: bool
    link: LinkSettings
    frame: TcpFrame

    @property
    def worst_gap_s(self) -> float:
        """The longest the manager can go between two C3 requests (PLAN 6)."""
        return self.heartbeat_period_s + self.c3_timeout_s + self.timeouts.health_s + self.link_jitter_allowance_s

    def to_dict(self) -> dict[str, Any]:
        """Plain values (for session.yaml)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (Timeouts, ClockSettings)):
                out[f.name] = dict(value.__dict__)
            elif isinstance(value, Path):
                out[f.name] = str(value)
            elif isinstance(value, LinkSettings):
                out[f.name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in value.__dict__.items()}
            elif isinstance(value, TcpFrame):
                out[f.name] = {"F_T_TCP": value.F_T_TCP.tolist(), "settled": bool(value.settled)}
            else:
                out[f.name] = value
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
        link = LinkSettings.load() if link is None else link
        frame = TcpFrame.load() if frame is None else frame
        config = load_config(NAME, station)
        problems: list[str] = []
        flat = _flatten(config)
        for key, (kind, rule) in _RULES.items():
            _check(key, flat.get(key), kind, rule, problems)
        for key in flat:
            if key not in _RULES:
                problems.append(f"config/manager.yaml: {key}: unknown key")
        box = None
        if flat.get("workspace_clamp_m") is not None:
            if flat["workspace_clamp_m"] < 0:
                problems.append(f"manager.workspace_clamp_m: must be >= 0 or null, got {flat['workspace_clamp_m']!r}")
            else:
                box = _workspace_box(float(flat["workspace_clamp_m"]), problems)
        if problems:
            raise StationError(problems)
        script = resolve_path(station, NAME, config, "operator_script")
        t = config["timeouts"]
        c = config["clock"]
        clock = ClockSettings(
            estimator=link.clock_estimator, hello_period_s=float(c["hello_period_s"]),
            startup_hello_period_s=float(c["startup_hello_period_s"]),
            round_trip_window_s=float(c["round_trip"]["window_s"]),
            round_trip_best_fraction=float(c["round_trip"]["best_fraction"]),
            one_way_window_s=float(c["one_way"]["window_s"]), one_way_best_fraction=float(c["one_way"]["best_fraction"]),
            drift_min_span_s=float(c["drift_min_span_s"]), max_drift_ppm=float(c["max_drift_ppm"]),
            min_samples_s=float(c["min_samples_s"]), min_samples=int(c["min_samples"]),
            max_sample_age_s=float(c["max_sample_age_s"]), startup_agree_ms=float(c["startup_agree_ms"]),
            jump_log_ms=float(c["jump_log_ms"]), mismatch_sets=int(c["mismatch_sets"]),
            mismatch_min_lead_ms=float(c["mismatch_min_lead_ms"]), future_state_ms=float(c["future_state_ms"]),
            log_every_s=float(c["log_every_s"]))
        mock = station.mock
        return cls(
            exec_h=config["exec_h"], lookahead_ms=config["lookahead_ms"], pose_source=config["pose_source"],
            odd_chunk_max_pos_m=float(config["odd_chunk_max_pos_m"]),
            odd_chunk_max_rot_deg=float(config["odd_chunk_max_rot_deg"]),
            anchor_fallback_m=float(config["anchor_fallback_m"]), status_ms=config["status_ms"],
            health_poll_ms=config["health_poll_ms"], history_s=float(config["history_s"]),
            max_image_age_ms=float(config["max_image_age_ms"]), state_recv_hwm=config["state_recv_hwm"],
            link_jitter_allowance_s=float(config["link_jitter_allowance_s"]),
            listener_poll_s=float(config["listener_poll_s"]),
            timeouts=Timeouts(**{k: float(t[k]) for k in Timeouts.__dataclass_fields__}),
            clock=clock, link_stats_window_s=float(config["link_stats"]["window_s"]),
            task_name=config["task_name"], prompt_index=config["prompt_index"], operator_script=script,
            operator_script_wait_s=float(config["operator_script_wait_s"]), skip_warn_s=float(config["skip_warn_s"]),
            log_steps=config["log_steps"], save_tap=config["save_tap"], tap_images_every=config["tap_images_every"],
            workspace_box=box,
            heartbeat_period_s=link.heartbeat_period_s, c3_timeout_s=link.c3_timeout_s,
            state_stale_s=link.state_stale_s, state_lost_s=link.state_lost_s, watchdog_limit_s=link.watchdog_limit_s,
            state_endpoint=state_endpoint or link.state_endpoint(local=mock),
            commands_endpoint=commands_endpoint or link.commands_endpoint(local=mock),
            frames_endpoint=station.frames_endpoint(), policy_endpoint=station.policy_endpoint(),
            session_root=station.session_root, station_file=station.path, mock=mock, link=link, frame=frame)


# key -> (kind, rule). kinds: int, float, bool, str, path. rules: "pos" (> 0), "nonneg" (>= 0), "exec_h", "ge1",
# "pose_source", "optional" (may be null), "fraction" (0 < x <= 1).
_RULES: dict[str, tuple[str, str]] = {
    "exec_h": ("int", "exec_h"),
    "lookahead_ms": ("int", "nonneg"),
    "pose_source": ("str", "pose_source"),
    "odd_chunk_max_pos_m": ("float", "pos"),
    "odd_chunk_max_rot_deg": ("float", "pos"),
    "anchor_fallback_m": ("float", "pos"),
    "status_ms": ("int", "pos"),
    "health_poll_ms": ("int", "pos"),
    "history_s": ("float", "pos"),
    "max_image_age_ms": ("float", "pos"),
    "state_recv_hwm": ("int", "pos"),
    "link_jitter_allowance_s": ("float", "nonneg"),
    "listener_poll_s": ("float", "pos"),
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
    "task_name": ("str", "optional"),
    "prompt_index": ("int", "nonneg"),
    "operator_script": ("path", "optional"),
    "operator_script_wait_s": ("float", "pos"),
    "skip_warn_s": ("float", "pos"),
    "log_steps": ("bool", ""),
    "save_tap": ("bool", ""),
    "tap_images_every": ("int", "ge1"),
    "workspace_clamp_m": ("float", "optional"),
}


def _workspace_box(inset_m: float, problems: list[str]) -> list[list[float]] | None:
    """The PC's safety.workspace_min_m / workspace_max_m (same commit on both machines), moved inset_m inwards."""
    where = f"manager.workspace_clamp_m: {PC_CONFIG}"
    try:
        safety = yaml.safe_load(PC_CONFIG.read_text(encoding="utf-8"))["safety"]
        lo = [float(v) + inset_m for v in safety["workspace_min_m"]]
        hi = [float(v) - inset_m for v in safety["workspace_max_m"]]
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        problems.append(f"{where}: cannot read safety.workspace_min_m / workspace_max_m ({exc})")
        return None
    if len(lo) != 3 or len(hi) != 3 or any(a >= b for a, b in zip(lo, hi)):
        problems.append(f"{where}: the box {lo}..{hi} (after the inset) is empty or not 3-D")
        return None
    return [lo, hi]


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _check(key: str, value: Any, kind: str, rule: str, problems: list[str]) -> None:
    where = f"manager.{key}"
    if value is None:
        if rule != "optional":
            problems.append(f"{where}: must be set")
        return
    if kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    elif kind == "bool":
        ok = isinstance(value, bool)
    else:                                   # str, path
        ok = isinstance(value, str) and value != ""
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
    elif rule == "exec_h" and not EXEC_H_RANGE[0] <= value <= EXEC_H_RANGE[1]:
        problems.append(f"{where}: must be between {EXEC_H_RANGE[0]} and {EXEC_H_RANGE[1]}, got {value!r}")
    elif rule == "pose_source" and value not in POSE_SOURCES:
        problems.append(f"{where}: must be one of {', '.join(POSE_SOURCES)}, got {value!r}")
