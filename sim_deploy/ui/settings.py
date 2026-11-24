"""Settings of sim_deploy.ui (ui/DESIGN.md §1 "Shared"): ports, endpoints, paths and rates.

Defaults are below; `ui/config.yaml` (or `--config FILE`) overrides any of them with the same flat key names. Paths
may use `~`. Keys the file has that are not fields here go to `Settings.extra` (a part that needs a knob of its own
reads `settings.extra.get("name", default)`), with a warning in the log.

Every part gets the same frozen Settings object: `Runner(settings)`, `Live(settings)`, `Catalog(settings)` (or the
module functions, see server.py). The page itself is served on 127.0.0.1 only (HTTP_HOST, not configurable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from sim_deploy import REPO, ROOT

#: The page binds here and nowhere else.
HTTP_HOST = "127.0.0.1"
#: The default config file (next to this module).
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
HOME = Path.home()

log = logging.getLogger("sim_deploy.ui")


class SettingsError(Exception):
    """Every problem of the settings, one line each."""

    def __init__(self, problems: str | list[str]) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class Settings:
    # --- the server (U1)
    http_port: int = 8770                     # 0: any free port
    status_hz: float = 5.0                    # status messages to every page
    telemetry_hz: float = 10.0                # telemetry messages (while Live has data)
    frame_fps: float = 10.0                   # binary frames, per channel, at most
    log_tail: int = 30                        # Status.log_tail lines
    shutdown_clean_s: float = 180.0           # Ctrl-C: wait this long for the clean stop (the runner auto-kills
                                              # 150 s after a first stop), then stop again (kill)
    shutdown_kill_s: float = 5.0              # ... and this long after the kill
    # --- paths (U2 runner, U5 catalog, /evals/)
    repo: Path = REPO                                           # ~/Capstone-main-sim
    run_sh: Path = ROOT / "run.sh"                              # what the runner starts
    python: Path = HOME / "envs" / "sim_deploy" / "bin" / "python"
    sim_evals: Path = REPO / "sim_evals"                        # results; served under /evals/
    policy_data_root: Path = HOME / "Capstone" / "data"         # <task>/policy/<run>/ (read only)
    datasets_root: Path = HOME / "data" / "umi" / "datasets"    # replay zarrs (read only)
    policy_station: Path = ROOT / "stations" / "policy_sim.yaml"
    scene_xml: Path = ROOT / "build" / "sim_rig" / "scene.xml"
    model_json: Path = ROOT / "build" / "sim_rig" / "model.json"
    rig_yaml: Path = REPO / "deployment" / "contracts" / "rigs" / "rig_pi_v3_640.yaml"
    # --- live endpoints (U3), all read only; the sim chain binds them, the ui only connects
    c0_endpoint: str = "tcp://127.0.0.1:5555"                   # pi_stream (camera JPEGs) SUB
    c2_endpoint: str = "tcp://127.0.0.1:5620"                   # robot state SUB
    state_endpoint: str = "tcp://127.0.0.1:5650"                # sim state SUB (MuJoCo mirror)
    scene_endpoint: str = "tcp://127.0.0.1:5651"                # scene REQ (ground_truth)
    ground_truth_hz: float = 5.0
    external_camera: str = "cam_external"
    external_width: int = 640
    external_height: int = 480
    jpeg_quality: int = 80                                      # re-encoded views (external, policy mosaic)
    # --- anything else config.yaml has
    extra: Mapping[str, Any] = field(default_factory=dict)
    config_file: Path | None = None
    stub: bool = False

    # ------------------------------------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: Path | str | None = None, **overrides: Any) -> "Settings":
        """Defaults <- the config file (DEFAULT_CONFIG when path is None and it exists) <- overrides (CLI)."""
        import yaml

        file = Path(path).expanduser() if path is not None else (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
        data: dict[str, Any] = {}
        if file is not None:
            try:
                loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
            except OSError as e:
                raise SettingsError(f"{file}: cannot read: {e.strerror or e}") from None
            except yaml.YAMLError as e:
                raise SettingsError(f"{file}: not valid YAML: {e}") from None
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise SettingsError(f"{file}: expected a map of settings, got {type(loaded).__name__}")
            data.update(loaded)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_mapping(data, config_file=file)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], config_file: Path | None = None) -> "Settings":
        problems: list[str] = []
        known = {f.name: f for f in fields(cls) if f.name not in ("extra", "config_file")}
        values: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        defaults = cls()
        for key, value in data.items():
            if key not in known:
                extra[key] = value
                continue
            default = getattr(defaults, key)
            try:
                values[key] = _coerce(key, value, default)
            except ValueError as e:
                problems.append(str(e))
        for key in extra:
            log.warning("config: %r is not a ui setting; kept in settings.extra", key)
        s = replace(defaults, **values, extra=extra, config_file=config_file)
        for key in ("status_hz", "telemetry_hz", "frame_fps", "ground_truth_hz", "shutdown_clean_s",
                    "shutdown_kill_s"):
            if not getattr(s, key) > 0:
                problems.append(f"{key}: expected a number > 0, got {getattr(s, key)!r}")
        if not 0 <= s.http_port <= 65535:
            problems.append(f"http_port: expected 0..65535, got {s.http_port}")
        if s.http_port in SIM_PORTS:
            problems.append(f"http_port: {s.http_port} is a sim chain port ({', '.join(map(str, SIM_PORTS))})")
        if not 1 <= s.jpeg_quality <= 100:
            problems.append(f"jpeg_quality: expected 1..100, got {s.jpeg_quality}")
        if s.log_tail < 1:
            problems.append(f"log_tail: expected >= 1, got {s.log_tail}")
        if problems:
            raise SettingsError(problems)
        return s

    def with_changes(self, **changes: Any) -> "Settings":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able (paths as strings)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else (dict(v) if f.name == "extra" else v)
        return out


#: Ports the sim chain uses (ui/DESIGN.md §6): the page never binds them.
SIM_PORTS = (5555, 5600, 5610, 5620, 5630, 5640, 5650, 5651)


def _coerce(key: str, value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{key}: expected true or false, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key}: expected an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key}: expected a number, got {value!r}")
        return float(value)
    if isinstance(default, Path):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key}: expected a path, got {value!r}")
        return Path(value).expanduser()
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{key}: expected a string, got {value!r}")
        return value
    return value


__all__ = ["HTTP_HOST", "DEFAULT_CONFIG", "SIM_PORTS", "Settings", "SettingsError"]
