"""The deployment launcher: `deploy check`, `deploy run`, and the geometry tool `python -m deploy.geometry`.

Modules read `deploy.DEPLOYMENT_ROOT` when they run (not at import), so one patch of it moves every default path.
"""
from __future__ import annotations

from pathlib import Path

DEPLOYMENT_ROOT: Path = Path(__file__).resolve().parents[2]   # deployment/ (this file is deployment/deploy/deploy/)

from deploy import check_helpers  # noqa: E402
from deploy.check_result import CheckResult, print_table  # noqa: E402
from deploy.config_merge import load_process_config, resolve_path  # noqa: E402
from deploy.errors import GeometryError, LaunchError, StationError  # noqa: E402
from deploy.launcher import Launcher  # noqa: E402
from deploy.loader import load_process_class  # noqa: E402
from deploy.managed_process import ManagedProcess  # noqa: E402
from deploy.session import Session  # noqa: E402
from deploy.station import Station, parse_endpoint  # noqa: E402

__all__ = [
    "DEPLOYMENT_ROOT", "CheckResult", "GeometryError", "LaunchError", "Launcher", "ManagedProcess", "Session",
    "Station", "StationError", "check_helpers", "load_process_class", "load_process_config", "parse_endpoint",
    "print_table", "resolve_path",
]
