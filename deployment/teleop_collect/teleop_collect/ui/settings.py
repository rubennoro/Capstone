"""Settings: config/ui.yaml merged with station.yaml's `ui:` section, validated (STRUCTURE.md 3.3).

The endpoints come from the station: the page on 127.0.0.1:<ports.http>, U1 on station.u1_endpoint(), U2 on
station.preview_endpoint().
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from teleop_collect.contracts import Station, StationError, load_config
from teleop_collect.contracts.station import LOCAL_HOST, is_number

#: The page binds here and nowhere else (PLAN.md 3).
HTTP_HOST = LOCAL_HOST


class SettingsError(Exception):
    """Every problem of the ui settings, each one line with its key path."""

    def __init__(self, problems: str | list[str]) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class Settings:
    status_hz: float
    preview_hz: float
    keys_resend_ms: float
    keys_stale_ms: float
    rpc_timeout_s: float
    open_browser: bool
    http_port: int                  # 0: any free port (tests only)
    u1_endpoint: str
    preview_endpoint: str
    session_root: Path

    @classmethod
    def load(cls, station_path: Path | str | None, mock: bool = False) -> Settings:
        try:
            station = Station.load(station_path, mock=mock)
        except StationError as e:
            raise SettingsError(e.problems) from None
        return cls.from_station(station)

    @classmethod
    def from_station(cls, station: Station) -> Settings:
        try:
            config = load_config("ui", station)
        except StationError as e:
            raise SettingsError(e.problems) from None
        return cls.from_config(config, http_port=station.http_port, u1_endpoint=station.u1_endpoint(),
                               preview_endpoint=station.preview_endpoint(), session_root=station.session_root)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, http_port: int, u1_endpoint: str, preview_endpoint: str,
                    session_root: Path) -> Settings:
        problems: list[str] = []

        def positive(key: str) -> float:
            value = config.get(key)
            if not is_number(value) or not value > 0:
                problems.append(f"ui.{key}: expected a number > 0, got {value!r}")
                return 1.0
            return float(value)

        status_hz = positive("status_hz")
        preview_hz = positive("preview_hz")
        keys_resend_ms = positive("keys_resend_ms")
        keys_stale_ms = positive("keys_stale_ms")
        rpc_timeout_s = positive("rpc_timeout_s")
        open_browser = config.get("open_browser")
        if not isinstance(open_browser, bool):
            problems.append(f"ui.open_browser: expected true or false, got {open_browser!r}")
            open_browser = False
        if not problems and keys_resend_ms >= keys_stale_ms:
            problems.append(f"ui.keys_resend_ms: {keys_resend_ms:g} must be below ui.keys_stale_ms "
                            f"{keys_stale_ms:g} (the page's resends keep the deadman from firing)")
        if problems:
            raise SettingsError(problems)
        return cls(status_hz=status_hz, preview_hz=preview_hz, keys_resend_ms=keys_resend_ms,
                   keys_stale_ms=keys_stale_ms, rpc_timeout_s=rpc_timeout_s, open_browser=open_browser,
                   http_port=int(http_port), u1_endpoint=u1_endpoint, preview_endpoint=preview_endpoint,
                   session_root=Path(session_root))

    def with_changes(self, **changes: Any) -> Settings:
        """A copy with some fields replaced (tests)."""
        return replace(self, **changes)
