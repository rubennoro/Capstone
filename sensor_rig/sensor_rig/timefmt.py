"""Time strings and episode names exactly as the iPhUMI iOS app writes them (R1 §1.1, §3.1).

- Time strings: ``YYYY-MM-DDTHH:MM:SS.mmmZ`` in UTC (ISO8601DateFormatter with fractional seconds).
  We truncate to milliseconds (floor); the app's rounding mode is unverified and immaterial (R1 §12.1).
- Episode NAME: ``<time with ':'->'-'>_<5 digits 1-9>_<session>_<demonstration|grippercalibration>_<left|right|head>``.
"""

from __future__ import annotations

import datetime as _dt
import random
import re

import numpy as np

ISO_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
NAME_TIME_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d\.\d{3}Z$")

KIND_TO_NAME = {"demonstration": "demonstration", "gripper_calibration": "grippercalibration"}
NAME_TO_KIND = {v: k for k, v in KIND_TO_NAME.items()}
SIDES = ("left", "right", "head")

_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


def iso_ms(unix_ns: int) -> str:
    """UNIX ns -> 'YYYY-MM-DDTHH:MM:SS.mmmZ' (UTC, milliseconds floored)."""
    ms = int(unix_ns) // 1_000_000
    sec, msr = divmod(ms, 1000)
    d = _EPOCH + _dt.timedelta(seconds=sec)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + f".{msr:03d}Z"


def iso_ms_array(unix_ns) -> list[str]:
    return [iso_ms(int(t)) for t in np.asarray(unix_ns, dtype=np.int64)]


def parse_iso(s: str) -> int:
    """ISO-8601 string (with 'Z' or an explicit offset) -> UNIX ns. Mirrors timecode_util.datetime_fromisoformat."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    d = _dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    delta = d - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def name_timestamp(unix_ns: int) -> str:
    return iso_ms(unix_ns).replace(":", "-")


def sanitize_session(s: str) -> str:
    """VC.swift:2446-2452: keep [A-Za-z0-9 _-], then '_' and ' ' -> '-'. Refuses '' and 'no-session'."""
    kept = "".join(c for c in s if (c.isascii() and c.isalnum()) or c in " _-")
    out = kept.replace("_", "-").replace(" ", "-")
    if out == "" or out == "no-session":
        raise ValueError(f"invalid session name {s!r} (sanitised to {out!r})")
    return out


def random_id(rng: random.Random | None = None) -> str:
    r = rng or random
    return "".join(r.choice("123456789") for _ in range(5))


def make_episode_name(start_unix_ns: int, rid: str, session: str, kind: str, side: str) -> str:
    if kind not in KIND_TO_NAME:
        raise ValueError(f"kind must be one of {list(KIND_TO_NAME)}, got {kind!r}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if not re.fullmatch(r"[1-9]{5}", rid):
        raise ValueError(f"random id must be 5 digits 1-9, got {rid!r}")
    if "_" in session or session != sanitize_session(session):
        raise ValueError(f"session {session!r} is not sanitised")
    name = f"{name_timestamp(start_unix_ns)}_{rid}_{session}_{KIND_TO_NAME[kind]}_{side}"
    assert len(name.split("_")) == 5
    return name


def parse_episode_name(name: str) -> dict:
    """NAME -> {time_str, rid, session, kind, kind_name, side, date}. ``kind`` uses the JSON spelling."""
    parts = name.split("_")
    if len(parts) != 5:
        raise ValueError(f"episode name must have 5 '_'-separated parts: {name!r}")
    time_str, rid, session, kind_name, side = parts
    if kind_name not in NAME_TO_KIND:
        raise ValueError(f"unknown recording type {kind_name!r} in {name!r}")
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r} in {name!r}")
    if not NAME_TIME_RE.match(time_str):
        raise ValueError(f"bad timestamp {time_str!r} in {name!r}")
    return {
        "time_str": time_str,
        "rid": rid,
        "session": session,
        "kind": NAME_TO_KIND[kind_name],
        "kind_name": kind_name,
        "side": side,
        "date": time_str[:10],
    }


def date_dir(name: str) -> str:
    return name[:10]
