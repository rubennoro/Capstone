"""Pi rig episodes (as the Pi 5 records them into ``/data/incoming``) -> contract episodes (``dataproc.raw.v1``).

The Pi's format is the ground truth: this module only reads a source directory, never writes to it, and writes a
converted copy that the rest of dataproc (intake, process) takes unchanged. Field mapping: docs/PI_ADAPTER.md.

::

    src = inspect_source(Path("/data/incoming/episode_000004"))     # verifies COMPLETE, manifest, meta
    if src.ok:
        out = adapt_into(src, root.incoming, rig_id="rig_pi_v1")   # -> incoming/<session>_episode_000004/

Source records (``$DATAPROC_ROOT/pi_sources.csv``) say which source episodes were already taken in, so a source is
never taken twice (keyed by the sha256 of its ``MANIFEST.sha256``).
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataproc.fsutil import fsync_dir, sha256_file, write_atomic
from dataproc.raw_episode import (COMPLETE, DATA_FILES, FRAMES_HEADER, IMU_HEADER, MANIFEST, SCHEMA, VERIFIED,
                                  RawEpisode)

__all__ = ["ADAPTER_VERSION", "SENSOR_ROLES", "DEFAULT_EXPOSURE_NS", "SOURCES_FILE", "SOURCES_COLUMNS",
           "PiEpisodeError", "PiSource", "inspect_source", "scan_sources", "session_id_for", "converted_name",
           "convert_episode", "adapt_into", "read_sources", "append_sources", "update_source_status"]

ADAPTER_VERSION = 1
#: Camera role by sensor model (the Pi's file names are not used: its cam0 is the IMX219).
SENSOR_ROLES = {"imx519": "main", "imx219": "ultrawide"}
ROLE_CAM = {"main": "cam0", "ultrawide": "cam1"}
IMU_SENSOR = "lsm6dsox"
#: Exposure written when the Pi does not record one (capped at half a frame period). Mid-exposure times are then
#: off by up to half the true exposure difference; the Pi's own ``exposure_ns`` column wins when present.
DEFAULT_EXPOSURE_NS = 5_000_000
STANDARD_GRAVITY = 9.80665
RAW_COUNTS = "raw int16 counts"
CLOCK = "CLOCK_BOOTTIME"
ADAPT_TMP = ".adapt-tmp"                 # incoming/<this>/<name>/ while converting (intake ignores dot-entries)
SOURCES_FILE = "pi_sources.csv"
SOURCES_COLUMNS = ("source_name", "source_manifest_sha256", "converted_name", "task", "kind", "status",
                   "adapted_unix_ns")


class PiEpisodeError(Exception):
    """The source episode cannot be converted (the message says why)."""


# --------------------------------------------------------------------------------------------- source

@dataclass
class PiSource:
    """One directory in the Pi's delivery folder. ``problem`` is None when it can be converted."""

    name: str
    path: Path
    manifest_sha256: str | None = None
    meta: dict | None = None
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.problem is None

    @property
    def episode_number(self) -> int:
        return int(self.meta["episode"])


def inspect_source(path: Path, verify_hashes: bool = True) -> PiSource:
    """Read-only checks of one Pi episode: a real directory with ``COMPLETE``, a ``MANIFEST.sha256`` that every
    file matches (``verify_hashes``), and a ``meta.json`` the Pi marked ``valid`` with no ``problems``."""
    path = Path(path)
    src = PiSource(path.name, path)
    if path.is_symlink() or not path.is_dir():
        src.problem = "not a directory"
        return src
    if not (path / COMPLETE).is_file():
        src.problem = "no COMPLETE (still being recorded or transferred)"
        return src
    if not (path / MANIFEST).is_file():
        src.problem = f"no {MANIFEST}"
        return src
    src.manifest_sha256 = sha256_file(path / MANIFEST)
    if verify_hashes:
        bad = [p for p in RawEpisode(path).verify_manifest() if p.severity == "error"]
        if bad:
            src.problem = "manifest check failed: " + "; ".join(p.message for p in bad[:3])
            return src
    try:
        src.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        src.problem = f"meta.json unreadable ({e})"
        return src
    m = src.meta
    if not isinstance(m, dict) or isinstance(m.get("episode"), bool) or not isinstance(m.get("episode"), int):
        src.problem = "meta.json has no integer 'episode'"
    elif m.get("valid") is False:
        src.problem = "the Pi marked it invalid (meta.json valid: false)"
    elif m.get("problems"):
        src.problem = f"the Pi reported problems: {m['problems']}"
    return src


def scan_sources(source_dir: Path, verify_hashes: bool = True) -> list[PiSource]:
    """Every non-dot entry of ``source_dir``, inspected, sorted by name."""
    return [inspect_source(p, verify_hashes) for p in sorted(Path(source_dir).iterdir())
            if not p.name.startswith(".")]


def session_id_for(src: PiSource) -> str:
    """``pi-YYYY-MM-DD``: the local date ``COMPLETE`` was written (the Pi has no session id)."""
    return "pi-" + time.strftime("%Y-%m-%d", time.localtime((src.path / COMPLETE).stat().st_mtime))


def converted_name(src: PiSource) -> str:
    return f"{session_id_for(src)}_episode_{src.episode_number:06d}"


# --------------------------------------------------------------------------------------------- CSV helpers

def _read_csv(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Integer columns by name (header-driven, so column order and extra columns do not matter)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise PiEpisodeError(f"{path.name} is empty")
    header = [h.strip() for h in rows[0]]
    missing = [c for c in required if c not in header]
    if missing:
        raise PiEpisodeError(f"{path.name}: missing column(s) {missing} (header {header})")
    body = [r for r in rows[1:] if r]
    if not body:
        raise PiEpisodeError(f"{path.name} has no data rows")
    try:
        a = np.array(body, dtype=np.int64)
    except ValueError as e:
        raise PiEpisodeError(f"{path.name}: non-integer value ({e})") from None
    if a.ndim != 2 or a.shape[1] != len(header):
        raise PiEpisodeError(f"{path.name}: rows do not all have {len(header)} columns")
    return {h: a[:, i] for i, h in enumerate(header)}


def _write_lines(path: Path, header: str, rows) -> None:
    buf = io.StringIO()
    buf.write(header + "\n")
    for r in rows:
        buf.write(r + "\n")
    path.write_text(buf.getvalue(), encoding="utf-8")


def _convert_frames(src_csv: Path, dst_csv: Path, fps: float, expected_frames: int | None) -> dict:
    cols = _read_csv(src_csv, ("frame_index", "t_ns"))
    idx, t = cols["frame_index"], cols["t_ns"]
    n = len(t)
    if not np.array_equal(idx, np.arange(n)):
        raise PiEpisodeError(f"{src_csv.name}: frame_index is not 0..{n - 1}")
    if expected_frames is not None and expected_frames != n:
        raise PiEpisodeError(f"{src_csv.name}: {n} rows but meta.json says {expected_frames} frames")
    period = 1e9 / fps
    synthesized = []
    if "sensor_seq" in cols:
        seq = cols["sensor_seq"]
    else:   # frame-period steps between stamps: a dropped frame shows up as a gap, as with a real sensor_seq
        steps = np.maximum(1, np.rint(np.diff(t) / period)).astype(np.int64)
        seq = np.concatenate([[0], np.cumsum(steps)])
        synthesized.append("sensor_seq")
    if "exposure_ns" in cols:
        exp = cols["exposure_ns"]
    else:
        exp = np.full(n, min(DEFAULT_EXPOSURE_NS, int(period) // 2), np.int64)
        synthesized.append("exposure_ns")
    _write_lines(dst_csv, FRAMES_HEADER, (f"{i},{a},{b},{c}" for i, a, b, c in zip(range(n), t, seq, exp)))
    return {"frame_count": n, "dropped_frames": int(max(0, seq[-1] - seq[0] - (n - 1))),
            "t_first": int(t[0]), "t_last": int(t[-1]), "synthesized": synthesized}


def _convert_imu(src_csv: Path, dst_csv: Path, imu_meta: dict) -> dict:
    units = imu_meta.get("units")
    if units != RAW_COUNTS:
        raise PiEpisodeError(f"imu units {units!r} not supported (expected {RAW_COUNTS!r})")
    try:
        g_lsb, dps_lsb = float(imu_meta["accel_g_per_lsb"]), float(imu_meta["gyro_dps_per_lsb"])
    except (KeyError, TypeError, ValueError):
        raise PiEpisodeError("meta.json imu needs numeric accel_g_per_lsb and gyro_dps_per_lsb") from None
    cols = _read_csv(src_csv, ("sample_index", "t_ns", "gx", "gy", "gz", "ax", "ay", "az"))
    t, k = cols["t_ns"], cols["sample_index"]
    gyro = np.stack([cols[c] for c in ("gx", "gy", "gz")], 1) * (dps_lsb * np.pi / 180.0)    # rad/s
    acc = np.stack([cols[c] for c in ("ax", "ay", "az")], 1) * (g_lsb * STANDARD_GRAVITY)    # m/s^2
    _write_lines(dst_csv, IMU_HEADER, (
        f"{a},{b},{g[0]:.9g},{g[1]:.9g},{g[2]:.9g},{c[0]:.9g},{c[1]:.9g},{c[2]:.9g}"
        for a, b, g, c in zip(t, k, gyro, acc)))
    n = len(t)
    return {"sample_count": n, "dropped_samples": int(max(0, k[-1] - k[0] - (n - 1))),
            "t_first": int(t[0]), "t_last": int(t[-1])}


# --------------------------------------------------------------------------------------------- conversion

def _camera_roles(meta: dict) -> dict[str, dict]:
    """``{role: pi camera entry}``, by sensor model."""
    cams = meta.get("cameras")
    if not isinstance(cams, list) or len(cams) != 2:
        raise PiEpisodeError("meta.json cameras must be a list of two cameras")
    out = {}
    for c in cams:
        model = str(c.get("model", "")).lower() if isinstance(c, dict) else ""
        role = SENSOR_ROLES.get(model)
        if role is None:
            raise PiEpisodeError(f"camera model {model!r} unknown (expected one of {sorted(SENSOR_ROLES)})")
        if role in out:
            raise PiEpisodeError(f"two cameras with model {model!r}")
        if not str(c.get("file", "")).endswith(".mp4"):
            raise PiEpisodeError(f"camera {model}: 'file' must name an .mp4, got {c.get('file')!r}")
        out[role] = c
    return out


def _listed_files(src: Path) -> list[str]:
    names = []
    for ln in (src / MANIFEST).read_text(encoding="utf-8-sig").splitlines():
        if ln.strip():
            name = ln.split(None, 1)[1].lstrip("*")
            names.append(name[2:] if name.startswith("./") else name)
    return names


def convert_episode(src: PiSource, out_dir: Path, rig_id: str) -> dict:
    """Write the contract episode for ``src`` into the new directory ``out_dir`` (data files, ``meta.json``,
    ``MANIFEST.sha256``, ``COMPLETE``; not ``.verified``). Returns the adapter summary stored in meta.json."""
    if not src.ok:
        raise PiEpisodeError(src.problem)
    pm, out = src.meta, Path(out_dir)
    out.mkdir(parents=True)
    roles = _camera_roles(pm)
    cams, synthesized, spans = {}, set(), []
    for role, c in roles.items():
        cam = ROLE_CAM[role]
        stem = c["file"][:-4]
        try:
            fps, size = float(c["fps"]), [int(x) for x in c["size"]]
        except (KeyError, TypeError, ValueError):
            raise PiEpisodeError(f"camera {c.get('model')}: needs numeric fps and size") from None
        fr = _convert_frames(src.path / f"{stem}_frames.csv", out / f"{cam}_frames.csv", fps, c.get("frames"))
        shutil.copyfile(src.path / c["file"], out / f"{cam}.mp4")
        synthesized |= {f"{cam}.{s}" for s in fr["synthesized"]}
        spans.append((fr["t_first"], fr["t_last"]))
        cams[cam] = {"role": role, "sensor": str(c["model"]).lower(), "resolution": size, "fps": fps,
                     "frame_count": fr["frame_count"], "dropped_frames": fr["dropped_frames"]}
    imu_meta = pm.get("imu") or {}
    im = _convert_imu(src.path / "imu.csv", out / "imu.csv", imu_meta)
    spans.append((im["t_first"], im["t_last"]))
    start, stop = max(s[0] for s in spans), min(s[1] for s in spans)
    if not start < stop:
        raise PiEpisodeError("camera and IMU streams do not overlap in time")
    extras = [n for n in _listed_files(src.path) if n not in DATA_FILES]
    for name in extras:                      # e.g. a Pi log (or truth.npz in tests): carried over verbatim
        shutil.copyfile(src.path / name, out / name)
    anchor_clock = pm.get("t_end_boottime_ns")
    if not isinstance(anchor_clock, int) or isinstance(anchor_clock, bool):
        anchor_clock = stop
    adapter = {"version": ADAPTER_VERSION, "source_path": str(src.path), "source_name": src.name,
               "source_manifest_sha256": src.manifest_sha256,
               "synthesized": sorted(synthesized) + ["button_start_ns", "button_stop_ns", "wall_anchor",
                                                     "session_id"],
               "notes": "button times = overlap of all streams (whole recording); wall_anchor = (t_end_boottime_ns, "
                        "COMPLETE mtime), approximate"}
    meta = {
        "schema": SCHEMA, "session_id": session_id_for(src), "episode_number": src.episode_number,
        "rig_id": rig_id, "clock": CLOCK,
        "wall_anchor": {"clock_ns": int(anchor_clock), "unix_ns": (src.path / COMPLETE).stat().st_mtime_ns},
        "button_start_ns": int(start), "button_stop_ns": int(stop),
        "cameras": dict(sorted(cams.items())),
        "imu": {"sensor": IMU_SENSOR, "rate_hz": float(imu_meta.get("odr_hz") or 0) or _rate(im),
                "sample_count": im["sample_count"], "dropped_samples": im["dropped_samples"]},
        "pi_meta": pm, "adapter": adapter,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    names = sorted([*DATA_FILES, *extras])
    (out / MANIFEST).write_text("".join(f"{sha256_file(out / n)}  {n}\n" for n in names), encoding="utf-8")
    (out / COMPLETE).write_bytes(b"")
    return adapter


def _rate(im: dict) -> float:
    return (im["sample_count"] - 1) * 1e9 / max(1, im["t_last"] - im["t_first"])


def adapt_into(src: PiSource, incoming: Path, rig_id: str) -> Path:
    """Convert ``src`` into ``incoming/<converted_name>/`` (contract-checked, then ``.verified``, then one rename).
    Built under ``incoming/.adapt-tmp/`` so intake never sees a half-written episode. Raises
    :class:`PiEpisodeError` (nothing left behind) if it cannot be converted or the result breaks the contract."""
    if not src.ok:
        raise PiEpisodeError(src.problem)
    incoming = Path(incoming)
    name = converted_name(src)
    final = incoming / name
    if os.path.lexists(final):
        raise PiEpisodeError(f"{final} already exists")
    tmp = incoming / ADAPT_TMP / name
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        convert_episode(src, tmp, rig_id)
        report = RawEpisode.open(tmp).validate(decode_videos=False)
        if not report.ok:
            raise PiEpisodeError("converted episode breaks the contract: "
                                 + "; ".join(f"{p.code}: {p.message}" for p in report.errors[:5]))
        (tmp / VERIFIED).write_bytes(b"")
        os.rename(tmp, final)
        fsync_dir(incoming)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        try:
            (incoming / ADAPT_TMP).rmdir()
        except OSError:
            pass
    return final


# --------------------------------------------------------------------------------------------- source records

def read_sources(path: Path) -> list[dict]:
    """Rows of ``pi_sources.csv`` (empty list if it does not exist)."""
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_sources(path: Path, rows: list[dict]) -> None:
    """Append rows (``SOURCES_COLUMNS``) to ``pi_sources.csv``, rewriting it atomically."""
    if not rows:
        return
    _write_sources(path, read_sources(path) + rows)


def update_source_status(path: Path, manifest_sha256: str, status: str) -> bool:
    """Set ``status`` of the row of ``manifest_sha256`` (``dataproc run``: ``ingested`` -> ``failed`` and back on
    ``--retry-failed``), rewriting the file atomically. Returns False (nothing written) when there is no such row
    or it already has that status."""
    rows = read_sources(path)
    hit = [r for r in rows if r.get("source_manifest_sha256") == manifest_sha256]
    if not hit or all(r.get("status") == status for r in hit):
        return False
    for r in hit:
        r["status"] = status
    _write_sources(path, rows)
    return True


def _write_sources(path: Path, rows: list[dict]) -> None:
    allrows = [{k: str(r.get(k, "")) for k in SOURCES_COLUMNS} for r in rows]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SOURCES_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(allrows)
    write_atomic(Path(path), buf.getvalue())
