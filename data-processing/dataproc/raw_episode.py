"""The single reader and validator of a raw episode directory (format: ``docs/CONTRACT.md``, ``dataproc.raw.v1``).

::

    ep = RawEpisode.open(path)                 # never raises for bad content; parse problems are kept
    report = ep.validate(decode_videos=True)   # manifest, meta, CSVs, counts, stamps, gaps, decoded frames
    if report.ok:
        ep.meta.button_start_ns, ep.frames("cam0").t_ns, ep.imu.gyro
        for i, rgb in ep.read_video_frames("cam1", indices=[0, 3, 6]): ...

Every check produces a :class:`Problem` with a stable ``code`` (see :data:`PROBLEM_CODES`), a message and a
severity. ``error`` means the episode breaks the contract; ``warning`` reports something processing must judge
(sensor_seq / sample_idx gaps with their sizes, button times outside a stream's span); thresholds are applied later.

Accessors (:attr:`RawEpisode.meta`, :meth:`RawEpisode.frames`, :attr:`RawEpisode.imu`) raise
:class:`RawEpisodeError` if that part could not be parsed. This module never writes to the episode directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from dataproc.paths import NAME_RE, validate_episode_dir_name

__all__ = [
    "SCHEMA", "CAMERAS", "CAMERA_ROLES", "DATA_FILES", "MANIFEST", "COMPLETE", "VERIFIED", "REQUIRED_FILES",
    "FRAMES_HEADER", "IMU_HEADER", "CLOCKS", "PROBLEM_CODES", "RawEpisodeError", "Problem", "ValidationReport",
    "WallAnchor", "CameraMeta", "ImuMeta", "EpisodeMeta", "FrameTable", "ImuTable", "RawEpisode", "parse_meta",
]

SCHEMA = "dataproc.raw.v1"
CAMERAS = ("cam0", "cam1")
CAMERA_ROLES = {"cam0": "main", "cam1": "ultrawide"}
CLOCKS = ("CLOCK_MONOTONIC", "CLOCK_BOOTTIME")
MANIFEST = "MANIFEST.sha256"
COMPLETE = "COMPLETE"
VERIFIED = ".verified"
DATA_FILES = ("cam0.mp4", "cam1.mp4", "cam0_frames.csv", "cam1_frames.csv", "imu.csv", "meta.json")
REQUIRED_FILES = DATA_FILES + (MANIFEST, COMPLETE)
FRAMES_HEADER = "frame_idx,t_ns,sensor_seq,exposure_ns"
IMU_HEADER = "t_ns,sample_idx,gx,gy,gz,ax,ay,az"

#: Every problem code, with its severity.
PROBLEM_CODES = {
    "episode_is_symlink": "error",      # the episode directory itself is a symlink
    "bad_episode_name": "error",        # the directory name is not [A-Za-z0-9][A-Za-z0-9._-]*
    "not_regular_file": "error",        # an entry is a symlink, directory or other non-regular file
    "missing_file": "error",            # a required file is absent
    "incomplete": "error",              # COMPLETE is absent
    "manifest_malformed": "error",      # a MANIFEST.sha256 line is not "<sha256>  <name>"
    "manifest_missing_entry": "error",  # a data file is not listed in the manifest
    "manifest_listed_missing": "error",  # the manifest lists a file that is absent
    "hash_mismatch": "error",           # sha256 differs from the manifest
    "extra_file": "error",              # a file or directory that is neither listed nor a marker
    "meta_invalid": "error",            # meta.json unreadable or a field is missing / mistyped
    "schema_version": "error",          # meta.json "schema" is not dataproc.raw.v1
    "camera_roles": "error",            # cameras are not exactly cam0=main, cam1=ultrawide
    "csv_header": "error",              # the CSV header line is not the contract header
    "csv_parse": "error",               # a CSV row is not parseable / has the wrong number of columns
    "empty_stream": "error",            # a CSV has no rows
    "frame_idx": "error",               # frame_idx is not 0..n-1
    "t_ns_not_monotonic": "error",      # t_ns is not strictly increasing
    "sequence_not_increasing": "error",  # sensor_seq / sample_idx is not strictly increasing
    "count_mismatch": "error",          # CSV rows != meta frame_count / sample_count
    "exposure_invalid": "error",        # exposure_ns <= 0 or >= one frame period (1e9 / fps)
    "video_unreadable": "error",        # PyAV cannot open or decode the MP4
    "video_resolution_mismatch": "error",  # decoded size != meta resolution
    "decoded_count_mismatch": "error",  # decoded frames != CSV rows
    "button_order": "error",            # button_start_ns >= button_stop_ns
    "button_no_overlap": "error",       # [button_start, button_stop] does not overlap a stream at all
    "button_outside_stream": "warning",  # a stream starts after button_start or ends before button_stop
    "sensor_seq_gap": "warning",        # sensor_seq jumps by more than 1 (dropped frames)
    "imu_sample_gap": "warning",        # sample_idx jumps by more than 1 (dropped samples)
    "drop_counter_mismatch": "warning",  # meta drop counter != total missing from the sequence gaps
}

_HASH_LINE = re.compile(r"([0-9a-fA-F]{64}) [ *](.+)")
_MARKERS = (MANIFEST, COMPLETE, VERIFIED)
_MAX_LISTED_GAPS = 50


class RawEpisodeError(Exception):
    """A part of the raw episode is missing or unparseable (see :meth:`RawEpisode.validate` for details)."""


@dataclass(frozen=True)
class Problem:
    """One finding. ``details`` holds machine-readable extras (gap sizes, counts, ...)."""

    code: str
    message: str
    severity: str
    file: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(code: str, message: str, file: str | None = None, **details) -> "Problem":
        return Problem(code, message, PROBLEM_CODES[code], file, details)

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "file": self.file, "message": self.message,
                "details": self.details}


@dataclass(frozen=True)
class ValidationReport:
    path: Path
    problems: tuple[Problem, ...]

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True if there is no error (warnings allowed)."""
        return not self.errors

    def codes(self) -> set[str]:
        return {p.code for p in self.problems}

    def to_dict(self) -> dict:
        return {"path": str(self.path), "ok": self.ok, "problems": [p.to_dict() for p in self.problems]}


# --------------------------------------------------------------------------------------------- meta.json

@dataclass(frozen=True)
class WallAnchor:
    """One simultaneous reading of the Pi stream clock (``clock_ns``) and CLOCK_REALTIME (``unix_ns``)."""

    clock_ns: int
    unix_ns: int

    def to_unix_ns(self, t_ns: int) -> int:
        return self.unix_ns + (int(t_ns) - self.clock_ns)


@dataclass(frozen=True)
class CameraMeta:
    role: str
    resolution: tuple[int, int]          # (width, height)
    fps: float                           # nominal
    frame_count: int
    dropped_frames: int
    sensor: str | None


@dataclass(frozen=True)
class ImuMeta:
    rate_hz: float                       # nominal
    sample_count: int
    dropped_samples: int
    sensor: str | None


@dataclass(frozen=True)
class EpisodeMeta:
    schema: str
    session_id: str
    episode_number: int
    rig_id: str
    clock: str
    wall_anchor: WallAnchor
    button_start_ns: int
    button_stop_ns: int
    cameras: dict[str, CameraMeta]       # keyed cam0, cam1
    imu: ImuMeta
    raw: dict                            # the full meta.json, including keys this schema does not read

    @property
    def start_unix_ns(self) -> int:
        """Wall-clock time of ``button_start_ns`` (intake orders episodes by this)."""
        return self.wall_anchor.to_unix_ns(self.button_start_ns)

    def camera(self, role: str) -> str:
        """The camera id (``cam0``/``cam1``) with this role (``main``/``ultrawide``)."""
        for cam, c in self.cameras.items():
            if c.role == role:
                return cam
        raise KeyError(f"no camera with role {role!r}")


class _Fields:
    """Reads typed fields from a dict, collecting ``meta_invalid`` problems instead of raising."""

    def __init__(self, d, path: str, problems: list[Problem]):
        """``d=None`` means the object is absent (already reported by the parent)."""
        self.path, self.problems = path, problems
        self.d = d if isinstance(d, dict) else None
        if d is not None and self.d is None:
            self._bad(path, f"must be an object, got {type(d).__name__}")

    def _bad(self, path: str, msg: str):
        self.problems.append(Problem.make("meta_invalid", f"meta.json {path}: {msg}", "meta.json"))
        return None

    def _get(self, key: str, required: bool):
        p = f"{self.path}.{key}" if self.path else key
        if self.d is None:
            return p, None, False
        if key not in self.d or self.d[key] is None:
            if required:
                self._bad(p, "missing required field")
            return p, None, False
        return p, self.d[key], True

    def int(self, key: str, min_value: int | None = None):
        p, v, ok = self._get(key, True)
        if not ok:
            return None
        if isinstance(v, bool) or not isinstance(v, int):
            return self._bad(p, f"expected an integer, got {v!r}")
        if min_value is not None and v < min_value:
            return self._bad(p, f"must be >= {min_value}, got {v}")
        return v

    def num(self, key: str, positive: bool = False):
        p, v, ok = self._get(key, True)
        if not ok:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            return self._bad(p, f"expected a finite number, got {v!r}")
        if positive and v <= 0:
            return self._bad(p, f"must be > 0, got {v}")
        return float(v)

    def str(self, key: str, required: bool = True, pattern: re.Pattern | None = None, choices=None):
        p, v, ok = self._get(key, required)
        if not ok:
            return None
        if not isinstance(v, str) or not v:
            return self._bad(p, f"expected a non-empty string, got {v!r}")
        if pattern is not None and not pattern.fullmatch(v):
            return self._bad(p, f"must match {pattern.pattern}, got {v!r}")
        if choices is not None and v not in choices:
            return self._bad(p, f"must be one of {list(choices)}, got {v!r}")
        return v

    def size(self, key: str):
        p, v, ok = self._get(key, True)
        if not ok:
            return None
        if (not isinstance(v, list) or len(v) != 2
                or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in v)):
            return self._bad(p, f"expected [width, height] positive integers, got {v!r}")
        return (v[0], v[1])

    def sub(self, key: str) -> "_Fields":
        p, v, _ = self._get(key, True)
        return _Fields(v, p, self.problems)


def parse_meta(d: Any) -> tuple[EpisodeMeta | None, list[Problem]]:
    """Validate a decoded meta.json. Returns ``(meta, problems)``; ``meta`` is None if any field is invalid.

    Unknown keys are allowed and kept in ``EpisodeMeta.raw``.
    """
    if not isinstance(d, dict):
        return None, [Problem.make("meta_invalid", f"meta.json: must be a JSON object, got {type(d).__name__}",
                                   "meta.json")]
    problems: list[Problem] = []
    top = _Fields(d, "", problems)
    schema = top.str("schema")
    if schema is not None and schema != SCHEMA:
        problems.append(Problem.make("schema_version", f"meta.json schema is {schema!r}, expected {SCHEMA!r}",
                                     "meta.json", found=schema))
    vals = dict(
        session_id=top.str("session_id"),
        episode_number=top.int("episode_number", min_value=0),
        rig_id=top.str("rig_id", pattern=NAME_RE),
        clock=top.str("clock", choices=CLOCKS),
        button_start_ns=top.int("button_start_ns"),
        button_stop_ns=top.int("button_stop_ns"),
    )
    wa = top.sub("wall_anchor")
    anchor = (wa.int("clock_ns"), wa.int("unix_ns"))

    cams: dict[str, CameraMeta] = {}
    cf = top.sub("cameras")
    if cf.d is not None:
        if set(cf.d) != set(CAMERAS):
            problems.append(Problem.make("camera_roles", f"meta.json cameras must be exactly {list(CAMERAS)}, "
                                         f"got {sorted(cf.d)}", "meta.json"))
        for cam in CAMERAS:
            if cam not in cf.d:
                continue
            c = cf.sub(cam)
            role = c.str("role")
            if role is not None and role != CAMERA_ROLES[cam]:
                problems.append(Problem.make("camera_roles", f"meta.json cameras.{cam}.role is {role!r}, "
                                             f"expected {CAMERA_ROLES[cam]!r}", "meta.json"))
            cvals = (role, c.size("resolution"), c.num("fps", positive=True), c.int("frame_count", 0),
                     c.int("dropped_frames", 0), c.str("sensor", required=False))
            cams[cam] = CameraMeta(*cvals)
    im = top.sub("imu")
    imu_fields = (im.num("rate_hz", positive=True), im.int("sample_count", 0), im.int("dropped_samples", 0),
                  im.str("sensor", required=False))

    if problems:
        return None, problems
    return EpisodeMeta(schema=schema, wall_anchor=WallAnchor(*anchor), cameras=cams, imu=ImuMeta(*imu_fields),
                       raw=d, **vals), problems


# --------------------------------------------------------------------------------------------- CSV tables

@dataclass(frozen=True, eq=False)
class FrameTable:
    """``camX_frames.csv``: row i describes decoded frame i. All int64."""

    frame_idx: np.ndarray
    t_ns: np.ndarray
    sensor_seq: np.ndarray
    exposure_ns: np.ndarray

    def __len__(self) -> int:
        return len(self.t_ns)


@dataclass(frozen=True, eq=False)
class ImuTable:
    """``imu.csv``. ``t_ns``/``sample_idx`` int64; ``gyro`` (N,3) rad/s and ``accel`` (N,3) m/s^2, float64."""

    t_ns: np.ndarray
    sample_idx: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray

    def __len__(self) -> int:
        return len(self.t_ns)


def _read_csv(path: Path, header: str, int_cols: int, n_cols: int) -> tuple[list[np.ndarray] | None, Problem | None]:
    """Columns of a headered numeric CSV: the first ``int_cols`` as int64, the rest float64."""
    name = path.name
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return None, Problem.make("csv_parse", f"{name}: cannot read ({e})", name)
    lines = text.splitlines()
    if not lines or lines[0].strip() != header:
        found = lines[0].strip() if lines else ""
        return None, Problem.make("csv_header", f"{name}: header is {found!r}, expected {header!r}", name)
    body = [ln for ln in lines[1:] if ln.strip()]
    if not body:
        return [np.zeros(0, np.int64)] * int_cols + [np.zeros(0)] * (n_cols - int_cols), None
    int_prefix = re.compile(r"(?:-?[0-9]{1,18},){%d}" % int_cols)     # <= 18 digits: always fits int64
    for i, ln in enumerate(body):
        if ln.count(",") != n_cols - 1:
            return None, Problem.make("csv_parse", f"{name}: data row {i} has {ln.count(',') + 1} columns, "
                                      f"expected {n_cols}", name, row=i)
        if not int_prefix.match(ln + ","):
            return None, Problem.make("csv_parse", f"{name}: data row {i}: the first {int_cols} columns must be "
                                      f"integers of at most 18 digits: {ln!r}", name, row=i)
    try:
        ints = np.loadtxt(io.StringIO("\n".join(body)), delimiter=",", dtype=np.int64, usecols=range(int_cols),
                          ndmin=2)
        floats = (np.loadtxt(io.StringIO("\n".join(body)), delimiter=",", dtype=np.float64,
                             usecols=range(int_cols, n_cols), ndmin=2) if n_cols > int_cols else None)
    except ValueError as e:
        return None, Problem.make("csv_parse", f"{name}: {e}", name)
    if floats is not None and not np.all(np.isfinite(floats)):
        return None, Problem.make("csv_parse", f"{name}: non-finite values", name)
    cols = [ints[:, j] for j in range(int_cols)]
    if floats is not None:
        cols += [floats[:, j] for j in range(n_cols - int_cols)]
    return cols, None


def _is_regular(path: Path) -> bool:
    """True for a regular file that is not a symlink."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gaps(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices i where ``seq[i+1] - seq[i] > 1`` and the number missing there."""
    d = np.diff(seq)
    idx = np.flatnonzero(d > 1)
    return idx, d[idx] - 1


# --------------------------------------------------------------------------------------------- the episode

class RawEpisode:
    """A raw episode directory. Use :meth:`open`."""

    def __init__(self, path: Path):
        self.path = path
        self._meta: EpisodeMeta | None = None
        self._frames: dict[str, FrameTable] = {}
        self._imu: ImuTable | None = None
        self._load_problems: list[Problem] = []

    @classmethod
    def open(cls, path: str | os.PathLike) -> "RawEpisode":
        """Parse meta.json and the three CSVs. Raises only if ``path`` is not a directory."""
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"not an episode directory: {path}")
        ep = cls(path)
        ep._load()
        return ep

    # ---- loading
    def _load(self) -> None:
        P = self._load_problems
        mp = self.path / "meta.json"
        if _is_regular(mp):
            try:
                raw = json.loads(mp.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
                P.append(Problem.make("meta_invalid", f"meta.json: not valid JSON ({e})", "meta.json"))
            else:
                self._meta, probs = parse_meta(raw)
                P.extend(probs)
        for cam in CAMERAS:
            f = self.path / f"{cam}_frames.csv"
            if _is_regular(f):
                cols, prob = _read_csv(f, FRAMES_HEADER, 4, 4)
                if prob:
                    P.append(prob)
                else:
                    self._frames[cam] = FrameTable(*cols)
        f = self.path / "imu.csv"
        if _is_regular(f):
            cols, prob = _read_csv(f, IMU_HEADER, 2, 8)
            if prob:
                P.append(prob)
            else:
                self._imu = ImuTable(cols[0], cols[1], np.stack(cols[2:5], 1), np.stack(cols[5:8], 1))

    # ---- accessors
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def meta(self) -> EpisodeMeta:
        if self._meta is None:
            raise RawEpisodeError(f"{self.path}: meta.json is missing or invalid")
        return self._meta

    def frames(self, cam: str) -> FrameTable:
        """The frame table of ``cam0`` or ``cam1``."""
        if cam not in CAMERAS:
            raise KeyError(f"unknown camera {cam!r} (expected one of {list(CAMERAS)})")
        if cam not in self._frames:
            raise RawEpisodeError(f"{self.path}: {cam}_frames.csv is missing or invalid")
        return self._frames[cam]

    @property
    def imu(self) -> ImuTable:
        if self._imu is None:
            raise RawEpisodeError(f"{self.path}: imu.csv is missing or invalid")
        return self._imu

    def video_path(self, cam: str) -> Path:
        if cam not in CAMERAS:
            raise KeyError(f"unknown camera {cam!r} (expected one of {list(CAMERAS)})")
        return self.path / f"{cam}.mp4"

    @property
    def is_complete(self) -> bool:
        return _is_regular(self.path / COMPLETE)

    @property
    def is_verified(self) -> bool:
        return _is_regular(self.path / VERIFIED)

    # ---- video
    def read_video_frames(self, cam: str, indices: Sequence[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, rgb)`` for every decoded frame, or only ``indices`` (strictly increasing, >= 0).

        ``rgb`` is (H, W, 3) uint8. Raises :class:`RawEpisodeError` if the video cannot be decoded or ends before
        the last requested index.
        """
        import av

        wanted = None
        if indices is not None:
            wanted = [int(i) for i in indices]
            if any(i < 0 for i in wanted) or any(b <= a for a, b in zip(wanted, wanted[1:])):
                raise ValueError("indices must be non-negative and strictly increasing")
            if not wanted:
                return
        pos = 0
        try:
            with av.open(str(self.video_path(cam))) as c:
                stream = c.streams.video[0]
                stream.thread_type = "AUTO"
                for i, frame in enumerate(c.decode(stream)):
                    if wanted is None:
                        yield i, frame.to_ndarray(format="rgb24")
                    elif i == wanted[pos]:
                        yield i, frame.to_ndarray(format="rgb24")
                        pos += 1
                        if pos == len(wanted):
                            return
        except (av.error.FFmpegError, IndexError, OSError) as e:
            raise RawEpisodeError(f"{self.video_path(cam)}: cannot decode ({e})") from e
        if wanted is not None:
            raise RawEpisodeError(f"{self.video_path(cam)}: video ends before frame {wanted[pos]}")

    def _probe_video(self, cam: str) -> tuple[int, tuple[int, int]]:
        """(decoded frame count, (width, height)) by decoding every frame without colour conversion."""
        import av

        n, size = 0, None
        with av.open(str(self.video_path(cam))) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in c.decode(stream):
                if size is None:
                    size = (frame.width, frame.height)
                n += 1
        return n, size

    # ---- checks
    def _entry_problems(self) -> list[Problem]:
        """``not_regular_file`` for every directory entry that is not a regular file (symlinks included)."""
        return [Problem.make("not_regular_file", f"{e} is not a regular file (symlink, directory, ...)", e)
                for e in sorted(os.listdir(self.path)) if not _is_regular(self.path / e)]

    def verify_manifest(self) -> list[Problem]:
        """Check ``MANIFEST.sha256`` (``sha256sum`` format) against the directory.

        Every entry must be a regular file; every data file must be listed; every listed file must exist with that
        hash; every other file except ``MANIFEST.sha256``, ``COMPLETE`` and ``.verified`` must be listed. A UTF-8
        BOM and a ``./`` prefix on names are accepted (``sha256sum -c`` accepts both).
        """
        problems = self._entry_problems()
        mpath = self.path / MANIFEST
        if not _is_regular(mpath):
            if not os.path.lexists(mpath):
                problems.append(Problem.make("missing_file", f"{MANIFEST} is missing", MANIFEST))
            return problems
        listed: dict[str, str] = {}
        try:
            lines = mpath.read_bytes().decode("utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            return problems + [Problem.make("manifest_malformed", f"{MANIFEST}: cannot read ({e})", MANIFEST)]
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            m = _HASH_LINE.fullmatch(ln.rstrip("\r"))
            name = m.group(2) if m else None
            if name is not None and name.startswith("./"):
                name = name[2:]
            if not m or "/" in name or name in ("", ".", "..") or name in _MARKERS or name in listed:
                problems.append(Problem.make("manifest_malformed", f"{MANIFEST} line {i + 1}: {ln!r}", MANIFEST,
                                             line=i + 1))
                continue
            listed[name] = m.group(1).lower()
        for name in DATA_FILES:
            if name not in listed:
                problems.append(Problem.make("manifest_missing_entry", f"{name} is not listed in {MANIFEST}", name))
        for name, digest in listed.items():
            p = self.path / name
            if not os.path.lexists(p):
                problems.append(Problem.make("manifest_listed_missing", f"{name} is listed in {MANIFEST} but "
                                             "absent", name))
            elif _is_regular(p) and _sha256(p) != digest:
                problems.append(Problem.make("hash_mismatch", f"{name}: sha256 does not match {MANIFEST}", name))
        for entry in sorted(os.listdir(self.path)):
            if entry not in listed and entry not in _MARKERS and _is_regular(self.path / entry):
                problems.append(Problem.make("extra_file", f"{entry} is not listed in {MANIFEST}", entry))
        return problems

    def validate(self, decode_videos: bool = True, check_manifest: bool = True) -> ValidationReport:
        """Run every contract check (see module docstring). ``decode_videos=False`` skips decoding the MP4s;
        ``check_manifest=False`` skips hashing (both are the slow parts)."""
        P: list[Problem] = []
        if self.path.is_symlink():
            P.append(Problem.make("episode_is_symlink", f"{self.path} is a symlink, not a directory", None))
        try:
            validate_episode_dir_name(self.path.name)
        except ValueError as e:
            P.append(Problem.make("bad_episode_name", str(e), None))
        for name in REQUIRED_FILES:
            if not os.path.lexists(self.path / name):
                code = "incomplete" if name == COMPLETE else "missing_file"
                P.append(Problem.make(code, f"{name} is missing", name))
        if check_manifest and _is_regular(self.path / MANIFEST):
            P.extend(self.verify_manifest())
        else:
            P.extend(self._entry_problems())
        P.extend(self._load_problems)
        meta = self._meta

        spans: dict[str, tuple[int, int]] = {}
        for cam in CAMERAS:
            ft = self._frames.get(cam)
            if ft is None:
                continue
            fname = f"{cam}_frames.csv"
            if len(ft) == 0:
                P.append(Problem.make("empty_stream", f"{fname} has no rows", fname))
                continue
            spans[cam] = (int(ft.t_ns[0]), int(ft.t_ns[-1]))
            if not np.array_equal(ft.frame_idx, np.arange(len(ft))):
                bad = int(np.flatnonzero(ft.frame_idx != np.arange(len(ft)))[0])
                P.append(Problem.make("frame_idx", f"{fname}: frame_idx must be 0..{len(ft) - 1}; row {bad} has "
                                      f"{int(ft.frame_idx[bad])}", fname, row=bad))
            P.extend(self._check_increasing(ft.t_ns, ft.sensor_seq, fname, "sensor_seq"))
            cm = meta.cameras[cam] if meta else None
            if cm is not None and cm.frame_count != len(ft):
                P.append(Problem.make("count_mismatch", f"{fname} has {len(ft)} rows, meta.json "
                                      f"cameras.{cam}.frame_count is {cm.frame_count}", fname,
                                      rows=len(ft), meta=cm.frame_count))
            if cm is not None:
                period_ns = 1e9 / cm.fps
                bad = np.flatnonzero((ft.exposure_ns <= 0) | (ft.exposure_ns >= period_ns))
                if len(bad):
                    P.append(Problem.make("exposure_invalid", f"{fname}: {len(bad)} row(s) with exposure_ns <= 0 or "
                                          f">= the frame period {period_ns:.0f} ns; first row {int(bad[0])} has "
                                          f"{int(ft.exposure_ns[bad[0]])}", fname, row=int(bad[0]), n_rows=len(bad)))
            P.extend(self._gap_problems(ft.t_ns, ft.sensor_seq, fname, "sensor_seq_gap",
                                        cm.dropped_frames if cm else None, f"cameras.{cam}.dropped_frames"))

        imu = self._imu
        if imu is not None:
            if len(imu) == 0:
                P.append(Problem.make("empty_stream", "imu.csv has no rows", "imu.csv"))
            else:
                spans["imu"] = (int(imu.t_ns[0]), int(imu.t_ns[-1]))
                P.extend(self._check_increasing(imu.t_ns, imu.sample_idx, "imu.csv", "sample_idx"))
                if meta is not None and meta.imu.sample_count != len(imu):
                    P.append(Problem.make("count_mismatch", f"imu.csv has {len(imu)} rows, meta.json "
                                          f"imu.sample_count is {meta.imu.sample_count}", "imu.csv",
                                          rows=len(imu), meta=meta.imu.sample_count))
                P.extend(self._gap_problems(imu.t_ns, imu.sample_idx, "imu.csv", "imu_sample_gap",
                                            meta.imu.dropped_samples if meta else None, "imu.dropped_samples"))

        if meta is not None:
            P.extend(self._button_problems(meta, spans))

        if decode_videos:
            for cam in CAMERAS:
                if _is_regular(self.video_path(cam)):
                    P.extend(self._video_problems(cam, meta))
        return ValidationReport(self.path, tuple(P))

    @staticmethod
    def _check_increasing(t_ns, seq, fname: str, seq_name: str) -> list[Problem]:
        out = []
        dt = np.diff(t_ns)
        if np.any(dt <= 0):
            i = int(np.flatnonzero(dt <= 0)[0])
            out.append(Problem.make("t_ns_not_monotonic", f"{fname}: t_ns not strictly increasing at row {i + 1} "
                                    f"({int(t_ns[i])} -> {int(t_ns[i + 1])})", fname, row=i + 1))
        ds = np.diff(seq)
        if np.any(ds <= 0):
            i = int(np.flatnonzero(ds <= 0)[0])
            out.append(Problem.make("sequence_not_increasing", f"{fname}: {seq_name} not strictly increasing at "
                                    f"row {i + 1} ({int(seq[i])} -> {int(seq[i + 1])})", fname, row=i + 1))
        return out

    @staticmethod
    def _gap_problems(t_ns, seq, fname: str, code: str, meta_drops: int | None, meta_key: str) -> list[Problem]:
        """One warning listing the gaps (row after which values are missing, how many, the time gap), plus a
        warning if the meta drop counter disagrees with the total."""
        idx, missing = _gaps(seq)
        total = int(missing.sum())
        out = []
        if len(idx):
            dt_ms = (t_ns[idx + 1] - t_ns[idx]) / 1e6
            gaps = [{"after_row": int(i), "missing": int(m), "dt_ms": float(d)}
                    for i, m, d in zip(idx[:_MAX_LISTED_GAPS], missing, dt_ms)]
            out.append(Problem.make(code, f"{fname}: {len(idx)} gap(s), {total} missing, largest {int(missing.max())}",
                                    fname, n_gaps=int(len(idx)), total_missing=total, max_missing=int(missing.max()),
                                    max_dt_ms=float(dt_ms.max()), gaps=gaps))
        if meta_drops is not None and meta_drops != total:
            out.append(Problem.make("drop_counter_mismatch", f"meta.json {meta_key} is {meta_drops} but {fname} "
                                    f"sequence gaps total {total}", fname, meta=meta_drops, from_gaps=total))
        return out

    @staticmethod
    def _button_problems(meta: EpisodeMeta, spans: dict[str, tuple[int, int]]) -> list[Problem]:
        b0, b1 = meta.button_start_ns, meta.button_stop_ns
        if b0 >= b1:
            return [Problem.make("button_order", f"meta.json button_start_ns ({b0}) must be < button_stop_ns ({b1})",
                                 "meta.json")]
        out = []
        for stream, (s0, s1) in spans.items():
            if s1 < b0 or s0 > b1:
                out.append(Problem.make("button_no_overlap", f"{stream} spans [{s0}, {s1}] ns, outside the button "
                                        f"interval [{b0}, {b1}]", "meta.json", stream=stream))
            elif s0 > b0 or s1 < b1:
                out.append(Problem.make("button_outside_stream", f"{stream} does not cover the button interval: "
                                        f"starts {max(0, s0 - b0) / 1e6:.1f} ms after start, ends "
                                        f"{max(0, b1 - s1) / 1e6:.1f} ms before stop", "meta.json", stream=stream,
                                        late_start_ms=max(0, s0 - b0) / 1e6, early_stop_ms=max(0, b1 - s1) / 1e6))
        return out

    def _video_problems(self, cam: str, meta: EpisodeMeta | None) -> list[Problem]:
        import av

        fname = f"{cam}.mp4"
        try:
            n, size = self._probe_video(cam)
        except (av.error.FFmpegError, IndexError, OSError) as e:
            return [Problem.make("video_unreadable", f"{fname}: cannot decode ({e})", fname)]
        out = []
        if meta is not None and size is not None and meta.cameras[cam].resolution != size:
            out.append(Problem.make("video_resolution_mismatch", f"{fname} decodes at {size[0]}x{size[1]}, meta.json "
                                    f"says {meta.cameras[cam].resolution[0]}x{meta.cameras[cam].resolution[1]}",
                                    fname))
        ft = self._frames.get(cam)
        if ft is not None and n != len(ft):
            out.append(Problem.make("decoded_count_mismatch", f"{fname} decodes to {n} frames, {cam}_frames.csv "
                                    f"has {len(ft)} rows", fname, decoded=n, rows=len(ft)))
        return out
