"""Raw rig episode: the single reader/writer of the on-disk format the Jetson records (P1 §3.1, brief D2).

Layout (EuRoC-like)::

    <raw_root>/<YYYY-MM-DD>/<NAME>/
      meta.json                 see default_meta()
      rig.yaml                  snapshot of the rig config (RigConfig.dump output)
      cam_main/ cam_ultrawide/
        data.csv                "#timestamp [ns],filename"     ascending, one row per frame file
        data/<t_ns>.jpg
        frames_meta.csv         "seq,t_ns,t_arrival_ns,sensor_seq,exposure_ns,gain,encode_ms,bytes"
        drops.csv               "seq,t_ns,reason,count"  (reason in DROP_REASONS)
      imu0/data.csv             EUROC_IMU_HEADER, rad/s and m/s^2 in the IMU frame
      imu0/aux.csv              "t_ns,chip_us,temp_c,batch,host_read_ns"
      logs/
      trajectory/               added by pose-import (trajectory.tum + trajectory.json)

All ``t_ns`` are raw CLOCK_MONOTONIC stamps; rig.yaml time offsets are applied later (pose-prep, export). While
recording, the folder is ``.<NAME>.recording`` (hidden); readers skip hidden folders.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from sensor_rig import __version__
from sensor_rig.timefmt import date_dir, parse_episode_name

RAW_FORMAT = "sensor_rig.raw.v1"
CAM_DIRS = {"main": "cam_main", "ultrawide": "cam_ultrawide"}
IMU_DIR = "imu0"
CAM_INDEX_HEADER = "#timestamp [ns],filename"
EUROC_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
)
FRAMES_META_COLUMNS = ["seq", "t_ns", "t_arrival_ns", "sensor_seq", "exposure_ns", "gain", "encode_ms", "bytes"]
DROPS_HEADER = "seq,t_ns,reason,count"
DROP_REASONS = ("sensor_gap", "queue_full", "encode_error", "write_error", "nonmonotonic_ts")
IMU_AUX_COLUMNS = ["t_ns", "chip_us", "temp_c", "batch", "host_read_ns"]


def _camera_sensor_template() -> dict:
    return {
        "backend": None, "width": None, "height": None, "fps_nominal": None, "sensor_mode": None,
        "exposure_us": None, "gain": None, "jpeg_quality": None, "frames": 0, "first_t_ns": None, "last_t_ns": None,
        "fps_measured": None, "drops": {r: 0 for r in DROP_REASONS},
        "latency_ms_median": None, "encode_ms_p50": None, "encode_ms_p99": None,
    }


def default_meta() -> dict:
    """Every meta.json key (P1 §3.2); null where unknown."""
    clock0 = {"monotonic_ns": None, "monotonic_raw_ns": None, "realtime_ns": None, "uncertainty_ns": None}
    return {
        "format": RAW_FORMAT,
        "name": None,
        "kind": None,
        "session": None,
        "side": None,
        "gripper_id": "default",
        "sides_present": None,
        "gripper_calibration_run_name": "",
        "task_label": None,
        "note": "",
        "device_identifier": None,
        "device_name": None,
        "rig_id": None,
        "rig_fingerprint": None,
        "rig_yaml_sha256": None,
        "status": "recording",
        "abort_reason": None,
        "clock": {"domain": "CLOCK_MONOTONIC", "start": dict(clock0), "stop": dict(clock0)},
        "record_start_mono_ns": None,
        "record_stop_mono_ns": None,
        "recording_start_time": None,
        "sensors": {
            "main": _camera_sensor_template(),
            "ultrawide": _camera_sensor_template(),
            "imu0": {
                "backend": None, "odr_hz": None, "accel_fs_g": None, "gyro_fs_dps": None, "samples": 0,
                "first_t_ns": None, "last_t_ns": None, "rate_measured": None, "fifo_overflows": 0,
                "invalid_samples": 0, "clock_fit": {"slope_ppm": None, "resid_us_p99": None},
            },
        },
        "host": {"hostname": None, "l4t": None, "sensor_rig_version": __version__, "git_commit": None,
                 "python": None},
        "sim": None,
    }


def deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge ``update`` into a copy of ``base`` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _to_jsonable(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json_atomic(path: Path, obj: Any, indent: int | None = 1) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_to_jsonable)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


class _CsvAppender:
    """Line-buffered-in-memory CSV appender with explicit flush/fsync (thread-safe)."""

    def __init__(self, path: Path, header: str):
        self.path = path
        new = not path.exists()
        self._f = open(path, "a")
        if new:
            self._f.write(header + "\n")
        self._lock = threading.Lock()

    def write_lines(self, lines: list[str]) -> None:
        with self._lock:
            self._f.write("".join(line + "\n" for line in lines))

    def flush(self, fsync: bool = False) -> None:
        with self._lock:
            if self._f.closed:
                return
            self._f.flush()
            if fsync:
                os.fsync(self._f.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                os.fsync(self._f.fileno())
                self._f.close()


class CameraIndexWriter:
    """Appends rows for frames whose JPEG already exists under ``data/``. Rows must come in ascending ``t_ns``."""

    def __init__(self, cam_dir: Path):
        self.cam_dir = cam_dir
        self.data_dir = cam_dir / "data"
        self._index = _CsvAppender(cam_dir / "data.csv", CAM_INDEX_HEADER)
        self._meta = _CsvAppender(cam_dir / "frames_meta.csv", ",".join(FRAMES_META_COLUMNS))
        self._lock = threading.Lock()
        self.last_t_ns: int | None = None
        self.first_t_ns: int | None = None
        self.count = 0

    def add(self, seq: int, t_ns: int, filename: str, frame_meta: dict | None = None) -> None:
        t_ns = int(t_ns)
        if "/" in filename or not (self.data_dir / filename).is_file():
            raise FileNotFoundError(f"frame file {self.data_dir / filename} does not exist (write the file first)")
        with self._lock:
            if self.last_t_ns is not None and t_ns <= self.last_t_ns:
                raise ValueError(f"{self.cam_dir.name}: t_ns {t_ns} not after previous {self.last_t_ns}")
            self.last_t_ns = t_ns
            if self.first_t_ns is None:
                self.first_t_ns = t_ns
            self.count += 1
            m = dict(frame_meta or {})
            m["seq"] = seq
            m["t_ns"] = t_ns
            self._index.write_lines([f"{t_ns},{filename}"])
            self._meta.write_lines([",".join(_fmt(m.get(c)) for c in FRAMES_META_COLUMNS)])

    def flush(self, fsync: bool = False) -> None:
        self._index.flush(fsync)
        self._meta.flush(fsync)

    def close(self) -> None:
        self._index.close()
        self._meta.close()


class ImuWriter:
    """Appends IMU batches (strictly increasing time across batches) to imu0/data.csv (+ aux.csv)."""

    def __init__(self, imu_dir: Path):
        self._data = _CsvAppender(imu_dir / "data.csv", EUROC_IMU_HEADER)
        self._aux = _CsvAppender(imu_dir / "aux.csv", ",".join(IMU_AUX_COLUMNS))
        self._lock = threading.Lock()
        self.last_t_ns: int | None = None
        self.first_t_ns: int | None = None
        self.count = 0

    def add_batch(self, t_ns, gyro, accel, aux: dict | None = None) -> None:
        t = np.asarray(t_ns, dtype=np.int64).reshape(-1)
        g = np.asarray(gyro, dtype=np.float64).reshape(-1, 3)
        a = np.asarray(accel, dtype=np.float64).reshape(-1, 3)
        if not (len(t) == len(g) == len(a)):
            raise ValueError("IMU batch arrays differ in length")
        if len(t) == 0:
            return
        with self._lock:
            if np.any(np.diff(t) <= 0) or (self.last_t_ns is not None and t[0] <= self.last_t_ns):
                raise ValueError("IMU timestamps must be strictly increasing")
            self.last_t_ns = int(t[-1])
            if self.first_t_ns is None:
                self.first_t_ns = int(t[0])
            self.count += len(t)
            lines = [
                f"{ti},{gi[0]:.9g},{gi[1]:.9g},{gi[2]:.9g},{ai[0]:.9g},{ai[1]:.9g},{ai[2]:.9g}"
                for ti, gi, ai in zip(t.tolist(), g, a)
            ]
            self._data.write_lines(lines)
            if aux:
                cols = {c: np.asarray(aux[c]).reshape(-1) if c in aux else None for c in IMU_AUX_COLUMNS[1:]}
                aux_lines = []
                for i, ti in enumerate(t.tolist()):
                    vals = [str(ti)] + [_fmt(cols[c][i].item()) if cols[c] is not None else ""
                                        for c in IMU_AUX_COLUMNS[1:]]
                    aux_lines.append(",".join(vals))
                self._aux.write_lines(aux_lines)

    def flush(self, fsync: bool = False) -> None:
        self._data.flush(fsync)
        self._aux.flush(fsync)

    def close(self) -> None:
        self._data.close()
        self._aux.close()


def _dump_rig(rig, path: Path) -> None:
    """Write the rig snapshot. ``rig`` is a RigConfig (``.dump(path)``) or a path to a yaml file (copied verbatim)."""
    if hasattr(rig, "dump"):
        rig.dump(path)
    elif isinstance(rig, (str, os.PathLike)):
        shutil.copyfile(rig, path)
    else:
        raise TypeError("rig must be a RigConfig (with .dump) or a path to a rig yaml")


class RawEpisodeWriter:
    def __init__(self, raw_root: Path, name: str, rig, meta_init: dict | None = None):
        parse_episode_name(name)  # validates the iPhone-style NAME
        self.raw_root = Path(raw_root)
        self.name = name
        self.date_dir = self.raw_root / date_dir(name)
        self.final_dir = self.date_dir / name
        self.dir = self.date_dir / f".{name}.recording"
        if self.final_dir.exists() or self.dir.exists():
            raise FileExistsError(f"episode {name} already exists under {self.date_dir}")
        self.dir.mkdir(parents=True)
        for cam_dir in CAM_DIRS.values():
            (self.dir / cam_dir / "data").mkdir(parents=True)
        (self.dir / IMU_DIR).mkdir()
        (self.dir / "logs").mkdir()
        _dump_rig(rig, self.dir / "rig.yaml")

        meta = default_meta()
        meta["name"] = name
        parts = parse_episode_name(name)
        meta.update({"kind": parts["kind"], "session": parts["session"], "side": parts["side"],
                     "sides_present": [parts["side"]]})
        meta["rig_yaml_sha256"] = _sha256_file(self.dir / "rig.yaml")
        if hasattr(rig, "fingerprint"):
            meta["rig_fingerprint"] = rig.fingerprint()
        export = getattr(rig, "export", None)
        if export is not None:
            meta["device_identifier"] = getattr(export, "device_identifier", None)
            meta["device_name"] = getattr(export, "device_name", None)
            meta["gripper_id"] = getattr(export, "gripper_id", None) or "default"
        meta["rig_id"] = getattr(rig, "rig_name", None)
        self.meta = deep_merge(meta, meta_init or {})
        self.meta["format"] = RAW_FORMAT
        self.meta["status"] = "recording"
        write_json_atomic(self.dir / "meta.json", self.meta)

        self._cams: dict[str, CameraIndexWriter] = {}
        self._drops: dict[str, _CsvAppender] = {}
        self._imu: ImuWriter | None = None
        self._closed = False

    # ---- streams ----
    def camera_writer(self, cam: str) -> CameraIndexWriter:
        if cam not in self._cams:
            self._cams[cam] = CameraIndexWriter(self.dir / CAM_DIRS[cam])
        return self._cams[cam]

    def frame_path(self, cam: str, t_ns: int) -> Path:
        return self.dir / CAM_DIRS[cam] / "data" / f"{int(t_ns)}.jpg"

    def add_drop(self, cam: str, seq: int, t_ns: int, reason: str, count: int = 1) -> None:
        if reason not in DROP_REASONS:
            raise ValueError(f"unknown drop reason {reason!r}")
        if cam not in self._drops:
            self._drops[cam] = _CsvAppender(self.dir / CAM_DIRS[cam] / "drops.csv", DROPS_HEADER)
        self._drops[cam].write_lines([f"{seq},{int(t_ns)},{reason},{int(count)}"])

    def imu_writer(self) -> ImuWriter:
        if self._imu is None:
            self._imu = ImuWriter(self.dir / IMU_DIR)
        return self._imu

    def flush(self) -> None:
        for w in list(self._cams.values()) + list(self._drops.values()):
            w.flush()
        if self._imu is not None:
            self._imu.flush()

    def update_meta(self, **fields) -> None:
        self.meta = deep_merge(self.meta, fields)
        write_json_atomic(self.dir / "meta.json", self.meta)

    # ---- end of episode ----
    def _close_streams(self) -> None:
        # make sure every stream file exists, even if nothing was written
        for cam in CAM_DIRS:
            self.camera_writer(cam)
            if cam not in self._drops:
                self._drops[cam] = _CsvAppender(self.dir / CAM_DIRS[cam] / "drops.csv", DROPS_HEADER)
        self.imu_writer()
        for w in list(self._cams.values()) + list(self._drops.values()) + [self._imu]:
            w.close()

    def finalize(self, status: str = "complete", stats: dict | None = None) -> Path:
        if self._closed:
            raise RuntimeError("episode already finalized/aborted")
        self._close_streams()
        for cam, w in self._cams.items():
            stats_cam = {"frames": w.count, "first_t_ns": w.first_t_ns, "last_t_ns": w.last_t_ns}
            self.meta = deep_merge(self.meta, {"sensors": {cam: stats_cam}})
        imu = self._imu
        self.meta = deep_merge(self.meta, {"sensors": {"imu0": {"samples": imu.count, "first_t_ns": imu.first_t_ns,
                                                                 "last_t_ns": imu.last_t_ns}}})
        if stats:
            self.meta = deep_merge(self.meta, stats)
        self.meta["status"] = status
        write_json_atomic(self.dir / "meta.json", self.meta)
        _fsync_dir(self.dir)
        os.replace(self.dir, self.final_dir)
        _fsync_dir(self.date_dir)
        self._closed = True
        return self.final_dir

    def abort(self, reason: str, keep: bool = False) -> Path | None:
        if self._closed:
            return None
        self._closed = True
        try:
            self._close_streams()
        except Exception:
            pass
        if not keep:
            shutil.rmtree(self.dir, ignore_errors=True)
            return None
        self.meta["status"] = "aborted"
        self.meta["abort_reason"] = reason
        write_json_atomic(self.dir / "meta.json", self.meta)
        dst = self.date_dir / f".{self.name}.aborted"
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(self.dir, dst)
        return dst


def _read_index(path: Path) -> tuple[np.ndarray, list[str]]:
    times: list[int] = []
    files: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t, fn = line.split(",", 1)
            times.append(int(t))
            files.append(fn.strip())
    return np.asarray(times, dtype=np.int64), files


class RawEpisode:
    """Reader for a finalized raw episode."""

    def __init__(self, path: Path):
        self.path = Path(path)
        meta_path = self.path / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} missing (not a raw rig episode)")
        with open(meta_path) as f:
            self.meta: dict = json.load(f)
        if self.meta.get("format") != RAW_FORMAT:
            raise ValueError(f"{self.path}: meta.format is {self.meta.get('format')!r}, expected {RAW_FORMAT!r}")
        required = [self.path / "rig.yaml", self.path / IMU_DIR / "data.csv"]
        required += [self.path / d / "data.csv" for d in CAM_DIRS.values()]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{self.path}: missing {missing}")
        self.name: str = self.meta.get("name") or self.path.name
        self._index_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._rig = None

    @property
    def kind(self) -> str:
        return self.meta["kind"]

    @property
    def is_calibration(self) -> bool:
        return self.meta["kind"] == "gripper_calibration"

    def rig(self, override: str | os.PathLike | None = None):
        from sensor_rig.rig_config import load_rig

        if override is not None:
            return load_rig(override)
        if self._rig is None:
            self._rig = load_rig(self.path / "rig.yaml")
        return self._rig

    def _index(self, cam: str) -> tuple[np.ndarray, list[str]]:
        if cam not in self._index_cache:
            t, files = _read_index(self.path / CAM_DIRS[cam] / "data.csv")
            if len(t) > 1 and np.any(np.diff(t) <= 0):
                raise ValueError(f"{self.path}/{CAM_DIRS[cam]}/data.csv: timestamps not strictly increasing")
            self._index_cache[cam] = (t, files)
        return self._index_cache[cam]

    def camera_times(self, cam: str) -> np.ndarray:
        return self._index(cam)[0].copy()

    def camera_files(self, cam: str) -> list[Path]:
        d = self.path / CAM_DIRS[cam] / "data"
        return [d / fn for fn in self._index(cam)[1]]

    def read_image(self, cam: str, i: int) -> np.ndarray:
        import cv2

        p = self.camera_files(cam)[i]
        buf = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot decode {p}")
        return img

    def imu(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.path / IMU_DIR / "data.csv"
        with open(p) as f:
            lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
        if not lines:
            return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3))
        # time column parsed as int64 (float64 would lose ns precision on large uptimes)
        t = np.array([int(ln.split(",", 1)[0]) for ln in lines], dtype=np.int64)
        vals = np.loadtxt(lines, delimiter=",", usecols=range(1, 7), dtype=np.float64, ndmin=2)
        return t, vals[:, 0:3].copy(), vals[:, 3:6].copy()

    def trajectory_dir(self) -> Path:
        return self.path / "trajectory"

    def trajectory_path(self) -> Path | None:
        p = self.trajectory_dir() / "trajectory.tum"
        return p if p.is_file() else None

    def trajectory_info(self) -> dict | None:
        p = self.trajectory_dir() / "trajectory.json"
        if not p.is_file():
            return None
        with open(p) as f:
            return json.load(f)

    def wall_offset_ns(self) -> int:
        """UNIX ns = CLOCK_MONOTONIC ns + this (one offset per episode, taken at record start, D5)."""
        s = self.meta["clock"]["start"]
        return int(s["realtime_ns"]) - int(s["monotonic_ns"])


def iter_raw_episodes(raw_root: Path) -> Iterator[RawEpisode]:
    """All finalized raw episodes under ``<root>/<date>/<NAME>/`` (hidden dirs skipped), sorted by name."""
    raw_root = Path(raw_root)
    for meta in sorted(raw_root.glob("*/*/meta.json")):
        ep_dir = meta.parent
        if ep_dir.name.startswith(".") or ep_dir.parent.name.startswith("."):
            continue
        yield RawEpisode(ep_dir)


def find_raw_episodes(path: Path) -> list[RawEpisode]:
    """``path`` may be one episode, a date dir, or a raw root."""
    path = Path(path)
    if (path / "meta.json").is_file():
        return [RawEpisode(path)]
    eps = list(iter_raw_episodes(path))
    if not eps:  # a <date> dir
        eps = [RawEpisode(m.parent) for m in sorted(path.glob("*/meta.json")) if not m.parent.name.startswith(".")]
    return eps
