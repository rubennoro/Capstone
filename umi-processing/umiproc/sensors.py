"""The rig, SLAM config and lens file the processing reads, and the **effective sensor values** made from them
(docs/DESIGN.md "Where it runs and the data root" and "Sensor values").

The processing never reads the repo's ``configs/`` directly: it reads the data-root copies in ``<root>/rigs/``::

    rigs/<rig>.yaml        copy of configs/rigs/<rig>.yaml   (schema dataproc.rig.v1; umiproc.dp.config.load_rig)
    rigs/<rig>.slam.yaml   copy of configs/slam/<rig>.yaml   (schema umiproc.slam.v1; umiproc.slam_config)
    rigs/<rig>.lens.yaml   optional, written by `umiproc lens-calibrate --apply` (schema umiproc.lens.v1)

:func:`ensure_rig_files` copies a missing rig file or SLAM config from the repo (never overwrites) and checks an
existing copy against the repo; :func:`check_rig_files` is the read-only check (``umiproc status``). A copy may be
edited on purpose (it is what the processing uses), so a difference from the repo is reported, not refused, except
when the copy predates a section the repo added (the copied ``check_rig_copy`` rule) or has a gripper mask in the rig
file (umiproc's mask lives in the SLAM config only).

:func:`load_sensors` returns :class:`SensorValues`:

* lens: the lens file's model when it exists (provenance ``calibrated``, with its date and source), else the rig
  file's nominal ultrawide model (provenance ``nominal, uncalibrated``);
* camera -> IMU: the rig file's CAD ``cameras.ultrawide.T_imu_cam``;
* time offset: 0 (refused otherwise; the measured offset is only a diagnostic);
* IMU noise: the rig file's datasheet densities and assumed random walks (``imu.vio_inflation`` applied for SLAM);
* IMU rate: the LSM6DSOX output data rate the Pi sets, :data:`IMU_RATE_HZ`.

``SensorValues.rig`` is the **effective rig**: the data-root rig with the ultrawide model replaced by the lens
file's, so every stage that reads ``rig.cameras["ultrawide"].model`` (SLAM settings, tag detection, the ultrawide
images) uses the same lens. :meth:`SensorValues.record` is the JSON record stored with every map and episode.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from umiproc.paths import repo_config, rig_file, rig_lens_file, rig_slam_file

__all__ = ["DEFAULT_RIG", "IMU_RATE_HZ", "NOMINAL", "CALIBRATED", "RigFilesError", "SensorError", "CopyState",
           "RigFiles", "check_copy", "check_rig_files", "ensure_rig_files", "SensorValues", "load_sensors",
           "recording_mismatches", "rig_copy_rules"]

DEFAULT_RIG = "rig_pi_v3"
#: Output data rate the Pi sets on the IMU (meta.json ``imu.odr_hz``), by sensor. ORB-SLAM3's ``IMU.Frequency`` (it
#: only scales the noise densities to per-sample values). The measured rate (about 105 Hz) is within the sensor's
#: oscillator tolerance; a recording whose meta.json says otherwise is not this rig's IMU setting.
IMU_RATE_HZ = {"lsm6dsox": 104.0}
NOMINAL, CALIBRATED = "nominal, uncalibrated", "calibrated"
#: Free-text keys ignored when comparing a copy with the repo file (notes, never values).
_IGNORED = {"rig": ("notes", "cameras.ultrawide.slam_mask.source"), "slam": ("mask.source",)}


class RigFilesError(ValueError):
    """A rig file or SLAM config is missing (in the data root and the repo) or its copy cannot be used."""


class SensorError(ValueError):
    """The rig, SLAM config or lens file load but do not fit together (or break a design rule)."""


# --------------------------------------------------------------------------------------------- data-root copies

@dataclass
class CopyState:
    """One data-root copy against its repo file.

    ``state``: ``missing`` (no copy yet: the next command that needs it copies it), ``same``, ``differs`` (see
    ``differences``: the dotted YAML paths whose values differ; the copy is what the processing uses),
    ``no repo file`` (only the copy exists) or ``unreadable`` (``note`` says why)."""

    what: str                       # "rig file" | "SLAM config"
    path: Path                      # the data-root copy
    repo: Path                      # the repo file it comes from
    state: str
    differences: list[str] = field(default_factory=list)
    note: str = ""

    def describe(self) -> str:
        if self.state == "missing":
            return f"not there yet (copied from {self.repo} by the first command that needs it)"
        if self.state == "same":
            return f"same as the repo's {self.repo}"
        if self.state == "differs":
            shown = ", ".join(self.differences[:6]) + (" ..." if len(self.differences) > 6 else "")
            return f"differs from the repo's {self.repo} in: {shown} (the processing uses this copy)"
        if self.state == "no repo file":
            return f"there is no repo file {self.repo} to compare with (the processing uses this copy)"
        return f"unreadable: {self.note}"


@dataclass
class RigFiles:
    """The files :func:`ensure_rig_files` / :func:`check_rig_files` looked at. ``lens`` is None without a lens
    file. ``warnings``: lines for the caller to print."""

    rig_id: str
    rig: CopyState
    slam: CopyState
    lens: Path | None
    warnings: list[str] = field(default_factory=list)


def _same_number(a, b) -> bool:
    return (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
            and not isinstance(b, bool) and float(a) == float(b))


def _equal(a, b) -> bool:
    """Equal YAML values, ``720 == 720.0`` (lists element by element)."""
    if _same_number(a, b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _diff(a, b, prefix: str, ignored: tuple[str, ...]) -> list[str]:
    """Dotted paths where the decoded YAML ``a`` (copy) and ``b`` (repo) differ; mappings are compared key by key,
    everything else as a whole."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            p = f"{prefix}.{k}" if prefix else str(k)
            if p in ignored:
                continue
            if k not in a:
                out.append(f"{p} (only in the repo)")
            elif k not in b:
                out.append(f"{p} (only in the copy)")
            else:
                out += _diff(a[k], b[k], p, ignored)
        return out
    return [] if _equal(a, b) else [prefix or "<top>"]


def check_copy(path: Path, repo: Path, what: str, ignored: tuple[str, ...] = ()) -> CopyState:
    """Compare the data-root copy ``path`` with the repo file ``repo`` (read only)."""
    import yaml

    path, repo = Path(path), Path(repo)
    if not path.is_file():
        return CopyState(what, path, repo, "missing")
    try:
        mine = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        return CopyState(what, path, repo, "unreadable", note=str(e))
    if not repo.is_file():
        return CopyState(what, path, repo, "no repo file")
    try:
        theirs = yaml.safe_load(repo.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        return CopyState(what, path, repo, "unreadable", note=f"the repo file: {e}")
    diffs = _diff(mine, theirs, "", ignored)
    return CopyState(what, path, repo, "differs" if diffs else "same", diffs)


def _repo_files(rig_id: str) -> tuple[Path, Path]:
    return repo_config("rigs", f"{rig_id}.yaml"), repo_config("slam", f"{rig_id}.yaml")


def _available(folder: Path) -> str:
    have = sorted(p.name[:-5] for p in folder.glob("*.yaml") if "." not in p.name[:-5]) if folder.is_dir() else []
    return ", ".join(have) or "none"


def rig_copy_rules(dst: Path, src: Path) -> list[str]:
    """The copied ``check_rig_copy`` (raises for a copy that predates a repo section; returns warnings) plus
    umiproc's rule: no ``cameras.ultrawide.slam_mask`` in the rig file (docs/DESIGN.md "SLAM": the mask is the SLAM
    config's, applied by the front-end; a rig mask would make the copied ``write_euroc`` mask the frames too)."""
    import yaml

    from umiproc.dp.pi_intake import PiIntakeError, _ultrawide_mask, check_rig_copy

    try:
        d = yaml.safe_load(dst.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []                   # load_rig reports it
    if isinstance(d, dict) and _ultrawide_mask(d) is not None:
        raise RigFilesError(f"{dst} has cameras.ultrawide.slam_mask: umiproc's gripper mask is the SLAM config's "
                            f"({rig_slam_file(dst.parent.parent, dst.stem)}), never the rig file's. Remove the "
                            "slam_mask block from the rig file (put polygons you want into the SLAM config's "
                            "mask.polygons instead).")
    try:
        return [f"rig: warning: {w}" for w in check_rig_copy(dst, src)]
    except PiIntakeError as e:
        raise RigFilesError(str(e)) from None


def check_rig_files(root: Path, rig_id: str = DEFAULT_RIG) -> RigFiles:
    """The data-root copies of ``rig_id`` against the repo, read only (nothing is copied or created)."""
    repo_rig, repo_slam = _repo_files(rig_id)
    lens = rig_lens_file(root, rig_id)
    return RigFiles(rig_id, check_copy(rig_file(root, rig_id), repo_rig, "rig file", _IGNORED["rig"]),
                    check_copy(rig_slam_file(root, rig_id), repo_slam, "SLAM config", _IGNORED["slam"]),
                    lens if lens.is_file() else None)


def _copy_new(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` atomically, never over an existing ``dst``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    tmp.write_bytes(src.read_bytes())
    try:
        os.link(tmp, dst)                     # fails if dst appeared meanwhile: never overwrite
    finally:
        tmp.unlink()


def ensure_rig_files(root: Path, rig_id: str = DEFAULT_RIG, log: Callable = print, *,
                     copy_missing: bool = True) -> RigFiles:
    """Make sure ``<root>/rigs/<rig_id>.yaml`` and ``<rig_id>.slam.yaml`` exist (copied from the repo when missing,
    unless ``copy_missing`` is False) and check them against the repo.

    Raises :class:`RigFilesError` when a file is in neither place (nothing is copied then), or a copy breaks a rule
    (see the module docstring). Returns the :class:`RigFiles` (differences become ``warnings``, also logged).
    With ``copy_missing=False`` a missing copy stays missing: the caller reads the repo file instead
    (``RigFiles.rig.repo``), as ``lens-calibrate`` does before the first ``map``."""
    from umiproc.dp.paths import validate_name

    try:
        validate_name(rig_id, "rig id")
    except ValueError as e:
        raise RigFilesError(str(e)) from None
    root = Path(root)
    repo_rig, repo_slam = _repo_files(rig_id)
    dst_rig, dst_slam = rig_file(root, rig_id), rig_slam_file(root, rig_id)
    for dst, src, what in ((dst_rig, repo_rig, "rig file"), (dst_slam, repo_slam, "SLAM config")):
        if not dst.is_file() and not src.is_file():
            raise RigFilesError(f"no {what} for {rig_id}: neither {dst} nor the repo's {src} exists (available in "
                                f"the repo: {_available(src.parent)})")
    if copy_missing:
        for dst, src, what in ((dst_rig, repo_rig, "rig file"), (dst_slam, repo_slam, "SLAM config")):
            if not dst.is_file():
                _copy_new(src, dst)
                log(f"{what}: copied the repo's {src} to {dst}")
    files = check_rig_files(root, rig_id)
    if files.rig.path.is_file():
        files.warnings += rig_copy_rules(files.rig.path, files.rig.repo)
    for st in (files.rig, files.slam):
        if st.state == "unreadable":
            raise RigFilesError(f"{st.what} {st.path}: {st.note}")
        if st.state == "differs":
            files.warnings.append(f"{st.what} {st.path}: {st.describe()}")
    for w in files.warnings:
        log(w)
    return files


# --------------------------------------------------------------------------------------------- effective values

@dataclass(frozen=True, eq=False)
class SensorValues:
    """The values every SLAM run (map and localization) uses. ``rig`` is the effective rig (lens applied);
    ``nominal_rig`` the data-root rig as written. Build it with :func:`load_sensors`."""

    rig_id: str
    rig: object                      # umiproc.dp.config.RigConfig, lens applied
    nominal_rig: object              # umiproc.dp.config.RigConfig, as in the rig file
    slam: object                     # umiproc.slam_config.SlamConfig
    lens: object | None              # umiproc.lens.lensfile.LensFile
    imu_rate_hz: float
    rig_path: Path
    slam_path: Path
    lens_path: Path | None

    @property
    def ultrawide(self):
        return self.rig.cameras["ultrawide"]

    @property
    def lens_provenance(self) -> str:
        return CALIBRATED if self.lens is not None else NOMINAL

    def lens_record(self) -> dict:
        """Where the ultrawide model comes from: ``{"provenance": "calibrated", "date", "source", "file_sha256"}``
        or ``{"provenance": "nominal, uncalibrated", "rig_file_provenance": ...}``."""
        if self.lens is None:
            return {"provenance": NOMINAL, "rig_file_provenance": self.nominal_rig.cameras["ultrawide"].provenance}
        from umiproc.dp.fsutil import sha256_file

        return {"provenance": CALIBRATED, "date": self.lens.date, "source": dict(self.lens.source),
                "file_sha256": sha256_file(Path(self.lens_path))}

    def record(self) -> dict:
        """The effective sensor values as JSON (stored with every map and episode; part of the SLAM fingerprint)."""
        cam = self.ultrawide
        noise = self.rig.imu.noise
        vio = noise.for_vio()
        names = ("gyroscope_noise_density", "gyroscope_random_walk", "accelerometer_noise_density",
                 "accelerometer_random_walk")
        return {
            "rig_id": self.rig_id,
            "ultrawide": {
                "sensor": cam.sensor, "resolution": [int(v) for v in cam.resolution], "fps": float(cam.fps),
                "model": {"type": cam.model.type, "intrinsics": [float(v) for v in cam.model.intrinsics],
                          "distortion": [float(v) for v in cam.model.distortion]},
                "lens": self.lens_record(),
                "T_imu_cam": [[float(v) for v in row] for row in cam.T_imu_cam],
                "T_imu_cam_provenance": f"CAD (rig file, provenance {cam.provenance})",
                "timestamp_ref": cam.timestamp_ref, "time_offset_ns": int(cam.time_offset_ns),
            },
            "imu": {
                "sensor": self.rig.imu.sensor, "rate_hz": float(self.imu_rate_hz),
                "time_offset_ns": int(self.rig.imu.time_offset_ns),
                "noise": {k: float(getattr(noise, k)) for k in names},
                "vio_inflation": {"noise_density": float(noise.inflation_noise),
                                  "random_walk": float(noise.inflation_random_walk)},
                "noise_for_slam": {k: float(getattr(vio, k)) for k in names},
                "provenance": "datasheet noise densities, assumed random walks (rig file comments)",
            },
        }


def recording_mismatches(sensors: SensorValues, pi_meta: dict) -> list[str]:
    """What in a Pi recording's own ``meta.json`` (``PiSource.meta``) does not fit the sensor values: the
    ultrawide's sensor model, size and frame rate, and the IMU's output data rate (``IMU.Frequency`` of the
    settings). Empty when it fits. A map or demo that does not fit must not be processed with these settings."""
    out = []
    cam = sensors.ultrawide
    cams = [c for c in (pi_meta.get("cameras") or []) if isinstance(c, dict)
            and str(c.get("model", "")).lower() == cam.sensor.lower()]
    if len(cams) != 1:
        out.append(f"meta.json has {len(cams)} {cam.sensor} cameras (expected 1)")
    else:
        c = cams[0]
        if [int(v) for v in c.get("size") or []] != [int(v) for v in cam.resolution]:
            out.append(f"the {cam.sensor} recorded {c.get('size')}, the rig file says {list(cam.resolution)}")
        if c.get("fps") is None or float(c["fps"]) != float(cam.fps):
            out.append(f"the {cam.sensor} recorded at {c.get('fps')} fps, the rig file says {cam.fps:g}")
    odr = (pi_meta.get("imu") or {}).get("odr_hz")
    if odr is None or float(odr) != float(sensors.imu_rate_hz):
        out.append(f"the IMU ran at {odr} Hz (meta.json imu.odr_hz), the settings assume {sensors.imu_rate_hz:g} Hz")
    return out


def _with_lens(rig, lens):
    """``rig`` with ``cameras.ultrawide.model`` replaced by the lens file's model (everything else unchanged)."""
    from umiproc.dp.config import CameraModel

    m = lens.model
    cam = dataclasses.replace(rig.cameras["ultrawide"],
                              model=CameraModel("fisheye-equidistant", (m.fx, m.fy, m.cx, m.cy),
                                                (m.k1, m.k2, m.k3, m.k4)))
    return dataclasses.replace(rig, cameras={**rig.cameras, "ultrawide": cam})


def load_sensors(root: Path, rig_id: str = DEFAULT_RIG, log: Callable = print, *,
                 ensure: bool = True) -> SensorValues:
    """Load the data-root rig file, SLAM config and optional lens file of ``rig_id`` (copied from the repo first
    when missing, unless ``ensure`` is False) and return the effective :class:`SensorValues`.

    Raises :class:`RigFilesError` (files) or :class:`SensorError` (they load but do not fit: SLAM config or lens
    file for another rig or resolution, a rig gripper mask, a time offset that is not 0, an IMU with no known
    rate). ``umiproc.dp.config.ConfigError`` / ``umiproc.slam_config.SlamConfigError`` /
    ``umiproc.lens.lensfile.LensFileError`` are raised as :class:`SensorError` with the file's own message."""
    from umiproc.dp.config import ConfigError, load_rig
    from umiproc.lens.lensfile import LensFileError, load_lens
    from umiproc.slam_config import SlamConfigError, load_slam_config

    root = Path(root)
    files = (ensure_rig_files(root, rig_id, log) if ensure else check_rig_files(root, rig_id))
    for st in (files.rig, files.slam):
        if st.state == "missing":
            raise RigFilesError(f"{st.what} {st.path} does not exist yet")
    try:
        rig = load_rig(files.rig.path)
        slam = load_slam_config(files.slam.path)
        lens = load_lens(files.lens) if files.lens is not None else None
    except (ConfigError, SlamConfigError, LensFileError) as e:
        raise SensorError(str(e)) from None
    cam = rig.cameras["ultrawide"]
    if cam.slam_mask is not None:
        raise SensorError(f"{files.rig.path}: cameras.ultrawide.slam_mask must not be set (umiproc's mask is the "
                          f"SLAM config's, {files.slam.path})")
    if slam.rig != rig_id or rig.rig_id != rig_id:
        raise SensorError(f"rig ids do not match: {files.rig.path} is {rig.rig_id}, {files.slam.path} is {slam.rig}, "
                          f"asked for {rig_id}")
    if tuple(slam.mask.resolution) != tuple(cam.resolution):
        raise SensorError(f"{files.slam.path}: mask.resolution {list(slam.mask.resolution)} is not the ultrawide's "
                          f"{list(cam.resolution)} ({files.rig.path})")
    for what, v in (("cameras.ultrawide.time_offset_ns", cam.time_offset_ns),
                    ("imu.time_offset_ns", rig.imu.time_offset_ns)):
        if v != 0:
            raise SensorError(f"{files.rig.path}: {what} is {v}: umiproc uses a time offset of 0 (docs/DESIGN.md "
                              "\"SLAM\"; a measured offset is only reported as a diagnostic). Set it to 0.")
    rate = IMU_RATE_HZ.get(rig.imu.sensor.lower())
    if rate is None:
        raise SensorError(f"{files.rig.path}: imu.sensor {rig.imu.sensor!r}: no known output data rate (known: "
                          f"{sorted(IMU_RATE_HZ)})")
    eff = rig
    if lens is not None:
        problems = []
        if lens.rig_id != rig_id:
            problems.append(f"it is for rig {lens.rig_id}")
        if lens.camera != "ultrawide":
            problems.append(f"it is for camera {lens.camera!r}")
        if lens.sensor.lower() != cam.sensor.lower():
            problems.append(f"it is for sensor {lens.sensor}, the rig's ultrawide is {cam.sensor}")
        if tuple(lens.resolution) != tuple(cam.resolution):
            problems.append(f"it was measured at {lens.resolution[0]}x{lens.resolution[1]}, the rig's ultrawide "
                            f"records {cam.resolution[0]}x{cam.resolution[1]}")
        if problems:
            raise SensorError(f"lens file {files.lens}: " + "; ".join(problems) + ". Re-run `umiproc lens-calibrate "
                              "... --apply` for this camera, or move the file away to use the nominal lens.")
        eff = _with_lens(rig, lens)
    return SensorValues(rig_id=rig_id, rig=eff, nominal_rig=rig, slam=slam, lens=lens, imu_rate_hz=rate,
                        rig_path=files.rig.path, slam_path=files.slam.path, lens_path=files.lens)
