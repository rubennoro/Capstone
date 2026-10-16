"""Import Kalibr results (camchain-imucam.yaml, imu.yaml) into a rig.yaml.

Kalibr's ``T_cam_imu`` maps IMU points into the camera (our ``T_cam_imu``); rig.yaml stores ``T_imu_cam``.
``timeshift_cam_imu`` (``t_imu = t_cam + shift``) is stored as ``time.kalibr_timeshift_cam_imu_s``; the camera's
``offset_prior_ns`` is kept from the input rig, because it is what the bag converter applied before Kalibr ran.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import re
from pathlib import Path

import numpy as np
import yaml

from sensor_rig.camera_model import model_from_kalibr
from sensor_rig.rig_config import ImuNoise, RigConfig, TimeOffsets
from sensor_rig.transforms import inv_T, make_T, orthonormalize, rotation_angle_deg

__all__ = ["read_kalibr_camchain", "read_kalibr_imu", "import_kalibr", "auto_cam_map", "parse_cam_map"]

CHAIN_TOL_M = 1e-3          # T_cn_cnm1 consistency limits (P2 §4.2)
CHAIN_TOL_DEG = 0.05
WARN = "WARNING: "


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_kalibr_camchain(path) -> dict[str, dict]:
    """Raw camchain(-imucam) yaml: ``{"cam0": {...}, "cam1": {...}}`` in index order."""
    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError(f"{path}: not a Kalibr camchain yaml")
    cams = {k: v for k, v in d.items() if re.fullmatch(r"cam\d+", str(k))}
    if not cams:
        raise ValueError(f"{path}: no cam0/cam1/... entries")
    return dict(sorted(cams.items(), key=lambda kv: int(kv[0][3:])))


def _read_imu_dict(path) -> dict:
    d = yaml.safe_load(Path(path).read_text())
    if isinstance(d, dict) and "imu0" in d and isinstance(d["imu0"], dict):
        d = d["imu0"]                                         # Kalibr results-style nesting
    keys = ("accelerometer_noise_density", "accelerometer_random_walk",
            "gyroscope_noise_density", "gyroscope_random_walk")
    if not isinstance(d, dict) or any(k not in d for k in keys):
        raise ValueError(f"{path}: expected Kalibr / allan_variance_ros imu yaml with keys {list(keys)}")
    return d


def read_kalibr_imu(path) -> ImuNoise:
    """Noise from a Kalibr input ``imu.yaml`` or an allan_variance_ros output (inflation 1)."""
    d = _read_imu_dict(path)
    return ImuNoise(float(d["accelerometer_noise_density"]), float(d["accelerometer_random_walk"]),
                    float(d["gyroscope_noise_density"]), float(d["gyroscope_random_walk"]))


def parse_cam_map(text: str | None) -> dict[str, str] | None:
    """``"cam0=ultrawide,cam1=main"`` -> ``{"cam0": "ultrawide", "cam1": "main"}``."""
    if not text:
        return None
    out = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        if not k.strip() or not v.strip():
            raise ValueError(f"bad --map entry {part!r}; expected camN=<camera name>")
        out[k.strip()] = v.strip()
    return out


def auto_cam_map(chain: dict[str, dict], camera_names) -> dict[str, str]:
    """Match each Kalibr ``rostopic`` against the rig camera names (``/cam_main/image_raw`` -> ``main``)."""
    out = {}
    for cid, entry in chain.items():
        tokens = set(re.split(r"[^A-Za-z0-9]+", str(entry.get("rostopic", ""))))
        hits = [n for n in camera_names if n in tokens]
        if len(hits) != 1:
            raise ValueError(f"{cid}: cannot match rostopic {entry.get('rostopic')!r} to one of the rig cameras "
                             f"{list(camera_names)}; pass --map cam0=<name>,cam1=<name>")
        out[cid] = hits[0]
    if len(set(out.values())) != len(out):
        raise ValueError(f"ambiguous camera mapping {out}; pass --map")
    return out


def import_kalibr(rig: RigConfig, camchain, imu=None, cam_map: dict[str, str] | None = None, *,
                  bag_offset_prior_ns: dict[str, int] | None = None,
                  date: str | None = None) -> tuple[RigConfig, list[str]]:
    """New rig with Kalibr intrinsics/extrinsics/timeshift (and IMU noise) plus a human-readable change log.

    Lines starting with ``"WARNING: "`` need attention (chain inconsistency, re-orthonormalisation, rate mismatch).
    Raises ``ValueError`` on a resolution mismatch (wrong sensor mode) and ``NotImplementedError`` for unsupported
    Kalibr models.
    """
    chain = read_kalibr_camchain(camchain)
    cam_map = cam_map or auto_cam_map(chain, list(rig.cameras))
    log: list[str] = []
    cams = dict(rig.cameras)
    for cid, entry in chain.items():
        if cid not in cam_map:
            raise ValueError(f"{cid}: missing from the camera map {cam_map}")
        name = cam_map[cid]
        if name not in cams:
            raise ValueError(f"{cid}: mapped to {name!r}, which is not a rig camera ({list(cams)})")
        old = cams[name]
        model = model_from_kalibr(entry)
        if model.size != tuple(old.resolution):
            raise ValueError(f"{cid} ({name}): Kalibr resolution {list(model.size)} != cameras.{name}.resolution "
                             f"{list(old.resolution)} (calibrated in a different sensor mode?)")
        log.append(f"cameras.{name}: model {old.model.kind} -> {model.kind}, intrinsics "
                   f"{[round(v, 3) for v in (model.fx, model.fy, model.cx, model.cy)]}, "
                   f"distortion {[float(v) for v in model.dist]}  (from {cid})")
        changes = dict(model=model)
        if "T_cam_imu" in entry:
            T_imu_cam = inv_T(np.asarray(entry["T_cam_imu"], dtype=np.float64))
            R = orthonormalize(T_imu_cam[:3, :3])
            corr = float(np.abs(R - T_imu_cam[:3, :3]).max())
            if corr > 1e-6:
                log.append(f"{WARN}cameras.{name}: T_cam_imu rotation re-orthonormalised (max change {corr:.2e})")
            T_imu_cam = make_T(R, T_imu_cam[:3, 3])
            shift = float(entry.get("timeshift_cam_imu", 0.0))
            changes.update(T_imu_cam=T_imu_cam, placeholder=False,
                           time=TimeOffsets(old.time.offset_prior_ns, shift))
            log.append(f"cameras.{name}: T_imu_cam translation {np.round(T_imu_cam[:3, 3], 5).tolist()} m, "
                       f"rotation changed by {rotation_angle_deg(old.T_imu_cam[:3, :3].T @ R):.3f} deg; "
                       f"timeshift_cam_imu {shift * 1e3:+.3f} ms (offset_prior_ns {old.time.offset_prior_ns} kept)")
            if abs(shift) > 0.010:
                log.append(f"{WARN}cameras.{name}: |timeshift_cam_imu| = {abs(shift) * 1e3:.1f} ms > 10 ms; fix "
                           "time.offset_prior_ns and rerun Kalibr")
        else:
            log.append(f"cameras.{name}: no T_cam_imu (camera-only calibration); extrinsics and time kept, camera "
                       "stays placeholder until kalibr_calibrate_imu_camera is imported")
        if bag_offset_prior_ns and name in bag_offset_prior_ns and \
                int(bag_offset_prior_ns[name]) != old.time.offset_prior_ns:
            log.append(f"{WARN}cameras.{name}: the bag was stamped with offset_prior_ns="
                       f"{bag_offset_prior_ns[name]} but the rig says {old.time.offset_prior_ns}")
        cams[name] = dataclasses.replace(old, **changes)
    for n in rig.cameras:
        if n not in cam_map.values():
            log.append(f"cameras.{n}: not in the camchain, unchanged")

    new = rig.replace(cameras=cams)

    # main <-> ultrawide consistency: Kalibr's T_cn_cnm1 vs our chain inv(T_imu_cn) @ T_imu_cnm1.
    ids = list(chain)
    for prev, cid in zip(ids, ids[1:]):
        if "T_cn_cnm1" not in chain[cid]:
            continue
        a, b = cam_map[cid], cam_map[prev]
        T_k = np.asarray(chain[cid]["T_cn_cnm1"], dtype=np.float64)
        T_ours = new.T(a, b)
        dt = float(np.linalg.norm(T_k[:3, 3] - T_ours[:3, 3]))
        dang = rotation_angle_deg(orthonormalize(T_k[:3, :3]).T @ T_ours[:3, :3])
        msg = f"T_cn_cnm1 {cid}<-{prev} ({a}<-{b}): |dt| = {dt * 1e3:.3f} mm, angle = {dang:.4f} deg"
        if dt > CHAIN_TOL_M or dang > CHAIN_TOL_DEG:
            log.append(f"{WARN}{msg} exceeds {CHAIN_TOL_M * 1e3:.0f} mm / {CHAIN_TOL_DEG} deg (inconsistent "
                       "camchain-imucam: were cam and cam-IMU calibrations from different mounts?)")
        else:
            log.append(f"{msg}: consistent")

    calib = {"source": "kalibr", "date": date or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
             "camchain": {"file": Path(camchain).name, "sha256": _sha256(camchain)}, "cam_map": dict(cam_map)}
    if imu is not None:
        d = _read_imu_dict(imu)
        noise = read_kalibr_imu(imu)
        noise = dataclasses.replace(noise, inflation_noise=rig.imu.noise.inflation_noise,
                                    inflation_random_walk=rig.imu.noise.inflation_random_walk)
        new = new.replace(imu=dataclasses.replace(new.imu, noise=noise, placeholder=False))
        log.append(f"imu.noise: {noise.kalibr_dict()} (vio_inflation kept: x{noise.inflation_noise} density, "
                   f"x{noise.inflation_random_walk} random walk)")
        rate = d.get("update_rate")
        if rate is not None and abs(float(rate) - rig.imu.output_rate_hz) > 0.01 * rig.imu.output_rate_hz:
            log.append(f"{WARN}imu: update_rate {rate} Hz in {Path(imu).name} != imu.output_rate_hz "
                       f"{rig.imu.output_rate_hz}")
        calib["imu"] = {"file": Path(imu).name, "sha256": _sha256(imu)}
    new = new.replace(calibration=calib)
    return new, log
