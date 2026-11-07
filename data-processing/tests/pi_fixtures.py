"""Pi-format episodes for tests: :func:`contract_to_pi` turns a contract episode (``tests/fixtures.build_episode``
or ``dataproc sim``) into what the Pi 5 recorder writes into ``/data/incoming`` (docs/PI_ADAPTER.md, "Pi format"):
the IMX219 is the Pi's ``cam0``, CSV headers ``frame_index,t_ns`` / ``sample_index,t_ns,...``, raw int16 IMU counts,
the Pi's ``meta.json`` layout. Extra files (``truth.npz``) are carried over and listed in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from dataproc.raw_episode import COMPLETE, MANIFEST, VERIFIED, RawEpisode

ACCEL_G_PER_LSB = 0.000122
GYRO_DPS_PER_LSB = 0.0175
#: The Pi's file stem per role (its cam0 is the ultrawide IMX219).
PI_CAM = {"ultrawide": "cam0", "main": "cam1"}
MODEL = {"main": "imx519", "ultrawide": "imx219"}


def write_pi_manifest(ep: Path) -> None:
    names = sorted(p.name for p in ep.iterdir() if p.is_file() and p.name not in (MANIFEST, COMPLETE, VERIFIED))
    (ep / MANIFEST).write_text("".join(f"{hashlib.sha256((ep / n).read_bytes()).hexdigest()}  {n}\n"
                                       for n in names))


def contract_to_pi(src: Path, dst: Path, episode: int, *, with_exposure: bool = False, valid: bool = True,
                   problems=(), complete: bool = True) -> Path:
    """Write the Pi-format version of contract episode ``src`` into ``dst`` (created)."""
    raw = RawEpisode.open(src)
    m = raw.meta
    dst.mkdir(parents=True)
    cams = []
    for cam_id, c in sorted(m.cameras.items()):
        stem = PI_CAM[c.role]
        shutil.copyfile(src / f"{cam_id}.mp4", dst / f"{stem}.mp4")
        ft = raw.frames(cam_id)
        head = "frame_index,t_ns" + (",exposure_ns" if with_exposure else "")
        rows = [f"{i},{t}" + (f",{e}" if with_exposure else "")
                for i, (t, e) in enumerate(zip(ft.t_ns, ft.exposure_ns))]
        (dst / f"{stem}_frames.csv").write_text("\n".join([head, *rows]) + "\n")
        cams.append({"file": f"{stem}.mp4", "picamera2_index": 0 if c.role == "main" else 1, "model": MODEL[c.role],
                     "size": list(c.resolution), "fps": int(round(c.fps)), "bitrate": 10_000_000,
                     "frames": c.frame_count, "max_gap_ns": 0, "max_gap_at_s": 0.0})
    cams.sort(key=lambda d: d["file"])
    imu = raw.imu
    g = np.rint(imu.gyro * 180.0 / np.pi / GYRO_DPS_PER_LSB).astype(np.int64)
    a = np.rint(imu.accel / 9.80665 / ACCEL_G_PER_LSB).astype(np.int64)
    assert np.abs(g).max() < 32768 and np.abs(a).max() < 32768
    rows = [f"{k},{t},{x[0]},{x[1]},{x[2]},{y[0]},{y[1]},{y[2]}"
            for k, t, x, y in zip(imu.sample_idx, imu.t_ns, g, a)]
    (dst / "imu.csv").write_text("\n".join(["sample_index,t_ns,gx,gy,gz,ax,ay,az", *rows]) + "\n")
    for p in src.iterdir():
        if p.is_file() and p.name not in {"cam0.mp4", "cam1.mp4", "cam0_frames.csv", "cam1_frames.csv",
                                          "imu.csv", "meta.json", MANIFEST, COMPLETE, VERIFIED}:
            shutil.copyfile(p, dst / p.name)
    t_all = [raw.frames(c).t_ns for c in m.cameras] + [imu.t_ns]
    meta = {"episode": episode, "valid": valid, "problems": list(problems),
            "t_start_boottime_ns": int(min(t[0] for t in t_all)), "t_end_boottime_ns": int(max(t[-1] for t in t_all)),
            "duration_s_requested": 15.0, "cameras": cams,
            "imu": {"address": "0x6a", "odr_hz": float(m.imu.rate_hz), "samples": len(imu.t_ns), "max_gap_ns": 0,
                    "i2c_errors": 0, "units": "raw int16 counts", "accel_g_per_lsb": ACCEL_G_PER_LSB,
                    "gyro_dps_per_lsb": GYRO_DPS_PER_LSB},
            "system_start": {"cpu_temp_c": 45.0, "throttled": "throttled=0x0"}}
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    write_pi_manifest(dst)
    if complete:
        (dst / COMPLETE).touch()
    return dst
