"""Tiny raw-episode fixtures (docs/CONTRACT.md) and helpers that break them in one specific way.

``build_episode(dir)`` writes a valid, verified episode with short H.264 MP4s; the ``corrupt_*`` / ``make_*``
helpers edit it in place and (unless the point is a bad hash) rewrite ``MANIFEST.sha256`` so that only the intended
problem remains.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np

from dataproc.raw_episode import COMPLETE, FRAMES_HEADER, IMU_HEADER, MANIFEST, VERIFIED

T0_NS = 1_000_000_000_000          # Pi CLOCK_MONOTONIC at the first sample
UNIX0_NS = 1_793_000_000_000_000_000
SIZES = {"cam0": (64, 36), "cam1": (64, 48)}
FPS = {"cam0": 60, "cam1": 30}
IMU_HZ = 200


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        st.options = {"crf": "10"}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def _frames(n: int, size: tuple[int, int], seed: int) -> list[np.ndarray]:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    return [np.stack([(xx * 4 + i * 8) % 256, (yy * 5 + seed * 40) % 256, np.full_like(xx, (i * 20) % 256)],
                     -1).astype(np.uint8) for i in range(n)]


def write_frames_csv(path: Path, t_ns, sensor_seq, exposure_ns=5_000_000, frame_idx=None) -> None:
    t_ns, sensor_seq = np.asarray(t_ns, np.int64), np.asarray(sensor_seq, np.int64)
    idx = np.arange(len(t_ns)) if frame_idx is None else np.asarray(frame_idx)
    rows = [f"{i},{t},{s},{exposure_ns}" for i, t, s in zip(idx, t_ns, sensor_seq)]
    path.write_text("\n".join([FRAMES_HEADER, *rows]) + "\n")


def write_imu_csv(path: Path, t_ns, sample_idx, gyro, accel) -> None:
    rows = [f"{t},{k},{g[0]:.6f},{g[1]:.6f},{g[2]:.6f},{a[0]:.6f},{a[1]:.6f},{a[2]:.6f}"
            for t, k, g, a in zip(t_ns, sample_idx, gyro, accel)]
    path.write_text("\n".join([IMU_HEADER, *rows]) + "\n")


def write_manifest(ep: Path) -> None:
    """``sha256sum``-format manifest of the six data files (plus any other non-marker file present)."""
    names = sorted(p.name for p in ep.iterdir() if p.is_file() and p.name not in (MANIFEST, COMPLETE, VERIFIED))
    lines = [f"{hashlib.sha256((ep / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (ep / MANIFEST).write_text("\n".join(lines) + "\n")


def build_episode(ep: Path, *, n_cam0: int = 12, n_cam1: int = 6, verified: bool = True,
                  session_id: str = "s01", episode_number: int = 1, rig_id: str = "rig_v1",
                  start_offset_ns: int = 0) -> Path:
    """A valid episode in ``ep`` (created). ``start_offset_ns`` shifts every stamp (for ordering tests)."""
    ep.mkdir(parents=True)
    t0 = T0_NS + start_offset_ns
    spans = []
    for k, (cam, n) in enumerate((("cam0", n_cam0), ("cam1", n_cam1))):
        _write_mp4(ep / f"{cam}.mp4", _frames(n, SIZES[cam], k), FPS[cam])
        t = t0 + np.arange(n, dtype=np.int64) * (1_000_000_000 // FPS[cam])
        write_frames_csv(ep / f"{cam}_frames.csv", t, 100 + np.arange(n))
        spans.append((t[0], t[-1]))
    dur = max(s[1] for s in spans) - t0
    n_imu = int(dur * IMU_HZ // 1_000_000_000) + 1
    t_imu = t0 + np.arange(n_imu, dtype=np.int64) * (1_000_000_000 // IMU_HZ)
    rng = np.random.default_rng(episode_number)
    write_imu_csv(ep / "imu.csv", t_imu, np.arange(n_imu), rng.normal(0, 0.01, (n_imu, 3)),
                  rng.normal(0, 0.05, (n_imu, 3)) + [0.0, 0.0, 9.81])
    end = min(s[1] for s in spans)
    meta = {
        "schema": "dataproc.raw.v1",
        "session_id": session_id,
        "episode_number": episode_number,
        "rig_id": rig_id,
        "clock": "CLOCK_MONOTONIC",
        "wall_anchor": {"clock_ns": T0_NS, "unix_ns": UNIX0_NS},
        "button_start_ns": int(t0 + 10_000_000),
        "button_stop_ns": int(end - 10_000_000),
        "cameras": {
            "cam0": {"role": "main", "sensor": "imx519", "resolution": list(SIZES["cam0"]), "fps": FPS["cam0"],
                     "frame_count": n_cam0, "dropped_frames": 0},
            "cam1": {"role": "ultrawide", "sensor": "imx219", "resolution": list(SIZES["cam1"]),
                     "fps": FPS["cam1"], "frame_count": n_cam1, "dropped_frames": 0},
        },
        "imu": {"sensor": "lsm6dsox", "rate_hz": IMU_HZ, "sample_count": n_imu, "dropped_samples": 0},
    }
    write_meta(ep, meta, manifest=False)
    write_manifest(ep)
    (ep / COMPLETE).touch()
    if verified:
        (ep / VERIFIED).touch()
    return ep


# --------------------------------------------------------------------------------------------- edits

def read_meta(ep: Path) -> dict:
    return json.loads((ep / "meta.json").read_text())


def write_meta(ep: Path, meta: dict, manifest: bool = True) -> None:
    (ep / "meta.json").write_text(json.dumps(meta, indent=2))
    if manifest:
        write_manifest(ep)


def _read_rows(path: Path) -> tuple[str, list[list[str]]]:
    lines = path.read_text().splitlines()
    return lines[0], [ln.split(",") for ln in lines[1:]]


def _write_rows(path: Path, header: str, rows: list[list[str]]) -> None:
    path.write_text("\n".join([header, *(",".join(r) for r in rows)]) + "\n")


def remove_file(ep: Path, name: str) -> None:
    """Delete a file without touching the manifest (a partial transfer)."""
    (ep / name).unlink()


def corrupt_hash(ep: Path, name: str = "imu.csv") -> None:
    """Change a file's bytes without updating the manifest."""
    with open(ep / name, "ab") as f:
        f.write(b"\n")


def drop_csv_row(ep: Path, name: str = "cam0_frames.csv") -> None:
    """Remove the last data row (CSV rows no longer match meta / decoded frames); manifest rewritten."""
    header, rows = _read_rows(ep / name)
    _write_rows(ep / name, header, rows[:-1])
    write_manifest(ep)


def make_non_monotonic(ep: Path, name: str = "cam0_frames.csv", row: int = 3) -> None:
    """Set ``t_ns`` of ``row`` equal to the previous row's; manifest rewritten."""
    header, rows = _read_rows(ep / name)
    col = header.split(",").index("t_ns")
    rows[row][col] = rows[row - 1][col]
    _write_rows(ep / name, header, rows)
    write_manifest(ep)


def make_seq_gap(ep: Path, name: str = "cam0_frames.csv", at: int = 4, missing: int = 2) -> None:
    """Add ``missing`` to the sequence column (``sensor_seq`` / ``sample_idx``) from row ``at`` on; manifest
    rewritten. Stamps and counts are left alone, so the only finding is the gap (and the drop-counter warning)."""
    header, rows = _read_rows(ep / name)
    col = 2 if name.endswith("_frames.csv") else 1
    for r in rows[at:]:
        r[col] = str(int(r[col]) + missing)
    _write_rows(ep / name, header, rows)
    write_manifest(ep)

