"""Launcher preflight checks. Each returns (ok: bool, detail: str); all are unit-tested with a failing input."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import numpy as np

from franka_deploy.paths import ForbiddenPathError, check_writable_dir

IMAGECODECS_PIN = "2023.9.18"


def dirs_outside_repo(*dirs) -> tuple[bool, str]:
    try:
        for d in dirs:
            check_writable_dir(d)
        return True, "ok"
    except ForbiddenPathError as e:
        return False, str(e)


def disk_free(path, budget_gb: float) -> tuple[bool, str]:
    p = Path(path)
    while not p.exists():
        p = p.parent
    free = shutil.disk_usage(p).free / 1e9
    return free >= budget_gb, f"{free:.1f} GB free (budget {budget_gb} GB)"


def shm_clean(pattern: str = "/dev/shm/rmq_*") -> tuple[bool, str]:
    left = glob.glob(pattern)
    return not left, "clean" if not left else f"{len(left)} leftover robotmq segments, e.g. {os.path.basename(left[0])}"


def ports_free(endpoints) -> tuple[bool, str]:
    busy = []
    for ep in endpoints:
        host, port = ep.replace("tcp://", "").rsplit(":", 1)
        host = "127.0.0.1" if host in ("localhost", "0.0.0.0", "*") else host
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, int(port)))
        except OSError:
            busy.append(ep)
        finally:
            s.close()
    return not busy, "free" if not busy else f"busy: {busy}"


def checkpoint_ok(path) -> tuple[bool, str]:
    p = Path(path)
    if p.suffix != ".ckpt":
        return False, f"{p} is not a .ckpt"
    if not p.resolve().is_file():
        return False, f"{p} not found"
    return True, f"{p.resolve().stat().st_size / 1e9:.2f} GB"


def checkpoint_staged(server_work, ckpt) -> tuple[bool, str]:
    link = Path(server_work) / "ckpt" / Path(ckpt).name
    if not link.is_symlink() and not link.exists():
        return False, f"not staged: {link} missing"
    if link.resolve() != Path(ckpt).resolve():
        return False, f"{link} -> {link.resolve()} != {Path(ckpt).resolve()}"
    return True, str(link)


def sha256_file(path, cache_dir=None) -> str:
    p = Path(path).resolve()
    st = p.stat()
    key = hashlib.sha256(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cf = Path(cache_dir) / f"sha256_{key}.txt" if cache_dir else None
    if cf and cf.exists():
        return cf.read_text().strip()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    d = h.hexdigest()
    if cf:
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(d)
    return d


def prompt_zarr_sha(path, expected: str | None, cache_dir=None) -> tuple[bool, str]:
    if not Path(path).is_file():
        return False, f"{path} not found"
    d = sha256_file(path, cache_dir)
    if expected and d != expected:
        return False, f"sha256 {d[:16]} != expected {expected[:16]}"
    return True, d


def imagecodecs_pin(python: str) -> tuple[bool, str]:
    r = subprocess.run([python, "-c", "import imagecodecs; print(imagecodecs.__version__)"], capture_output=True,
                       text=True, timeout=60, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    v = r.stdout.strip()
    return v == IMAGECODECS_PIN, f"imagecodecs {v or r.stderr.strip()[-200:]} (pin {IMAGECODECS_PIN})"


def rig_not_default(rig_path) -> tuple[bool, str]:
    from sensor_rig.rig_config import load_rig

    if rig_path is None or Path(str(rig_path)).name == "rig_default.yaml":
        return False, f"rig config is the default ({rig_path}); use the calibrated rig.yaml"
    try:
        rig = load_rig(str(rig_path))
    except Exception as e:
        return False, f"cannot load {rig_path}: {e}"
    if getattr(rig, "placeholder", False):
        return False, f"{rig_path} still has placeholder calibration"
    return True, f"{rig.source_path} fingerprint {rig.fingerprint()[:12]}"


def agent_hz_ok(hz) -> tuple[bool, str]:
    return float(hz) == 20.0, f"agent_update_freq_hz {hz} (trained: 20)"


def camera_ok(camera_client, seconds: float = 1.0, min_fps_ratio: float = 0.9) -> tuple[bool, str]:
    """Packets keep flowing at the configured rate and the policy's frame selection succeeds."""
    import time

    from franka_deploy.tasks.frame_select import get_downsampled_camera_images

    fps = camera_client.config["camera_configs"]["main"]["fps"]
    _, ts0 = camera_client.get_latest_images_dict_THWC(2, extra=0)
    time.sleep(seconds)
    _, ts = camera_client.get_latest_images_dict_THWC(int(fps * seconds * 0.5), extra=0)
    rate = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
    try:
        get_downsampled_camera_images(camera_client, 2, 20.0, {"main": 20.0, "ultrawide": 10.0})
    except Exception as e:
        return False, f"frame selection failed: {e}"
    if ts[-1] <= ts0[-1]:
        return False, "no new packets"
    ok = rate >= min_fps_ratio * fps
    return ok, f"packet rate {rate:.1f} Hz (configured {fps}), frame selection ok"


def bridge_not_reflex(state: dict) -> tuple[bool, str]:
    if state.get("error") or state.get("bridge_state") == "reflex":
        return False, f"bridge in reflex: {state.get('error')}"
    return True, f"bridge {state.get('bridge_state')}"


def fk_near_home(fk_xyz_wxyz, home_xyz, max_dist_m: float) -> tuple[bool, str]:
    d = float(np.linalg.norm(np.asarray(fk_xyz_wxyz)[:3] - np.asarray(home_xyz, float)))
    return d <= max_dist_m, f"FK {1000 * d:.1f} mm from home (max {1000 * max_dist_m:.0f} mm)"


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))
