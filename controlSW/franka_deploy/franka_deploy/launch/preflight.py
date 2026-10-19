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


# ------------------------------------------------------------------------------------------------ station checks


def _covered(path: str, allow) -> bool:
    return any(path == a or path.startswith(str(a) + ".") for a in allow)


def station_placeholders(station, run_mode: str, allow=()) -> tuple[bool, str]:
    """Critical placeholder/hw values of the station. Real mode fails on those not listed in `allow_placeholders`;
    sim mode passes and lists them."""
    crit = [path for path, _ in station.placeholders(critical_only=True)]
    if run_mode == "sim":
        return True, f"sim mode, {len(crit)} critical placeholder/hw value(s): {crit}" if crit else "none"
    bad = [c for c in crit if not _covered(c, allow)]
    if bad:
        return False, f"real mode with critical placeholder/hw values (set them, or list them in allow_placeholders): {bad}"
    return True, f"{len(crit)} critical value(s), all allowed" if crit else "none"


def tooling_sensor_compat(station) -> tuple[bool, str]:
    tool, sens = station.ids.get("tooling"), station.ids.get("sensor")
    if tool is None or sens is None:
        return True, "tooling or sensor group not loaded"
    compat = station.get("tooling.sensor_compat", [])
    return sens in compat, f"tooling {tool} / sensor {sens} (sensor_compat: {list(compat)})"


# ------------------------------------------------------------------------------------------------ wireless checks


def websocket_available(python: str, env: dict | None = None) -> tuple[bool, str]:
    """`import simple_websocket` in the receiver's python: without it engineio stays on long-polling (design finding 10)."""
    r = subprocess.run([python, "-c", "import simple_websocket, flask_socketio; print(simple_websocket.__file__)"],
                       capture_output=True, text=True, timeout=60, env=env or {**os.environ, "PYTHONNOUSERSITE": "1"})
    if r.returncode == 0:
        return True, f"simple_websocket + flask_socketio importable ({r.stdout.strip()})"
    return False, (f"{python}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'import failed'}; install "
                   "requirements-wireless.txt (simple-websocket, flask-socketio) in python.realenv")


def read_stats(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def wireless_health_eval(first: dict, last: dict, physical_fps: dict, max_age_s: float, socket_streams,
                         min_fps_ratio: float = 0.9) -> tuple[bool, str]:
    """Judge two receiver stats snapshots (peripherals/wireless_camera.py stats()) taken a check window apart.

    socket_streams: the streams with their own socket (v1: main + ultrawide; v2: main), whose transport must be
    websocket. physical_fps: {stream: device fps}; fps_in of each must be >= min_fps_ratio * it. age_s p95 of main
    <= max_age_s; no reconnect during the window; no size mismatch; no fatal error."""
    probs, parts = [], []
    if last.get("fatal"):
        probs.append(f"receiver fatal: {last['fatal']}")
    for s in socket_streams:
        tr = last["streams"][s].get("transport")
        if tr != "websocket":
            probs.append(f"{s} transport stuck on {tr or 'nothing (no client)'}: install simple-websocket"
                         if tr == "polling" else f"{s} transport {tr or 'none (no client connected)'}")
    for s, fps in physical_fps.items():
        st, st0 = last["streams"][s], first["streams"][s]
        fin = float(st.get("fps_in") or 0.0)
        parts.append(f"{s} {fin:.1f}/{fps:g} fps")
        if fin < min_fps_ratio * float(fps):
            probs.append(f"{s} fps_in {fin:.1f} < {min_fps_ratio:g} x {fps:g}")
        if st.get("reconnects", 0) != st0.get("reconnects", 0):
            probs.append(f"{s} reconnected during the check ({st0.get('reconnects')} -> {st.get('reconnects')})")
        if st.get("size_mismatches", 0):
            probs.append(f"{s} {st['size_mismatches']} size mismatch(es)")
    age = (last["streams"]["main"].get("age_s") or {}).get("p95")
    parts.append(f"age p95 {'-' if age is None else f'{1000 * age:.0f} ms'} (max {1000 * max_age_s:.0f})")
    if age is None or age > max_age_s:
        probs.append(f"main age p95 {age} s > {max_age_s} s" if age is not None else "no published frames (age unknown)")
    parts.append(f"transport {'/'.join(str(last['streams'][s].get('transport')) for s in socket_streams)}")
    parts.append(f"clock {(last.get('clock') or {}).get('mode')}")
    return (not probs), ("; ".join(probs) + " | " if probs else "") + ", ".join(parts)


def wireless_health_ok(stats_path, physical_fps: dict, max_age_s: float, socket_streams, upgrade_timeout_s: float = 5.0,
                       window_s: float = 2.0, min_fps_ratio: float = 0.9, stats_period_s: float = 1.0,
                       sleep=None, clock=None) -> tuple[bool, str]:
    """Link health after the camera is up: within upgrade_timeout_s every socket stream is on websocket, then over a
    window (>= 2 stats periods) fps/age/reconnects/size mismatches are judged by wireless_health_eval."""
    import time

    sleep, clock = sleep or time.sleep, clock or time.monotonic
    t0 = clock()
    first = None
    while True:
        st = read_stats(stats_path)
        if st and all(st["streams"][s].get("transport") == "websocket" for s in socket_streams):
            first = st
            break
        if clock() - t0 > upgrade_timeout_s:
            if st is None:
                return False, f"no receiver stats at {stats_path} within {upgrade_timeout_s} s"
            return wireless_health_eval(st, st, physical_fps, max_age_s, socket_streams, min_fps_ratio)
        sleep(0.1)
    t1 = clock()
    while clock() - t1 < max(window_s, 2 * stats_period_s) or (read_stats(stats_path) or {}).get("t") == first.get("t"):
        if clock() - t1 > window_s + 5 * stats_period_s:
            return False, f"receiver stats at {stats_path} stopped updating"
        sleep(0.1)
    return wireless_health_eval(first, read_stats(stats_path) or first, physical_fps, max_age_s, socket_streams,
                                min_fps_ratio)


def clock_eval(first: dict, last: dict, max_rtt_ms: float = 20.0, max_offset_change_s: float = 0.002) -> tuple[bool, str]:
    """v2 clock sync: min ping RTT < max_rtt_ms and the offset stable within max_offset_change_s between snapshots."""
    c0, c1 = first.get("clock") or {}, last.get("clock") or {}
    if c1.get("mode") != "sender_stamp_ping":
        return True, f"clock mode {c1.get('mode')}: no ping sync to check"
    rtt, o0, o1 = c1.get("rtt_ms_min"), c0.get("offset_s"), c1.get("offset_s")
    if rtt is None or o0 is None or o1 is None:
        return False, f"clock not synced (rtt_ms_min {rtt}, offsets {o0} -> {o1}, pings {c1.get('n_pings')})"
    d = abs(float(o1) - float(o0))
    ok = float(rtt) < max_rtt_ms and d <= max_offset_change_s
    return ok, (f"ping RTT min {float(rtt):.1f} ms (max {max_rtt_ms:g}), offset {1000 * float(o1):.2f} ms, "
                f"change {1000 * d:.2f} ms (max {1000 * max_offset_change_s:g}), drift {c1.get('drift_ppm')}")


def clock_ok(stats_path, window_s: float = 3.0, sleep=None) -> tuple[bool, str]:
    import time

    sleep = sleep or time.sleep
    first = read_stats(stats_path)
    if first is None:
        return False, f"no receiver stats at {stats_path}"
    sleep(window_s)
    return clock_eval(first, read_stats(stats_path) or first)


def hello_check(hello_problems, run_mode: str) -> tuple[bool, str]:
    """v2 hello vs the sensor yaml. Sim: a warning only (the replay sends each episode's own K); real: FAIL."""
    probs = list(hello_problems or [])
    if not probs:
        return True, "hello consistent with the sensor yaml (or v1: no hello)"
    if run_mode == "sim":
        return True, f"WARNING (sim, expected for replays with the episode's own K): {probs}"
    return False, f"hello does not match the sensor yaml: {probs}"
