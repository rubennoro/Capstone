"""Simulated raw rig episode from a real iPhone export (brief D10a, P1 §4.8).

For one iPhone episode E:
1. Truth times: E's slot times (``rgbTimes``, or reconstructed for calibrations) mapped onto a fake CLOCK_MONOTONIC
   ``mono(t) = MONO_BASE + (t - recordingStartTime)``; meta ``clock.start`` is chosen so export reproduces E's UNIX
   times exactly.
2. Main stream: every slot (60 fps rig) or every 2nd slot (30 fps rig), minus optional drops. Recorded stamp =
   true capture time - rig offset (+ jitter). Image = E's main frame warped into the physical main camera with the
   inverse remap built for E's *own* main K (``rig.with_virtual_reference``), JPEG q95. Calibrations (no main video)
   get a constant grey frame per slot, so the grid exists.
3. Ultrawide: one frame per non-empty iPhone UW slot (``native``), or a free-running 30 fps clock holding the latest
   iPhone UW frame (``30fps_hold``, timing tests only).
4. Truth poses: ``T_Wz_S = arkit_main_to_vio(T_arkit, rig.T_S_virtual_main_arkit(S), T_Wz_change)``, i.e. the
   pose of sensor S in a z-up VIO world with a different origin and heading.
5. VIO-like TUM trajectory (``<ep>/sim/vio_trajectory.tum``; 60 Hz or UW-rate keys, optional init delay / gap),
   optionally imported with the real ``pose-import`` code (``trajectory/``).
6. IMU synthesised from the truth trajectory (``imu_synth``), static IMU for calibrations.
7. ``meta.sim`` + ``<ep>/sim/truth.npz`` (slot times, ARKit poses, source indices) for the tests.

Raw episodes are written only through ``raw_episode.RawEpisodeWriter``.
"""

from __future__ import annotations

import dataclasses
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from sensor_rig import __version__
from sensor_rig.pose.frames import arkit_main_to_vio, hom, rig_frame
from sensor_rig.pose.trajectory import Trajectory, save_tum
from sensor_rig.raw_episode import RawEpisode, RawEpisodeWriter
from sensor_rig.sim.imu_synth import synth_imu, synth_static_imu
from sensor_rig.sim.iphone_reader import IphoneEpisode
from sensor_rig.timefmt import KIND_TO_NAME, iso_ms, sanitize_session
from sensor_rig.transforms import rot_x, rot_z

SIM_DIR = Path(__file__).resolve().parent
SIM_RIG_PATH = SIM_DIR / "sim_rig.yaml"
SIM_RIG_MAIN30_PATH = SIM_DIR / "sim_rig_main30.yaml"
MONO_BASE = 1_234_567_890_000_000  # ~14 days of uptime; any large positive value works
ARRIVAL_DELAY_NS = 25_000_000
SIM_SUBDIR = "sim"
VIO_TRAJ_FILE = "vio_trajectory.tum"
TRUTH_FILE = "truth.npz"


@dataclass
class SimOptions:
    main_fps: float | None = None          # None: from the rig (>= 50 -> every slot, else every 60/fps-th slot)
    uw_mode: str = "native"                # "native" | "30fps_hold"
    traj_rate: str = "main"                # "main" (one pose per iPhone slot, 60 Hz) | "uw" (UW slots, ~10 Hz)
    traj_frame: str | None = None          # sensor frame of the TUM poses (default rig.vio.trajectory_frame)
    vio_init_delay_s: float = 0.0          # no poses before t0 + delay (VIO initialising)
    vio_gap: tuple[float, float] | None = None   # (start_s, duration_s) without poses (tracking lost)
    ts_jitter_ns: int = 0                  # Gaussian sigma added to recorded camera stamps
    drop_rate: float = 0.0                 # fraction of main frames dropped (never the first/last)
    imu_rate_hz: float = 400.0
    imu_noise: str = "nominal"             # "nominal" (rig.imu.noise) | "none"
    imu_jitter_ns: int = 2000
    max_frames: int | None = None          # use only the first N main slots
    seed: int = 0
    session: str | None = None             # replace the session in NAME (and in the calibration run name)
    vio_world_yaw_deg: float = 73.0
    vio_world_t: tuple[float, float, float] = (0.4, -1.2, 0.3)
    jpeg_quality: int = 95
    import_trajectory: bool = True         # run pose-import's import_trajectory on the simulated TUM file
    force_import: bool = False
    meta_overrides: dict = field(default_factory=dict)


def _replace_session(name: str, session: str | None) -> str:
    if not name or not session:
        return name
    parts = name.split("_")
    parts[2] = sanitize_session(session)
    return "_".join(parts)


def _video_size(path: Path, default: tuple[int, int]) -> tuple[int, int]:
    if not path.is_file():
        return default
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return (int(s.codec_context.width), int(s.codec_context.height))


def _increasing(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.int64).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1
    return t


def _encode(img: np.ndarray, q: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        raise IOError("JPEG encoding failed")
    return buf.tobytes()


def _camera_meta(rig, cam: str, opts: SimOptions, fps: float) -> dict:
    w, h = rig.cameras[cam].resolution
    return {"backend": "sim", "width": int(w), "height": int(h), "fps_nominal": float(fps),
            "sensor_mode": rig.cameras[cam].sensor_mode, "jpeg_quality": int(opts.jpeg_quality)}


def simulate_raw_from_iphone(iphone_dir, rig, raw_root, opts: SimOptions | None = None) -> Path:
    """Write one simulated raw rig episode for the iPhone export ``iphone_dir``; returns the final episode dir.

    ``rig`` is a RigConfig or a path to a rig yaml (default sim rig: ``SIM_RIG_PATH``).
    """
    from sensor_rig.rig_config import load_rig

    opts = opts or SimOptions()
    rig = load_rig(rig) if not hasattr(rig, "cameras") else rig
    if opts.main_fps and float(opts.main_fps) != float(rig.cameras["main"].fps):
        # the raw episode's rig.yaml snapshot must state the simulated main rate (export's fill-mode fps rule and
        # the recorder's live rules read it): T3 found 30 fps sims carrying fps: 60
        import dataclasses as _dc

        rig = rig.replace(cameras={**rig.cameras, "main": _dc.replace(rig.cameras["main"], fps=float(opts.main_fps))})
    E = IphoneEpisode(iphone_dir)
    rng = np.random.default_rng(opts.seed)
    demo = not E.is_calibration
    traj_frame = rig_frame(opts.traj_frame or rig.vio.trajectory_frame)

    # ---- 1. truth times on the fake monotonic clock
    t_unix = E.main_times_unix_ns()
    if opts.max_frames:
        t_unix = t_unix[: int(opts.max_frames)]
    N = len(t_unix)
    t_ref = E.recording_start_unix_ns()
    mono = MONO_BASE + (t_unix - t_ref)
    T_arkit = E.poses_arkit()[:N] if demo else np.zeros((0, 4, 4))
    if demo and len(T_arkit) != N:
        raise ValueError(f"{E.name}: {len(T_arkit)} poses for {N} slots")

    name = _replace_session(E.name, opts.session)
    calib_run = _replace_session(E.json.get("gripperCalibrationRunName") or "", opts.session)

    # per-episode virtual references (the iPhone K of *this* episode, P2 §11.8)
    main_size = _video_size(E.rgb_path, (1920, 1440))
    uw_size = _video_size(E.uw_path, (640, 480))
    rig_src = rig.with_virtual_reference("main", E.K_main, main_size)
    rig_src = rig_src.with_virtual_reference("ultrawide", E.K_uw, uw_size)
    rm_main = rig_src.inverse_remap("main")
    rm_uw = rig_src.inverse_remap("ultrawide")

    # ---- main stream indices and stamps
    main_fps = float(opts.main_fps or rig.cameras["main"].fps)
    step = 1 if main_fps >= 50 else max(1, int(round(60.0 / main_fps)))
    idx = np.arange(0, N, step, dtype=np.int64)
    if opts.drop_rate > 0 and len(idx) > 2:
        n_drop = int(round(opts.drop_rate * (len(idx) - 2)))
        drop = rng.choice(np.arange(1, len(idx) - 1), size=n_drop, replace=False)
        idx = np.delete(idx, drop)
    off_main = int(rig.capture_time_ns("main", 0))
    off_uw = int(rig.capture_time_ns("ultrawide", 0))
    off_imu = int(rig.capture_time_ns("imu", 0))

    def stamps(T_cap: np.ndarray, off: int) -> np.ndarray:
        t = T_cap - off
        if opts.ts_jitter_ns:
            t = t + np.round(rng.normal(0.0, float(opts.ts_jitter_ns), len(t))).astype(np.int64)
        return _increasing(t)

    t_main_raw = stamps(mono[idx], off_main)

    # ---- ultrawide stamps and source frames
    uw_slots = E.uw_slots()
    uw_slots = uw_slots[uw_slots < N]
    if len(uw_slots) == 0:
        raise ValueError(f"{E.name}: no ultrawide frame within the first {N} slots")
    if opts.uw_mode == "native":
        T_uw = mono[uw_slots]
        uw_src = np.arange(len(uw_slots), dtype=np.int64)      # UW video frame per rig UW frame
    elif opts.uw_mode == "30fps_hold":
        period = 1e9 / 30.0
        phase = float(rng.uniform(0.0, period))
        n = int(np.floor((mono[-1] - mono[0] - phase) / period)) + 1
        T_uw = mono[0] + np.round(phase + np.arange(n) * period).astype(np.int64)
        uw_src = np.clip(np.searchsorted(mono[uw_slots], T_uw, side="right") - 1, 0, len(uw_slots) - 1)
    else:
        raise ValueError(f"unknown uw_mode {opts.uw_mode!r}")
    t_uw_raw = stamps(T_uw, off_uw)

    # ---- truth poses, VIO trajectory, IMU
    C = hom(rot_z(opts.vio_world_yaw_deg), opts.vio_world_t)
    X = rig.T_S_virtual_main_arkit(traj_frame)
    noise = rig.imu.noise if opts.imu_noise == "nominal" else None
    if demo:
        T_Wz_S = arkit_main_to_vio(T_arkit, X, C)
        if opts.traj_rate == "main":
            key = np.arange(N)
        elif opts.traj_rate == "uw":
            key = uw_slots.copy()
        else:
            raise ValueError(f"unknown traj_rate {opts.traj_rate!r}")
        keep = mono[key] >= mono[0] + int(round(opts.vio_init_delay_s * 1e9))
        if opts.vio_gap is not None:
            a = mono[0] + int(round(opts.vio_gap[0] * 1e9))
            d = int(round(opts.vio_gap[1] * 1e9))
            keep &= ~((mono[key] >= a) & (mono[key] < a + d))
        key = key[keep]
        traj = Trajectory.from_T(mono[key], T_Wz_S[key], frame=traj_frame)
        T_W_imu = T_Wz_S @ rig.T(traj_frame, "imu")
        t_imu, gyro, acc = synth_imu(mono, T_W_imu, opts.imu_rate_hz, noise, opts.seed, opts.imu_jitter_ns)
    else:
        traj = None
        # gripper held still-ish while the fingers move: static IMU, tilted so no axis is trivially aligned
        R_W_imu = (rot_z(20.0) @ rot_x(-70.0))
        t_imu, gyro, acc = synth_static_imu(int(mono[0]) - 500_000_000, int(mono[-1]) + 500_000_000,
                                            opts.imu_rate_hz, R_W_imu, noise, opts.seed, opts.imu_jitter_ns)

    # ---- meta
    t_last = int(max(t_main_raw[-1], t_uw_raw[-1]))
    clock_start = {"monotonic_ns": MONO_BASE, "monotonic_raw_ns": MONO_BASE, "realtime_ns": int(t_ref),
                   "uncertainty_ns": 0}
    stop_mono = t_last + 50_000_000
    clock_stop = {"monotonic_ns": stop_mono, "monotonic_raw_ns": stop_mono,
                  "realtime_ns": int(t_ref) + (stop_mono - MONO_BASE), "uncertainty_ns": 0}
    sim_meta = {
        "source_episode": str(E.folder), "source_name": E.name,
        "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(opts).items()
                    if k != "meta_overrides"},
        "T_Wz_change": C.tolist(), "traj_frame": traj_frame,
        "K_main_src": E.K_main.tolist(), "K_uw_src": E.K_uw.tolist(),
        "n_slots": int(N), "n_main": int(len(idx)), "n_uw": int(len(t_uw_raw)),
        "coverage_inverse": {"main": float(rm_main.coverage), "ultrawide": float(rm_uw.coverage)},
        "truth_note": ("raw stamps = true capture time - rig offset (+jitter); true capture times = MONO_BASE + "
                       "(iPhone slot UNIX time - recordingStartTime); see sim/truth.npz"),
    }
    meta_init = {
        "kind": E.kind, "session": name.split("_")[2], "side": E.parts["side"],
        "gripper_id": E.json.get("gripperID") or rig.export.gripper_id,
        "sides_present": list(E.json.get("sidesPresent") or [E.parts["side"]]),
        "gripper_calibration_run_name": calib_run if demo else "",
        "note": f"simulated from {E.name}",
        "clock": {"start": clock_start, "stop": clock_stop},
        "record_start_mono_ns": MONO_BASE, "record_stop_mono_ns": stop_mono,
        "recording_start_time": E.json.get("recordingStartTime") or iso_ms(int(t_ref)),
        "sensors": {"main": _camera_meta(rig, "main", opts, main_fps),
                    "ultrawide": _camera_meta(rig, "ultrawide", opts, float(rig.cameras["ultrawide"].fps)),
                    "imu0": {"backend": "sim", "odr_hz": float(opts.imu_rate_hz)}},
        "host": {"hostname": socket.gethostname(), "sensor_rig_version": __version__,
                 "python": sys.version.split()[0]},
        "sim": sim_meta,
    }
    if opts.meta_overrides:
        from sensor_rig.raw_episode import deep_merge

        meta_init = deep_merge(meta_init, opts.meta_overrides)

    w = RawEpisodeWriter(raw_root, name, rig, meta_init)
    try:
        # ---- main frames
        cw = w.camera_writer("main")
        q = opts.jpeg_quality
        if demo:
            want = {int(i): k for k, i in enumerate(idx)}
            frames = E.iter_main_frames(bgr=True)
            for i, img in enumerate(frames):
                if i >= N:
                    break
                k = want.get(i)
                if k is None:
                    continue
                t_raw = int(t_main_raw[k])
                p = w.frame_path("main", t_raw)
                p.write_bytes(_encode(rm_main.apply(np.ascontiguousarray(img)), q))
                cw.add(k, t_raw, p.name, {"t_arrival_ns": t_raw + ARRIVAL_DELAY_NS, "sensor_seq": i})
            if cw.count != len(idx):
                raise ValueError(f"{E.name}: main video has fewer frames than slots ({cw.count} < {len(idx)})")
        else:
            mw, mh = rig.cameras["main"].resolution
            grey = _encode(np.full((mh, mw, 3), 128, np.uint8), q)
            for k, i in enumerate(idx):
                t_raw = int(t_main_raw[k])
                p = w.frame_path("main", t_raw)
                p.write_bytes(grey)
                cw.add(k, t_raw, p.name, {"t_arrival_ns": t_raw + ARRIVAL_DELAY_NS, "sensor_seq": int(i)})

        # ---- ultrawide frames (each source frame warped + encoded once, reused while held)
        uw = w.camera_writer("ultrawide")
        need = int(uw_src.max()) + 1
        cache: dict[int, bytes] = {}
        k_out = 0
        for j, img in enumerate(E.iter_uw_frames(bgr=True)):
            if j >= need:
                break
            cache[j] = _encode(rm_uw.apply(np.ascontiguousarray(img)), q)
            while k_out < len(uw_src) and uw_src[k_out] <= j:
                t_raw = int(t_uw_raw[k_out])
                p = w.frame_path("ultrawide", t_raw)
                p.write_bytes(cache[int(uw_src[k_out])])
                uw.add(k_out, t_raw, p.name, {"t_arrival_ns": t_raw + ARRIVAL_DELAY_NS,
                                              "sensor_seq": int(uw_src[k_out])})
                k_out += 1
            for old in [x for x in cache if x < (uw_src[k_out] if k_out < len(uw_src) else need)]:
                del cache[old]
        if k_out != len(uw_src):
            raise ValueError(f"{E.name}: ultrawide video has too few frames ({k_out} < {len(uw_src)})")

        # ---- IMU
        w.imu_writer().add_batch(t_imu - off_imu, gyro, acc)

        # ---- sim artefacts (inside the episode, hidden from nothing: readers ignore unknown dirs)
        sd = w.dir / SIM_SUBDIR
        sd.mkdir()
        np.savez_compressed(
            sd / TRUTH_FILE, t_unix_ns=t_unix, t_ns=mono, T_arkit=T_arkit, main_src_index=idx,
            uw_src_slot=uw_slots[uw_src], uw_src_frame=uw_src, t_main_raw=t_main_raw, t_uw_raw=t_uw_raw,
            T_Wz_change=C, K_main_src=E.K_main, K_uw_src=E.K_uw)
        if traj is not None:
            save_tum(traj, sd / VIO_TRAJ_FILE)
        final = w.finalize()
    except BaseException:
        w.abort("sim_failed", keep=False)
        raise

    if traj is not None and opts.import_trajectory:
        from sensor_rig.pose.stage import import_trajectory

        import_trajectory(RawEpisode(final), final / SIM_SUBDIR / VIO_TRAJ_FILE, fmt="tum", frame=traj_frame,
                          force=opts.force_import, rig=rig)
    return final


def load_truth(raw_episode_dir) -> dict:
    """The simulator's ground truth for a simulated raw episode (``sim/truth.npz``) as a dict of arrays."""
    with np.load(Path(raw_episode_dir) / SIM_SUBDIR / TRUTH_FILE) as z:
        return {k: z[k] for k in z.files}


def find_iphone_episodes(paths) -> list[Path]:
    """Expand episode folders / parent folders into iPhone episode folders (holding exactly one ``<NAME>.json``)."""
    out = []
    for p in map(Path, paths):
        if any(p.glob("*.json")):
            out.append(p)
        else:
            out.extend(sorted(d for d in p.iterdir() if d.is_dir() and any(d.glob("*.json"))))
    return out


def kind_order(path: Path) -> tuple[bool, str]:
    """Sort key: calibrations first."""
    return (f"_{KIND_TO_NAME['gripper_calibration']}_" not in path.name, path.name)
