"""Write one synthetic raw episode in the exact ``dataproc.raw.v1`` format (docs/CONTRACT.md), plus ``truth.npz``.

See docs/SIM.md for the scene, the trajectory, the time-stamp convention and the ``truth.npz`` keys. In short:

* true time = the IMU time base of the rig (the clock all of processing works on); the recording starts at
  ``t0_ns`` (first IMU sample) and lasts ``duration_s``;
* camera frame ``k`` is rendered at its true mid-exposure time ``t_mid``; its CSV ``t_ns`` is chosen so that
  ``rig.camera_time_ns(role, t_ns, exposure_ns) == t_mid`` exactly; IMU CSV ``t_ns`` so that
  ``rig.imu_time_ns(t_ns)`` is the true sample time.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np
import yaml

from dataproc.config import RigConfig, parse_rig
from dataproc.geometry.camera_model import model_from_dict
from dataproc.geometry.transforms import make_T, rot_x, rot_y
from dataproc.raw_episode import COMPLETE, FRAMES_HEADER, IMU_HEADER, MANIFEST, SCHEMA, VERIFIED
from dataproc.sim.imu_synth import add_noise, imu_times
from dataproc.sim.render import CameraRenderer, Quad, build_scene, camera_model, quad_half_extent
from dataproc.sim.trajectory import make_trajectory

__all__ = ["Faults", "SimEpisode", "generate_episode", "finger_tag_poses_virt", "fast_rig", "TRUTH_VERSION",
           "TRUTH_FILE", "EXPOSURE_NS", "FRAME_PHASE_NS", "DEFAULT_IMU_RATE_HZ", "TAG_Y_M", "TAG_TILT_DEG",
           "default_t0_ns", "read_truth", "run_sim", "next_episode_number", "DEFAULT_RIG_PATH"]

TRUTH_VERSION = "dataproc.sim.truth.v1"
TRUTH_FILE = "truth.npz"
CAMS = {"cam0": "main", "cam1": "ultrawide"}
#: Constant exposure per role (ns) and the true mid-exposure time of frame 0 after the recording start (ns).
EXPOSURE_NS = {"main": 4_000_000, "ultrawide": 8_000_000}
FRAME_PHASE_NS = {"main": 3_000_000, "ultrawide": 7_000_000}
DEFAULT_IMU_RATE_HZ = 416.0          # LSM6DSOX ODR
TRUTH_RATE_HZ = 1000.0
#: Finger tags in the virtual ultrawide frame: centre (-+width/2, TAG_Y_M, aruco.nominal_z_m); orientation
#: rot_y(+-TAG_TILT_DEG[1]) @ rot_x(TAG_TILT_DEG[0]) (left: +, right: -), i.e. facing the camera, tilted like
#: iPhUMI's fingers (sensor_rig tests/test_calib_tools.py TAG_POSES).
TAG_Y_M = 0.045
TAG_TILT_DEG = (150.0, 15.0)
_T0_BASE_NS = 1_000_000_000_000     # CLOCK_MONOTONIC ~16 min after boot
_UNIX_BASE_NS = 1_793_000_000_000_000_000
_SEQ0 = {"cam0": 1000, "cam1": 500}
_CRF = 10                            # libx264 constant quality, near-lossless (<= 12), yuv420p
_PRESET = "veryfast"


def default_t0_ns(episode_number: int) -> int:
    """Deterministic, well separated recording start per episode number."""
    return _T0_BASE_NS + 100_000_000_000 * int(episode_number)


@dataclass
class Faults:
    """Faults to inject (for downstream tests).

    * ``drop_frames``: ``{cam: [capture indices]}``: those frames are neither encoded nor listed (CSV and video stay
      consistent; ``sensor_seq`` has gaps; meta ``dropped_frames`` counts them) -> ``sensor_seq_gap`` warning;
    * ``imu_gaps``: ``[(start_s, dur_s)]`` relative to the recording start: samples removed, ``sample_idx`` gap,
      ``dropped_samples`` counts them -> ``imu_sample_gap`` warning;
    * ``count_mismatch``: ``[cam]``: the last CSV row is removed (video and meta keep it) -> ``count_mismatch`` and
      ``decoded_count_mismatch`` errors.
    """

    drop_frames: dict = field(default_factory=dict)
    imu_gaps: list = field(default_factory=list)
    count_mismatch: list = field(default_factory=list)

    @classmethod
    def from_any(cls, f) -> "Faults":
        if f is None:
            return cls()
        if isinstance(f, cls):
            return f
        if isinstance(f, dict):
            unknown = set(f) - {"drop_frames", "imu_gaps", "count_mismatch"}
            if unknown:
                raise ValueError(f"unknown fault(s): {sorted(unknown)}")
            return cls(**{k: v for k, v in f.items()})
        raise TypeError(f"faults must be a Faults, a dict or None, got {type(f).__name__}")

    def check(self) -> None:
        for cam in list(self.drop_frames) + list(self.count_mismatch):
            if cam not in CAMS:
                raise ValueError(f"unknown camera {cam!r} in faults (expected cam0 or cam1)")


@dataclass
class SimEpisode:
    path: Path
    kind: str
    seed: int
    frame_counts: dict
    imu_count: int
    button_start_ns: int
    button_stop_ns: int
    timings_s: dict


def finger_tag_poses_virt(width_m: float, nominal_z_m: float) -> dict:
    """``{"left": T_virt_tag, "right": T_virt_tag}`` (4x4) at tag-centre separation ``width_m``."""
    tx, ty = TAG_TILT_DEG
    return {"left": make_T(rot_y(ty) @ rot_x(tx), [-width_m / 2, TAG_Y_M, nominal_z_m]),
            "right": make_T(rot_y(-ty) @ rot_x(tx), [width_m / 2, TAG_Y_M, nominal_z_m])}


# --------------------------------------------------------------------------------------------- io helpers

def _save_npz(path: Path, arrays: dict) -> None:
    """``np.savez``-compatible, byte-deterministic (fixed zip timestamps)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(v), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, buf.getvalue())


def read_truth(ep_dir) -> dict:
    """``truth.npz`` of an episode as a dict (scalars unwrapped)."""
    with np.load(Path(ep_dir) / TRUTH_FILE, allow_pickle=False) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def _write_manifest(ep: Path) -> None:
    names = sorted(p.name for p in ep.iterdir() if p.is_file() and p.name not in (MANIFEST, COMPLETE, VERIFIED))
    lines = [f"{hashlib.sha256((ep / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (ep / MANIFEST).write_text("\n".join(lines) + "\n")


class _Mp4:
    def __init__(self, path: Path, size: tuple[int, int], fps: float, crf: int, preset: str):
        self.c = av.open(str(path), "w")
        rate = int(round(fps)) if float(fps).is_integer() else fps
        self.st = self.c.add_stream("libx264", rate=rate)
        self.st.width, self.st.height, self.st.pix_fmt = size[0], size[1], "yuv420p"
        self.st.options = {"crf": str(crf), "preset": preset}
        self.n = 0

    def write(self, rgb: np.ndarray) -> None:
        for pkt in self.st.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.c.mux(pkt)
        self.n += 1

    def close(self) -> None:
        for pkt in self.st.encode():
            self.c.mux(pkt)
        self.c.close()


_RENDERERS: dict = {}


def _renderer(cam, ss: int) -> CameraRenderer:
    m = camera_model(cam)
    key = (m.kind, m.width, m.height, tuple(cam.model.intrinsics), tuple(cam.model.distortion), ss)
    if key not in _RENDERERS:
        if len(_RENDERERS) > 4:
            _RENDERERS.clear()
        _RENDERERS[key] = CameraRenderer(m, ss)
    return _RENDERERS[key]


def _noise_bank(rng, shape, sigma: float, n: int = 8):
    """``n`` (pos, neg) uint8 pairs of rounded Gaussian noise, added with saturating cv2.add / cv2.subtract."""
    bank = []
    for _ in range(n):
        z = np.rint(rng.normal(0.0, sigma, shape)).astype(np.int16)
        bank.append((np.clip(z, 0, 255).astype(np.uint8), np.clip(-z, 0, 255).astype(np.uint8)))
    return bank


# --------------------------------------------------------------------------------------------- generator

def _n_frames(dur_ns: int, role: str, fps: float) -> int:
    """Frames whose mid-exposure ``phase + k / fps`` lies within the recording."""
    return int(np.floor((dur_ns - FRAME_PHASE_NS[role]) / (1e9 / fps))) + 1


def generate_episode(out_dir, rig: RigConfig, kind: str = "demonstration", seed: int = 0, duration_s: float = 15.0,
                     preroll_s: float = 2.5, session_id: str = "sim", episode_number: int = 0, faults=None, *,
                     verified: bool = True, imu_rate_hz: float = DEFAULT_IMU_RATE_HZ, imu_noise: bool = True,
                     noise_sigma: float = 1.0, ss: int = 2, scene_seed: int = 0, texels_per_m: float = 400.0,
                     t0_ns: int | None = None, unix_anchor_ns: int | None = None, crf: int = _CRF,
                     preset: str = _PRESET, extra_quads=()) -> SimEpisode:
    """Render and write one episode into ``out_dir`` (created; must not exist or be empty).

    ``kind``: ``demonstration`` (reach, grasp, lift, move; width closed->open->closed) or ``calibration`` (fixed
    pose + wobble, width swept over the full range). ``seed`` fixes the trajectory, IMU noise and image noise;
    ``scene_seed`` the room (the same room for all episodes by default). ``faults``: :class:`Faults` or its dict.
    ``extra_quads``: world-fixed :class:`~dataproc.sim.render.Quad` s (tests). Returns a :class:`SimEpisode`.
    """
    tt = time.perf_counter()
    faults = Faults.from_any(faults)
    faults.check()
    if not 1 <= crf <= 12:
        raise ValueError(f"crf must be in [1, 12] (near-lossless), got {crf}")
    traj = make_trajectory(rig, kind, seed, duration_s, preroll_s)          # validates the arguments
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    t0 = default_t0_ns(episode_number) if t0_ns is None else int(t0_ns)
    dur_ns = int(round(duration_s * 1e9))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x1A6E]))
    scene = build_scene(scene_seed, texels_per_m, rig.aruco.dict,
                        tuple(sorted({rig.aruco.left_id, rig.aruco.right_id, *range(8)})))
    half = quad_half_extent(rig.aruco.dict, rig.aruco.marker_size_m)
    T_uw_virt = make_T(rig.virtual_ultrawide.R_phys_virt)
    T_imu_uw = rig.cameras["ultrawide"].T_imu_cam
    tag_ids = {"left": rig.aruco.left_id, "right": rig.aruco.right_id}
    timings = {}
    for cam in CAMS:                                                         # validate faults before writing
        n_cap = _n_frames(dur_ns, CAMS[cam], rig.cameras[CAMS[cam]].fps)
        if any(not 0 <= int(i) < n_cap for i in faults.drop_frames.get(cam, [])):
            raise ValueError(f"drop_frames[{cam}] indices must be in [0, {n_cap})")
    out.mkdir(parents=True, exist_ok=True)

    # ---- cameras
    frames = {}
    for cam, role in CAMS.items():
        c = rig.cameras[role]
        t1 = time.perf_counter()
        period = 1e9 / c.fps
        n_cap = _n_frames(dur_ns, role, c.fps)
        t_mid = t0 + FRAME_PHASE_NS[role] + np.round(np.arange(n_cap) * period).astype(np.int64)
        exp = EXPOSURE_NS[role]
        if not 0 < exp < period:
            raise ValueError(f"exposure {exp} ns does not fit the {role} frame period")
        s = 1 if c.timestamp_ref == "start_of_exposure" else -1
        t_csv = t_mid - s * (exp // 2) - c.time_offset_ns           # inverse of RigConfig.camera_time_ns
        drop = sorted({int(i) for i in faults.drop_frames.get(cam, [])})
        if any(i < 0 or i >= n_cap for i in drop):
            raise ValueError(f"drop_frames[{cam}] indices must be in [0, {n_cap})")
        keep = np.setdiff1d(np.arange(n_cap), drop)
        ts = (t_mid[keep] - t0) * 1e-9
        T_W_imu = traj.T_W_imu(ts)
        T_W_cam = T_W_imu @ c.T_imu_cam
        T_W_uw = T_W_imu @ T_imu_uw
        widths = traj.width_m(ts)
        rend = _renderer(c, ss)
        bank = _noise_bank(rng, (c.resolution[1], c.resolution[0], 3), noise_sigma) if noise_sigma > 0 else None
        mp4 = _Mp4(out / f"{cam}.mp4", c.resolution, c.fps, crf, preset)
        try:
            for j in range(len(keep)):
                quads = list(extra_quads)
                for side, T_virt_tag in finger_tag_poses_virt(float(widths[j]), rig.aruco.nominal_z_m).items():
                    quads.append(Quad(T_W_uw[j] @ T_uw_virt @ T_virt_tag, half, scene.marker_rects[tag_ids[side]]))
                img = rend.render(scene, T_W_cam[j], quads)
                if bank is not None:
                    pos, neg = bank[int(rng.integers(len(bank)))]
                    img = cv2.subtract(cv2.add(img, pos), neg)
                mp4.write(img)
        finally:
            mp4.close()
        rows = keep
        if cam in faults.count_mismatch:
            rows = keep[:-1]
        seq = _SEQ0[cam] + rows
        lines = [f"{i},{t},{q},{exp}" for i, (t, q) in enumerate(zip(t_csv[rows], seq))]
        (out / f"{cam}_frames.csv").write_text("\n".join([FRAMES_HEADER, *lines]) + "\n")
        frames[cam] = dict(t_mid=t_mid[keep], n_video=len(keep), dropped=len(drop))
        timings[cam] = time.perf_counter() - t1

    # ---- IMU
    t1 = time.perf_counter()
    t_imu = imu_times(t0, t0 + dur_ns, imu_rate_hz)
    gyro, acc = traj.imu((t_imu - t0) * 1e-9)
    if imu_noise:
        n = rig.imu.noise
        nd = {"gyroscope_noise_density": n.gyroscope_noise_density, "gyroscope_random_walk": n.gyroscope_random_walk,
              "accelerometer_noise_density": n.accelerometer_noise_density,
              "accelerometer_random_walk": n.accelerometer_random_walk}
        gyro, acc = add_noise(t_imu, gyro, acc, nd, seed=int(seed) + 0x13579)
    idx = np.arange(len(t_imu))
    keep = np.ones(len(t_imu), bool)
    for start_s, dur_s in faults.imu_gaps:
        a, b = t0 + int(round(start_s * 1e9)), t0 + int(round((start_s + dur_s) * 1e9))
        keep &= ~((t_imu >= a) & (t_imu < b))
    t_csv_imu = t_imu - rig.imu.time_offset_ns                     # inverse of RigConfig.imu_time_ns
    lines = [f"{t},{k},{g[0]:.9g},{g[1]:.9g},{g[2]:.9g},{a_[0]:.9g},{a_[1]:.9g},{a_[2]:.9g}"
             for t, k, g, a_ in zip(t_csv_imu[keep], idx[keep], gyro[keep], acc[keep])]
    (out / "imu.csv").write_text("\n".join([IMU_HEADER, *lines]) + "\n")
    n_imu = int(keep.sum())
    timings["imu"] = time.perf_counter() - t1

    # ---- meta
    b0 = t0 + int(round(traj.button_start_s * 1e9))
    b1 = t0 + int(round(traj.button_stop_s * 1e9))
    anchor_clock = t0 - 500_000_000
    unix = (_UNIX_BASE_NS + (anchor_clock - _T0_BASE_NS)) if unix_anchor_ns is None else int(unix_anchor_ns)

    def _num(x):
        return int(x) if float(x).is_integer() else float(x)

    meta = {
        "schema": SCHEMA, "session_id": session_id, "episode_number": int(episode_number), "rig_id": rig.rig_id,
        "clock": "CLOCK_MONOTONIC", "wall_anchor": {"clock_ns": int(anchor_clock), "unix_ns": int(unix)},
        "button_start_ns": int(b0), "button_stop_ns": int(b1),
        "cameras": {cam: {"role": role, "sensor": rig.cameras[role].sensor,
                          "resolution": list(rig.cameras[role].resolution), "fps": _num(rig.cameras[role].fps),
                          "frame_count": frames[cam]["n_video"], "dropped_frames": frames[cam]["dropped"]}
                    for cam, role in CAMS.items()},
        "imu": {"sensor": rig.imu.sensor, "rate_hz": _num(imu_rate_hz), "sample_count": n_imu,
                "dropped_samples": int(len(t_imu) - n_imu)},
        "sim": {"generator": "dataproc.sim", "truth": TRUTH_FILE, "kind": kind, "seed": int(seed),
                "scene_seed": int(scene_seed), "duration_s": float(duration_s), "preroll_s": float(preroll_s)},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # ---- truth
    t1 = time.perf_counter()
    n_truth = int(np.floor(dur_ns * TRUTH_RATE_HZ / 1e9)) + 1
    t_truth = t0 + np.round(np.arange(n_truth) * (1e9 / TRUTH_RATE_HZ)).astype(np.int64)
    ts = (t_truth - t0) * 1e-9
    tv = finger_tag_poses_virt(0.0, rig.aruco.nominal_z_m)
    _save_npz(out / TRUTH_FILE, {
        "version": np.str_(TRUTH_VERSION), "rig_id": np.str_(rig.rig_id), "seed": np.int64(seed),
        "kind": np.str_(kind), "session_id": np.str_(session_id), "episode_number": np.int64(episode_number),
        "scene_seed": np.int64(scene_seed),
        "t_ns": t_truth, "T_W_imu": traj.T_W_imu(ts), "width_m": traj.width_m(ts),
        "cam0_t_ns": frames["cam0"]["t_mid"], "cam1_t_ns": frames["cam1"]["t_mid"],
        "imu_t_ns": t_imu[keep],
        "T_imu_cam0": rig.cameras["main"].T_imu_cam, "T_imu_cam1": rig.cameras["ultrawide"].T_imu_cam,
        "T_virt_tag_left0": tv["left"], "T_virt_tag_right0": tv["right"],
        "button_start_ns": np.int64(b0), "button_stop_ns": np.int64(b1),
    })
    timings["truth"] = time.perf_counter() - t1

    _write_manifest(out)
    (out / COMPLETE).touch()
    if verified:
        (out / VERIFIED).touch()
    timings["total"] = time.perf_counter() - tt
    return SimEpisode(path=out, kind=kind, seed=int(seed),
                      frame_counts={cam: frames[cam]["n_video"] for cam in CAMS}, imu_count=n_imu,
                      button_start_ns=int(b0), button_stop_ns=int(b1), timings_s=timings)


# --------------------------------------------------------------------------------------------- fast rig

def fast_rig(rig_yaml_path, main_scale: float = 0.25, ultrawide_scale: float = 0.5,
             suffix: str = "_fast") -> tuple[dict, RigConfig]:
    """Low-resolution copy of a rig yaml for tests: camera resolutions scaled (intrinsics via the camera model's
    ``scaled()``, half-pixel convention), ``rig_id`` + ``suffix``; everything else (virtual ultrawide, tags, IMU,
    extrinsics, time model) unchanged. Returns ``(yaml dict, RigConfig)``; write the dict as
    ``rigs/<rig_id>.yaml`` so processing can load it."""
    path = Path(rig_yaml_path)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    d["rig_id"] = f"{d['rig_id']}{suffix}"
    d["notes"] = (f"Synthetic low-resolution copy of {path.name} (dataproc sim --fast: main x{main_scale}, "
                  f"ultrawide x{ultrawide_scale}). " + str(d.get("notes", "") or "")).strip()
    for role, s in (("main", main_scale), ("ultrawide", ultrawide_scale)):
        c = d["cameras"][role]
        w, h = c["resolution"]
        w1, h1 = int(round(w * s)), int(round(h * s))
        if w1 % 2 or h1 % 2 or not np.isclose(w1 / w, h1 / h):
            raise ValueError(f"{role}: scale {s} gives {w1}x{h1} (needs even sizes and the same x/y scale)")
        m = model_from_dict({"type": c["model"]["type"], "intrinsics": c["model"]["intrinsics"],
                             "distortion": c["model"]["distortion"]}, (w, h)).scaled(w1, h1)
        c["resolution"] = [w1, h1]
        c["model"]["intrinsics"] = [float(m.fx), float(m.fy), float(m.cx), float(m.cy)]
    return d, parse_rig(d, f"{path} ({suffix.strip('_')})")



# --------------------------------------------------------------------------------------------- CLI driver

DEFAULT_RIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "rigs" / "rig_v1.yaml"
_EP_NUM_RE = re.compile(r"(?:^|_)episode_(\d+)$")


def next_episode_number(out_dir) -> int:
    """1 + the largest ``..._episode_NNNNNN`` / ``episode_NNNNNN`` number among the entries of ``out_dir`` (0 if
    none)."""
    out = Path(out_dir)
    nums = [int(m.group(1)) for p in (out.iterdir() if out.is_dir() else []) if (m := _EP_NUM_RE.search(p.name))]
    return max(nums) + 1 if nums else 0


def _ensure_rig_file(rigs_dir: Path, rig_id: str, text: str, log) -> None:
    """Copy the rig yaml used for rendering to ``rigs_dir/<rig_id>.yaml`` if missing; never overwrite."""
    dst = Path(rigs_dir) / f"{rig_id}.yaml"
    if dst.exists():
        same = yaml.safe_load(dst.read_text(encoding="utf-8")) == yaml.safe_load(text)
        if not same:
            log(f"warning: {dst} differs from the rig used to render (left unchanged)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    log(f"wrote {dst}")


def run_sim(out, episodes: int = 1, calibration: bool = False, seed: int = 0, duration_s: float | None = None,
            rig_path=None, fast: bool = False, session_id: str | None = None, rigs_dir=None, verified: bool = True,
            log=print) -> list[SimEpisode]:
    """``dataproc sim``: write ``episodes`` episodes into ``out`` as ``<session>_episode_NNNNNN`` (numbering
    continues after the largest existing number in ``out``). ``fast``: the ``<rig_id>_fast`` low-resolution rig
    (:func:`fast_rig`) and 5 s episodes with a 2 s pre-roll. The rig yaml used is copied to ``rigs_dir`` if missing.
    """
    from dataproc.config import load_rig
    from dataproc.paths import validate_episode_dir_name

    rig_path = Path(rig_path) if rig_path is not None else DEFAULT_RIG_PATH
    if fast:
        d, rig = fast_rig(rig_path)
        rig_text = ("# Written by dataproc sim --fast (dataproc.sim.episode.fast_rig).\n"
                    + yaml.safe_dump(d, sort_keys=False))
    else:
        rig = load_rig(rig_path)
        rig_text = rig_path.read_text(encoding="utf-8")
    duration = duration_s if duration_s is not None else (5.0 if fast else 15.0)
    preroll = 2.0 if fast else 2.5
    make_trajectory(rig, "calibration" if calibration else "demonstration", seed, duration, preroll)  # validate
    now = time.time_ns()
    session = session_id or f"sim-{time.strftime('%Y%m%dT%H%M%S', time.gmtime(now / 1e9))}{now // 1000 % 10**6:06d}"
    kind = "calibration" if calibration else "demonstration"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if rigs_dir is not None:
        _ensure_rig_file(Path(rigs_dir), rig.rig_id, rig_text, log)
    first = next_episode_number(out)
    done = []
    for k in range(episodes):
        n = first + k
        name = validate_episode_dir_name(f"{session}_episode_{n:06d}")
        ep = generate_episode(out / name, rig, kind=kind, seed=seed * 1_000_003 + n, duration_s=duration,
                              preroll_s=preroll, session_id=session, episode_number=n, verified=verified,
                              unix_anchor_ns=time.time_ns())
        log(f"{ep.path}  {kind}, {duration:g} s, cam0 {ep.frame_counts['cam0']} / cam1 {ep.frame_counts['cam1']} "
            f"frames, {ep.imu_count} IMU samples, {ep.timings_s['total']:.1f} s")
        done.append(ep)
    return done
