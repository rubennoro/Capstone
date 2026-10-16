"""The recorder (P1 §2.3, §3): cameras + IMU run continuously from ``open()`` to ``close()``; episodes only gate what is
written. Raw episodes are written exclusively through ``sensor_rig.raw_episode.RawEpisodeWriter``.

Threads: one capture pump per camera (``backend.read()`` -> active ``CameraStreamWriter.submit``), one IMU pump
(``read_batch`` -> pre-roll ring + active ``ImuStreamWriter``), the ``Monitor`` (1 Hz status + live abort rules), and
per episode K encode workers + 1 index writer per camera. Timestamps come from the backends; Python scheduling only
adds latency.

Usage::

    rec = Recorder(rig, raw_root)            # backends from rig.capture.backends (or pass cameras=/imu=)
    rec.open()                               # start sensors, warm up, check rates
    name = rec.start_episode("demonstration", session="rig-pilot", side="left")
    ...
    path = rec.stop_episode()                # finalize -> <raw_root>/<date>/<NAME>/
    rec.close()
"""

from __future__ import annotations

import collections
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from sensor_rig.drivers.base import BackendError, CameraBackend, ImuBackend, RealClock, make_camera, make_imu
from sensor_rig.raw_episode import RawEpisodeWriter
from sensor_rig.record.imu_writer import ImuStreamWriter
from sensor_rig.record.jpeg import make_encoder
from sensor_rig.record.monitor import LiveAbortConfig, Monitor, disk_free_gb
from sensor_rig.record.session_state import SessionState
from sensor_rig.record.stream_writer import CameraStreamWriter
from sensor_rig.timefmt import KIND_TO_NAME, SIDES, iso_ms, make_episode_name, random_id, sanitize_session

CAMERAS = ("main", "ultrawide")


class RecorderError(RuntimeError):
    pass


class EpisodeAborted(RecorderError):
    """Raised by ``stop_episode`` when the episode was already aborted (e.g. by a live abort rule)."""


def _host_info() -> dict:
    l4t = None
    try:
        with open("/etc/nv_tegra_release") as f:
            l4t = f.readline().strip() or None
    except OSError:
        pass
    commit = None
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=2)
        if out.returncode == 0:
            commit = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    from sensor_rig import __version__

    return {"hostname": socket.gethostname(), "l4t": l4t, "sensor_rig_version": __version__, "git_commit": commit,
            "python": platform.python_version()}


class _CameraPump:
    """Reads one backend continuously; dispatches frames to the active stream writer."""

    def __init__(self, rec: "Recorder", name: str, cam: CameraBackend):
        self.rec = rec
        self.name = name
        self.cam = cam
        self.recent: collections.deque[int] = collections.deque(maxlen=max(16, int(cam.fps * 4)))
        self.frames = 0
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._loop, name=f"pump-{name}", daemon=True)

    def _loop(self) -> None:
        try:
            self._loop_inner()
        except Exception as e:   # anything else (writer bug, disk error): fail loudly, abort the episode
            self.error = e
            self.rec._pump_failed(self.name, e)

    def _loop_inner(self) -> None:
        rec = self.rec
        while not rec._closing.is_set():
            try:
                f = self.cam.read(timeout_s=0.2)
            except BackendError as e:
                self.error = e
                rec._pump_failed(self.name, e)
                return
            if f is None:
                if self.cam.exhausted:
                    return
                continue
            self.frames += 1
            self.recent.append(int(f.t_ns))
            stream = rec._streams.get(self.name)
            if stream is not None:
                stream.submit(f)

    def fps_since(self, t0_ns: int) -> float | None:
        t = np.array(self.recent, dtype=np.int64)
        t = t[t >= t0_ns]
        if len(t) < 2 or t[-1] <= t[0]:
            return None
        return (len(t) - 1) / ((t[-1] - t[0]) / 1e9)


class _ImuPump:
    def __init__(self, rec: "Recorder", imu: ImuBackend, keep_s: float):
        self.rec = rec
        self.imu = imu
        self.keep_ns = int(keep_s * 1e9)
        self.ring: collections.deque = collections.deque()
        self.samples = 0
        self.first_t: int | None = None
        self.last_t: int | None = None
        self.error: BaseException | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="pump-imu", daemon=True)

    def _loop(self) -> None:
        try:
            self._loop_inner()
        except Exception as e:   # e.g. ImuWriter rejecting non-increasing stamps: fail loudly, abort the episode
            self.error = e
            self.rec._pump_failed("imu", e)

    def _loop_inner(self) -> None:
        rec = self.rec
        while not rec._closing.is_set():
            try:
                b = self.imu.read_batch(timeout_s=0.05)
            except BackendError as e:
                self.error = e
                rec._pump_failed("imu", e)
                return
            if b is None or not len(b):
                continue
            with self.lock:
                self.samples += len(b)
                if self.first_t is None:
                    self.first_t = int(b.t_ns[0])
                self.last_t = int(b.t_ns[-1])
                self.ring.append(b)
                while self.ring and self.ring[0].t_ns[-1] < self.last_t - self.keep_ns:
                    self.ring.popleft()
                stream = rec._imu_stream
                if stream is not None:
                    stream.submit(b)

    def start_stream(self, make_stream: Callable[[list], ImuStreamWriter]) -> ImuStreamWriter:
        """Create the episode's IMU stream atomically with respect to the pump (no batch lost or duplicated)."""
        with self.lock:
            stream = make_stream(list(self.ring))
            self.rec._imu_stream = stream
            return stream

    def rate_since(self, t0_ns: int) -> float | None:
        with self.lock:
            ts = [b.t_ns for b in self.ring]
        if not ts:
            return None
        t = np.concatenate(ts)
        t = t[t >= t0_ns]
        if len(t) < 2 or t[-1] <= t[0]:
            return None
        return (len(t) - 1) / ((t[-1] - t[0]) / 1e9)


class Recorder:
    def __init__(self, rig, raw_root, cameras: dict[str, CameraBackend] | None = None, imu: ImuBackend | None = None,
                 clock=None, encoder=None, warmup_s: float = 2.0, n_workers: int | None = None,
                 queue_frames: int | dict | None = None, preroll_s: float = 1.0, postroll_s: float = 1.0,
                 live_abort: LiveAbortConfig | dict | None = None, keep_aborted: bool = False,
                 log: Callable[[str], None] | None = None, monitor_period_s: float = 1.0,
                 status_lines: bool = False, check_resolution: bool = True, stop_timeout_s: float = 1.0):
        self.rig = rig
        self.raw_root = Path(raw_root)
        self.clock = clock if clock is not None else RealClock()
        cap = getattr(rig, "capture", None)
        cap = cap if isinstance(cap, dict) else {}
        jpeg = dict(cap.get("jpeg") or {})
        self.jpeg_quality = int(jpeg.get("quality", 90))
        self.encoder = encoder if encoder is not None else make_encoder(jpeg.get("encoder", "opencv"),
                                                                        self.jpeg_quality)
        self.n_workers = int(n_workers or jpeg.get("workers_per_camera", 2))
        self.queue_seconds = float(jpeg.get("queue_seconds", 0.75))
        self.queue_frames = queue_frames
        if isinstance(live_abort, LiveAbortConfig):
            self.abort_cfg = live_abort
        else:
            self.abort_cfg = LiveAbortConfig.from_dict(live_abort if live_abort is not None
                                                       else cap.get("live_abort"))
        self.warmup_s = float(warmup_s)
        self.preroll_s = float(preroll_s)
        self.postroll_s = float(postroll_s)
        self.keep_aborted = bool(keep_aborted)
        self.monitor_period_s = monitor_period_s
        self.status_lines = status_lines
        self.check_resolution = check_resolution
        self.stop_timeout_s = float(stop_timeout_s)
        self._log_fn = log or (lambda s: None)
        self.cameras = cameras if cameras is not None else {c: make_camera(rig, c, clock=self._clock_opt())
                                                            for c in CAMERAS}
        self.imu = imu if imu is not None else make_imu(rig, clock=self._clock_opt())
        self.session_state = SessionState(self.raw_root)
        self._closing = threading.Event()
        self._state_lock = threading.RLock()
        self._streams: dict[str, CameraStreamWriter] = {}
        self._imu_stream: ImuStreamWriter | None = None
        self._writer: RawEpisodeWriter | None = None
        self._episode: dict | None = None
        self._stopping = False
        self._opened = False
        self._pumps: dict[str, _CameraPump] = {}
        self._imu_pump: _ImuPump | None = None
        self._monitor: Monitor | None = None
        self._ep_log_lock = threading.Lock()
        self.last_abort_reason: str | None = None
        self.pump_errors: dict[str, str] = {}   # source -> repr(exception) of a pump that died
        self.episodes: list[dict] = []   # history: {"name", "status", "path", "reason"}

    def _clock_opt(self):
        return None if isinstance(self.clock, RealClock) else self.clock

    # ---- logging ----
    def _log(self, msg: str) -> None:
        self._log_fn(msg)
        w = self._writer
        if w is not None:
            with self._ep_log_lock:
                try:
                    with open(w.dir / "logs" / "recorder.log", "a") as f:
                        f.write(f"{self.clock.now_ns()} {msg}\n")
                except OSError:
                    pass

    # ---- lifecycle ----
    def open(self) -> None:
        if self._opened:
            return
        self.raw_root.mkdir(parents=True, exist_ok=True)
        if self.check_resolution:
            for name, cam in self.cameras.items():
                exp = tuple(int(v) for v in self.rig.cameras[name].resolution)
                if (cam.width, cam.height) != exp:
                    raise RecorderError(f"camera {name}: backend delivers {cam.width}x{cam.height} but rig.yaml "
                                        f"calibration resolution is {exp[0]}x{exp[1]}")
        self._closing.clear()
        started = []
        try:
            for cam in self.cameras.values():
                cam.start()
                started.append(cam)
            self.imu.start()
            started.append(self.imu)
        except Exception:
            for b in started:
                b.stop()
            raise
        self._pumps = {n: _CameraPump(self, n, c) for n, c in self.cameras.items()}
        self._imu_pump = _ImuPump(self, self.imu, keep_s=self.preroll_s + 0.5)
        for p in list(self._pumps.values()) + [self._imu_pump]:
            p.thread.start()
        self._opened = True
        t_warm = self.clock.now_ns()
        self.clock.sleep(self.warmup_s)
        problems = self._rate_problems(t_warm + int(0.25 * self.warmup_s * 1e9))
        if problems:
            self.close()
            raise RecorderError("sensors not ready after warm-up: " + "; ".join(problems))
        self._monitor = Monitor(self, self.abort_cfg, period_s=self.monitor_period_s,
                                status=self._log if self.status_lines else None)
        self._monitor.start()
        self._log("recorder open: " + ", ".join(f"{n} {c.width}x{c.height}@{c.fps:g}" for n, c in
                                                self.cameras.items()) + f", imu {self.imu.rate_hz:g} Hz")

    def _rate_problems(self, t0_ns: int) -> list[str]:
        out = []
        ratio = self.abort_cfg.warmup_min_fps_ratio
        for n, p in self._pumps.items():
            if p.error is not None:
                out.append(f"{n}: {p.error}")
                continue
            fps = p.fps_since(t0_ns)
            if fps is None or fps < ratio * p.cam.fps:
                out.append(f"{n}: warm-up fps {fps if fps is None else round(fps, 1)} < {ratio} x {p.cam.fps:g}")
        ip = self._imu_pump
        if ip.error is not None:
            out.append(f"imu: {ip.error}")
        else:
            r = ip.rate_since(t0_ns)
            if r is None or r < self.abort_cfg.imu_min_ratio * self.imu.rate_hz:
                out.append(f"imu: warm-up rate {r if r is None else round(r)} < "
                           f"{self.abort_cfg.imu_min_ratio} x {self.imu.rate_hz:g}")
        return out

    def close(self) -> None:
        with self._state_lock:
            if self._writer is not None:
                self._abort_locked("recorder closed", keep=self.keep_aborted)
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        self._closing.set()
        for p in list(self._pumps.values()) + ([self._imu_pump] if self._imu_pump else []):
            p.thread.join(timeout=5.0)
        for cam in self.cameras.values():
            cam.stop()
        self.imu.stop()
        self._opened = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def recording(self) -> bool:
        return self._writer is not None

    # ---- episodes ----
    def start_episode(self, kind: str, session: str, side: str, gripper_id: str = "default",
                      calibration_run: str | None = "auto", task_label: str | None = None,
                      sides_present: list[str] | None = None, note: str = "",
                      allow_no_calibration: bool = False) -> str:
        if not self._opened:
            raise RecorderError("call open() first")
        if kind not in KIND_TO_NAME:
            raise RecorderError(f"kind must be one of {sorted(KIND_TO_NAME)}")
        if side not in SIDES:
            raise RecorderError(f"side must be one of {SIDES}")
        try:
            session_s = sanitize_session(session)
        except ValueError as e:
            raise RecorderError(str(e)) from e
        with self._state_lock:
            if self._writer is not None:
                raise RecorderError("an episode is already recording")
            cal = ""
            if kind == "demonstration" and side != "head":
                if calibration_run == "auto":
                    cal = self.session_state.get_calibration(session_s, side, gripper_id) or ""
                elif calibration_run not in (None, "", "none"):
                    cal = str(calibration_run)
                if not cal and not allow_no_calibration:
                    raise RecorderError(f"no gripper calibration recorded for session={session_s} side={side} "
                                        f"gripper={gripper_id}; record one first or pass allow_no_calibration")
            problems = [f"{n}: {p.error}" for n, p in self._pumps.items() if p.error is not None]
            if self._imu_pump.error is not None:
                problems.append(f"imu: {self._imu_pump.error}")
            if problems:
                raise RecorderError("backend failure: " + "; ".join(problems))
            c = self.clock.sample()
            name = make_episode_name(c.realtime_ns, random_id(), session_s, kind, side)
            sensors = {}
            for n, cam in self.cameras.items():
                info = dict(cam.info())
                pc = self.rig.cameras[n]
                info.setdefault("sensor_mode", getattr(pc, "sensor_mode", None))
                argus = getattr(pc, "argus", None) or {}
                if "exposure_time_range_ns" in argus:
                    lo, hi = argus["exposure_time_range_ns"]
                    info.setdefault("exposure_us", hi / 1000.0 if lo == hi else None)
                info["jpeg_quality"] = self.jpeg_quality
                sensors[n] = info
            sensors["imu0"] = dict(self.imu.info())
            meta_init = {
                "gripper_id": gripper_id,
                "sides_present": list(sides_present) if sides_present else [side],
                "gripper_calibration_run_name": cal,
                "task_label": task_label,
                "note": note,
                "clock": {"start": c.to_dict()},
                "recording_start_time": iso_ms(c.realtime_ns),
                "sensors": sensors,
                "host": _host_info(),
            }
            writer = RawEpisodeWriter(self.raw_root, name, self.rig, meta_init)
            self._writer = writer
            self._stopping = False
            streams = {}
            for n, cam in self.cameras.items():
                qf = self.queue_frames.get(n) if isinstance(self.queue_frames, dict) else self.queue_frames
                streams[n] = CameraStreamWriter(n, writer, self.encoder, cam.fps, None, n_workers=self.n_workers,
                                                queue_frames=qf, queue_seconds=self.queue_seconds, log=self._log)
            self._streams = streams
            # Start all camera streams atomically: hold every stream lock while reading the clock.
            locks = [s.lock for s in streams.values()]
            for lk in locks:
                lk.acquire()
            try:
                start_ns = self.clock.now_ns()
                for s in streams.values():
                    s.begin_locked(start_ns)
            finally:
                for lk in reversed(locks):
                    lk.release()
            self._imu_pump.start_stream(lambda ring: ImuStreamWriter(writer, start_ns, self.preroll_s,
                                                                     self.postroll_s, ring))
            writer.update_meta(record_start_mono_ns=start_ns)
            self._episode = {"name": name, "kind": kind, "session": session_s, "side": side,
                             "gripper_id": gripper_id, "start_ns": start_ns}
            self._log(f"start {name} (calibration run: {cal or '-'})")
            return name

    def stop_episode(self) -> Path:
        with self._state_lock:
            if self._writer is None:
                if self.last_abort_reason:
                    raise EpisodeAborted(f"episode was aborted: {self.last_abort_reason}")
                raise RecorderError("no episode is recording")
            self._stopping = True
            stop_ns = self.clock.now_ns()
            streams = dict(self._streams)
            imu_stream = self._imu_stream
            for s in streams.values():
                s.request_stop(stop_ns)
            imu_stream.request_stop(stop_ns)
            deadline = stop_ns + int(max(self.stop_timeout_s, self.postroll_s + 0.5) * 1e9)
            while self.clock.now_ns() < deadline:
                if all(s.done for s in streams.values()) and imu_stream.done:
                    break
                time.sleep(0.001)
            late = [n for n, s in streams.items() if not s.done] + ([] if imu_stream.done else ["imu"])
            self._streams = {}
            with self._imu_pump.lock:
                self._imu_stream = None
            cam_stats = {n: s.finish() for n, s in streams.items()}
            imu_stats = imu_stream.finish()
            c_stop = self.clock.sample()
            writer = self._writer
            sensors = {n: st for n, st in cam_stats.items()}
            imu_info = dict(self.imu.info())
            imu_info.update(imu_stats)
            sensors["imu0"] = imu_info
            if late:
                self._log(f"stop: no data after the stop time from {late} within the timeout")
            path = writer.finalize("complete", {"clock": {"stop": c_stop.to_dict()}, "record_stop_mono_ns": stop_ns,
                                                "sensors": sensors})
            ep = self._episode
            self._writer = None
            self._episode = None
            self._stopping = False
            self.last_abort_reason = None
            if ep["kind"] == "gripper_calibration":
                self.session_state.set_calibration(ep["session"], ep["side"], ep["gripper_id"], ep["name"])
            self.episodes.append({"name": ep["name"], "status": "complete", "path": path, "reason": None})
            self._log_fn(f"saved {path}")
            return path

    def abort_episode(self, reason: str, keep: bool | None = None, from_monitor: bool = False) -> None:
        with self._state_lock:
            if self._writer is None or (from_monitor and self._stopping):
                return
            self._abort_locked(reason, keep=self.keep_aborted if keep is None else keep)

    def _abort_locked(self, reason: str, keep: bool) -> None:
        self._log(f"abort: {reason}")
        streams = dict(self._streams)
        self._streams = {}
        if self._imu_pump is not None:
            with self._imu_pump.lock:
                imu_stream, self._imu_stream = self._imu_stream, None
        else:
            imu_stream, self._imu_stream = self._imu_stream, None
        for s in streams.values():
            s.finish()
        if imu_stream is not None:
            imu_stream.finish()
        writer, ep = self._writer, self._episode
        path = writer.abort(reason, keep=keep)
        self._writer = None
        self._episode = None
        self._stopping = False
        self.last_abort_reason = reason
        self.episodes.append({"name": ep["name"] if ep else None, "status": "aborted", "path": path,
                              "reason": reason})
        self._log_fn(f"aborted {ep['name'] if ep else '?'}: {reason}")

    def _pump_failed(self, source: str, err: BaseException) -> None:
        """Called from a pump thread that is about to exit: record the cause and abort the active episode."""
        reason = f"{source}_error: {err!r}"
        self.pump_errors[source] = repr(err)
        self._log(f"[{source}] pump stopped: {err!r}")
        if not self._closing.is_set():
            self.abort_episode(reason)

    # ---- live status (Monitor / CLI) ----
    def live_snapshot(self) -> dict | None:
        ep = self._episode
        streams = self._streams
        imu_stream = self._imu_stream
        if ep is None or self._stopping or not streams or imu_stream is None:
            return None
        now = self.clock.now_ns()
        cams = {}
        for n, s in streams.items():
            ft = np.sort(np.array(list(s.frame_times[-600:]) + list(s.queue_full_times[-600:]), dtype=np.int64))
            cams[n] = {"fps_nominal": s.fps, "is_main": n == "main", "frame_times": ft,
                       "queue_full_times": np.array(list(s.queue_full_times[-600:]), dtype=np.int64)}
        st = list(imu_stream.sample_times[-1000:])   # >= the 2 s rate window at 5 ms batches
        imu_t = np.concatenate(st) if st else np.zeros(0, np.int64)
        return {"elapsed_s": (now - ep["start_ns"]) / 1e9, "now_ns": now,
                "disk_free_gb": disk_free_gb(self.raw_root), "cameras": cams,
                "imu": {"rate_nominal": self.imu.rate_hz, "sample_times": imu_t}}

    @property
    def status(self) -> dict:
        ep = self._episode
        d = {"recording": ep is not None, "episode": ep["name"] if ep else None,
             "elapsed_s": (self.clock.now_ns() - ep["start_ns"]) / 1e9 if ep else None,
             "last_abort_reason": self.last_abort_reason, "pump_errors": dict(self.pump_errors),
             "cameras": {}, "imu": {}}
        for n, p in self._pumps.items():
            s = self._streams.get(n)
            d["cameras"][n] = {"fps": p.fps_since(self.clock.now_ns() - 2_000_000_000), "frames": p.frames,
                               "internal_drops": p.cam.stats.get("internal_drops", 0),
                               "episode_frames": s.submitted if s else None,
                               "episode_drops": dict(s.drops) if s else None,
                               "queue": s._encode_q.qsize() if s else None}
        if self._imu_pump is not None:
            d["imu"] = {"rate": self._imu_pump.rate_since(self.clock.now_ns() - 1_000_000_000),
                        "samples": self._imu_pump.samples}
        return d

    def status_line(self) -> str:
        st = self.status
        parts = [f"REC {st['elapsed_s']:.1f}s" if st["recording"] else "idle"]
        for n, c in st["cameras"].items():
            fps = "-" if c["fps"] is None else f"{c['fps']:.1f}"
            extra = ""
            if c["episode_drops"] is not None:
                extra = f" q{c['queue']} drops {sum(c['episode_drops'].values())}"
            parts.append(f"{n} {fps} fps{extra}")
        rate = st["imu"].get("rate")
        parts.append(f"imu {'-' if rate is None else f'{rate:.0f}'} Hz")
        return " | ".join(parts)
