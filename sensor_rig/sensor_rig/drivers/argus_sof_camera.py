"""ArgusSofCamera: CSI camera via the ``argus_sof`` libargus helper (sensor start-of-frame stamps; T5b).

The helper (``sensor_rig/native/argus_sof``, C++) is started as a subprocess per camera. It prints one line per
frame (seq, sensor timestamp, exposure, gain, ring slot) and writes NV12 pixels into a POSIX shared-memory ring;
this backend parses the lines, reads the slot (seqlock check), converts NV12 -> BGR and maps the sensor timestamp
(CLOCK_MONOTONIC_RAW domain on L4T) to CLOCK_MONOTONIC with ``clock.RawToMono`` (refitted every 2 s from paired clock
samples, the same logic the rest of the package uses). Errors surface as ``BackendError`` through ``read()``:
helper exits / crashes (with its stderr tail), no ``READY`` within ``startup_timeout_s``, protocol errors.
Frames lost before userspace show up as seq gaps (``stats["sensor_gaps"]``), slots overwritten before we read them
as ``stats["shm_overruns"]``.

NOT RUN ON HARDWARE YET (the helper has never been built). Tests: ``tests/test_argus_sof.py`` with a fake helper.
"""

from __future__ import annotations

import collections
import os
import select
import subprocess
import time
import uuid
from pathlib import Path

from sensor_rig.clock import RawToMono, sample_clocks
from sensor_rig.drivers.argus_sof_logic import FrameMsg, ProtocolError, Ready, ShmRing, helper_args, nv12_to_bgr, \
    parse_line
from sensor_rig.drivers.base import BackendError, CameraBackend, Frame

DEFAULT_HELPER = Path(__file__).resolve().parents[2] / "native" / "argus_sof" / "build" / "argus_sof"


def helper_path(rig=None, override: str | None = None) -> Path:
    if override:
        return Path(override)
    cap = getattr(rig, "capture", None) if rig is not None else None
    h = ((cap or {}).get("argus_sof") or {}).get("helper") if isinstance(cap, dict) else None
    return Path(h or os.environ.get("SENSOR_RIG_ARGUS_SOF") or DEFAULT_HELPER)


class ArgusSofCamera(CameraBackend):
    def __init__(self, name: str, cmd: list[str], width: int, height: int, fps: float, clock=None,
                 queue_size: int = 8, ring_size: int = 16, startup_timeout_s: float = 10.0,
                 clock_sampler=sample_clocks, refit_s: float = 2.0, shm_dir: str = "/dev/shm",
                 description: dict | None = None):
        super().__init__(name, width, height, fps, queue_size=queue_size, ring_size=ring_size, clock=clock)
        self.cmd = list(cmd)
        self.startup_timeout_s = float(startup_timeout_s)
        self.clock_sampler = clock_sampler
        self.refit_s = float(refit_s)
        self.shm_dir = shm_dir
        self._description = description or {}
        self._proc: subprocess.Popen | None = None
        self._shm: ShmRing | None = None
        self._ready: Ready | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
        self._buf = b""
        self.stats.update({"sensor_gaps": 0, "shm_overruns": 0})

    @classmethod
    def from_rig(cls, rig, name: str, helper: str | None = None, **opts) -> "ArgusSofCamera":
        from sensor_rig.drivers.argus_logic import argus_params_from_camera

        params = argus_params_from_camera(rig.cameras[name])
        shm = f"sensor_rig_{name}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        exe = helper_path(rig, helper)
        cmd = [str(exe), *helper_args(params, shm)]
        desc = {"helper": str(exe), "sensor_id": params["sensor_id"], "sensor_mode": params["sensor_mode"]}
        return cls(name, cmd, params["width"], params["height"], params["fps"], description=desc, **opts)

    # ---- helper process ----
    def _readline(self, timeout_s: float) -> str | None:
        """One stdout line, '' on EOF, None on timeout (non-blocking reads, so stop() stays responsive)."""
        deadline = time.monotonic() + timeout_s
        out = self._proc.stdout
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            r, _, _ = select.select([out], [], [], min(remaining, 0.2))
            if not r:
                continue
            chunk = os.read(out.fileno(), 65536)
            if not chunk:
                rest, self._buf = self._buf, b""
                return rest.decode(errors="replace") if rest else ""
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode(errors="replace") + "\n"

    def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            while select.select([self._proc.stderr], [], [], 0)[0]:
                chunk = os.read(self._proc.stderr.fileno(), 65536)
                if not chunk:
                    break
                self._stderr_tail.extend(chunk.decode(errors="replace").splitlines())
        except (OSError, ValueError):
            pass

    def _died(self, what: str) -> BackendError:
        rc = self._proc.poll() if self._proc else None
        self._drain_stderr()
        tail = " | ".join(list(self._stderr_tail)[-5:])
        return BackendError(f"camera {self.name}: argus_sof helper {what} (rc={rc}){': ' + tail if tail else ''}")

    def _open(self) -> None:
        exe = self.cmd[0]
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            raise BackendError(f"camera {self.name}: argus_sof helper {exe} not found or not executable (build it "
                               "with sensor_rig/native/argus_sof/build.sh on the Jetson)")
        self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            line = self._readline(max(0.0, deadline - time.monotonic()))
            if line is None:
                self._kill()
                raise BackendError(f"camera {self.name}: argus_sof helper sent no READY within "
                                   f"{self.startup_timeout_s:g} s (nvargus-daemon hung? camera busy?)")
            if line == "":
                err = self._died("exited before READY")
                self._kill()
                raise err
            msg = parse_line(line)
            if isinstance(msg, tuple):
                self._kill()
                raise BackendError(f"camera {self.name}: argus_sof helper: {msg[1]}")
            if isinstance(msg, Ready):
                if (msg.width, msg.height) != (self.width, self.height):
                    self._kill()
                    raise BackendError(f"camera {self.name}: helper delivers {msg.width}x{msg.height}, expected "
                                       f"{self.width}x{self.height}")
                self._ready = msg
                self._shm = ShmRing(msg, self.shm_dir)
                return

    def _kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(2.0)

    def _run(self) -> None:
        first = self.clock_sampler()
        r2m = RawToMono(first)
        last_fit = time.monotonic()
        last_seq = None
        out_seq = 0
        while not self._stopping.is_set():
            line = self._readline(0.2)
            if line is None:
                continue
            if line == "":
                if self._stopping.is_set():
                    return
                raise self._died("exited")
            try:
                msg = parse_line(line)
            except ProtocolError as e:
                raise BackendError(f"camera {self.name}: {e}") from e
            if isinstance(msg, tuple):
                raise BackendError(f"camera {self.name}: argus_sof helper: {msg[1]}")
            if not isinstance(msg, FrameMsg):
                continue
            if time.monotonic() - last_fit > self.refit_s:
                r2m = RawToMono(first, self.clock_sampler())
                last_fit = time.monotonic()
            if last_seq is not None and msg.seq > last_seq + 1:
                self.stats["sensor_gaps"] += msg.seq - last_seq - 1
            last_seq = msg.seq
            buf = self._shm.read(msg.slot, msg.seq)
            if buf is None:
                self.stats["shm_overruns"] += 1
                continue
            img = nv12_to_bgr(buf, self._ready.width, self._ready.height)
            self._push(Frame(image=img, t_ns=int(r2m(msg.sensor_ts_ns)), seq=int(msg.seq),
                             t_arrival_ns=self.clock.now_ns(),
                             meta={"sensor_ts_raw_ns": msg.sensor_ts_ns, "exposure_ns": msg.exposure_ns,
                                   "gain": msg.gain}))
            out_seq += 1

    def stop(self) -> None:
        self._stopping.set()
        self._kill()            # unblocks the reader; it then sees EOF with _stopping set
        super().stop()

    def _close(self) -> None:
        self._kill()
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self._ready is not None:
            try:
                os.unlink(os.path.join(self.shm_dir, self._ready.shm_name))
            except OSError:
                pass

    def info(self) -> dict:
        d = {"backend": "argus_sof", "width": self.width, "height": self.height, "fps_nominal": self.fps,
             "stamp": "sensor start-of-frame (ICaptureMetadata::getSensorTimestamp), RAW->MONOTONIC"}
        d.update(self._description)
        d.update({k: self.stats[k] for k in ("sensor_gaps", "shm_overruns")})
        return d
