"""SimPi: the live stand-in for the Pi 5 camera sender (DESIGN.md section 6, PI_PAYLOAD.md section 0).

Three threads:

* main ("GL"): a 30 Hz schedule on CLOCK_MONOTONIC. For tick k the main frame's mid-exposure is
  ``t0 + k / fps`` and the ultrawide's ``uw_phase_ms`` later (the live Pi's pair: ultrawide 16.3 ms newer). For each
  camera in turn it waits for the state export to cover that instant (at most ``max_hold_ms``), interpolates the qpos
  there (state.StateBuffer: never extrapolated; a newer time holds the newest state and counts as stale when held
  more than ``max_hold_ms``), renders that camera's views into a free buffer set and hands them to the encoders.
* "sim-pi-enc" (2 workers, one per camera at a time): remap to the native sensor image, INTER_AREA to 640x480,
  JPEG (quality 90). Opt-in (settings ``camera_look``, W2): the camera model of ``sim_deploy/look/camera_look.py``
  applied to the 640x480 frame just before the JPEG, with a numpy Generator seeded per camera and frame
  (``[look seed, camera index, tick k]``, so frames do not depend on which worker ran first); off by default (frames
  byte-identical to before).
* "sim-pi-send": in tick order, builds the header and publishes each packet ``send_delay_ms`` after its main frame's
  ``t_sensor_ns`` (live Pi: 45.8 ms), stamping ``t_send_ns`` on the Pi clock just before sending.

Numbering as the live Pi: ``packet_no`` = ``main.frame_no`` = tick index (a skipped tick leaves a gap in both: the
receiver counts it as lost, which is what a dropped main frame looks like); a new ultrawide frame has the tick index
as its ``frame_no``. Overruns: a tick whose GL stage cannot start before the next tick is skipped (counted); a tick
that starts more than half a period late renders only the main camera and repeats the last ultrawide frame
(``uw_reused`` true, same frame_no and stamps; never twice in a row, so the laptop's reuse limit of 2 holds).
Pi clock: ``t_mono + offset + drift_ppm * 1e-6 * (t_mono - start)`` (FakePiSender's model).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import zmq

from sim_deploy.sim_pi._deploy import PiFrameStamp, PiStreamHeader, PiStreamPacket, load_live_rig
from sim_deploy.sim_pi.cameras import CameraChain
from sim_deploy.sim_pi.model_file import load_render_model, wait_for
from sim_deploy.sim_pi.render import GLRenderer, Lighting
from sim_deploy.sim_pi.settings import SimPiSettings
from sim_deploy.sim_pi.state import StateBuffer, StateSubscriber, quat_slices

log = logging.getLogger("sim_pi")
ROLES = ("main", "ultrawide")


class PiClock:
    def __init__(self, offset_s: float, drift_ppm: float, start_ns: int):
        self.offset_ns = int(round(offset_s * 1e9))
        self.drift = float(drift_ppm) * 1e-6
        self.start_ns = int(start_ns)

    def __call__(self, t_mono_ns: int) -> int:
        return int(t_mono_ns) + self.offset_ns + int(round(self.drift * (int(t_mono_ns) - self.start_ns)))


@dataclass
class Stats:
    packets: int = 0
    skipped: int = 0             # ticks with no packet (GL stage could not start in time)
    uw_reused: int = 0
    stale: int = 0               # frames rendered from a state held more than max_hold_ms
    waiting: int = 0             # ticks before the first state arrived
    late_send: int = 0           # packets that left after their due time (+2 ms)
    gl_ms: list = field(default_factory=list)          # GL stage per tick (both cameras' render + readback)
    render_ms: dict = field(default_factory=lambda: {"main": [], "ultrawide": []})
    cpu_ms: list = field(default_factory=list)
    hold_ms: list = field(default_factory=list)
    send_after_mid_ms: list = field(default_factory=list)

    def summary(self) -> dict:
        def st(v):
            return {"mean": round(float(np.mean(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                    "max": round(float(np.max(v)), 2)} if v else None
        return {"packets": self.packets, "skipped": self.skipped, "uw_reused": self.uw_reused, "stale": self.stale,
                "waiting_for_state": self.waiting, "late_send": self.late_send, "gl_ms": st(self.gl_ms),
                "render_ms": {r: st(v) for r, v in self.render_ms.items()},
                "cpu_ms": st(self.cpu_ms), "state_hold_ms": st(self.hold_ms),
                "send_after_main_mid_ms": st(self.send_after_mid_ms)}


@dataclass
class _Job:
    k: int
    t_mid_mono: dict             # role -> CLOCK_MONOTONIC mid-exposure of the frame in this job
    futs: dict                   # role -> future of (stream image, JPEG) (ultrawide absent when reused)
    slot: int


class SimPi:
    def __init__(self, s: SimPiSettings):
        self.s = s
        self.stats = Stats()
        self._stop = threading.Event()
        self.rig = load_live_rig(s.rig_file)
        self.fps = float(self.rig.cameras["main"].fps)
        if float(self.rig.cameras["ultrawide"].fps) != self.fps:
            raise ValueError("main and ultrawide fps differ in the rig file; the sim Pi runs one schedule")
        self.period_ns = int(round(1e9 / self.fps))
        self.chains = {r: CameraChain(r, self.rig, density=s.density) for r in ROLES}
        W = max(c.max_view[0] for c in self.chains.values())
        H = max(c.max_view[1] for c in self.chains.values())
        wait_for(s.model, s.model_wait_s, log=log.info)
        self.lighting = Lighting(shadows=s.shadows)
        self.rm = load_render_model(s.model, self.lighting, W, H, keep_model_lights=s.keep_model_lights)
        for r, c in self.chains.items():
            c.set_hide([self.rm.own_board[r]])
        self.gl = GLRenderer(self.rm.model, W, H)
        self.gl.set_shadows(s.shadows)
        self.buffer = StateBuffer(quat_slices(self.rm.model))
        self.sub = StateSubscriber(s.state, self.buffer, self.rm.sha256, self.rm.model.nq)
        self._slots = [{r: c.new_buffer() for r, c in self.chains.items()} for _ in range(2)]
        self._free = queue.Queue()
        for i in range(len(self._slots)):
            self._free.put(i)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sim-pi-enc")
        self._out: queue.Queue = queue.Queue()
        self._last_uw: tuple[PiFrameStamp, bytes] | None = None
        self._save_dir = Path(s.save_dir) if s.save_dir else None
        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)
        self.look = None
        if s.camera_look is not None:
            from sim_deploy.look.camera_look import load_camera_look
            self.look = load_camera_look(s.camera_look)
        self.endpoint = None
        self.clock: PiClock | None = None

    def describe(self) -> str:
        lines = [f"model {self.rm.scene_xml} (sha {self.rm.sha256[:16]}..., nq {self.rm.model.nq}); GL: "
                 f"{self.gl.gl_renderer}; rig file {self.s.rig_file.name} ({self.rig.rig_id})"]
        for r, c in self.chains.items():
            i = c.info()
            lines.append(f"  {r}: native {i.native[0]}x{i.native[1]} -> {i.out[0]}x{i.out[1]}, "
                         f"{len(i.views)} view(s) {[(v['w'], v['h']) for v in i.views]} = {i.rendered_px / 1e6:.2f} Mpx")
        if self.look is not None:
            lines.append(f"  {self.look.describe()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------------------------------------ run
    def stop(self) -> None:
        self._stop.set()

    def run(self) -> dict:
        s = self.s
        ctx = zmq.Context.instance()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.setsockopt(zmq.SNDHWM, s.send_hwm)
        pub.bind(s.bind)
        self.endpoint = pub.getsockopt_string(zmq.LAST_ENDPOINT)
        self.sub.start()
        start = time.monotonic_ns()
        self.clock = PiClock(s.pi_clock_offset_s, s.pi_clock_drift_ppm, start)
        self.wall_offset_ns = time.time_ns() - self.clock(time.monotonic_ns())
        snd = threading.Thread(target=self._send_loop, args=(pub,), name="sim-pi-send", daemon=True)
        snd.start()
        log.info("[sim_pi] PUB %s (pi_stream/v1), state SUB %s, %g fps", self.endpoint, s.state, self.fps)
        try:
            self._gl_loop(start)
        finally:
            self._stop.set()
            self._out.put(None)
            snd.join(timeout=2)
            self.sub.stop()
            pub.close(linger=0)
            self.gl.close()
            self._write_stats()
        return self.stats.summary()

    def _gl_loop(self, start: int) -> None:
        s, P = self.s, self.period_ns
        phase = int(round(s.uw_phase_ms * 1e6))
        hold = int(round(s.max_hold_ms * 1e6))
        t0 = start + 200_000_000
        end = start + int(s.seconds * 1e9) if s.seconds > 0 else None
        last_stats = time.monotonic_ns()
        reused_last = False
        k = 0
        while not self._stop.is_set():
            if self.sub.error is not None:
                raise RuntimeError(f"state export: {self.sub.error}")
            tick = t0 + k * P
            t_mid = {"main": tick, "ultrawide": tick + phase}
            t_need = tick
            now = time.monotonic_ns()
            if end is not None and now >= end:
                break
            if now < t_need:
                self._stop.wait((t_need - now) / 1e9)
            self.buffer.wait_for(t_need, t_need + hold)
            now = time.monotonic_ns()
            if s.stats_s > 0 and now - last_stats >= s.stats_s * 1e9:
                self._status_line()
                last_stats = now
            if self.buffer.newest_t_ns() is None:              # nothing from the sim PC yet: no frames
                self.stats.waiting += 1
                if self.stats.waiting % int(self.fps * 5) == 1:
                    log.info("[sim_pi] waiting for the state export on %s ...", s.state)
                k += 1
                continue
            late = now - (t_need + hold)
            if late > P:                                       # cannot start in time: the tick is dropped
                self.stats.skipped += 1
                k += 1
                continue
            reuse = (late > P // 2 and not reused_last and self._last_uw is not None)
            slot = self._free.get()
            if self._stop.is_set():
                break
            futs, gl_ms = {}, 0.0
            for role in (("main",) if reuse else ROLES):
                if role != "main":
                    self.buffer.wait_for(t_mid[role], t_mid[role] + hold)
                g0 = time.perf_counter()
                st = self.buffer.at(t_mid[role])
                if st.held_ns > hold:
                    self.stats.stale += 1
                self.stats.hold_ms.append(max(st.held_ns, 0) / 1e6)
                self.rm.set_qpos(st.qpos)
                self.gl.update(self.rm.data)
                buf = self._slots[slot][role]
                self.chains[role].render_into(self.gl, *self.rm.camera_pose(role), buf)
                futs[role] = self._pool.submit(self._encode, role, buf, k)
                dt = (time.perf_counter() - g0) * 1e3
                self.stats.render_ms[role].append(dt)
                gl_ms += dt
            self.stats.gl_ms.append(gl_ms)
            reused_last = reuse
            self._out.put(_Job(k, t_mid, futs, slot))
            k += 1
        self._pool.shutdown(wait=True)

    # ------------------------------------------------------------------------------------------------ stages
    def _stamp(self, role: str, k: int, t_mid_mono: int) -> PiFrameStamp:
        exp_us = int(self.s.exposure_us[0 if role == "main" else 1])
        t_mid = self.clock(t_mid_mono)
        t_sensor = t_mid + exp_us * 1000 // 2
        return PiFrameStamp(frame_no=k, t_sensor_ns=t_sensor, exposure_us=exp_us,
                            t_mid_ns=t_sensor - exp_us * 1000 // 2)

    def _jpeg(self, rgb: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(self.s.jpeg_quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    def _encode(self, role: str, atlas: np.ndarray, k: int = 0) -> tuple[np.ndarray, bytes, float]:
        c0 = time.perf_counter()
        img = self.chains[role].stream_from(atlas)
        if self.look is not None:
            img = self.look.apply(role, img, np.random.default_rng([self.look.seed, ROLES.index(role), int(k)]))
        return img, self._jpeg(img), (time.perf_counter() - c0) * 1e3

    def _send_loop(self, pub) -> None:
        while True:
            job = self._out.get()
            if job is None:
                return
            res = {r: f.result() for r, f in job.futs.items()}
            self._free.put(job.slot)                           # the GL stage may reuse the buffers now
            self.stats.cpu_ms.append(max(v[2] for v in res.values()))
            ms = self._stamp("main", job.k, job.t_mid_mono["main"])
            if "ultrawide" in res:
                us = self._stamp("ultrawide", job.k, job.t_mid_mono["ultrawide"])
                self._last_uw = (us, res["ultrawide"][1])
                reused = False
            else:
                us = self._last_uw[0]
                reused = True
            uj = self._last_uw[1]
            if self._save_dir is not None and job.k % max(1, self.s.save_every) == 0:
                for r, v in res.items():
                    cv2.imwrite(str(self._save_dir / f"packet_{job.k:08d}_{r}.png"), v[0][:, :, ::-1])
            mid_mono = job.t_mid_mono["main"]
            send_at = mid_mono + self.s.exposure_us[0] * 1000 // 2 + int(round(self.s.send_delay_ms * 1e6))
            now = time.monotonic_ns()
            if send_at > now:
                time.sleep((send_at - now) / 1e9)
            now = time.monotonic_ns()
            if now > send_at + 2_000_000:
                self.stats.late_send += 1
            header = PiStreamHeader(version=1, packet_no=job.k, t_send_ns=self.clock(now),
                                    clock_offset_ns=self.wall_offset_ns, main=ms, ultrawide=us,
                                    skew_ns=us.t_mid_ns - ms.t_mid_ns,      # the live Pi's sign (PI_PAYLOAD.md 0.6)
                                    uw_reused=reused)
            parts = PiStreamPacket(header, res["main"][1], uj).to_parts()
            PiStreamPacket.from_parts(parts)                               # the laptop's own contract check
            try:
                pub.send_multipart(parts, zmq.NOBLOCK)
            except zmq.Again:
                pass
            self.stats.packets += 1
            if reused:
                self.stats.uw_reused += 1
            self.stats.send_after_mid_ms.append((now - mid_mono) / 1e6)

    # ------------------------------------------------------------------------------------------------ report
    def _status_line(self) -> None:
        st = self.stats
        n = min(len(st.gl_ms), int(self.fps))
        gl = np.mean(st.gl_ms[-n:]) if n else float("nan")
        cpu = np.mean(st.cpu_ms[-n:]) if st.cpu_ms else float("nan")
        newest = self.buffer.newest_t_ns()
        age = (time.monotonic_ns() - newest) / 1e6 if newest else float("nan")
        log.info("[sim_pi] packets %d | skipped %d | uw_reused %d | stale %d | GL %.1f ms | encode %.1f ms | "
                 "state age %.1f ms (%d msgs)", st.packets, st.skipped, st.uw_reused, st.stale, gl, cpu, age,
                 self.buffer.received)
        self._write_stats()
        for v in (st.gl_ms, st.cpu_ms, st.hold_ms, st.send_after_mid_ms, *st.render_ms.values()):   # keep short
            if len(v) > 20000:
                del v[:10000]

    def _write_stats(self) -> None:
        if self.s.stats_json:
            Path(self.s.stats_json).write_text(json.dumps(self.stats.summary(), indent=1))
