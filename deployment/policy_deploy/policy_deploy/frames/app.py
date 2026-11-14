# adapted from rnoro/deployment:deployment/process_frames/process_frames/app.py @ 91b178b
# changes: no ClockSync; always PiStreamPacketSource + OneWayClockMap (pi_v1, JPEG); the fake Pi binds
# ports.mock_pi; the receiver gets a PairRule (fix 1); the stats add uw_reused and the pair refusals; the waiting
# line names the Pi endpoint.
"""FramesApp: builds everything from Settings, runs the threads, writes the ready file and stats, shuts down
(STRUCTURE.md 4.1, 4.5)."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from policy_deploy.contracts import OBS_STEPS, OneWayClockMap

from policy_deploy.frames._dataproc.config import ConfigError
from policy_deploy.frames._dataproc.imageprep import ImagePrep
from policy_deploy.frames.camera_health import OK, CameraHealth
from policy_deploy.frames.errors import FatalError
from policy_deploy.frames.fake_pi_sender import FakePiSender
from policy_deploy.frames.frame_decoder import make_decoder
from policy_deploy.frames.frames_server import FramesServer
from policy_deploy.frames.packet_buffer import PacketBuffer
from policy_deploy.frames.packet_selector import PacketSelector
from policy_deploy.frames.pair_rule import PairRule
from policy_deploy.frames.pi_recording import PiRecording, PiRecordingError
from policy_deploy.frames.pi_stream_source import PiStreamPacketSource
from policy_deploy.frames.receiver import CAMERAS, Receiver
from policy_deploy.frames.rig_file import camera_sizes, load_live_rig, rig_summary
from policy_deploy.frames.settings import Settings

log = logging.getLogger("frames")

#: While not ready, say what it is waiting for this often.
_WAITING_LOG_EVERY_S = 5.0


class FramesApp:
    def __init__(self, settings: Settings, ready_file: Path | None, out_dir: Path,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.settings = settings
        self.ready_file = ready_file
        self.out_dir = Path(out_dir)
        self._clock = clock
        self._stop = threading.Event()
        self._fatal: FatalError | None = None
        self._fatal_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        # built by run(); public for logs, tests and the stats line
        self.rig = None
        self.server: FramesServer | None = None
        self.fake_pi: FakePiSender | None = None
        self.buffer: PacketBuffer | None = None
        self.health: dict[str, CameraHealth] = {}
        self.clock_map: OneWayClockMap | None = None
        self.source: PiStreamPacketSource | None = None
        self.receiver: Receiver | None = None
        self.ready = threading.Event()

    @property
    def frames_endpoint(self) -> str | None:
        """The bound C1 address (None until bound)."""
        return self.server.endpoint if self.server is not None else None

    def request_stop(self) -> None:
        self._stop.set()

    def _store_fatal(self, e: FatalError) -> None:
        with self._fatal_lock:
            if self._fatal is None:
                self._fatal = e
        self._stop.set()

    def _thread(self, name: str, body: Callable[[], None]) -> threading.Thread:
        def wrapped() -> None:
            try:
                body()
            except FatalError as e:
                self._store_fatal(e)
            except Exception as e:
                log.exception("thread %s failed", name)
                self._store_fatal(FatalError(f"internal error in thread {name}: {type(e).__name__}: {e}",
                                             "see the traceback above"))
        t = threading.Thread(target=wrapped, name=name, daemon=True)
        t.start()
        self._threads.append(t)
        return t

    # ---------------------------------------------------------------------------------------------- run

    def run(self) -> None:
        """Returns on stop (SIGINT / SIGTERM); raises FatalError for an unrecoverable problem."""
        try:
            self._start()
            self._main_loop()
        except FatalError as e:
            self._store_fatal(e)
        finally:
            self._shutdown()
        if self._fatal is not None:
            raise self._fatal

    def _start(self) -> None:
        s = self.settings
        try:
            rig = load_live_rig(s.rig_file)
        except ConfigError as e:
            raise FatalError(f"rig file invalid: {e}", "check frames.rig_file") from e
        self.rig = rig
        log.info("rig %s", rig_summary(rig))
        prep = ImagePrep(rig)

        self.health = {cam: CameraHealth(cam, s.health.stale_s, s.health.down_s, s.health.fps_window_s, self._clock)
                       for cam in CAMERAS}
        self.buffer = PacketBuffer(s.buffer.keep_s)
        selector = PacketSelector(fps={cam: rig.cameras[cam].fps for cam in CAMERAS},
                                  spacing_ns={"main": int(s.select.main_spacing_ms * 1e6),
                                              "ultrawide": int(s.select.ultrawide_spacing_ms * 1e6)},
                                  count=OBS_STEPS, tolerance_frames=s.select.tolerance_frames)
        try:
            self.server = FramesServer(s.frames, self.buffer, self.health, selector, prep, self._clock)
        except Exception as e:          # zmq.ZMQError: address in use, bad endpoint
            raise FatalError(f"cannot bind C1 on {s.frames} ({e})", "is another frames process running?") from e

        pi_frames = s.pi_frames
        if s.mock:
            m = s.mock_settings
            try:
                recording = PiRecording.load(s.mock_episode, max_seconds=m.max_seconds, jpeg_quality=m.jpeg_quality,
                                             sizes=camera_sizes(rig))
            except PiRecordingError as e:
                raise FatalError(f"mock episode unusable: {e}", "check frames.mock_episode") from e
            self.fake_pi = FakePiSender(recording, s.mock_pi_frames,
                                        pi_clock_offset_ns=int(round(m.pi_clock_offset_s * 1e9)),
                                        drift_ppm=m.drift_ppm, send_hwm=m.send_hwm, clock=self._clock)
            try:
                self.fake_pi.start()
            except Exception as e:
                raise FatalError(f"fake Pi cannot bind {s.mock_pi_frames} ({e})",
                                 "check station.yaml ports.mock_pi (another process on that port?)") from e
            pi_frames = self.fake_pi.frames_endpoint
            log.info("FakePiSender (stand-in Pi) playing %s on %s", s.mock_episode, pi_frames)

        c = s.clock
        # the Pi's 3-part stream; clock one-way from its t_send_ns (no time sync, PI_PAYLOAD.md 0.5)
        self.clock_map = OneWayClockMap(c.window_s, c.best_fraction, c.drift_min_span_s, c.max_drift_ppm,
                                        c.max_sync_age_s)
        decoders = {cam: make_decoder("jpeg", cam) for cam in CAMERAS}
        self.source = PiStreamPacketSource(pi_frames, s.receive.recv_hwm, self.clock_map, self._clock)
        pair_rule = PairRule(s.select.pair_max_age_ms, s.select.pair_max_lead_ms, s.select.pair_max_reuse)
        self.receiver = Receiver(self.source, decoders, rig, self.clock_map, self.buffer, self.health,
                                 s.default_exposure_ns, s.buffer.keep_s, self._clock, pair_rule)
        log.info("C1 on %s; Pi frames %s (pi_stream/v1, JPEG, clock from t_send_ns)", self.server.endpoint,
                 pi_frames)
        self._thread("receiver", lambda: self.receiver.run(self._stop, s.receive.timeout_s))

    def _main_loop(self) -> None:
        s = self.settings
        stats_every_ns = int(s.log.stats_every_s * 1e9)
        waiting_every_ns = int(_WAITING_LOG_EVERY_S * 1e9)
        next_stats = self._clock() + stats_every_ns
        next_waiting = self._clock() + waiting_every_ns
        synced_logged = False
        while not self._stop.is_set():
            self.server.serve_once(s.server.poll_s)
            now = self._clock()
            if not synced_logged and self.clock_map.is_fresh(now):
                synced_logged = True
                st = self.clock_map.stats(now)
                log.info("clock synced: offset %.6f s (one-way, from t_send_ns)", st.offset_ns / 1e9)
            if not self.ready.is_set():
                if all(self.health[cam].status(now) == OK for cam in CAMERAS):
                    self._write_ready()
                elif now >= next_waiting:
                    next_waiting = now + waiting_every_ns
                    log.info("not ready yet: %s", self._waiting_for(now))
            if now >= next_stats:
                next_stats = now + stats_every_ns
                self._write_stats(now)
        try:
            self._write_stats(self._clock())      # a last line at shutdown, so a short run has stats.jsonl too
        except Exception as exc:                  # never let the shutdown fail on a stats line
            log.warning("final stats line: %s", exc)

    def _write_ready(self) -> None:
        if self.ready_file is not None:
            self.ready_file.parent.mkdir(parents=True, exist_ok=True)
            self.ready_file.touch()
        self.ready.set()
        log.info("ready")

    def _waiting_for(self, now: int) -> str:
        why = []
        if self.receiver.packets.received and not self.clock_map.is_fresh(now):
            why.append(f"no packet for {self.settings.clock.max_sync_age_s:g} s (clock)")
        if self.receiver.packets.received == 0:
            why.append(f"no packets yet from {self.source.endpoint}: is the Pi on ASUS_5g and its sender running?")
        for cam in CAMERAS:
            if self.receiver.packets.received and self.receiver.counters[cam].received == 0:
                why.append(f"no {cam} frames yet")
            elif self.health[cam].status(now) != OK:
                why.append(f"{cam} {self.health[cam].status(now)}")
        return ", ".join(why) or "both cameras ok"

    def stats(self, now: int) -> dict:
        """The stats record (also the JSON line of stats.jsonl)."""
        out: dict = {"t_ns": now}
        for cam in CAMERAS:
            r = self.health[cam].report(now)
            c = self.receiver.counters[cam]
            out[cam] = {"status": r.status, "fps": r.fps, "age_ms": r.last_frame_age_ms, "dropped": r.dropped,
                        "received": c.received, "usable": c.usable, "unusable": c.unusable}
            if cam == "ultrawide":
                out[cam].update(unpaired=c.unpaired, pair_missing=c.pair_missing)
        pc = self.receiver.packets
        period = pc.take_stats()
        lat, gap = period["latency_ms"], period["pair_gap_ms"]
        out["packets"] = {"received": pc.received, "buffered": pc.buffered, "lost": pc.lost, "unsynced": pc.unsynced,
                          "out_of_order": pc.out_of_order, "ignored": pc.ignored, "uw_reused": pc.uw_reused,
                          "uw_reused_headers": self.source.converter.uw_reused if self.source else 0,
                          "pair_too_old": pc.pair_too_old, "pair_too_new": pc.pair_too_new,
                          "pair_reused": pc.pair_reused,
                          "latency_mean_ms": lat[0] if lat else None, "latency_max_ms": lat[1] if lat else None,
                          "pair_gap_mean_ms": gap[0] if gap else None, "pair_gap_max_ms": gap[1] if gap else None}
        st = self.clock_map.stats(now)
        out["clock"] = {"synced": st.synced, "offset_ns": st.offset_ns, "drift_ppm": st.drift_ppm,
                        "last_sync_age_s": st.last_sync_age_s, "samples": st.samples}
        out["served"], out["misses"] = self.server.served, self.server.misses
        return out

    @staticmethod
    def stats_line(rec: dict) -> str:
        def mean_max(d: dict, name: str) -> str:
            mean, mx = d[f"{name}_mean_ms"], d[f"{name}_max_ms"]
            return "-" if mean is None else f"{mean:.0f}/{mx:.0f}ms"

        parts = []
        for cam in CAMERAS:
            c = rec[cam]
            line = (f"{cam} {c['status']} {c['fps']:.1f}fps age {c['age_ms']:.0f}ms drop {c['dropped']} "
                    f"unusable {c['unusable']}")
            if cam == "ultrawide":
                line += f" unpaired {c['unpaired']}"
            parts.append(line)
        p = rec["packets"]
        parts.append(f"packets lost {p['lost']} unsynced {p['unsynced']} lat {mean_max(p, 'latency')} "
                     f"pair gap {mean_max(p, 'pair_gap')} uw_reused {p['uw_reused']} refused old/new/reused "
                     f"{p['pair_too_old']}/{p['pair_too_new']}/{p['pair_reused']}")
        k = rec["clock"]
        if k["offset_ns"] is None:
            parts.append("clock not synced")
        else:
            parts.append(f"clock offset {k['offset_ns'] / 1e9:.6f}s drift {k['drift_ppm']:.1f}ppm "
                         f"one-way sync {k['last_sync_age_s']:.1f}s ago")
        parts.append(f"served {rec['served']} misses {rec['misses']}")
        return "stats " + " | ".join(parts)

    def _write_stats(self, now: int) -> None:
        rec = self.stats(now)
        log.info("%s", self.stats_line(rec))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "stats.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _shutdown(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(self.settings.server.join_timeout_s)
            if t.is_alive():
                log.warning("thread %s did not stop within %.1f s", t.name, self.settings.server.join_timeout_s)
        self._threads = []
        if self.fake_pi is not None:
            self.fake_pi.stop()
        if self.server is not None:
            self.server.close()
