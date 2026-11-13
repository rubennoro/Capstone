"""FramesServer: the serving side, contract C1, main thread (STRUCTURE.md §5.13, §6.2)."""

from __future__ import annotations

import logging
import time
from typing import Callable, Mapping

import numpy as np

from contracts import (CameraHealthReport, FramesReply, GetFramesRequest, GetHealthRequest, HealthReply,
                       HealthReport, RpcServer)

from process_frames._dataproc.imageprep import ImagePrep
from process_frames.camera_health import CameraHealth, OK, STALE
from process_frames.packet_buffer import PacketBuffer
from process_frames.packet_selector import PacketSelector

log = logging.getLogger("process_frames.frames_server")

#: At most one "pick failed" log line this often.
_MISS_LOG_EVERY_NS = 1_000_000_000


class FramesServer:
    def __init__(self, endpoint: str, buffer: PacketBuffer, health: Mapping[str, CameraHealth],
                 selector: PacketSelector, prep: ImagePrep, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._buffer = buffer
        self._health = health
        self._selector = selector
        self._prep = prep
        self._clock = clock
        self._last_miss_log: int | None = None
        self.served = 0
        self.misses = 0
        # bound here, in the main thread that also serves it, so a busy port fails at once
        self._server = RpcServer(endpoint, {GetFramesRequest: self.get_frames, GetHealthRequest: self.get_health})
        self.endpoint: str = self._server.endpoint

    def _report(self, now: int) -> dict[str, CameraHealthReport]:
        return {cam: self._health[cam].report(now) for cam in ("main", "ultrawide")}

    @staticmethod
    def _no_images(report: Mapping[str, CameraHealthReport]) -> FramesReply:
        return FramesReply(main=None, main_t_ns=None, ultrawide=None, ultrawide_t_ns=None,
                           health=HealthReport(main=report["main"], ultrawide=report["ultrawide"]))

    def get_frames(self, req: GetFramesRequest) -> FramesReply:
        now = self._clock()
        report = self._report(now)
        if any(r.status != OK for r in report.values()):                 # both or nothing
            return self._no_images(report)
        picks = self._selector.pick(self._buffer.snapshot())            # anchored on the newest packet
        failed = [cam for cam, sel in picks.items() if not sel.ok]
        if failed:
            self.misses += 1
            for cam in failed:                    # stale in this reply only: age, fps and dropped unchanged
                r = report[cam]
                report[cam] = CameraHealthReport(status=STALE, last_frame_age_ms=r.last_frame_age_ms, fps=r.fps,
                                                 dropped=r.dropped)
            if self._last_miss_log is None or now - self._last_miss_log >= _MISS_LOG_EVERY_NS:
                self._last_miss_log = now
                log.info("no images (%d misses so far): %s", self.misses,
                         "; ".join(f"{cam}: {picks[cam].reason}" for cam in failed))
            return self._no_images(report)
        main_p, uw_p = picks["main"].packets, picks["ultrawide"].packets
        reply = FramesReply(                                             # ultrawide times: their packets' times
            main=np.stack([self._prep.prep_main(p.main_rgb) for p in main_p]),
            main_t_ns=np.array([p.t_pc_ns for p in main_p], dtype=np.int64),
            ultrawide=np.stack([self._prep.prep_ultrawide(p.ultrawide_rgb) for p in uw_p]),
            ultrawide_t_ns=np.array([p.t_pc_ns for p in uw_p], dtype=np.int64),
            health=HealthReport(main=report["main"], ultrawide=report["ultrawide"]))
        self.served += 1
        return reply

    def get_health(self, req: GetHealthRequest) -> HealthReply:
        report = self._report(self._clock())
        return HealthReply(health=HealthReport(main=report["main"], ultrawide=report["ultrawide"]))

    def serve_once(self, timeout_s: float) -> bool:
        return self._server.serve_once(timeout_s)

    def close(self) -> None:
        self._server.close()
