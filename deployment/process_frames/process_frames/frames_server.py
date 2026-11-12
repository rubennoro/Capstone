"""FramesServer: the serving side, contract C1, main thread (STRUCTURE.md §5.12, §6.2)."""

from __future__ import annotations

import logging
import time
from typing import Callable, Mapping

import numpy as np

from contracts import (OBS_STEPS, CameraHealthReport, FramesReply, GetFramesRequest, GetHealthRequest, HealthReply,
                       HealthReport, RpcServer)

from process_frames._dataproc.imageprep import ImagePrep
from process_frames.camera_health import CameraHealth, OK, STALE
from process_frames.frame_buffer import FrameBuffer
from process_frames.frame_selector import FrameSelector, Selection

log = logging.getLogger("process_frames.frames_server")

#: At most one "pick failed" log line this often.
_MISS_LOG_EVERY_NS = 1_000_000_000


class FramesServer:
    def __init__(self, endpoint: str, buffers: Mapping[str, FrameBuffer], health: Mapping[str, CameraHealth],
                 selector: FrameSelector, prep: ImagePrep, main_spacing_ns: int, ultrawide_spacing_ns: int,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._buffers = buffers
        self._health = health
        self._selector = selector
        self._prep = prep
        self._spacing = {"main": int(main_spacing_ns), "ultrawide": int(ultrawide_spacing_ns)}
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
        snaps = {cam: self._buffers[cam].snapshot() for cam in ("main", "ultrawide")}
        if not snaps["main"]:
            picks = {"main": Selection(None, "no frames"), "ultrawide": Selection(None, "no main anchor")}
        else:
            anchor = snaps["main"][-1].t_pc_ns                           # the newest main frame
            picks = {cam: self._selector.pick(snaps[cam], anchor, self._spacing[cam], OBS_STEPS)
                     for cam in ("main", "ultrawide")}
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
        main_f, uw_f = picks["main"].frames, picks["ultrawide"].frames
        reply = FramesReply(
            main=np.stack([self._prep.prep_main(f.rgb) for f in main_f]),
            main_t_ns=np.array([f.t_pc_ns for f in main_f], dtype=np.int64),
            ultrawide=np.stack([self._prep.prep_ultrawide(f.rgb) for f in uw_f]),
            ultrawide_t_ns=np.array([f.t_pc_ns for f in uw_f], dtype=np.int64),
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
