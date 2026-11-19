"""FramesServers: R1 (teleop -> frames) and U2 (ui -> frames), two REP sockets served from the main thread
(STRUCTURE.md 4.2, 4.3, 5.3).

New, in the manner of pd:frames/frames_server.py (bound in the main thread that serves them, so a busy port fails at
once). R1: start_recording (ack, or refused with the reason), stop_recording (the summary; error/v1 when not
recording), get_health (always answered). U2: get_preview (the newest JPEGs the ui does not have yet, and health).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Mapping

from deploy_contracts import Ack, ErrorReply, RpcServer

from teleop_collect.contracts import (CAMERAS, ClockMap, FramesHealth, GetFramesHealth, GetPreview, Preview,
                                      RecordingSummary, StartRecording, StopRecording)

from teleop_collect.frames.camera_health import OK, CameraHealth
from teleop_collect.frames.errors import RecordingRefused
from teleop_collect.frames.preview import LatestFrames, health_report
from teleop_collect.frames.recorder import FrameRecorder

log = logging.getLogger("frames.servers")


class FramesServers:
    def __init__(self, r1_endpoint: str, preview_endpoint: str, recorder: FrameRecorder, latest: LatestFrames,
                 health: Mapping[str, CameraHealth], clock_map: ClockMap,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._recorder = recorder
        self._latest = latest
        self._health = health
        self._clock_map = clock_map
        self._clock = clock
        self.requests = {"r1": 0, "u2": 0}
        self._r1 = RpcServer(r1_endpoint, {StartRecording: self.start_recording, StopRecording: self.stop_recording,
                                           GetFramesHealth: self.get_health})
        try:
            self._u2 = RpcServer(preview_endpoint, {GetPreview: self.get_preview})
        except BaseException:
            self._r1.close()
            raise
        self.r1_endpoint: str = self._r1.endpoint
        self.preview_endpoint: str = self._u2.endpoint

    # ------------------------------------------------------------------------------------------------ R1

    def start_refusal(self, episode_dir: Path) -> str:
        """Why a recording cannot start now ("" if it can), before the folder checks of FrameRecorder.start."""
        if self._recorder.recording:
            return "already recording"
        now = self._clock()
        for cam in CAMERAS:
            status = self._health[cam].status(now)
            if status != OK:
                return f"{cam} camera {status}"
        if not self._clock_map.is_fresh(now):
            return "Pi clock not synced"
        return ""

    def start_recording(self, req: StartRecording) -> Ack:
        folder = Path(req.episode_dir)
        reason = self.start_refusal(folder)
        if not reason:
            try:
                self._recorder.start(folder)
            except RecordingRefused as e:
                reason = str(e)
        if reason:
            log.warning("start_recording %s refused: %s", folder, reason)
            return Ack(ok=False, reason=reason)
        return Ack(ok=True, reason="")

    def stop_recording(self, req: StopRecording) -> RecordingSummary | ErrorReply:
        if not self._recorder.recording:
            return ErrorReply(request=StopRecording.SCHEMA, error=f"{StopRecording.SCHEMA}: not recording", field="")
        return self._recorder.stop()

    def get_health(self, req: GetFramesHealth) -> FramesHealth:
        now = self._clock()
        r = self._recorder
        return FramesHealth(health=health_report(self._health, now), synced=self._clock_map.is_fresh(now),
                            recording=r.recording, recording_error=r.error,
                            frames_main=r.frames_written("main"), frames_ultrawide=r.frames_written("ultrawide"))

    # ------------------------------------------------------------------------------------------------ U2

    def get_preview(self, req: GetPreview) -> Preview:
        return self._latest.get(req.main_after, req.ultrawide_after)

    # ------------------------------------------------------------------------------------------------ serving

    def serve_once(self, timeout_s: float) -> bool:
        """Answer at most one R1 and one U2 request, waiting up to timeout_s in all (R1 first, half each)."""
        half = max(0.0, timeout_s / 2)
        r1 = self._r1.serve_once(half)
        u2 = self._u2.serve_once(0.0 if r1 else half)
        self.requests["r1"] += int(r1)
        self.requests["u2"] += int(u2)
        return r1 or u2

    def close(self) -> None:
        self._r1.close()
        self._u2.close()
