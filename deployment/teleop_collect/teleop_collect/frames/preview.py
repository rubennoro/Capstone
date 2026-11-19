"""LatestFrames: the newest JPEG of each camera, exactly as received, for the page's preview (U2, STRUCTURE.md 5.3).

New. The receiver thread calls update() with every RecordItem; the main thread answers u2.get_preview with get().
A camera's JPEG goes into the reply only when its frame_no is above what the ui already has, so an unchanged
preview costs nothing to send. The health comes along with every reply. One lock.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Mapping

from teleop_collect.contracts import NO_FRAME, CameraHealthReport, HealthReport, Preview

from teleop_collect.frames.camera_health import CameraHealth
from teleop_collect.frames.recorder import RecordItem


def health_report(health: Mapping[str, CameraHealth], now_ns: int) -> HealthReport:
    """Both cameras' reports at now_ns."""
    reports: dict[str, CameraHealthReport] = {cam: health[cam].report(now_ns) for cam in ("main", "ultrawide")}
    return HealthReport(main=reports["main"], ultrawide=reports["ultrawide"])


class LatestFrames:
    def __init__(self, health: Mapping[str, CameraHealth], clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._health = health
        self._clock = clock
        self._lock = threading.Lock()
        self._jpeg: dict[str, bytes | None] = {"main": None, "ultrawide": None}
        self._frame_no: dict[str, int] = {"main": NO_FRAME, "ultrawide": NO_FRAME}

    def update(self, item: RecordItem) -> None:
        """Keep the item's main frame and, when it carries one, its new ultrawide frame."""
        with self._lock:
            self._jpeg["main"] = item.main.data
            self._frame_no["main"] = int(item.main.info.frame_seq)
            if item.ultrawide is not None:
                self._jpeg["ultrawide"] = item.ultrawide.data
                self._frame_no["ultrawide"] = int(item.ultrawide.info.frame_seq)

    def newest(self) -> dict[str, int]:
        """The newest frame_no per camera (-1 if none yet)."""
        with self._lock:
            return dict(self._frame_no)

    def get(self, main_after: int, ultrawide_after: int) -> Preview:
        """The u2.preview/v1 reply: each camera's JPEG only if its newest frame_no is above *_after."""
        with self._lock:
            jpeg, no = dict(self._jpeg), dict(self._frame_no)
        after = {"main": int(main_after), "ultrawide": int(ultrawide_after)}
        send = {cam: jpeg[cam] if jpeg[cam] is not None and no[cam] > after[cam] else None for cam in no}
        return Preview(main_jpeg=send["main"], ultrawide_jpeg=send["ultrawide"], main_frame_no=no["main"],
                       ultrawide_frame_no=no["ultrawide"], health=health_report(self._health, self._clock()))
