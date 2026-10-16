"""Read-only access to one real iPhone export folder (R1): JSON fields, slot times, ARKit poses, decoded frames.

Used by the simulator (D10a) and the acceptance tests. Never writes anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from sensor_rig.timefmt import parse_episode_name, parse_iso


class IphoneEpisode:
    def __init__(self, folder):
        self.folder = Path(folder)
        jsons = sorted(p for p in self.folder.glob("*.json") if not p.name.startswith("."))
        if len(jsons) != 1:
            raise FileNotFoundError(f"{self.folder}: expected exactly one <NAME>.json, found {len(jsons)}")
        self.json_path = jsons[0]
        self.name = self.json_path.stem
        self.parts = parse_episode_name(self.name)
        with open(self.json_path) as f:
            self.json: dict = json.load(f)
        self.K_main = np.asarray(self.json["mainCameraIntrinsics"], dtype=np.float64)
        self.K_uw = np.asarray(self.json["ultrawideCameraIntrinsics"], dtype=np.float64)

    # ---- metadata
    @property
    def kind(self) -> str:
        return self.json.get("type") or self.parts["kind"]

    @property
    def is_calibration(self) -> bool:
        return self.kind == "gripper_calibration"

    @property
    def frame_count(self) -> int:
        return int(self.json.get("frameCount") or len(self.json["ultrawideRGBTimes"]))

    @property
    def rgb_path(self) -> Path:
        return self.folder / f"{self.name}_rgb.mp4"

    @property
    def uw_path(self) -> Path:
        return self.folder / f"{self.name}_ultrawidergb.mp4"

    def recording_start_unix_ns(self) -> int:
        return parse_iso(self.json["recordingStartTime"])

    # ---- times
    def uw_slots(self) -> np.ndarray:
        """Indices of the main-grid slots that carry an ultrawide frame (the k-th is UW video frame k)."""
        return np.array([i for i, s in enumerate(self.json["ultrawideRGBTimes"]) if s], dtype=np.int64)

    def main_times_unix_ns(self) -> np.ndarray:
        """UNIX ns of every main-grid slot (ms precision, from the strings).

        Demonstrations: ``rgbTimes``. Calibrations have no ``rgbTimes``: the slot times are reconstructed by a least
        squares fit ``t = a + b*i`` to the non-empty ``ultrawideRGBTimes`` slots, keeping the exact value at those slots
        (P1 §4.8 step 1). The result is made strictly increasing.
        """
        rgb = self.json.get("rgbTimes") or []
        if rgb:
            t = np.array([parse_iso(s) for s in rgb], dtype=np.int64)
        else:
            slots = self.uw_slots()
            if len(slots) < 2:
                raise ValueError(f"{self.name}: cannot reconstruct slot times from {len(slots)} ultrawide slots")
            known = np.array([parse_iso(self.json["ultrawideRGBTimes"][i]) for i in slots], dtype=np.int64)
            t0 = int(known[0])
            b, a = np.polyfit(slots.astype(np.float64), (known - t0).astype(np.float64), 1)
            t = t0 + np.round(a + b * np.arange(self.frame_count)).astype(np.int64)
            t[slots] = known
        for i in range(1, len(t)):  # enforce strict increase (never needed for real exports, cheap safety)
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + 1
        return t

    def poses_arkit(self) -> np.ndarray:
        """(N,4,4) float64 row-major ``camera.transform`` (empty (0,4,4) for calibrations)."""
        P = self.json.get("poseTransforms") or []
        return np.asarray(P, dtype=np.float64).reshape(-1, 4, 4)

    # ---- frames
    @staticmethod
    def _iter_video(path: Path, fmt: str) -> Iterator[np.ndarray]:
        import av

        with av.open(str(path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for frame in c.decode(s):
                yield frame.to_ndarray(format=fmt)

    def iter_main_frames(self, bgr: bool = True) -> Iterator[np.ndarray]:
        """Decoded main frames in order (frame i belongs to slot i)."""
        if not self.rgb_path.is_file():
            raise FileNotFoundError(f"{self.rgb_path} missing (calibration episodes have no main video)")
        return self._iter_video(self.rgb_path, "bgr24" if bgr else "rgb24")

    def iter_uw_frames(self, bgr: bool = True) -> Iterator[np.ndarray]:
        """Decoded ultrawide frames in order (frame k belongs to slot ``uw_slots()[k]``)."""
        return self._iter_video(self.uw_path, "bgr24" if bgr else "rgb24")
