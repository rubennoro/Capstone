# adapted from behavior_prompting/deps/real-env/real_env/utils/umi_utils.py @ d0298b9 (get_downsampled_camera_images)
# kept: the latest frame_num packets, anchor on the newest, step back by 1/target_freq, nearest packet (first on ties),
# error beyond 2 frame periods; differs: one call over packets for both cameras, ns, a miss returns a reason instead
# of raising, a chosen packet without a usable image or the same packet twice is a miss too
"""PacketSelector and Selection: iPhUMI's picking rule over packets (STRUCTURE.md §5.10, §6.2). Pure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from process_frames.packet_buffer import BufferedPacket

CAMERAS = ("main", "ultrawide")


@dataclass(frozen=True)
class Selection:
    packets: tuple[BufferedPacket, ...] | None    # oldest first; None if the pick failed
    reason: str                                   # "" when ok

    @property
    def ok(self) -> bool:
        return self.packets is not None


class PacketSelector:
    def __init__(self, fps: Mapping[str, float], spacing_ns: Mapping[str, int], count: int,
                 tolerance_frames: float) -> None:
        """``fps``: each camera's rig frame rate; ``spacing_ns``: the wanted spacing of that camera's images (50 /
        100 ms); ``count``: images per camera (OBS_STEPS); ``tolerance_frames``: iPhUMI's limit (2)."""
        self.count = int(count)
        self.tolerance_frames = float(tolerance_frames)
        self.spacing_ns = {cam: int(spacing_ns[cam]) for cam in CAMERAS}
        self.tolerance_ns = {cam: tolerance_frames * 1e9 / float(fps[cam]) for cam in CAMERAS}
        # iPhUMI's frame_num: the packets that cover the obs window at nominal fps; it looks only at the newest ones
        frame_num = {cam: round((self.count - 1) * float(fps[cam]) * self.spacing_ns[cam] / 1e9) + 1
                     for cam in CAMERAS}
        self.window = max(frame_num.values())

    def targets(self, anchor_ns: int, camera: str) -> np.ndarray:
        """int64, oldest first: ``anchor - (count-1-i) * spacing``."""
        i = np.arange(self.count, dtype=np.int64)
        return np.int64(anchor_ns) - (self.count - 1 - i) * np.int64(self.spacing_ns[camera])

    def pick(self, packets: Sequence[BufferedPacket]) -> dict[str, Selection]:
        """Per camera, the packets nearest the targets measured back from the newest packet."""
        if not packets:
            return {cam: Selection(None, "no packets") for cam in CAMERAS}
        recent = list(packets)[-self.window:]
        times = np.array([p.t_pc_ns for p in recent], dtype=np.int64)
        anchor = int(times[-1])
        return {cam: self._pick_one(recent, times, anchor, cam) for cam in CAMERAS}

    def _pick_one(self, recent: list[BufferedPacket], times: np.ndarray, anchor: int, cam: str) -> Selection:
        targets = self.targets(anchor, cam)
        dist = np.abs(times[:, None] - targets[None, :])
        indices = np.argmin(dist, axis=0)                        # first minimum: ties go to the earlier packet
        deviation = dist[indices, np.arange(self.count)]
        worst = int(np.argmax(deviation))
        if deviation[worst] > self.tolerance_ns[cam]:
            return Selection(None, f"nearest packet {deviation[worst] / 1e6:.1f} ms from its target (max "
                                   f"{self.tolerance_ns[cam] / 1e6:.1f} ms = {self.tolerance_frames:g} frame periods)")
        chosen = [recent[int(i)] for i in indices]
        if len({id(p) for p in chosen}) != len(chosen):
            return Selection(None, f"the same packet picked for two targets (packet_seq {chosen[0].packet_seq})")
        for p in chosen:
            if p.image(cam) is None:
                why = "no paired ultrawide image" if cam == "ultrawide" else "no usable main image"
                return Selection(None, f"packet_seq {p.packet_seq} has {why}")
        return Selection(tuple(chosen), "")
