"""H.264 writer/prober for the iPhone-format MP4s (R1 §6, P1 §4.6).

Like the app: QuickTime container (``.mp4`` extension, brand ``qt  ``), timescale 600, VFR PTS from real capture
times, first PTS 0, B-frames. The container's sample table gives ``stream.frames`` (gen_replay_buffer_zarr.py:588 skips
frame indices >= ``in_stream.frames`` and leaves them black, so the header count must be right).

Pixel format ``yuvj420p`` (full range, BT.601), like the iPhone. Frames are converted with OpenCV's full-range
BT.601 YCrCb (the inverse of what PyAV's ``to_ndarray('rgb24')`` does on decode) and chroma is area-downsampled.
PyAV's own ``reformat`` darkened frames by ~2 levels on the round trip; this path keeps the decode within ~1 level
of the input on average, so exported (training) images stay close to deployment images, which skip H.264.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

# AVColorRange / AVColorSpace / AVColorPrimaries / AVColorTransferCharacteristic values
_COLOR_RANGE_JPEG = 2
_COLORSPACE_SMPTE170M = 6
_PRIMARIES_BT709 = 1
_TRC_BT709 = 1


def bgr_to_yuvj420p(bgr: np.ndarray) -> np.ndarray:
    """(H,W,3) BGR uint8 -> (H*3/2, W) planar full-range BT.601 Y, U(Cb), V(Cr), chroma area-downsampled 2x2."""
    import cv2

    h, w = bgr.shape[:2]
    ycc = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2YCrCb)
    u = cv2.resize(ycc[..., 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(ycc[..., 1], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return np.concatenate([ycc[..., 0].reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(h * 3 // 2, w)


class Mp4Writer:
    def __init__(self, path, width: int, height: int, time_base_den: int = 600, container: str = "mov",
                 crf: int = 18, preset: str = "faster", gop: int = 30, bframes: int = 2, nominal_rate: int = 60):
        import av

        if width % 2 or height % 2:
            raise ValueError(f"libx264 yuvj420p needs even dimensions, got {width}x{height}")
        self.path = Path(path)
        self.width, self.height = int(width), int(height)
        self.den = int(time_base_den)
        self._tb = Fraction(1, self.den)
        opts = {"video_track_timescale": str(self.den)} if container == "mov" else {}
        self._c = av.open(str(self.path), "w", format=container, options=opts)
        s = self._c.add_stream("libx264", rate=nominal_rate)
        s.width, s.height = self.width, self.height
        s.pix_fmt = "yuvj420p"
        s.time_base = self._tb
        cc = s.codec_context
        cc.time_base = self._tb
        cc.color_range = _COLOR_RANGE_JPEG
        cc.colorspace = _COLORSPACE_SMPTE170M
        cc.color_primaries = _PRIMARIES_BT709
        cc.color_trc = _TRC_BT709
        s.options = {"crf": str(crf), "preset": preset, "profile": "high", "g": str(gop), "bf": str(bframes)}
        self._s = s
        self._last_pts: int | None = None
        self.frames_in = 0
        self.packets_out = 0
        self.pts: list[int] = []

    def write(self, bgr: np.ndarray, pts: int) -> None:
        import av

        pts = int(pts)
        if self._last_pts is not None and pts <= self._last_pts:
            raise ValueError(f"PTS must be strictly increasing ({pts} after {self._last_pts})")
        if bgr.shape != (self.height, self.width, 3) or bgr.dtype != np.uint8:
            raise ValueError(f"expected ({self.height},{self.width},3) uint8, got {bgr.shape} {bgr.dtype}")
        frame = av.VideoFrame.from_ndarray(bgr_to_yuvj420p(bgr), format="yuvj420p")
        frame.pts = pts
        frame.time_base = self._tb
        for pkt in self._s.encode(frame):
            self._c.mux(pkt)
            self.packets_out += 1
        self._last_pts = pts
        self.frames_in += 1
        self.pts.append(pts)

    def close(self) -> int:
        if self._c is None:
            return self.frames_in
        for pkt in self._s.encode(None):
            self._c.mux(pkt)
            self.packets_out += 1
        self._c.close()
        self._c = None
        if self.packets_out != self.frames_in:
            raise RuntimeError(f"{self.path}: encoder emitted {self.packets_out} packets for {self.frames_in} frames")
        return self.frames_in

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        elif self._c is not None:
            self._c.close()
            self._c = None


def probe_video(path) -> dict:
    """Demux (no decoding): header frame count, sorted packet PTS (in stream ticks), size, time base."""
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        pts = sorted(int(p.pts) for p in c.demux(st) if p.pts is not None)
        return {
            "frames_header": int(st.frames),
            "pts_list": pts,
            "n_packets": len(pts),
            "width": int(st.codec_context.width),
            "height": int(st.codec_context.height),
            "time_base": st.time_base,
        }
