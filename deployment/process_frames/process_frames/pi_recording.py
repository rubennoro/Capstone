# adapted from rnoro/data-work:data-processing/dataproc/raw_episode.py @ 96ff703 (read_video_frames: the same PyAV
# decode and to_ndarray("rgb24")), and the Pi file layout of docs/PI_ADAPTER.md and dataproc/pi_format.py @ aac065c
# (read only; pi_format.py writes the format and depends on RawEpisode, so it is not copied)
"""PiRecording and RecordedCamera: a recorded Pi episode as C0 frames (STRUCTURE.md §5.13).

Used only by the two stand-ins for the Pi (ReplaySource in unit tests, FakePiSender in unit tests and --mock); it
never feeds the real arm. Each frame is re-encoded the way the live Pi sends it (H.264: libx264, zerolatency, a
keyframe every ``keyframe_interval`` frames with SPS/PPS before it, Annex-B, one access unit per message; JPEG: one
image per frame).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping

import av
import cv2
import numpy as np

from contracts import Frame

#: Role by sensor model (the Pi's own cam0/cam1 names are not trusted).
ROLE_BY_MODEL = {"imx519": "main", "imx219": "ultrawide"}
ENCODINGS = ("h264", "jpeg")


class PiRecordingError(ValueError):
    """A recorded Pi episode that can't be used; names the file and the problem."""


@dataclass(frozen=True)
class RecordedCamera:
    camera: str
    sensor: str
    width: int
    height: int
    fps: float
    frames: tuple[Frame, ...]          # encoded, seq from 0, t_pi_ns and exposure_ns as recorded (0 if no column)

    @property
    def period_ns(self) -> int:
        return int(round(1e9 / self.fps))


def _read_frames_csv(path: Path, expected_rows: int) -> tuple[list[int], list[int]]:
    """(t_ns, exposure_ns) per row; exposure 0 when there is no column."""
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except OSError as e:
        raise PiRecordingError(f"{path}: cannot read ({e})") from None
    if not rows or rows[0] not in (["frame_index", "t_ns"], ["frame_index", "t_ns", "exposure_ns"]):
        raise PiRecordingError(f"{path}: header must be frame_index,t_ns[,exposure_ns], got "
                               f"{','.join(rows[0]) if rows else 'nothing'}")
    has_exp = len(rows[0]) == 3
    body = [r for r in rows[1:] if r]
    if len(body) != expected_rows:
        raise PiRecordingError(f"{path}: {len(body)} rows, meta.json says {expected_rows} frames")
    t_ns, exp = [], []
    for i, r in enumerate(body):
        try:
            if len(r) != len(rows[0]) or int(r[0]) != i:
                raise ValueError("bad row")
            t_ns.append(int(r[1]))
            exp.append(int(r[2]) if has_exp else 0)
        except ValueError:
            raise PiRecordingError(f"{path}: row {i + 1} is not 'frame_index,t_ns[,exposure_ns]': {r}") from None
        if exp[-1] < 0:
            raise PiRecordingError(f"{path}: row {i + 1}: exposure_ns must be >= 0")
    if any(b <= a for a, b in zip(t_ns, t_ns[1:])):
        raise PiRecordingError(f"{path}: t_ns is not strictly increasing")
    return t_ns, exp


def _decode_video(path: Path, keep: int, total: int, size: tuple[int, int]) -> list[np.ndarray]:
    """The first ``keep`` frames as RGB, decoded as dataproc's read_video_frames does; the video must have at least
    ``total`` frames (its CSV rows)."""
    out: list[np.ndarray] = []
    n = 0
    try:
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in c.decode(stream):
                if n < keep:
                    out.append(frame.to_ndarray(format="rgb24"))
                n += 1
    except (av.error.FFmpegError, IndexError, OSError) as e:
        raise PiRecordingError(f"{path}: cannot decode ({e})") from None
    if n < total:
        raise PiRecordingError(f"{path}: {n} frames, shorter than its CSV ({total} rows)")
    if out and out[0].shape[:2] != (size[1], size[0]):
        raise PiRecordingError(f"{path}: frames are {out[0].shape[1]}x{out[0].shape[0]}, meta.json says "
                               f"{size[0]}x{size[1]}")
    return out


def encode_h264(images: list[np.ndarray], fps: float, keyframe_interval: int, bitrate: int) -> list[tuple[bytes, bool]]:
    """(Annex-B access unit, keyframe) per image, as the live Pi encoder sends them."""
    h, w = images[0].shape[:2]
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = w, h, "yuv420p"
    rate = Fraction(fps).limit_denominator(1001)
    ctx.framerate = rate
    ctx.time_base = 1 / rate
    ctx.bit_rate = int(bitrate)
    ctx.options = {"preset": "ultrafast", "tune": "zerolatency",       # no B-frames, no look-ahead
                   "x264-params": f"keyint={keyframe_interval}:scenecut=0:repeat-headers=1"}
    out: list[tuple[bytes, bool]] = []
    for i, rgb in enumerate(images):
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24").reformat(format="yuv420p")
        vf.pts = i
        packets = ctx.encode(vf)
        if len(packets) != 1:
            raise PiRecordingError(f"libx264 gave {len(packets)} packets for frame {i}, expected 1")
        out.append((bytes(packets[0]), bool(packets[0].is_keyframe)))
    if ctx.encode(None):
        raise PiRecordingError("libx264 held frames back (expected none with tune=zerolatency)")
    return out


def encode_jpeg(images: list[np.ndarray], quality: int) -> list[tuple[bytes, bool]]:
    out = []
    for rgb in images:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise PiRecordingError("cv2.imencode failed")
        out.append((buf.tobytes(), True))
    return out


@dataclass(frozen=True)
class PiRecording:
    path: Path
    cameras: Mapping[str, RecordedCamera]    # "main", "ultrawide"
    duration_ns: int                         # from the first frame to one period after the last

    @classmethod
    def load(cls, path: Path, encoding: str, max_seconds: float, keyframe_interval: int, bitrate: int,
             jpeg_quality: int) -> "PiRecording":
        path = Path(path)
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
        meta_path = path / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except OSError:
            raise PiRecordingError(f"{meta_path}: not found (not a recorded Pi episode)") from None
        except json.JSONDecodeError as e:
            raise PiRecordingError(f"{meta_path}: invalid JSON ({e})") from None
        cams = meta.get("cameras") if isinstance(meta, dict) else None
        if not isinstance(cams, list):
            raise PiRecordingError(f"{meta_path}: no 'cameras' list")
        found: dict[str, dict] = {}
        for i, c in enumerate(cams):
            try:
                model, stem = c["model"], Path(c["file"]).stem
                size, fps, n = (int(c["size"][0]), int(c["size"][1])), float(c["fps"]), int(c["frames"])
            except (KeyError, TypeError, ValueError, IndexError):
                raise PiRecordingError(f"{meta_path}: cameras[{i}]: needs file, model, size, fps, frames") from None
            if model not in ROLE_BY_MODEL:
                raise PiRecordingError(f"{meta_path}: cameras[{i}]: unknown sensor model {model!r} "
                                       f"(expected one of {sorted(ROLE_BY_MODEL)})")
            role = ROLE_BY_MODEL[model]
            if role in found:
                raise PiRecordingError(f"{meta_path}: two cameras with model {model!r}")
            found[role] = {"model": model, "stem": stem, "size": size, "fps": fps, "frames": n}
        for role in ("main", "ultrawide"):
            if role not in found:
                model = next(m for m, r in ROLE_BY_MODEL.items() if r == role)
                raise PiRecordingError(f"{meta_path}: no {role} camera (model {model})")

        recorded: dict[str, RecordedCamera] = {}
        for role, c in found.items():
            t_ns, exp = _read_frames_csv(path / f"{c['stem']}_frames.csv", c["frames"])
            limit = t_ns[0] + int(max_seconds * 1e9)
            keep = max(1, sum(1 for t in t_ns if t < limit))
            images = _decode_video(path / f"{c['stem']}.mp4", keep, len(t_ns), c["size"])
            if encoding == "h264":
                encoded = encode_h264(images, c["fps"], keyframe_interval, bitrate)
            else:
                encoded = encode_jpeg(images, jpeg_quality)
            w, h = c["size"]
            frames = tuple(Frame(camera=role, sensor=c["model"], seq=i, t_pi_ns=t_ns[i], exposure_ns=exp[i], width=w,
                                 height=h, encoding=encoding, keyframe=key, data=data)
                           for i, (data, key) in enumerate(encoded))
            recorded[role] = RecordedCamera(camera=role, sensor=c["model"], width=w, height=h, fps=c["fps"],
                                            frames=frames)
        start = min(cam.frames[0].t_pi_ns for cam in recorded.values())
        end = max(cam.frames[-1].t_pi_ns + cam.period_ns for cam in recorded.values())
        return cls(path=path, cameras=recorded, duration_ns=end - start)

    @property
    def start_ns(self) -> int:
        """The Pi time of the first frame of either camera."""
        return min(cam.frames[0].t_pi_ns for cam in self.cameras.values())
