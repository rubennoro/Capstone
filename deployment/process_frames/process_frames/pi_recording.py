# adapted from rnoro/data-work:data-processing/dataproc/raw_episode.py @ 96ff703 (read_video_frames: the same PyAV
# decode and to_ndarray("rgb24")), and the Pi file layout of docs/PI_ADAPTER.md and dataproc/pi_format.py @ aac065c
# (read only; pi_format.py writes the format and depends on RawEpisode, so it is not copied)
"""PiRecording, RecordedCamera and CaptureEvent: a recorded Pi episode as C0 frames and packets (STRUCTURE.md §5.14).

Used only by the two stand-ins for the Pi (RecordingPacketSource in unit tests, FakePiSender in unit tests and
--mock); it never feeds the real arm. Each frame is re-encoded the way the live Pi sends it (PI_PAYLOAD.md §5 rule 4,
H.264: libx264, Constrained Baseline, no B-frames, Annex-B, SPS/PPS before every keyframe, a keyframe every
``keyframe_interval`` frames, one access unit per frame, at the recording's bit rate; JPEG: one image per frame).
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Mapping

import av
import cv2
import numpy as np

from contracts import EncodedFrame, FrameInfo, Packet

from process_frames.packet_pairer import PAIR_AHEAD_NS, PacketPairer

#: Role by sensor model (the Pi's own cam0/cam1 names are not trusted).
ROLE_BY_MODEL = {"imx519": "main", "imx219": "ultrawide"}
CAMERAS = ("main", "ultrawide")
ENCODINGS = ("h264", "jpeg")
#: The frames CSV: these two columns first, then any of the optional ones (the real Pi writes in_episode).
CSV_REQUIRED = ["frame_index", "t_ns"]
CSV_OPTIONAL = ("exposure_ns", "in_episode")


class PiRecordingError(ValueError):
    """A recorded Pi episode that can't be used; names the file and the problem."""


@dataclass(frozen=True)
class RecordedCamera:
    camera: str
    sensor: str
    width: int
    height: int
    fps: float
    bitrate: int
    frames: tuple[EncodedFrame, ...]   # encoded, frame_seq from 0, t_pi_ns and exposure_ns as recorded (0 if none)

    @property
    def period_ns(self) -> int:
        return int(round(1e9 / self.fps))


@dataclass(frozen=True)
class CaptureEvent:
    """One step of the Pi's work, in the order the Pi does it (PiRecording.events)."""

    camera: str
    frame_seq: int            # counts on over loops of the clip
    capture_ns: int           # the recorded t_pi_ns, plus one clip duration per loop
    due_ns: int               # ultrawide: its capture (it is encoded); main: capture + 5 ms (its packet is sent)
    frame: EncodedFrame       # the recorded frame


def _read_frames_csv(path: Path, expected_rows: int) -> tuple[list[int], list[int]]:
    """(t_ns, exposure_ns) per row; exposure 0 when there is no exposure_ns column."""
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
    except OSError as e:
        raise PiRecordingError(f"{path}: cannot read ({e})") from None
    head = rows[0] if rows else []
    extra = head[2:]
    if head[:2] != CSV_REQUIRED or any(c not in CSV_OPTIONAL for c in extra) or len(set(extra)) != len(extra):
        raise PiRecordingError(f"{path}: header must be frame_index,t_ns[,exposure_ns][,in_episode], got "
                               f"{','.join(head) if rows else 'nothing'}")
    exp_col = head.index("exposure_ns") if "exposure_ns" in head else None
    body = [r for r in rows[1:] if r]
    if len(body) != expected_rows:
        raise PiRecordingError(f"{path}: {len(body)} rows, meta.json says {expected_rows} frames")
    t_ns, exp = [], []
    for i, r in enumerate(body):
        try:
            if len(r) != len(head) or int(r[0]) != i:
                raise ValueError("bad row")
            t_ns.append(int(r[1]))
            exp.append(int(r[exp_col]) if exp_col is not None else 0)
        except ValueError:
            raise PiRecordingError(f"{path}: row {i + 1} is not '{','.join(head)}': {r}") from None
        if exp[-1] < 0:
            raise PiRecordingError(f"{path}: row {i + 1}: exposure_ns must be >= 0")
    if any(b <= a for a, b in zip(t_ns, t_ns[1:])):
        raise PiRecordingError(f"{path}: t_ns is not strictly increasing")
    return t_ns, exp


class H264Encoder:
    """The live Pi's H.264 settings through PyAV's libx264: one Annex-B access unit per frame."""

    def __init__(self, width: int, height: int, fps: float, keyframe_interval: int, bitrate: int) -> None:
        ctx = av.CodecContext.create("libx264", "w")
        ctx.width, ctx.height, ctx.pix_fmt = int(width), int(height), "yuv420p"
        rate = Fraction(fps).limit_denominator(1001)
        ctx.framerate = rate
        ctx.time_base = 1 / rate
        ctx.bit_rate = int(bitrate)
        k = int(keyframe_interval)
        ctx.options = {"preset": "ultrafast", "tune": "zerolatency", "profile": "baseline",   # Constrained Baseline
                       "x264-params": f"keyint={k}:min-keyint={k}:scenecut=0:bframes=0:repeat-headers=1:annexb=1"}
        self._ctx = ctx
        self._n = 0

    def encode(self, rgb: np.ndarray) -> tuple[bytes, bool]:
        """(access unit, keyframe) for one RGB image."""
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24").reformat(format="yuv420p")
        vf.pts = self._n
        packets = self._ctx.encode(vf)
        if len(packets) != 1:
            raise PiRecordingError(f"libx264 gave {len(packets)} packets for frame {self._n}, expected 1")
        self._n += 1
        return bytes(packets[0]), bool(packets[0].is_keyframe)

    def close(self) -> None:
        if self._ctx.encode(None):
            raise PiRecordingError("libx264 held frames back (expected none with tune=zerolatency)")


class JpegEncoder:
    def __init__(self, quality: int) -> None:
        self.quality = int(quality)

    def encode(self, rgb: np.ndarray) -> tuple[bytes, bool]:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise PiRecordingError("cv2.imencode failed")
        return buf.tobytes(), True

    def close(self) -> None:
        pass


def encode_h264(images: list[np.ndarray], fps: float, keyframe_interval: int, bitrate: int) -> list[tuple[bytes, bool]]:
    """(Annex-B access unit, keyframe) per image, as the live Pi encoder sends them."""
    h, w = images[0].shape[:2]
    enc = H264Encoder(w, h, fps, keyframe_interval, bitrate)
    out = [enc.encode(rgb) for rgb in images]
    enc.close()
    return out


def encode_jpeg(images: list[np.ndarray], quality: int) -> list[tuple[bytes, bool]]:
    enc = JpegEncoder(quality)
    return [enc.encode(rgb) for rgb in images]


def _reencode_video(path: Path, keep: int, size: tuple[int, int], encoder) -> list[tuple[bytes, bool]]:
    """Decode the first ``keep`` frames as dataproc's read_video_frames does, and encode each one as it comes (so
    no more than one decoded image is held). The video must have at least ``keep`` frames."""
    out: list[tuple[bytes, bool]] = []
    try:
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in c.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.shape[:2] != (size[1], size[0]):
                    raise PiRecordingError(f"{path}: frames are {rgb.shape[1]}x{rgb.shape[0]}, meta.json says "
                                           f"{size[0]}x{size[1]}")
                out.append(encoder.encode(rgb))
                if len(out) == keep:
                    break
    except (av.error.FFmpegError, IndexError, OSError) as e:
        raise PiRecordingError(f"{path}: cannot decode ({e})") from None
    encoder.close()
    if len(out) < keep:
        raise PiRecordingError(f"{path}: {len(out)} frames, shorter than its CSV ({keep} rows used)")
    return out


@dataclass(frozen=True)
class PiRecording:
    path: Path
    cameras: Mapping[str, RecordedCamera]    # "main", "ultrawide"
    duration_ns: int                         # from the first frame to one period after the last

    @classmethod
    def load(cls, path: Path, encoding: str, max_seconds: float, keyframe_interval: int,
             jpeg_quality: int) -> "PiRecording":
        path = Path(path)
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}")
        found = cls._read_meta(path / "meta.json")
        times = {role: _read_frames_csv(path / f"{c['stem']}_frames.csv", c["frames"]) for role, c in found.items()}
        limit = min(t[0][0] for t in times.values()) + int(max_seconds * 1e9)
        recorded: dict[str, RecordedCamera] = {}
        for role in CAMERAS:
            c, (t_ns, exp) = found[role], times[role]
            keep = max(1, sum(1 for t in t_ns if t < limit))
            w, h = c["size"]
            if encoding == "h264":
                encoder = H264Encoder(w, h, c["fps"], keyframe_interval, c["bitrate"])
            else:
                encoder = JpegEncoder(jpeg_quality)
            encoded = _reencode_video(path / f"{c['stem']}.mp4", keep, c["size"], encoder)
            frames = tuple(
                EncodedFrame(FrameInfo(sensor=c["model"], frame_seq=i, t_pi_ns=t_ns[i], exposure_ns=exp[i], width=w,
                                       height=h, encoding=encoding, keyframe=key), data)
                for i, (data, key) in enumerate(encoded))
            recorded[role] = RecordedCamera(camera=role, sensor=c["model"], width=w, height=h, fps=c["fps"],
                                            bitrate=c["bitrate"], frames=frames)
        start = min(cam.frames[0].info.t_pi_ns for cam in recorded.values())
        end = max(cam.frames[-1].info.t_pi_ns + cam.period_ns for cam in recorded.values())
        return cls(path=path, cameras=recorded, duration_ns=end - start)

    @staticmethod
    def _read_meta(meta_path: Path) -> dict[str, dict]:
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
                bitrate = int(c["bitrate"])
            except (KeyError, TypeError, ValueError, IndexError):
                raise PiRecordingError(f"{meta_path}: cameras[{i}]: needs file, model, size, fps, frames, "
                                       f"bitrate") from None
            if model not in ROLE_BY_MODEL:
                raise PiRecordingError(f"{meta_path}: cameras[{i}]: unknown sensor model {model!r} "
                                       f"(expected one of {sorted(ROLE_BY_MODEL)})")
            role = ROLE_BY_MODEL[model]
            if role in found:
                raise PiRecordingError(f"{meta_path}: two cameras with model {model!r}")
            found[role] = {"model": model, "stem": stem, "size": size, "fps": fps, "frames": n, "bitrate": bitrate}
        for role in CAMERAS:
            if role not in found:
                model = next(m for m, r in ROLE_BY_MODEL.items() if r == role)
                raise PiRecordingError(f"{meta_path}: no {role} camera (model {model})")
        return found

    @property
    def start_ns(self) -> int:
        """The Pi time of the first frame of either camera."""
        return min(cam.frames[0].info.t_pi_ns for cam in self.cameras.values())

    def _event(self, camera: str, seq: int) -> CaptureEvent:
        frames = self.cameras[camera].frames
        loop, i = divmod(seq, len(frames))
        capture = frames[i].info.t_pi_ns + loop * self.duration_ns
        due = capture + (PAIR_AHEAD_NS if camera == "main" else 0)
        return CaptureEvent(camera, seq, capture, due, frames[i])

    def events(self) -> Iterator[CaptureEvent]:
        """Forever, looping the clip: each ultrawide frame at its capture (encoded, waiting for the next packet), and
        each main frame 5 ms after its capture (its packet is sent then, so every ultrawide frame that can be its pair
        under PI_PAYLOAD.md §5 rule 3 is already encoded). Ties: the ultrawide frame first."""
        nxt = {"ultrawide": 0, "main": 0}
        while True:
            ev = min((self._event(cam, nxt[cam]) for cam in ("ultrawide", "main")), key=lambda e: e.due_ns)
            nxt[ev.camera] += 1
            yield ev

    def packets(self) -> list[Packet]:
        """One pass over the clip as the Pi would send it, with the recorded times: one packet per main frame,
        paired by PacketPairer. Ultrawide frames captured after the last main frame are not sent."""
        pairer: PacketPairer[EncodedFrame] = PacketPairer()
        n_main = len(self.cameras["main"].frames)
        out: list[Packet] = []
        for ev in self.events():
            if ev.camera == "ultrawide":
                if ev.frame_seq < len(self.cameras["ultrawide"].frames):
                    pairer.add_ultrawide(ev.frame_seq, ev.capture_ns, ev.frame)
                continue
            uw, pair = pairer.take(ev.capture_ns)
            out.append(Packet.build(ev.frame_seq, ev.frame, uw, pair))
            if len(out) == n_main:
                return out
        raise AssertionError("unreachable")


def with_seq_and_time(frame: EncodedFrame, frame_seq: int, t_pi_ns: int) -> EncodedFrame:
    """The same encoded frame with another frame_seq and t_pi_ns (the fake Pi's numbering and clock)."""
    return EncodedFrame(dataclasses.replace(frame.info, frame_seq=frame_seq, t_pi_ns=t_pi_ns), frame.data)
