# adapted from rnoro/deployment:deployment/process_frames/tests/conftest.py @ 91b178b (the helpers; the fixtures are
# in tests/frames/conftest.py)
# changes: importable module (test folders have no __init__.py); RawDecoder reads "jpeg" frames (the H.264 keyframe
# rule is gone with H.264); raw_packets pairs by the nearest frame, each main packet sent half an ultrawide period after
# capture; V2_RIG is rigs/rig_pi_v2_640.yaml (both cameras 640x480: the only rig carried over); station files use the
# laptop station schema (station_helpers.write_station with a frames: section); fast_overrides without the dropped
# keys.
"""Helpers and test-only fakes for the frames tests (STRUCTURE.md 12)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import av
import numpy as np
import yaml
from station_helpers import write_station as _write_station

from policy_deploy.contracts import EncodedFrame, FrameInfo, Packet, paths
from policy_deploy.frames.frame_decoder import FrameDecoder
from policy_deploy.frames.packet_pairer import PacketPairer

TESTS = Path(__file__).resolve().parent / "frames"
FOLDER = paths.folder()
V2_RIG = FOLDER / "rigs" / "rig_pi_v2_640.yaml"
SMALL_RIG = TESTS / "data" / "rig_small.yaml"
MAIN_SIZE = (160, 90)          # the small rig, (width, height)
UW_SIZE = (128, 96)
V2_MAIN_SIZE = (640, 480)      # the real rig file
V2_UW_SIZE = (640, 480)
PERIOD_NS = 33_333_333
HALF_PERIOD_NS = PERIOD_NS // 2


# ------------------------------------------------------------------------------------------------ RawDecoder

class RawDecoder(FrameDecoder):
    """Test decoder: ``data`` is raw RGB bytes of height x width x 3 (labelled jpeg, as every frame now is). Records
    the frame_seq of every frame it is given, so "every frame decoded exactly once, in order" can be checked."""

    ENCODING = "jpeg"

    def __init__(self) -> None:
        self.decoded: list[int] = []           # frame_seq of every frame given to decode(), in order
        self.gaps = 0

    def decode(self, frame: EncodedFrame) -> np.ndarray | None:
        self.decoded.append(frame.info.frame_seq)
        return np.frombuffer(frame.data, np.uint8).reshape(frame.info.height, frame.info.width, 3).copy()

    def notify_gap(self) -> None:
        self.gaps += 1


def raw_frame(camera: str = "main", seq: int = 0, t_pi_ns: int = 10_000_000_000, *, exposure_ns: int = 0,
              size: tuple[int, int] | None = None, sensor: str | None = None, encoding: str = "jpeg",
              value: int | None = None, data: bytes | None = None) -> EncodedFrame:
    """A valid C0 frame (map + bytes) with raw RGB data (for RawDecoder), at the small rig's size by default.
    The image's value is ``seq * 7 % 256`` for main and ``(seq * 7 + 100) % 256`` for ultrawide, unless given."""
    w, h = size or (MAIN_SIZE if camera == "main" else UW_SIZE)
    if data is None:
        v = (seq * 7 + (0 if camera == "main" else 100)) if value is None else value
        data = np.full((h, w, 3), v % 256, np.uint8).tobytes()
    info = FrameInfo(sensor=sensor or ("imx519" if camera == "main" else "imx219"), frame_seq=seq, t_pi_ns=t_pi_ns,
                     exposure_ns=exposure_ns, width=w, height=h, encoding=encoding, keyframe=True)
    return EncodedFrame(info, data)


def raw_packet(main: EncodedFrame, ultrawide=(), pair_seq: int = -1, packet_seq: int | None = None) -> Packet:
    """One packet by hand (packet_seq = the main frame_seq unless given)."""
    seq = main.info.frame_seq if packet_seq is None else packet_seq
    return Packet.build(seq, main, list(ultrawide), pair_seq)


def raw_packets(main_t: list[int], uw_t: list[int], *, skip_main: tuple[int, ...] = (),
                skip_uw: tuple[int, ...] = (), **kw) -> list[Packet]:
    """Packets as the fake Pi sends them, from capture times (Pi ns): main frame i at ``main_t[i]``, ultrawide frame j
    at ``uw_t[j]``; ultrawide frames encoded at capture, each main packet sent half an ultrawide period after its
    capture, paired by PacketPairer (nearest frame). ``skip_*``: frame_seq values the Pi drops before encoding (gaps).
    ``kw`` goes to raw_frame."""
    events = sorted([(t, 0, j) for j, t in enumerate(uw_t)] +
                    [(t + HALF_PERIOD_NS, 1, i) for i, t in enumerate(main_t)])
    pairer: PacketPairer[EncodedFrame] = PacketPairer()
    out = []
    for _, is_main, seq in events:
        if not is_main:
            if seq not in skip_uw:
                pairer.add_ultrawide(seq, uw_t[seq], raw_frame("ultrawide", seq, uw_t[seq], **kw))
            continue
        if seq in skip_main:
            continue
        uw, pair = pairer.take(main_t[seq])
        out.append(raw_packet(raw_frame("main", seq, main_t[seq], **kw), uw, pair))
    return out


# ------------------------------------------------------------------------------------------------ clocks, waits

class FakeClock:
    """A settable ns clock."""

    def __init__(self, start_ns: int = 1_000_000_000_000) -> None:
        self.now = int(start_ns)

    def __call__(self) -> int:
        return self.now

    def advance(self, ns: int) -> None:
        self.now += int(ns)

    def advance_s(self, s: float) -> None:
        self.now += int(round(s * 1e9))


def wait_for(condition: Callable[[], bool], timeout_s: float = 10.0, what: str = "condition") -> None:
    """Poll ``condition`` until true; fail after ``timeout_s`` (no fixed sleeps)."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.005)


# ------------------------------------------------------------------------------------------------ rigs

def write_rig(dst_dir: Path, rig_id: str, change: Callable[[dict], None], src: Path = SMALL_RIG) -> Path:
    """A copy of ``src`` named ``<rig_id>.yaml`` (rig_id set), changed by ``change(d)``."""
    d = yaml.safe_load(src.read_text())
    d["rig_id"] = rig_id
    change(d)
    path = Path(dst_dir) / f"{rig_id}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


# ------------------------------------------------------------------------------------------------ Pi episodes

def pattern(i: int, size: tuple[int, int], salt: int) -> np.ndarray:
    """A moving pattern: every frame differs."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 3 + i * 5 + salt) % 256
    g = (yy * 4 + i * 3) % 256
    b = ((xx + yy) * 2 + i * 7 + salt) % 256
    img = np.stack([r, g, b], axis=2).astype(np.uint8)
    img[(i * 2) % h, :, :] = 255                                   # a moving bright line
    return img


def write_mp4(path: Path, images: list[np.ndarray], fps: int = 30) -> None:
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = images[0].shape[1], images[0].shape[0], "yuv420p"
        s.options = {"preset": "ultrafast", "crf": "18"}
        for img in images:
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode(None):
            c.mux(pkt)


def write_pi_episode(dst: Path, n_frames: int = 30, with_exposure: bool = True, *,
                     main_model: str = "imx519", uw_model: str = "imx219", t0_ns: int = 5_000_000_000,
                     main_exposure_ns: int = 8_000_000, uw_exposure_ns: int = 4_000_000,
                     main_size: tuple[int, int] = MAIN_SIZE, uw_size: tuple[int, int] = UW_SIZE,
                     uw_offset_ns: int = 11_000_000, in_episode: bool = False) -> Path:
    """A tiny Pi-format episode: meta.json, cam0 = imx219 (ultrawide), cam1 = imx519 (main), cam*_frames.csv with or
    without exposure_ns (and with the real Pi's in_episode column if asked), at the small rig's sizes by default.
    The ultrawide is captured ``uw_offset_ns`` after the main camera."""
    dst.mkdir(parents=True)
    cams = []
    for stem, model, size, salt, offset, exp, rate in (
            ("cam0", uw_model, uw_size, 40, uw_offset_ns, uw_exposure_ns, 10_000_000),
            ("cam1", main_model, main_size, 0, 0, main_exposure_ns, 15_000_000)):
        write_mp4(dst / f"{stem}.mp4", [pattern(i, size, salt) for i in range(n_frames)])
        head = "frame_index,t_ns" + (",exposure_ns" if with_exposure else "") + (",in_episode" if in_episode else "")
        rows = [f"{i},{t0_ns + offset + i * PERIOD_NS}" + (f",{exp}" if with_exposure else "")
                + (f",{int(i >= 2)}" if in_episode else "") for i in range(n_frames)]
        (dst / f"{stem}_frames.csv").write_text("\n".join([head, *rows]) + "\n")
        cams.append({"file": f"{stem}.mp4", "picamera2_index": 0, "model": model, "size": list(size), "fps": 30,
                     "bitrate": rate, "frames": n_frames, "max_gap_ns": 0, "max_gap_at_s": 0.0})
    (dst / "meta.json").write_text(json.dumps({"episode": 1, "valid": True, "problems": [], "cameras": cams}))
    return dst


# ------------------------------------------------------------------------------------------------ station files

def write_station(path: Path, *, frames: dict | None = None, **sections) -> Path:
    """A valid laptop station at ``path`` (free ports on 127.0.0.1, session_root next to it) with ``frames`` as its
    frames: section; other keyword arguments replace top-level keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames is not None:
        sections["frames"] = frames
    return _write_station(path.parent, name=path.name, **sections)


def fast_overrides(**more) -> dict:
    """Short limits so socket tests run fast."""
    o = {"rig_file": str(SMALL_RIG),
         "health": {"stale_s": 0.2, "down_s": 0.5, "fps_window_s": 1.0},
         "clock": {"max_sync_age_s": 1.0},
         "server": {"poll_s": 0.01, "join_timeout_s": 2.0},
         "receive": {"timeout_s": 0.02},
         "log": {"stats_every_s": 0.5},
         "mock": {"max_seconds": 10.0}}
    for k, v in more.items():
        if isinstance(v, dict) and isinstance(o.get(k), dict):
            o[k] = {**o[k], **v}
        else:
            o[k] = v
    return o
