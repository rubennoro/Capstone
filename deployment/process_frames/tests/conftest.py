"""Fixtures and test-only fakes for process_frames (STRUCTURE.md §13.1)."""

from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import numpy as np
import pytest
import yaml

from contracts import Frame

from process_frames import FOLDER
from process_frames.frame_decoder import FrameDecoder
from process_frames.rig_file import load_live_rig

TESTS = Path(__file__).resolve().parent
STANDIN_RIG = FOLDER / "rigs" / "rig_pi_v1_standin.yaml"
SMALL_RIG = TESTS / "data" / "rig_small.yaml"
MAIN_SIZE = (160, 90)          # the small rig, (width, height)
UW_SIZE = (128, 96)
PERIOD_NS = 33_333_333


# ------------------------------------------------------------------------------------------------ RawDecoder

class RawDecoder(FrameDecoder):
    """Test decoder: ``data`` is raw RGB bytes of height x width x 3. Copies H.264's keyframe rule (unusable after
    notify_gap() until a keyframe), so the receiver's drop handling is tested without PyAV."""

    ENCODING = "h264"

    def __init__(self) -> None:
        self._waiting = True

    def decode(self, frame: Frame) -> np.ndarray | None:
        if self._waiting:
            if not frame.keyframe:
                return None
            self._waiting = False
        return np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 3).copy()

    def notify_gap(self) -> None:
        self._waiting = True


def raw_frame(camera: str = "main", seq: int = 0, t_pi_ns: int = 10_000_000_000, *, exposure_ns: int = 0,
              size: tuple[int, int] | None = None, sensor: str | None = None, keyframe: bool = True,
              encoding: str = "h264", value: int | None = None, data: bytes | None = None) -> Frame:
    """A valid C0 Frame with raw RGB data (for RawDecoder), at the small rig's size by default."""
    w, h = size or (MAIN_SIZE if camera == "main" else UW_SIZE)
    if data is None:
        data = np.full((h, w, 3), (seq * 7 if value is None else value) % 256, np.uint8).tobytes()
    return Frame(camera=camera, sensor=sensor or ("imx519" if camera == "main" else "imx219"), seq=seq,
                 t_pi_ns=t_pi_ns, exposure_ns=exposure_ns, width=w, height=h, encoding=encoding, keyframe=keyframe,
                 data=data)


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


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def wait_for(condition: Callable[[], bool], timeout_s: float = 10.0, what: str = "condition") -> None:
    """Poll ``condition`` until true; fail after ``timeout_s`` (no fixed sleeps)."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout_s} s waiting for {what}")
        time.sleep(0.005)


@pytest.fixture
def waiter():
    return wait_for


# ------------------------------------------------------------------------------------------------ rigs

@pytest.fixture(scope="session")
def standin_rig():
    return load_live_rig(STANDIN_RIG)


@pytest.fixture(scope="session")
def small_rig():
    return load_live_rig(SMALL_RIG)


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
                     main_exposure_ns: int = 8_000_000, uw_exposure_ns: int = 4_000_000) -> Path:
    """A tiny Pi-format episode (PI_ADAPTER.md "Pi format"): meta.json, cam0 = imx219 (ultrawide), cam1 = imx519
    (main), cam*_frames.csv with or without exposure_ns, at the small rig's sizes."""
    dst.mkdir(parents=True)
    cams = []
    for stem, model, size, salt, offset, exp in (("cam0", uw_model, UW_SIZE, 40, 11_000_000, uw_exposure_ns),
                                                 ("cam1", main_model, MAIN_SIZE, 0, 0, main_exposure_ns)):
        write_mp4(dst / f"{stem}.mp4", [pattern(i, size, salt) for i in range(n_frames)])
        head = "frame_index,t_ns" + (",exposure_ns" if with_exposure else "")
        rows = [f"{i},{t0_ns + offset + i * PERIOD_NS}" + (f",{exp}" if with_exposure else "")
                for i in range(n_frames)]
        (dst / f"{stem}_frames.csv").write_text("\n".join([head, *rows]) + "\n")
        cams.append({"file": f"{stem}.mp4", "picamera2_index": 0, "model": model, "size": list(size), "fps": 30,
                     "bitrate": 10_000_000, "frames": n_frames, "max_gap_ns": 0, "max_gap_at_s": 0.0})
    (dst / "meta.json").write_text(json.dumps({"episode": 1, "valid": True, "problems": [], "cameras": cams}))
    return dst


@pytest.fixture(scope="session")
def pi_episode(tmp_path_factory) -> Path:
    """30 frames per camera (1 s), with the exposure_ns column."""
    return write_pi_episode(tmp_path_factory.mktemp("pi") / "ep_0001")


@pytest.fixture(scope="session")
def pi_episode_no_exposure(tmp_path_factory) -> Path:
    return write_pi_episode(tmp_path_factory.mktemp("pi") / "ep_0002", n_frames=12, with_exposure=False)


@pytest.fixture
def make_pi_episode(tmp_path) -> Callable[..., Path]:
    counter = iter(range(1000))

    def make(**kw) -> Path:
        return write_pi_episode(tmp_path / f"ep_{next(counter):04d}", **kw)
    return make


# ------------------------------------------------------------------------------------------------ station files

def write_station(path: Path, *, process_frames: dict | None = None, endpoints: dict | None = None,
                  session_root: str | Path | None = None, extra: dict | None = None) -> Path:
    """A station.yaml for tests: every endpoint on 127.0.0.1 with a free port (``*``) unless given."""
    ep = {"pi_frames": "tcp://127.0.0.1:*", "pi_time_sync": "tcp://127.0.0.1:*",
          "mock_pi_frames": "tcp://127.0.0.1:*", "mock_pi_time_sync": "tcp://127.0.0.1:*",
          "frames": "tcp://127.0.0.1:*", "state": "tcp://127.0.0.1:5620", "commands": "tcp://127.0.0.1:5630",
          "policy": "tcp://127.0.0.1:5640"}
    ep.update(endpoints or {})
    d = {"station_version": 1, "session_root": str(session_root or path.parent / "sessions"), "endpoints": ep}
    if process_frames is not None:
        d["process_frames"] = process_frames
    d.update(extra or {})
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


def fast_overrides(**more) -> dict:
    """Short limits so socket tests run fast (STRUCTURE.md §13)."""
    o = {"rig_file": str(SMALL_RIG),
         "health": {"stale_s": 0.2, "down_s": 0.5, "fps_window_s": 1.0},
         "clock": {"interval_s": 0.1, "timeout_s": 0.2, "startup_exchanges": 3, "max_sync_age_s": 1.0},
         "server": {"poll_s": 0.01, "join_timeout_s": 2.0},
         "receive": {"timeout_s": 0.02},
         "log": {"stats_every_s": 0.5},
         "mock": {"keyframe_interval": 10, "max_seconds": 10.0}}
    for k, v in more.items():
        if isinstance(v, dict) and isinstance(o.get(k), dict):
            o[k] = {**o[k], **v}
        else:
            o[k] = v
    return o
