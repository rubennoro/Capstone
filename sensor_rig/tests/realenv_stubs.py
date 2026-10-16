"""Light stand-ins for real-env's third-party imports, for the RigCamera tests ONLY (never shipped into real-env).

``install()`` puts ``behavior_prompting/deps/real-env`` (namespace package ``real_env``) and ``behavior_prompting``
on ``sys.path`` and, **only for modules whose real import fails**, registers minimal fakes in ``sys.modules``:

* ``robotmq``: ``RMQServer`` / ``RMQClient`` recording calls; ``serialize`` / ``deserialize`` = pickle.
* ``loguru.logger``.
* ``robot_utils.logging_utils.echo_exception``, ``robot_utils.image_utils.resize_frame_without_distortion``.
* ``robologger.loggers.video_logger.VideoLogger`` (holds ``codec``; never records),
  ``robologger.utils.huecodec.depth2logrgb``.

On the Jetson (real packages installed) nothing is stubbed and the same tests run against the real modules.
``RecordingRMQServer`` is used by the tests regardless (it replaces ``camera.rmq_server`` after construction).
"""

from __future__ import annotations

import importlib
import pickle
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_ENV_ROOT = REPO_ROOT / "behavior_prompting" / "deps" / "real-env"
REAL_ENV_CONFIGS = REAL_ENV_ROOT / "real_env" / "configs" / "peripherals"
BP_ROOT = REPO_ROOT / "behavior_prompting"

STUBBED: list[str] = []


class StopRun(Exception):
    """Raised by RecordingRMQServer to end BaseCamera.run()'s infinite loop."""


class RecordingRMQServer:
    """RMQServer stand-in: records every ``put_data``; raises ``StopRun`` after ``stop_after_timestamps`` puts on the
    ``timestamp`` topic."""

    def __init__(self, server_name: str = "", server_endpoint: str = "", stop_after_timestamps: int | None = None):
        self.server_name = server_name
        self.server_endpoint = server_endpoint
        self.stop_after_timestamps = stop_after_timestamps
        self.topics: dict[str, dict] = {}
        self.puts: list[tuple[str, bytes]] = []

    def add_shared_memory_topic(self, topic, message_remaining_time_s=1.0, shared_memory_size_gb=0.5):
        self.topics[topic] = {"shm": True, "remaining_s": message_remaining_time_s}

    def add_topic(self, topic, message_remaining_time_s=1.0):
        self.topics[topic] = {"shm": False, "remaining_s": message_remaining_time_s}

    def put_data(self, topic, data):
        self.puts.append((topic, data))
        if topic == "timestamp" and self.stop_after_timestamps is not None:
            if sum(1 for t, _ in self.puts if t == "timestamp") >= self.stop_after_timestamps:
                raise StopRun()

    def pop_data(self, topic, n=0):
        return [], []

    def peek_data(self, topic, n=-1):
        return [], []

    def data(self, topic) -> list[bytes]:
        return [d for t, d in self.puts if t == topic]


def _serialize(obj) -> bytes:
    return pickle.dumps(obj)


def _deserialize(data: bytes):
    return pickle.loads(data)


class _RMQClient:
    def __init__(self, client_name="", server_endpoint=""):
        self.client_name = client_name
        self.server_endpoint = server_endpoint

    def peek_data(self, topic, n=-1):
        return [], []

    def pop_data(self, topic, n=0):
        return [], []

    def put_data(self, topic, data):
        pass


class _Logger:
    def __getattr__(self, name):
        def log(*args, **kwargs):
            pass
        return log


class _VideoLogger:
    def __init__(self, name, endpoint, attr, codec="h264_nvenc", depth_range=(0.0, 4.0)):
        self.name = name
        self.endpoint = endpoint
        self.attr = attr
        self.codec = codec
        for cam, cfg in attr["camera_configs"].items():
            for key in ("width", "height", "fps", "type"):
                if key not in cfg:
                    raise ValueError(f"Camera config for '{cam}' missing '{key}'")

    def update_recording_state(self) -> bool:
        return False

    def log_frames(self, frames):
        raise AssertionError("stub VideoLogger never records")


def _echo_exception() -> str:
    import traceback

    return traceback.format_exc()


def _resize_frame_without_distortion(source_frame, source_wh, display_wh):
    import cv2

    return cv2.resize(source_frame, tuple(display_wh))


def _depth2logrgb(depth, zrange=(0.0, 4.0)):
    import numpy as np

    d = np.clip((np.asarray(depth, np.float32) - zrange[0]) / (zrange[1] - zrange[0]), 0, 1)
    g = (d * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    m.__stub__ = True
    return m


def _register(name: str, **attrs) -> types.ModuleType:
    m = _module(name, **attrs)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent:
        if parent not in sys.modules:
            _register(parent, __path__=[])
        setattr(sys.modules[parent], child, m)
    STUBBED.append(name)
    return m


def _importable(*names: str) -> bool:
    try:
        for n in names:
            importlib.import_module(n)
        return True
    except Exception:
        return False


def install() -> list[str]:
    """Make ``real_env`` importable; stub only what is missing. Returns the stubbed module names."""
    for p in (REAL_ENV_ROOT, BP_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if not _importable("robotmq"):
        _register("robotmq", RMQServer=RecordingRMQServer, RMQClient=_RMQClient, serialize=_serialize,
                  deserialize=_deserialize, __path__=[])
    if not _importable("loguru"):
        _register("loguru", logger=_Logger(), __path__=[])
    if not _importable("robot_utils.logging_utils", "robot_utils.image_utils"):
        for k in [k for k in sys.modules if k == "robot_utils" or k.startswith("robot_utils.")]:
            del sys.modules[k]
        _register("robot_utils", __path__=[])
        _register("robot_utils.logging_utils", echo_exception=_echo_exception)
        _register("robot_utils.image_utils", resize_frame_without_distortion=_resize_frame_without_distortion)
    if not _importable("robologger.loggers.video_logger", "robologger.utils.huecodec"):
        for k in [k for k in sys.modules if k == "robologger" or k.startswith("robologger.")]:
            del sys.modules[k]
        _register("robologger", __path__=[])
        _register("robologger.loggers", __path__=[])
        _register("robologger.loggers.video_logger", VideoLogger=_VideoLogger)
        _register("robologger.utils", __path__=[])
        _register("robologger.utils.huecodec", depth2logrgb=_depth2logrgb)
    return list(STUBBED)
