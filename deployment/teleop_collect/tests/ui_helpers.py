"""Test helpers for the ui (STRUCTURE.md 12 row E5): fake U1 (teleop) and U2 (frames) servers on free ports, and the
ui server running on a thread. Nothing here binds a station's fixed ports."""
from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

from deploy_contracts import Ack, Message, RpcServer
from station_helpers import free_port

from teleop_collect.contracts import (CreateTask, DeleteEpisode, GetPreview, GetStatus, GripperCommand, HomeRequest,
                                      KeysUpdate, Preview, SelectTask, StartEpisode, StopArm, StopEpisode,
                                      TeleopStatus, UiMessage)
from teleop_collect.contracts.examples import health, teleop_status
from teleop_collect.ui.server import UiApp
from teleop_collect.ui.settings import Settings

U1_REQUESTS = (GetStatus, KeysUpdate, GripperCommand, SelectTask, CreateTask, HomeRequest, StartEpisode, StopEpisode,
               DeleteEpisode, StopArm)


def wait_until(pred: Callable[[], bool], timeout_s: float = 3.0, what: str = "the condition") -> None:
    deadline = time.monotonic() + timeout_s
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def idle_status(**change) -> TeleopStatus:
    """teleop at home, idle, ready to start (no messages)."""
    d = dict(phase="idle", recording_episode="", episode_s=0.0, arm_mode="idle", at_home=True, start_allowed=True,
             start_refusal="", moving=False, messages=())
    d.update(change)
    return teleop_status(**d)


class _Served:
    """An RpcServer served on its own thread until stop()."""

    def __init__(self, handlers: dict) -> None:
        self.endpoint = f"tcp://127.0.0.1:{free_port()}"
        self._handlers = handlers
        self._server: RpcServer | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        started = threading.Event()

        def serve() -> None:
            self._server = RpcServer(self.endpoint, self._handlers)
            started.set()
            try:
                while not self._stop.is_set():
                    self._server.serve_once(0.01)
            finally:
                self._server.close()

        self._stop.clear()
        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
        assert started.wait(2.0)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None


class FakeTeleop(_Served):
    """U1 REP: records every request; status, acks and a delay are set by the test."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[Message] = []
        self.status: TeleopStatus = idle_status()
        self.messages: list[UiMessage] = []
        self.delay_s = 0.0
        self.refuse: dict[type, str] = {}                  # request class -> refusal reason
        super().__init__({cls: self._handle for cls in U1_REQUESTS})

    def _handle(self, request: Message) -> Message:
        if self.delay_s:
            time.sleep(self.delay_s)
        with self.lock:
            self.requests.append(request)
            if isinstance(request, GetStatus):
                msgs = tuple(m for m in self.messages if m.id > request.after_message)[:50]
                d = {k: getattr(self.status, k) for k in self.status.to_dict() if k != "schema"}
                d["messages"] = msgs
                return TeleopStatus(**d)
            reason = self.refuse.get(type(request))
            return Ack(ok=reason is None, reason=reason or "")

    def add_message(self, text: str, level: str = "info") -> None:
        with self.lock:
            self.messages.append(UiMessage(id=len(self.messages) + 1, level=level, text=text))

    def of(self, cls: type) -> list:
        with self.lock:
            return [r for r in self.requests if isinstance(r, cls)]

    def last_keys(self) -> KeysUpdate | None:
        keys = self.of(KeysUpdate)
        return keys[-1] if keys else None

    def clear(self) -> None:
        with self.lock:
            self.requests.clear()


JPEG_MAIN = b"\xff\xd8main" + bytes(range(40)) + b"\xff\xd9"
JPEG_UW = b"\xff\xd8uw" + bytes(range(60)) + b"\xff\xd9"


class FakeFrames(_Served):
    """U2 REP: a new frame of each camera every `every` requests; frame_no counts up from 100."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[GetPreview] = []
        self.frame_no = {"main": 100, "ultrawide": 100}
        super().__init__({GetPreview: self._handle})

    def jpeg(self, camera: str, frame_no: int) -> bytes:
        return (JPEG_MAIN if camera == "main" else JPEG_UW) + frame_no.to_bytes(4, "big")

    def _handle(self, request: GetPreview) -> Preview:
        with self.lock:
            self.requests.append(request)
            for cam in self.frame_no:
                self.frame_no[cam] += 1
            main, uw = self.frame_no["main"], self.frame_no["ultrawide"]
        return Preview(main_jpeg=self.jpeg("main", main) if main > request.main_after else None,
                       ultrawide_jpeg=self.jpeg("ultrawide", uw) if uw > request.ultrawide_after else None,
                       main_frame_no=main, ultrawide_frame_no=uw, health=health())


def ui_settings(u1_endpoint: str, preview_endpoint: str, tmp_path: Path, **change) -> Settings:
    d = dict(status_hz=20.0, preview_hz=20.0, keys_resend_ms=50.0, keys_stale_ms=150.0, rpc_timeout_s=0.3,
             open_browser=False, http_port=0, u1_endpoint=u1_endpoint, preview_endpoint=preview_endpoint,
             session_root=Path(tmp_path) / "sessions")
    d.update(change)
    return Settings(**d)


class UiThread:
    """UiApp.run on a thread; .url once bound; .error holds what run() raised."""

    def __init__(self, settings: Settings, ready_file: Path | None = None) -> None:
        self.app = UiApp(settings, ready_file=ready_file)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self.app.run()
        except BaseException as e:                      # noqa: BLE001 (the test reads it)
            self.error = e

    def start(self) -> UiThread:
        self.thread.start()
        wait_until(lambda: self.app.bound.is_set() or not self.thread.is_alive(), 5.0, "the ui to bind")
        if self.error is not None:
            raise self.error
        return self

    @property
    def url(self) -> str:
        return self.app.url()

    def stop(self, timeout_s: float = 5.0) -> None:
        self.app.request_stop()
        self.thread.join(timeout_s)
        assert not self.thread.is_alive(), "the ui did not stop"


@contextlib.contextmanager
def running_ui(settings: Settings, ready_file: Path | None = None) -> Iterator[UiThread]:
    ui = UiThread(settings, ready_file).start()
    try:
        yield ui
    finally:
        ui.stop()
