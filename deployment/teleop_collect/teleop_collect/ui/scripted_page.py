"""ScriptedPage: a test client that speaks the page's WebSocket protocol exactly as static/app.js does
(STRUCTURE.md 9.2, 9.4). Used by the ui's tests and by E7's end-to-end tests.

Synchronous API for plain pytest: the aiohttp client runs on its own thread with its own event loop. Key names are
teleop's names (MOTION_KEYS: "x+", "tilt_left", ...), not keyboard letters; the letters are the page's alone.

Like the page, `hold` sends the key state at once, again every keys_resend_ms (from the server's hello) while held,
and a release (held=[], focus=true) at the end. Unlike the page it does not hide the keys outside the "recording"
phase, so a test can check that teleop ignores them.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Callable, Sequence

import aiohttp

from teleop_collect.contracts import GRIPPER_COMMANDS, MOTION_KEYS

DEFAULT_KEYS_RESEND_MS = 50.0


class ScriptedPage:
    """One page: connect(), drive it, close()."""

    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._lock = threading.Condition()
        self._status: dict | None = None
        self._statuses = 0
        self._hello: dict | None = None
        self._acks: dict[int, dict] = {}
        self._errors: list[str] = []
        self._previews: dict[str, int] = {}
        self._jpegs: dict[str, bytes] = {}
        self._pending_preview: dict | None = None
        self._next_id = 1
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._closed = threading.Event()
        self._held: tuple[str, ...] = ()
        self._slow = False
        self._focus = False

    # ------------------------------------------------------------------------------------------ connect / close

    def connect(self) -> None:
        """Open the WebSocket and wait for the server's hello."""
        if self._thread is not None:
            raise RuntimeError("already connected")
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="scripted-page", daemon=True)
        self._thread.start()
        try:
            self._call(self._open(), self.timeout_s)
        except BaseException:
            self.close()
            raise
        self._wait(lambda: self._hello is not None, self.timeout_s, "the server's hello")

    def close(self) -> None:
        """Close the socket (the server's deadman then releases the keys) and stop the thread."""
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            if loop.is_running():
                self._call(self._shutdown(), self.timeout_s)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        thread.join(self.timeout_s)
        loop.close()
        self._loop = None
        self._thread = None

    def __enter__(self) -> ScriptedPage:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        """False once the server closed the socket (or it was never opened)."""
        return self._ws is not None and not self._closed.is_set()

    @property
    def hello(self) -> dict:
        """The server's hello: keys_resend_ms, keys_stale_ms, status_hz, preview_hz, motion_keys."""
        with self._lock:
            if self._hello is None:
                raise RuntimeError("not connected")
            return dict(self._hello)

    # -------------------------------------------------------------------------------------------------- reading

    def status(self) -> dict:
        """The newest status message (TeleopStatus fields plus the ui's link fields)."""
        with self._lock:
            if self._status is None:
                raise RuntimeError("no status received yet")
            return dict(self._status)

    def wait_for(self, pred: Callable[[dict], bool], timeout_s: float, what: str) -> dict:
        """Wait until pred(status) is true and return that status. A KeyError / TypeError in pred counts as false
        (a status before teleop ever answered has only the link fields). TimeoutError names `what` and the last
        status."""
        def ok() -> bool:
            if self._status is None:
                return False
            try:
                return bool(pred(self._status))
            except (KeyError, TypeError, IndexError):
                return False
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while not ok():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last = {k: self._status.get(k) for k in ("phase", "start_refusal", "arm_mode", "task",
                                                             "recording_episode", "teleop_answering")} \
                        if self._status else None
                    raise TimeoutError(f"ScriptedPage: no {what} within {timeout_s:.1f} s (last status: {last})")
                self._lock.wait(min(remaining, 0.05))
            return dict(self._status)                                     # type: ignore[arg-type]

    def previews(self) -> dict[str, int]:
        """The newest frame_no received per camera ("main", "ultrawide"); a camera is absent until its first."""
        with self._lock:
            return dict(self._previews)

    def preview_jpeg(self, camera: str) -> bytes | None:
        """The newest JPEG received for that camera, exactly as the server sent it."""
        with self._lock:
            return self._jpegs.get(camera)

    def errors(self) -> list[str]:
        """The server's {"type": "error"} texts so far (a message this client sent was not understood)."""
        with self._lock:
            return list(self._errors)

    # -------------------------------------------------------------------------------------------------- sending

    def command(self, name: str, /, **args: Any) -> dict:
        """Send a button command and return its ack {"type", "id", "ok", "reason"} (TimeoutError if none).
        `name` is positional-only, so command("select_task", name="pick_block") works."""
        with self._lock:
            cid = self._next_id
            self._next_id += 1
        self._send({"type": "command", "name": name, "args": args, "id": cid})
        self._wait(lambda: cid in self._acks, self.timeout_s, f"ack to {name} (id {cid})")
        with self._lock:
            return self._acks.pop(cid)

    def keys(self, held: Sequence[str], slow: bool = False, focus: bool = True) -> None:
        """One key state message, as the page sends it on every change."""
        held = tuple(held)
        unknown = [k for k in held if k not in MOTION_KEYS]
        if unknown:
            raise ValueError(f"unknown key name(s) {unknown}; known: {', '.join(MOTION_KEYS)}")
        self._held, self._slow, self._focus = held, bool(slow), bool(focus)
        self._send({"type": "keys", "held": list(held), "slow": bool(slow), "focus": bool(focus)})

    def hold(self, keys: Sequence[str], seconds: float, slow: bool = False) -> None:
        """Hold these keys for `seconds` like the page: the state at once, again every keys_resend_ms, then the
        release (held=[], focus=true)."""
        period = float(self.hello.get("keys_resend_ms", DEFAULT_KEYS_RESEND_MS)) / 1000.0
        end = time.monotonic() + seconds
        self.keys(keys, slow=slow, focus=True)
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(period, remaining))
            if time.monotonic() < end:
                self.keys(keys, slow=slow, focus=True)
        self.keys((), slow=False, focus=True)

    def gripper(self, command: str) -> None:
        """N (close) or M (open): one message, as on key down."""
        if command not in GRIPPER_COMMANDS:
            raise ValueError(f"unknown gripper command {command!r}; known: {', '.join(GRIPPER_COMMANDS)}")
        self._send({"type": "gripper", "command": command})

    def blur(self) -> None:
        """The page lost focus: an empty held set with focus=false (then silence, as the page)."""
        self.keys((), slow=False, focus=False)

    def send_raw(self, message: Any) -> None:
        """Send any JSON value (tests of the server's checks)."""
        self._send(message)

    # ------------------------------------------------------------------------------------------------ internals

    def _call(self, coro, timeout_s: float):
        if self._loop is None:
            raise RuntimeError("not connected")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout_s)

    def _wait(self, cond: Callable[[], bool], timeout_s: float, what: str) -> None:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while not cond():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"ScriptedPage: no {what} within {timeout_s:.1f} s")
                if self._closed.is_set() and self._ws is not None:
                    raise ConnectionError(f"ScriptedPage: the server closed the page while waiting for {what}")
                self._lock.wait(min(remaining, 0.05))

    def _send(self, message: Any) -> None:
        if not self.connected:
            raise ConnectionError("ScriptedPage: not connected")
        self._call(self._ws.send_str(json.dumps(message)), self.timeout_s)   # type: ignore[union-attr]

    async def _open(self) -> None:
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(f"{self.base_url}/ws", timeout=self.timeout_s,
                                                  max_msg_size=8 << 20)
        asyncio.get_running_loop().create_task(self._reader())

    async def _shutdown(self) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()

    async def _reader(self) -> None:
        ws = self._ws
        assert ws is not None
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._on_text(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self._on_binary(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            with self._lock:
                self._closed.set()
                self._lock.notify_all()

    def _on_text(self, text: str) -> None:
        msg = json.loads(text)
        kind = msg.get("type")
        with self._lock:
            if kind == "status":
                self._status = msg
                self._statuses += 1
            elif kind == "hello":
                self._hello = msg
            elif kind == "ack":
                self._acks[msg.get("id")] = msg
            elif kind == "preview":
                self._pending_preview = msg
            elif kind == "error":
                self._errors.append(str(msg.get("text")))
            self._lock.notify_all()

    def _on_binary(self, data: bytes) -> None:
        with self._lock:
            header, self._pending_preview = self._pending_preview, None
            if header is None:
                self._errors.append("a binary message without a preview header")
            else:
                camera = header["camera"]
                self._previews[camera] = int(header["frame_no"])
                self._jpegs[camera] = bytes(data)
            self._lock.notify_all()
