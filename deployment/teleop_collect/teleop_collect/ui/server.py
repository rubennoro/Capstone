"""The ui's web server: the page, its WebSocket, the key arbiter with the deadman, and the U1/U2 pollers
(STRUCTURE.md 9.1, 9.2).

One asyncio event loop on the main thread runs everything; every U1/U2 request runs on a Bridge lane (a worker
thread), so the loop never blocks. The server binds 127.0.0.1:<ports.http> only.

Tasks on the loop:
- status poller: u1.get_status at status_hz -> one status message to every page;
- preview poller: u2.get_preview at preview_hz (only while a page is connected) -> a preview header plus the JPEG;
- keys sender: forwards the newest key state to teleop, in order, coalescing while a request is in flight;
- deadman: when the page driving the keys is silent for keys_stale_ms, sends keys(held=[], slow=false, focus=false);
- one handler and one writer per page (WebSocket).
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import numpy as np
from aiohttp import WSCloseCode, WSMsgType, web
from deploy_contracts import ContractError, Message, RemoteError, TransportTimeout

from teleop_collect.contracts import (CAMERAS, GRIPPER_COMMANDS, MOTION_KEYS, STATUS_MAX_MESSAGES, CreateTask,
                                      DeleteEpisode, HomeRequest, SelectTask, StartEpisode, StopArm, StopEpisode,
                                      TeleopStatus)
from teleop_collect.ui import NAME
from teleop_collect.ui.bridge import Bridge, FatalError
from teleop_collect.ui.settings import HTTP_HOST, Settings

log = logging.getLogger(NAME)

STATIC_DIR = Path(__file__).resolve().parent / "static"
#: Host names the page may be opened under (the Host and Origin checks below).
LOCAL_NAMES = ("127.0.0.1", "localhost")
TELEOP_NOT_ANSWERING = "teleop not answering"
FRAMES_NOT_ANSWERING = "frames not answering"
#: A page's outbox: status and previews are dropped beyond this many waiting sends (acks never are).
OUTBOX_MAX = 6
#: How long the shutdown waits for the last "no keys" to reach teleop.
RELEASE_ON_STOP_S = 0.5


# ------------------------------------------------------------------------------------------------- the commands

def _arg(args: Mapping[str, Any], key: str) -> Any:
    if key not in args:
        raise ValueError(f"missing argument {key!r}")
    return args[key]


#: Page command name -> (its arguments, builder of the U1 request). STRUCTURE.md 9.2, except that the arguments travel
#: in their own map, {"type": "command", "name": "select_task", "args": {"name": "pick_block"}, "id": 3}: flat, a
#: task's `name` would collide with the command's `name`.
COMMANDS: dict[str, tuple[tuple[str, ...], Callable[[Mapping[str, Any]], Message]]] = {
    "home": ((), lambda a: HomeRequest()),
    "start_episode": ((), lambda a: StartEpisode()),
    "stop_arm": ((), lambda a: StopArm()),
    "select_task": (("name",), lambda a: SelectTask(name=_arg(a, "name"))),
    "create_task": (("name", "description"),
                    lambda a: CreateTask(name=_arg(a, "name"), description=a.get("description", ""))),
    "stop_episode": (("outcome",), lambda a: StopEpisode(outcome=_arg(a, "outcome"))),
    "delete_episode": (("episode",), lambda a: DeleteEpisode(episode=_arg(a, "episode"))),
}


def build_command(name: Any, args: Mapping[str, Any]) -> Message:
    """The U1 request of a page command; ValueError / ContractError (one line) when it is not a valid one."""
    if name not in COMMANDS:
        raise ValueError(f"unknown command {name!r} (known: {', '.join(COMMANDS)})")
    allowed, builder = COMMANDS[name]
    extra = sorted(set(args) - set(allowed))
    if extra:
        raise ValueError(f"{name}: unknown argument(s) {', '.join(map(repr, extra))}")
    return builder(args)


def jsonable(value: Any) -> Any:
    """A contract dict as JSON values: arrays become lists, numpy scalars Python numbers."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return None                                        # never sent as JSON (previews are binary messages)
    return value


# ------------------------------------------------------------------------------------------------------- pages

@dataclass(eq=False)
class Page:
    """One connected page (WebSocket)."""

    id: int
    ws: web.WebSocketResponse
    outbox: collections.deque = field(default_factory=collections.deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    last_keys_s: float | None = None

    def send_soon(self, parts: list[str | bytes], droppable: bool) -> None:
        """Queue messages sent together (a preview header and its JPEG); dropped when the page is behind."""
        if droppable and len(self.outbox) >= OUTBOX_MAX:
            return
        self.outbox.append(parts)
        self.wake.set()


class KeysArbiter:
    """Which page drives the keys, and the deadman (STRUCTURE.md 9.1).

    The page that last sent a keys message with focus=true drives them; a focus=false message counts only from that
    page. When the driving page sends nothing for keys_stale_ms, or its socket closes, or it reports focus=false, the
    key state becomes "no keys" (held=[], slow=false, focus=false) at once and no page drives until one sends
    focus=true again. `forward` gets every key state to send to teleop.
    """

    RELEASED = ((), False, False)

    def __init__(self, keys_stale_s: float, forward: Callable[[tuple[str, ...], bool, bool], None]) -> None:
        self.keys_stale_s = keys_stale_s
        self._forward = forward
        self.active: Page | None = None
        self.releases = 0                                  # deadman / close / blur releases (tests, log)

    def on_keys(self, page: Page, held: tuple[str, ...], slow: bool, focus: bool, now_s: float) -> None:
        page.last_keys_s = now_s
        if focus:
            if self.active is not page and self.active is not None:
                log.info("page %d now drives the keys (was page %d)", page.id, self.active.id)
            self.active = page
            self._forward(held, slow, True)
        elif page is self.active:
            self._release("the page lost focus")

    def on_close(self, page: Page) -> None:
        if page is self.active:
            self._release("the page closed")

    def tick(self, now_s: float) -> None:
        page = self.active
        if page is not None and page.last_keys_s is not None and now_s - page.last_keys_s >= self.keys_stale_s:
            self._release(f"no keys from the page for {self.keys_stale_s * 1000:.0f} ms")

    def release(self, why: str) -> None:
        self._release(why)

    def _release(self, why: str) -> None:
        if self.active is not None:
            log.info("keys released: %s", why)
        self.active = None
        self.releases += 1
        self._forward(*self.RELEASED)


class KeysSender:
    """Sends the newest key state to teleop, one request at a time and in order (a newer state replaces one that
    has not gone out yet)."""

    def __init__(self, bridge: Bridge, on_link: Callable[[Exception | None], None]) -> None:
        self._bridge = bridge
        self._on_link = on_link
        self._pending: tuple[tuple[str, ...], bool, bool] | None = None
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self.sent = 0

    def set(self, held: tuple[str, ...], slow: bool, focus: bool) -> None:
        self._pending = (tuple(held), bool(slow), bool(focus))
        self._idle.clear()
        self._wake.set()

    async def wait_idle(self, timeout_s: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._idle.wait(), timeout_s)

    async def run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            state, self._pending = self._pending, None
            if state is None:
                continue
            try:
                await self._bridge.keys(*state)
                self.sent += 1
                self._on_link(None)
            except (TransportTimeout, RemoteError, ContractError) as e:
                self._on_link(e)
            finally:
                if self._pending is None:
                    self._idle.set()


# ------------------------------------------------------------------------------------------------------- server

class UiServer:
    """The server on one event loop. start() binds and starts the tasks; stop() ends them; `fatal` is set when the
    ui must end with a FATAL line."""

    def __init__(self, settings: Settings, bridge: Bridge, ready: Callable[[], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.settings = settings
        self.bridge = bridge
        self._ready_cb = ready
        self._clock = clock
        self.fatal: FatalError | None = None
        self.stopped = asyncio.Event()                     # set on stop request or fatal
        self.port: int | None = None
        self.pages: dict[int, Page] = {}
        self._next_page = 1
        self._tasks: list[asyncio.Task] = []
        self._runner: web.AppRunner | None = None
        self._ready_done = False
        # teleop's side
        self.status: TeleopStatus | None = None
        self.teleop_answering = False
        self.teleop_error = TELEOP_NOT_ANSWERING
        self.messages: collections.deque = collections.deque(maxlen=STATUS_MAX_MESSAGES)
        self._last_message_id = 0
        # frames' side
        self.frames_answering = False
        self.frames_error = FRAMES_NOT_ANSWERING
        self.preview_health: dict | None = None
        self.latest: dict[str, tuple[int, bytes]] = {}
        # keys
        self.keys_sender = KeysSender(bridge, self._on_keys_link)
        self.arbiter = KeysArbiter(settings.keys_stale_ms / 1000.0, self.keys_sender.set)

    # ------------------------------------------------------------------------------------------- start / stop

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self._local_only])
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._websocket)
        app.router.add_static("/static/", STATIC_DIR, append_version=False)
        return app

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app(), access_log=None, handle_signals=False, shutdown_timeout=1.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, HTTP_HOST, self.settings.http_port)
        try:
            await site.start()
        except OSError as e:
            await self._runner.cleanup()
            self._runner = None
            raise FatalError(f"cannot bind the page on {HTTP_HOST}:{self.settings.http_port}: {e.strerror or e}",
                             "another program uses ports.http (an old teleop_collect run?); change ports.http in "
                             "station.yaml or stop it") from None
        self.port = int(self._runner.addresses[0][1])
        log.info("page on http://%s:%d", HTTP_HOST, self.port)
        loop = asyncio.get_running_loop()
        for coro in (self.keys_sender.run(), self._poll_status(), self._poll_previews(), self._deadman()):
            task = loop.create_task(coro)
            task.add_done_callback(self._task_done)
            self._tasks.append(task)

    async def stop(self) -> None:
        """Release the keys, close every page, stop the tasks and the HTTP server."""
        if self._tasks:
            self.arbiter.release("the ui is stopping")
            await self.keys_sender.wait_idle(RELEASE_ON_STOP_S)
        for page in list(self.pages.values()):
            with contextlib.suppress(Exception):
                await page.ws.close(code=WSCloseCode.GOING_AWAY, message=b"ui stopping")
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def request_stop(self) -> None:
        self.stopped.set()

    def _set_fatal(self, error: FatalError) -> None:
        if self.fatal is None:
            log.error("%s", error)
            self.fatal = error
        self.stopped.set()

    def _task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            if isinstance(exc, FatalError):
                self._set_fatal(exc)
            else:
                log.error("ui task failed", exc_info=exc)
                self._set_fatal(FatalError(f"internal error: {type(exc).__name__}: {exc}", "see the log above"))

    def _contract_fatal(self, e: ContractError, process: str) -> FatalError:
        return FatalError(f"{process} sent a message that breaks its contract: {e}",
                          f"teleop_collect's {process} and ui must come from the same checkout")

    # ------------------------------------------------------------------------------------------------- HTTP

    @web.middleware
    async def _local_only(self, request: web.Request, handler):
        """Only a page opened as 127.0.0.1 / localhost may use the server: the Host check stops DNS rebinding, the
        Origin check stops another web site's script from opening the WebSocket (browsers allow that). An Origin
        header must be exactly http://127.0.0.1:<port> or http://localhost:<port>; "null" (a sandboxed iframe, a
        file:// page) is refused like any other. No Origin header at all (ScriptedPage, curl) is allowed: browsers
        always send one on a WebSocket."""
        host = urlsplit(f"//{request.host}").hostname if request.host else None
        if host not in LOCAL_NAMES:
            raise web.HTTPForbidden(text="the ui answers only http://127.0.0.1")
        origin = request.headers.get("Origin")
        if origin is not None and origin not in {f"http://{name}:{self.port}" for name in LOCAL_NAMES}:
            raise web.HTTPForbidden(text="another site may not use the ui")
        return await handler(request)

    async def _index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=None, max_msg_size=1 << 20)
        await ws.prepare(request)
        page = Page(id=self._next_page, ws=ws)
        self._next_page += 1
        self.pages[page.id] = page
        log.info("page %d connected (%d open)", page.id, len(self.pages))
        writer = asyncio.get_running_loop().create_task(self._writer(page))
        try:
            page.send_soon([json.dumps(self._hello())], droppable=False)
            page.send_soon([json.dumps(self._status_message())], droppable=False)
            for camera in CAMERAS:
                if camera in self.latest:
                    page.send_soon(self._preview_parts(camera, *self.latest[camera]), droppable=False)
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self._on_page_message(page, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.pages.pop(page.id, None)
            self.arbiter.on_close(page)
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await writer
            log.info("page %d closed (%d open)", page.id, len(self.pages))
        return ws

    async def _writer(self, page: Page) -> None:
        while not page.ws.closed:
            await page.wake.wait()
            page.wake.clear()
            while page.outbox and not page.ws.closed:
                for part in page.outbox.popleft():
                    if isinstance(part, bytes):
                        await page.ws.send_bytes(part)
                    else:
                        await page.ws.send_str(part)

    def _hello(self) -> dict:
        s = self.settings
        return {"type": "hello", "keys_resend_ms": s.keys_resend_ms, "keys_stale_ms": s.keys_stale_ms,
                "status_hz": s.status_hz, "preview_hz": s.preview_hz, "motion_keys": list(MOTION_KEYS)}

    # ------------------------------------------------------------------------------------------- page -> server

    def _on_page_message(self, page: Page, text: str) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            msg = None
        if not isinstance(msg, dict):
            self._page_error(page, "not a JSON object")
            return
        kind = msg.get("type")
        if kind == "keys":
            self._on_keys(page, msg)
        elif kind == "gripper":
            command = msg.get("command")
            if command not in GRIPPER_COMMANDS:
                self._page_error(page, f"gripper: unknown command {command!r}")
                return
            self._spawn(self._send_gripper(command))
        elif kind == "command":
            self._spawn(self._run_command(page, msg))
        else:
            self._page_error(page, f"unknown message type {kind!r}")

    def _on_keys(self, page: Page, msg: dict) -> None:
        held, slow, focus = msg.get("held"), msg.get("slow"), msg.get("focus")
        ok = (isinstance(held, list) and all(isinstance(k, str) and k in MOTION_KEYS for k in held)
              and isinstance(slow, bool) and isinstance(focus, bool))
        now = self._clock()
        if not ok:
            # a broken key state never moves the arm: from the page driving the keys it counts as no keys
            self._page_error(page, f"keys: expected held (names of {', '.join(MOTION_KEYS)}), slow and focus "
                                   f"(true/false), got {msg!r}")
            if page is self.arbiter.active:
                self.arbiter.release("a broken key state from the page")
            return
        unique = tuple(dict.fromkeys(held))
        self.arbiter.on_keys(page, unique if focus else (), slow if focus else False, focus, now)

    def _page_error(self, page: Page, text: str) -> None:
        log.warning("page %d: %s", page.id, text)
        page.send_soon([json.dumps({"type": "error", "text": text})], droppable=False)

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        task.add_done_callback(self._task_done)

    async def _send_gripper(self, command: str) -> None:
        try:
            ack = await self.bridge.gripper(command)
        except (TransportTimeout, RemoteError) as e:
            self._on_teleop_link(e)
            return
        except ContractError as e:
            raise self._contract_fatal(e, "teleop") from None
        if not ack.ok:
            log.info("gripper %s refused: %s", command, ack.reason)

    async def _run_command(self, page: Page, msg: dict) -> None:
        cid = msg.get("id")
        name = msg.get("name")
        args = msg.get("args", {})
        extra = sorted(set(msg) - {"type", "name", "args", "id"})

        def answer(ok: bool, reason: str) -> None:
            if not page.ws.closed:
                page.send_soon([json.dumps({"type": "ack", "id": cid, "ok": ok, "reason": reason})], droppable=False)

        if isinstance(cid, bool) or not isinstance(cid, int):
            answer(False, f"command: 'id' must be an integer, got {cid!r}")
            return
        if not isinstance(args, dict) or extra:
            answer(False, f"command: expected type, name, args (a map) and id, got {msg!r}")
            return
        try:
            request = build_command(name, args)
        except (ValueError, ContractError) as e:
            answer(False, str(e))
            return
        log.info("page %d: %s", page.id, name)
        try:
            ack = await self.bridge.command(request)
        except TransportTimeout as e:
            self._on_teleop_link(e)
            answer(False, TELEOP_NOT_ANSWERING)
            return
        except RemoteError as e:
            answer(False, str(e))
            return
        except ContractError as e:
            answer(False, str(e))
            raise self._contract_fatal(e, "teleop") from None
        answer(ack.ok, ack.reason)

    # ----------------------------------------------------------------------------------------------- pollers

    def _on_keys_link(self, error: Exception | None) -> None:
        if isinstance(error, ContractError):
            self._set_fatal(self._contract_fatal(error, "teleop"))
        elif isinstance(error, TransportTimeout):
            self._on_teleop_link(error)

    def _on_teleop_link(self, error: Exception | None) -> None:
        answering = error is None
        if answering != self.teleop_answering:
            log.info("teleop %s", "answering" if answering else f"not answering: {error}")
        self.teleop_answering = answering
        if error is None:
            self.teleop_error = ""
        elif isinstance(error, TransportTimeout):
            self.teleop_error = TELEOP_NOT_ANSWERING
        else:
            self.teleop_error = f"teleop: {error}"

    async def _poll_status(self) -> None:
        period = 1.0 / self.settings.status_hz
        while True:
            started = self._clock()
            try:
                status = await self.bridge.get_status(self._last_message_id)
            except (TransportTimeout, RemoteError) as e:
                self._on_teleop_link(e)
            except ContractError as e:
                raise self._contract_fatal(e, "teleop") from None
            else:
                self._on_status(status)
            self._broadcast([json.dumps(self._status_message())], droppable=True)
            await asyncio.sleep(max(0.0, period - (self._clock() - started)))

    def _on_status(self, status: TeleopStatus) -> None:
        self._on_teleop_link(None)
        self.status = status
        for m in status.messages:
            if m.id > self._last_message_id:
                self.messages.append(m.to_dict())
                self._last_message_id = m.id
        if not self._ready_done:
            self._ready_done = True
            if self._ready_cb is not None:
                self._ready_cb()

    def _status_message(self) -> dict:
        """{"type": "status", ...TeleopStatus fields..., plus the ui's own link fields}. Before teleop ever
        answered, only the link fields. `messages` holds the newest STATUS_MAX_MESSAGES the ui has seen (a page
        keeps those it has not shown, by id)."""
        out: dict[str, Any] = {"type": "status"}
        if self.status is not None:
            d = self.status.to_dict()
            d.pop("schema", None)
            out.update(jsonable(d))
            out["messages"] = jsonable(list(self.messages))
        out.update(teleop_answering=self.teleop_answering, teleop_error=self.teleop_error,
                   frames_answering=self.frames_answering, frames_error=self.frames_error,
                   preview_health=self.preview_health)
        return out

    def _broadcast(self, parts: list[str | bytes], droppable: bool) -> None:
        for page in list(self.pages.values()):
            page.send_soon(parts, droppable)

    @staticmethod
    def _preview_parts(camera: str, frame_no: int, jpeg: bytes) -> list[str | bytes]:
        return [json.dumps({"type": "preview", "camera": camera, "frame_no": frame_no}), jpeg]

    async def _poll_previews(self) -> None:
        period = 1.0 / self.settings.preview_hz
        after = {camera: -1 for camera in CAMERAS}
        while True:
            started = self._clock()
            if self.pages:
                try:
                    preview = await self.bridge.get_preview(after["main"], after["ultrawide"])
                except (TransportTimeout, RemoteError) as e:
                    if self.frames_answering:
                        log.info("frames not answering: %s", e)
                    self.frames_answering = False
                    self.frames_error = (FRAMES_NOT_ANSWERING if isinstance(e, TransportTimeout)
                                         else f"frames: {e}")
                except ContractError as e:
                    raise self._contract_fatal(e, "frames") from None
                else:
                    if not self.frames_answering:
                        log.info("frames answering")
                    self.frames_answering = True
                    self.frames_error = ""
                    self.preview_health = jsonable(preview.health.to_dict())
                    for camera in CAMERAS:
                        jpeg = getattr(preview, f"{camera}_jpeg")
                        newest = getattr(preview, f"{camera}_frame_no")
                        if jpeg is not None:
                            self.latest[camera] = (newest, jpeg)
                            after[camera] = newest
                            self._broadcast(self._preview_parts(camera, newest, jpeg), droppable=True)
                        elif newest < after[camera]:
                            after[camera] = -1                 # the Pi stream restarted: its numbers start again
            await asyncio.sleep(max(0.0, period - (self._clock() - started)))

    async def _deadman(self) -> None:
        period = min(0.025, self.settings.keys_stale_ms / 4000.0)
        while True:
            self.arbiter.tick(self._clock())
            await asyncio.sleep(period)


# ---------------------------------------------------------------------------------------------------------- app

class UiApp:
    """The ui process around UiServer: its own event loop (run() blocks), the ready file, a thread-safe stop.

    run() returns on request_stop() and raises FatalError when the ui must end with a FATAL line. Teleop or frames
    being slow or gone never ends it (the page shows "teleop not answering")."""

    def __init__(self, settings: Settings, ready_file: Path | None = None,
                 bridge_factory: Callable[[Settings], Bridge] | None = None,
                 write_ready: Callable[[], None] | None = None) -> None:
        self.settings = settings
        self.ready_file = ready_file
        self._bridge_factory = bridge_factory or (lambda s: Bridge(s.u1_endpoint, s.preview_endpoint,
                                                                   s.rpc_timeout_s))
        self._write_ready = write_ready
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: UiServer | None = None
        self._stop_requested = threading.Event()
        self.bound = threading.Event()                      # set once the HTTP socket is bound
        self.ready = threading.Event()                      # set once ready (bound and teleop answered once)

    @property
    def server(self) -> UiServer | None:
        return self._server

    @property
    def port(self) -> int | None:
        return None if self._server is None else self._server.port

    def url(self) -> str:
        return f"http://{HTTP_HOST}:{self.port}"

    def request_stop(self) -> None:
        """Thread-safe (and safe from a signal handler)."""
        self._stop_requested.set()
        loop, server = self._loop, self._server
        if loop is not None and server is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(server.request_stop)

    def run(self) -> None:
        asyncio.run(self._main())

    def _on_ready(self) -> None:
        if self._write_ready is not None:
            self._write_ready()
        elif self.ready_file is not None:
            path = Path(self.ready_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ready\n", encoding="utf-8")
        log.info("ready: http://%s:%s", HTTP_HOST, self.port)
        self.ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        bridge = self._bridge_factory(self.settings)
        # ready = bound and teleop answered get_status once: the status poller starts only once the socket is bound
        server = UiServer(self.settings, bridge, ready=self._on_ready)
        self._server = server
        try:
            await server.start()
            self.bound.set()
            if self._stop_requested.is_set():
                server.request_stop()
            await server.stopped.wait()
        finally:
            await server.stop()
            await asyncio.get_running_loop().run_in_executor(None, bridge.close)
        if server.fatal is not None:
            raise server.fatal
