"""sim_deploy.ui's web server (ui/DESIGN.md §2-§3): the page, the REST API, the WebSocket, and the wiring of the
runner (U2), live data (U3) and catalog (U5).

Modelled on teleop_collect/ui/server.py: one asyncio event loop on the main thread runs everything; every call into
a part that is not a coroutine runs on a worker thread (asyncio.to_thread), so the loop never blocks. The server
binds 127.0.0.1 only and answers only pages opened as 127.0.0.1 / localhost (Host and Origin checks).

Expected APIs of the other parts. Each method may be sync or `async def` (or return an awaitable); the adapters below
accept both. A constructor that takes a parameter gets the Settings (ui/settings.py), else it is called bare.

  Runner (ui/runner.py, U2):  Runner(settings)
    start(req: dict) -> run_id: str   req = the validated RunRequest (§2, every key present, defaults filled) plus
                                      "target": the eval driver's --policy value (run folder | hold_still |
                                      replay:<zarr>#<i>). Raise ValueError for a request it refuses (-> 400),
                                      anything else (e.g. busy) -> 409 if the runner then reports a busy state, else 500.
    stop() -> None                    the first: clean stop (n / q); a second: kill. Called also when idle (a no-op).
    status() -> dict                  Status (§3) without "type"; polled at status_hz.
    events()                          event dicts {"kind": ...}. Called ONCE at start; may return a queue.Queue, an
                                      asyncio.Queue, an async iterator or a (blocking) iterator. If it returns a list,
                                      it is treated as "new events since the last call" and polled at status_hz.
    session_dir() -> Path | None      the running policy_deploy session folder; passed to Live.set_session_dir on
                                      every change (None when nothing runs).
    close() / shutdown() (optional)   called last on exit.
  Live (ui/live.py, U3):  Live(settings)
    start(), stop()                   started with the server, stopped on exit (both may block; worker thread).
    latest_frames() -> {channel: jpeg | (seq, jpeg)}
                                      channel = 1..4 or "external" / "main" / "ultrawide" / "policy". Polled at
                                      frame_fps while a page is connected; a frame is sent once (same seq, or the same
                                      bytes object when no seq, is not sent again; equal bytes in a new object are
                                      sent, so a static scene keeps updating).
      or subscribe(callback)          callback(channel, jpeg) from any thread; the newest per channel is kept and
                                      sent at frame_fps (used only when Live has no latest_frames).
    telemetry() -> dict | None        §3 telemetry without "type"; polled at telemetry_hz, sent when not empty.
    set_session_dir(path | None)
  Catalog (ui/catalog.py, U5):  a class Catalog(settings) with these methods, or the module's own functions
    policies() -> {"policies": [Policy], "replays": [Replay], "builtins": [str]}
    evals() -> {"evals": [EvalRow]}
    (a function / method with a required parameter gets the Settings; policies(replays=True) is called bare)

`--stub` (all parts) or `--stub runner,live` (some) uses ui/_stubs.py instead.

Tasks on the loop: status pump (5 Hz), telemetry pump (10 Hz), frame pump (<= 10 fps per channel), event pump, one
handler and one writer per page. A page has one slot for status, one for telemetry and one per channel: a slow page
gets the newest of each when its writer is free (older ones are overwritten, never buffered); events queue (bounded).
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import html
import importlib
import inspect
import json
import logging
import math
import queue as queue_mod
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlsplit

from aiohttp import WSCloseCode, WSMsgType, web

from sim_deploy.ui import NAME
from sim_deploy.ui.settings import HTTP_HOST, Settings

log = logging.getLogger(NAME)

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOCAL_NAMES = ("127.0.0.1", "localhost")
BUSY_STATES = ("starting", "running", "stopping")
CHANNELS = {"external": 1, "main": 2, "ultrawide": 3, "policy": 4}
ALL_CHANNELS = frozenset(CHANNELS.values())
#: kernel send buffer of a page's socket (Linux doubles it)
WS_SNDBUF = 256 * 1024
#: events waiting for one page beyond this are dropped (oldest first), with a warning
EVENTS_MAX = 1000
#: /evals/ files shown in the browser as text instead of downloaded
TEXT_TYPES = {".jsonl": "text/plain; charset=utf-8", ".yaml": "text/plain; charset=utf-8",
              ".yml": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8",
              ".csv": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8",
              ".json": "application/json; charset=utf-8", ".md": "text/plain; charset=utf-8"}
PARTS = ("runner", "live", "catalog")


# ------------------------------------------------------------------------------------------------ the parts

async def call(fn: Callable, *args: Any) -> Any:
    """fn(*args) without blocking the loop: a coroutine function is awaited, anything else runs on a worker thread
    (and an awaitable it returns is awaited)."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    result = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(result):
        result = await result
    return result


def _params(fn: Callable) -> list:
    try:
        return [p for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)]
    except (TypeError, ValueError):
        return []


def _takes_args(fn: Callable) -> bool:
    """Any positional parameter (constructors: Runner(settings), Live(settings=None) both get the settings)."""
    return bool(_params(fn))


def _requires_arg(fn: Callable) -> bool:
    """A positional parameter without a default (catalog functions: policies(settings) yes, policies(replays=True)
    no)."""
    return any(p.default is p.empty and p.kind != p.VAR_POSITIONAL for p in _params(fn))


def construct(cls: Callable, settings: Settings) -> Any:
    return cls(settings) if _takes_args(cls) else cls()


class CatalogAdapter:
    """A Catalog object or the catalog module's functions, with the settings passed when they take a parameter."""

    def __init__(self, target: Any, settings: Settings) -> None:
        self._target = target
        self._settings = settings

    async def _get(self, name: str) -> dict:
        fn = getattr(self._target, name, None)
        if fn is None:
            raise RuntimeError(f"the catalog has no {name}()")
        result = await (call(fn, self._settings) if _requires_arg(fn) else call(fn))
        if not isinstance(result, dict):
            raise RuntimeError(f"catalog.{name}() returned {type(result).__name__}, expected a dict")
        return result

    async def policies(self) -> dict:
        out = await self._get("policies")
        out.setdefault("policies", [])
        out.setdefault("replays", [])
        out.setdefault("builtins", ["hold_still"])
        return out

    async def evals(self) -> dict:
        out = await self._get("evals")
        out.setdefault("evals", [])
        return out


def load_parts(settings: Settings, stub: Iterable[str] = ()) -> tuple[Any, Any, CatalogAdapter]:
    """(runner, live, catalog): the stubs for the parts named in `stub`, the real modules for the rest."""
    stub = set(stub)
    if stub:
        from sim_deploy.ui import _stubs
    if "runner" in stub:
        runner = _stubs.StubRunner(settings)
    else:
        runner = construct(_import("runner").Runner, settings)
    if "live" in stub:
        live = _stubs.StubLive(settings)
    else:
        live = construct(_import("live").Live, settings)
    if "catalog" in stub:
        catalog = _stubs.StubCatalog(settings)
    else:
        mod = _import("catalog")
        catalog = construct(mod.Catalog, settings) if hasattr(mod, "Catalog") else mod
    return runner, live, CatalogAdapter(catalog, settings)


class PartMissing(Exception):
    pass


def _import(name: str):
    try:
        return importlib.import_module(f"sim_deploy.ui.{name}")
    except ModuleNotFoundError as e:
        if e.name == f"sim_deploy.ui.{name}":
            raise PartMissing(f"ui/{name}.py is missing (run with --stub {name} to use a fake one)") from None
        raise


def jsonable(value: Any) -> Any:
    """For json.dumps(default=...): paths and numpy values become JSON values; bytes are not sent."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return None
    return str(value)


def dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, default=jsonable, allow_nan=False)
    except ValueError:                                 # NaN / inf somewhere: they become null
        return json.dumps(_clean_nan(obj), default=jsonable)


def _clean_nan(obj: Any) -> Any:
    """NaN / inf are not JSON: they become null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def channel_id(key: Any) -> int | None:
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key if key in ALL_CHANNELS else None
    if isinstance(key, str):
        if key.isdigit():
            return channel_id(int(key))
        return CHANNELS.get(key)
    return None


# ------------------------------------------------------------------------------------------------ RunRequest

TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$")
REPLAY_RE = re.compile(r"^replay:(?P<zarr>.+)#(?P<index>\d+)$")
HOMES = ("start", "deploy")          # P2: start pose (PC key p, default) | deploy home (h only)
HOME_ALIASES = {"trainstart": "start"}
LOOKS = ("robot", "train", "current")   # run.sh --look (sim_deploy/looks.py), robot the default
REQUEST_KEYS = ("policy", "rollouts", "seed", "home", "look", "timeout_s", "tap", "tag", "precision", "diffusion_steps",
                "exec_h", "starts")
STARTS = ("random", "set")           # random = rng([seed, n]) (default) | set = the driver's --starts set (w3.yaml)


def validate_request(body: Any) -> tuple[dict, list[str]]:
    """The RunRequest with defaults filled, and its problems (one line each; empty = valid). The policy id is only
    checked for its type here (resolve_target checks it against the catalog)."""
    if not isinstance(body, dict):
        return {}, [f"the body must be a JSON object (a RunRequest), got {type(body).__name__}"]
    problems: list[str] = []
    extra = sorted(set(body) - set(REQUEST_KEYS))
    if extra:
        problems.append(f"unknown key(s) {', '.join(map(repr, extra))} (a RunRequest has {', '.join(REQUEST_KEYS)})")

    def integer(key: str, default: int | None, lo: int, hi: int, nullable: bool = False) -> int | None:
        v = body.get(key, default)
        if v is None and nullable:
            return None
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            problems.append(f"{key}: expected an integer {lo}..{hi}{' or null' if nullable else ''}, got {v!r}")
            return default
        return v

    def choice(key: str, default: str | None, options: tuple, nullable: bool = False) -> str | None:
        v = body.get(key, default)
        if v is None and nullable:
            return None
        if v not in options:
            problems.append(f"{key}: expected one of {', '.join(map(repr, options))}"
                            f"{' or null' if nullable else ''}, got {v!r}")
            return default
        return v

    policy = body.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        problems.append(f"policy: expected a policy id (GET /api/policies), 'hold_still' or a replay id, "
                        f"got {policy!r}")
        policy = None
    else:
        policy = policy.strip()
    timeout = body.get("timeout_s", 60.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1.0 <= timeout <= 3600.0:
        problems.append(f"timeout_s: expected seconds per rollout, 1..3600, got {timeout!r}")
        timeout = 60.0
    tap = body.get("tap", False)
    if not isinstance(tap, bool):
        problems.append(f"tap: expected true or false, got {tap!r}")
        tap = False
    tag = body.get("tag", "ui")
    if tag is None or tag == "":
        tag = "ui"
    if not isinstance(tag, str) or not TAG_RE.match(tag):
        problems.append(f"tag: expected 1..60 characters of letters, digits, '_', '-', '.' (it ends the eval "
                        f"folder's name), starting with a letter or digit, got {tag!r}")
        tag = "ui"
    home = body.get("home", "start")
    if home is None or home == "":
        home = "start"
    home = HOME_ALIASES.get(home, home) if isinstance(home, str) else home   # old requests: trainstart = start
    if home not in HOMES:
        problems.append(f"home: expected one of {', '.join(map(repr, HOMES))} (or 'trainstart' = 'start'), "
                        f"got {home!r}")
        home = "start"
    req = {
        "policy": policy,
        "rollouts": integer("rollouts", 1, 1, 100),
        "seed": integer("seed", 0, 0, 2**32 - 1),
        "home": home,
        "look": choice("look", "robot", LOOKS) if body.get("look") not in (None, "") else "robot",
        "timeout_s": float(timeout),
        "tap": tap,
        "tag": tag,
        # 2026-11-21 (the user): the arm's settings (bf16, 8 steps, exec_h 12) unless the request sets others
        "precision": choice("precision", "bf16", ("fp32", "bf16")) if body.get("precision") is not None else "bf16",
        "diffusion_steps": integer("diffusion_steps", 8, 1, 1000) if body.get("diffusion_steps") is not None else 8,
        "exec_h": integer("exec_h", 12, 1, 16) if body.get("exec_h") is not None else 12,
        # the rules for "set" (home start, no replay, a checkpoint listed in w3.yaml) are the runner's pre-check
        "starts": choice("starts", "random", STARTS) if body.get("starts") not in (None, "") else "random",
    }
    return req, problems


_START_POSE_CACHE: dict = {}


def start_pose_info() -> dict:
    """GET /api/start_pose: sim_deploy.scene.start_pose.show() trimmed for the page, cached until start_pose.yaml or
    model.json changes (the first call loads the MuJoCo model, ~8 s)."""
    from sim_deploy.scene import start_pose as sp

    key = tuple((p.stat().st_mtime_ns if p.exists() else 0) for p in (sp.START_POSE_YAML, sp.DEFAULT_MODEL))
    if _START_POSE_CACHE.get("key") != key:
        info = sp.show()
        keep = ("set", "file", "source", "text", "q_start_rad", "tcp_O_m", "tilt_deg", "heading_words",
                "height_above_table_m", "ok", "vs_demos", "region", "move_time_s")
        out = {k: info[k] for k in keep if k in info}
        if info.get("checks"):
            out["checks"] = {k: {"ok": c["ok"], "detail": c["detail"]} for k, c in info["checks"].items()}
            out["text"] = sp.summary_line(info)
        _START_POSE_CACHE.update(key=key, info=out)
    # the real arm's (the upstream station.yaml, sim_deploy/upstream.py), read only; not cached (git show, a few ms)
    return {**_START_POSE_CACHE["info"], "vs_git": sp.compare_git()}


def resolve_target(policy_id: str, catalog: dict) -> tuple[str | None, str | None]:
    """(the eval driver's --policy value, None) or (None, why not)."""
    if policy_id in catalog.get("builtins", []) or policy_id == "hold_still":
        return policy_id, None
    for p in catalog.get("policies", []):
        if isinstance(p, dict) and p.get("id") == policy_id:
            if not p.get("checkpoint"):
                return None, f"policy {policy_id!r} has no checkpoint (not fetched: train/fetch.sh it first)"
            if not p.get("run_folder"):
                return None, f"policy {policy_id!r} has no run_folder in the catalog"
            return str(p["run_folder"]), None
    for r in catalog.get("replays", []):
        if isinstance(r, dict) and r.get("id") == policy_id:
            return policy_id, None
    m = REPLAY_RE.match(policy_id)
    if m:
        zarr = Path(m["zarr"]).expanduser()
        if not zarr.exists():
            return None, f"replay dataset {zarr} does not exist"
        return f"replay:{zarr}#{int(m['index'])}", None
    return None, (f"unknown policy {policy_id!r}: expected an id from GET /api/policies (runnable: "
                  f"{sum(1 for p in catalog.get('policies', []) if isinstance(p, dict) and p.get('checkpoint'))} "
                  f"policies), 'hold_still' or replay:<zarr>#<index>")


# ------------------------------------------------------------------------------------------------------ pages

@dataclass(eq=False)
class Page:
    """One connected page. Status, telemetry and each channel are single slots (the newest wins); events queue."""

    id: int
    ws: web.WebSocketResponse
    channels: set = field(default_factory=lambda: set(ALL_CHANNELS))
    events: collections.deque = field(default_factory=collections.deque)
    status: str | None = None
    telemetry: str | None = None
    frames: dict = field(default_factory=dict)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    dropped_frames: int = 0
    dropped_events: int = 0

    def push_event(self, text: str) -> None:
        if len(self.events) >= EVENTS_MAX:
            self.events.popleft()
            self.dropped_events += 1
        self.events.append(text)
        self.wake.set()

    def push_status(self, text: str) -> None:
        self.status = text
        self.wake.set()

    def push_telemetry(self, text: str) -> None:
        self.telemetry = text
        self.wake.set()

    def push_frame(self, ch: int, data: bytes) -> None:
        if ch not in self.channels:
            return
        if ch in self.frames:
            self.dropped_frames += 1                 # the page has not taken the previous one yet
        self.frames[ch] = data
        self.wake.set()


# ----------------------------------------------------------------------------------------------------- server

class UiServer:
    """The server on one event loop. start() binds and starts the tasks; shutdown() stops the run, Live and the
    server."""

    def __init__(self, settings: Settings, runner: Any, live: Any, catalog: CatalogAdapter) -> None:
        self.settings = settings
        self.runner = runner
        self.live = live
        self.catalog = catalog
        self.port: int | None = None
        self.pages: dict[int, Page] = {}
        self._next_page = 1
        self._tasks: list[asyncio.Task] = []
        self._app_runner: web.AppRunner | None = None
        self._run_lock = asyncio.Lock()
        self.status: dict = {"state": "idle", "run_id": None, "request": None, "rollout": 0, "rollouts": 0,
                             "phase": "", "elapsed_s": 0.0, "outcomes": [], "last_error": None, "log_tail": []}
        self._runner_error: str | None = None
        self._session_dir: Any = object()            # not None: the first status pass always sets it
        self._frame_seen: dict[int, Any] = {}
        self.latest: dict[int, bytes] = {}
        self._pushed: dict[int, bytes] = {}           # Live.subscribe(callback) frames
        self._pushed_lock = threading.Lock()
        self._live_started = False
        self._closing = False
        self._evals_root = settings.sim_evals.expanduser().resolve()

    # --------------------------------------------------------------------------------------- start / stop

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self._local_only], client_max_size=1 << 20)
        r = app.router
        r.add_get("/", self._index)
        r.add_get("/ws", self._websocket)
        r.add_get("/api/policies", self._api_policies)
        r.add_get("/api/evals", self._api_evals)
        r.add_get("/api/status", self._api_status)
        r.add_get("/api/start_pose", self._api_start_pose)
        r.add_post("/api/run", self._api_run)
        r.add_post("/api/stop", self._api_stop)
        r.add_post("/api/frames", self._api_frames)
        r.add_get("/evals", self._evals_root_redirect)
        r.add_get("/evals/{tail:.*}", self._evals_file)
        if STATIC_DIR.is_dir():
            r.add_static("/static/", STATIC_DIR, append_version=False)
        return app

    async def start(self) -> None:
        self._app_runner = web.AppRunner(self.app(), access_log=None, handle_signals=False, shutdown_timeout=1.0)
        await self._app_runner.setup()
        site = web.TCPSite(self._app_runner, HTTP_HOST, self.settings.http_port)
        try:
            await site.start()
        except OSError as e:
            await self._app_runner.cleanup()
            self._app_runner = None
            raise SystemExit(f"FATAL {NAME}: cannot bind {HTTP_HOST}:{self.settings.http_port}: "
                             f"{e.strerror or e}: another program uses it (an old ui?); use --port") from None
        self.port = int(self._app_runner.addresses[0][1])
        log.info("page on http://%s:%d", HTTP_HOST, self.port)
        try:
            await call(self.live.start)
            self._live_started = True
        except Exception:
            log.exception("live.start failed: the page gets no frames or telemetry")
        if not hasattr(self.live, "latest_frames") and hasattr(self.live, "subscribe"):
            await call(self.live.subscribe, self._on_pushed_frame)
        loop = asyncio.get_running_loop()
        for coro in (self._status_pump(), self._telemetry_pump(), self._frame_pump(), self._event_pump()):
            task = loop.create_task(coro)
            task.add_done_callback(self._task_done)
            self._tasks.append(task)

    async def shutdown(self) -> None:
        """Stop the run (clean, then kill after shutdown_clean_s), Live, every page and the HTTP server."""
        self._closing = True
        await self.stop_run_for_exit()
        if self._live_started:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(call(self.live.stop), 5.0)
        for page in list(self.pages.values()):
            with contextlib.suppress(Exception):
                await page.ws.close(code=WSCloseCode.GOING_AWAY, message=b"ui stopping")
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        for name in ("close", "shutdown"):
            fn = getattr(self.runner, name, None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(call(fn), 5.0)
                break
        if self._app_runner is not None:
            await self._app_runner.cleanup()
            self._app_runner = None

    async def stop_run_for_exit(self, force: bool = False) -> None:
        """A clean stop (none sent when the run is already stopping: a second one would kill it), then wait
        shutdown_clean_s for it; then a kill. force (a second Ctrl-C): the kill at once."""
        state = await self._runner_state()
        if state not in BUSY_STATES:
            return
        if force:
            log.warning("killing the run")
            with contextlib.suppress(Exception):
                await call(self.runner.stop)
            if not await self._wait_not_busy(self.settings.shutdown_kill_s):
                log.error("the run is still %s after the kill", await self._runner_state())
            return
        if state == "stopping":
            log.info("the run is already stopping: waiting for it (up to %.0f s)", self.settings.shutdown_clean_s)
        else:
            log.info("stopping the run (%s) before exit (up to %.0f s; Ctrl-C again kills it)", state,
                     self.settings.shutdown_clean_s)
            with contextlib.suppress(Exception):
                await call(self.runner.stop)
        if not await self._wait_not_busy(self.settings.shutdown_clean_s):
            if await self._runner_state() not in BUSY_STATES:
                return
            log.warning("the run did not stop in %.0f s: killing it", self.settings.shutdown_clean_s)
            with contextlib.suppress(Exception):
                await call(self.runner.stop)
            if not await self._wait_not_busy(self.settings.shutdown_kill_s):
                log.error("the run is still %s after the kill", await self._runner_state())

    async def _wait_not_busy(self, timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if await self._runner_state() not in BUSY_STATES:
                return True
            await asyncio.sleep(0.2)
        return False

    async def _runner_state(self) -> str | None:
        try:
            st = await call(self.runner.status)
        except Exception:
            return None
        return st.get("state") if isinstance(st, dict) else None

    def _task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("ui task %s failed", task.get_coro().__name__, exc_info=exc)

    # ------------------------------------------------------------------------------------------------ HTTP

    @web.middleware
    async def _local_only(self, request: web.Request, handler):
        """As teleop_collect/ui: the Host check stops DNS rebinding, the Origin check stops another site's script
        (no Origin at all, e.g. curl, is allowed)."""
        host = urlsplit(f"//{request.host}").hostname if request.host else None
        if host not in LOCAL_NAMES:
            raise web.HTTPForbidden(text="the ui answers only http://127.0.0.1")
        origin = request.headers.get("Origin")
        if origin is not None and origin not in {f"http://{name}:{self.port}" for name in LOCAL_NAMES}:
            raise web.HTTPForbidden(text="another site may not use the ui")
        return await handler(request)

    @staticmethod
    def _json(obj: Any, status: int = 200) -> web.Response:
        return web.Response(text=dumps(obj), status=status, content_type="application/json",
                            headers={"Cache-Control": "no-store"})

    def _error(self, status: int, error: str, **extra: Any) -> web.Response:
        return self._json({"ok": False, "error": error, **extra}, status)

    async def _index(self, request: web.Request) -> web.StreamResponse:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            return web.Response(text="sim_deploy.ui: ui/static/index.html is not there yet\n", status=503)
        return web.FileResponse(index, headers={"Cache-Control": "no-store"})

    async def _api_policies(self, request: web.Request) -> web.Response:
        try:
            return self._json(await self.catalog.policies())
        except Exception as e:
            log.exception("catalog.policies failed")
            return self._error(500, f"catalog: {type(e).__name__}: {e}")

    async def _api_evals(self, request: web.Request) -> web.Response:
        try:
            return self._json(await self.catalog.evals())
        except Exception as e:
            log.exception("catalog.evals failed")
            return self._error(500, f"catalog: {type(e).__name__}: {e}")

    async def _api_status(self, request: web.Request) -> web.Response:
        await self._refresh_status()
        return self._json(self.status)

    async def _api_start_pose(self, request: web.Request) -> web.Response:
        """P2: the start pose `--home start` uses (stations/start_pose.yaml), as `scene start-pose --show`."""
        try:
            return self._json(await asyncio.to_thread(start_pose_info))
        except Exception as e:
            log.exception("start pose failed")
            return self._error(500, f"start pose: {type(e).__name__}: {e}")

    async def _api_run(self, request: web.Request) -> web.Response:
        try:
            body = await request.json() if request.can_read_body else None
        except ValueError as e:
            return self._error(400, f"the body is not JSON: {e}")
        req, problems = validate_request(body)
        if problems:
            return self._error(400, "; ".join(problems), problems=problems)
        async with self._run_lock:
            await self._refresh_status()
            state = self.status.get("state")
            if state in BUSY_STATES:
                return self._error(409, f"a run is already {state} (run {self.status.get('run_id')}): stop it "
                                        f"first (POST /api/stop)")
            try:
                catalog = await self.catalog.policies()
            except Exception as e:
                log.exception("catalog.policies failed")
                return self._error(500, f"catalog: {type(e).__name__}: {e} (cannot check the policy)")
            target, why = resolve_target(req["policy"], catalog)
            if why is not None:
                return self._error(400, why)
            req["target"] = target
            try:
                run_id = await call(self.runner.start, req)
            except ValueError as e:
                return self._error(400, str(e))
            except Exception as e:
                await self._refresh_status()
                if self.status.get("state") in BUSY_STATES and self.status.get("run_id") is not None:
                    return self._error(409, f"{e}")
                log.exception("runner.start failed")
                return self._error(500, f"runner: {type(e).__name__}: {e}")
            log.info("run %s: %s x%d (seed %d, tag %s)", run_id, target, req["rollouts"], req["seed"], req["tag"])
            await self._refresh_status()
            self._broadcast_status()
            return self._json({"ok": True, "run_id": None if run_id is None else str(run_id)})

    async def _api_stop(self, request: web.Request) -> web.Response:
        if request.can_read_body:
            with contextlib.suppress(Exception):
                await request.read()
        try:
            await call(self.runner.stop)
        except Exception as e:
            log.exception("runner.stop failed")
            return self._error(500, f"runner: {type(e).__name__}: {e}")
        await self._refresh_status()
        self._broadcast_status()
        return self._json({"ok": True, "state": self.status.get("state")})

    async def _api_frames(self, request: web.Request) -> web.Response:
        """{"eval_rel": <eval folder under sim_evals/>} -> `evaltool frames` (runner.frames) -> {"ok", "frames": [url]}."""
        try:
            body = await request.json() if request.can_read_body else None
        except ValueError as e:
            return self._error(400, f"the body is not JSON: {e}")
        rel = body.get("eval_rel") if isinstance(body, dict) else None
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            return self._error(400, "eval_rel: an eval folder under sim_evals/ expected")
        root = self._evals_root
        try:
            d = (root / rel).resolve()
        except (OSError, RuntimeError):
            return self._error(400, "bad eval_rel")
        if root not in d.parents or not (d / "eval.yaml").is_file():
            return self._error(404, f"no eval folder {rel} under sim_evals/")
        fn = getattr(self.runner, "frames", None)
        if fn is None:
            return self._error(501, "this runner cannot make frames")
        try:
            paths = await asyncio.to_thread(fn, d)
        except Exception as e:
            return self._error(500, str(e))
        urls = []
        for p in paths:
            with contextlib.suppress(ValueError, OSError):
                urls.append("/evals/" + quote(str(Path(p).resolve().relative_to(root))))
        return self._json({"ok": True, "frames": urls})

    # ---------------------------------------------------------------------------------------------- /evals/

    async def _evals_root_redirect(self, request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/evals/")

    async def _evals_file(self, request: web.Request) -> web.StreamResponse:
        tail = request.match_info["tail"]
        root = self._evals_root
        if "\x00" in tail or "\\" in tail:
            raise web.HTTPBadRequest(text="bad path")
        try:
            path = (root / tail).resolve()
        except (OSError, RuntimeError):
            raise web.HTTPNotFound() from None
        if path != root and root not in path.parents:
            raise web.HTTPForbidden(text="outside sim_evals/")
        if path.is_dir():
            if tail and not tail.endswith("/"):
                raise web.HTTPFound(f"/evals/{quote(tail)}/")
            return web.Response(text=self._listing(path, tail), content_type="text/html",
                                headers={"Cache-Control": "no-store"})
        if not path.is_file():
            raise web.HTTPNotFound(text=f"no {tail} in sim_evals/")
        headers = {"Cache-Control": "no-cache"}
        ctype = TEXT_TYPES.get(path.suffix.lower())
        if ctype:
            headers["Content-Type"] = ctype
        return web.FileResponse(path, headers=headers)

    def _listing(self, folder: Path, tail: str) -> str:
        rows = []
        if tail:
            rows.append('<li><a href="../">../</a></li>')
        for p in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            name = p.name + ("/" if p.is_dir() else "")
            rows.append(f'<li><a href="{quote(name)}">{html.escape(name)}</a></li>')
        title = html.escape(f"sim_evals/{tail}")
        return (f"<!doctype html><meta charset=utf-8><title>{title}</title>"
                f"<style>body{{font:14px system-ui,sans-serif;margin:16px}}"
                f"@media(prefers-color-scheme:dark){{body{{background:#15171a;color:#ddd}}a{{color:#8cf}}}}</style>"
                f"<h3>{title}</h3><ul>{''.join(rows)}</ul>")

    # ------------------------------------------------------------------------------------------- WebSocket

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=1 << 16)
        await ws.prepare(request)
        sock = request.transport.get_extra_info("socket") if request.transport is not None else None
        if sock is not None:
            # a fixed, small kernel send buffer (Linux would grow it to MBs): a page that does not read makes the
            # writer wait, so its slots are overwritten (dropped) instead of seconds of frames queueing in the kernel
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, WS_SNDBUF)
        page = Page(id=self._next_page, ws=ws)
        self._next_page += 1
        self.pages[page.id] = page
        log.info("page %d connected (%d open)", page.id, len(self.pages))
        writer = asyncio.get_running_loop().create_task(self._writer(page))
        try:
            page.push_status(self._status_text())
            for ch, data in self.latest.items():
                page.push_frame(ch, data)
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self._on_page_message(page, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.pages.pop(page.id, None)
            writer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await writer
            log.info("page %d closed (%d open; %d frames dropped for it)", page.id, len(self.pages),
                     page.dropped_frames)
        return ws

    async def _writer(self, page: Page) -> None:
        ws = page.ws
        while not ws.closed:
            await page.wake.wait()
            page.wake.clear()
            while not ws.closed:
                if page.events:
                    await ws.send_str(page.events.popleft())
                elif page.status is not None:
                    text, page.status = page.status, None
                    await ws.send_str(text)
                elif page.telemetry is not None:
                    text, page.telemetry = page.telemetry, None
                    await ws.send_str(text)
                elif page.frames:
                    ch = next(iter(page.frames))
                    data = page.frames.pop(ch)
                    await ws.send_bytes(bytes([ch]) + data)
                else:
                    break

    def _on_page_message(self, page: Page, text: str) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            msg = None
        if not isinstance(msg, dict):
            return self._page_error(page, "not a JSON object")
        if msg.get("type") != "subscribe":
            return self._page_error(page, f"unknown message type {msg.get('type')!r} (only 'subscribe'; control "
                                          f"is REST)")
        chans = msg.get("channels", sorted(ALL_CHANNELS))
        if not isinstance(chans, list) or any(channel_id(c) is None for c in chans):
            return self._page_error(page, f"subscribe: channels must be a list of 1..4 (or names "
                                          f"{', '.join(CHANNELS)}), got {chans!r}")
        page.channels = {channel_id(c) for c in chans}
        for ch in list(page.frames):
            if ch not in page.channels:
                page.frames.pop(ch)
        for ch in page.channels:
            if ch in self.latest and ch not in page.frames:
                page.push_frame(ch, self.latest[ch])
        log.info("page %d subscribes to %s", page.id, sorted(page.channels))

    def _page_error(self, page: Page, text: str) -> None:
        log.warning("page %d: %s", page.id, text)
        page.push_event(dumps({"type": "error", "text": text}))

    # ----------------------------------------------------------------------------------------------- pumps

    def _status_text(self) -> str:
        return dumps({"type": "status", **self.status})

    def _broadcast_status(self) -> None:
        text = self._status_text()
        for page in list(self.pages.values()):
            page.push_status(text)

    def _broadcast_event(self, event: Any) -> None:
        if not isinstance(event, dict):
            event = {"kind": "error", "text": f"runner sent a non-dict event: {event!r}"}
        text = dumps({**event, "type": "event"})
        for page in list(self.pages.values()):
            page.push_event(text)

    async def _refresh_status(self) -> None:
        try:
            st = await call(self.runner.status)
        except Exception as e:
            msg = f"runner.status failed: {type(e).__name__}: {e}"
            if msg != self._runner_error:
                log.exception("%s", msg)
            self._runner_error = msg
            return
        if not isinstance(st, dict):
            self._runner_error = f"runner.status returned {type(st).__name__}"
            return
        self._runner_error = None
        st = dict(st)
        st.pop("type", None)
        tail = st.get("log_tail")
        if isinstance(tail, (list, tuple)):
            st["log_tail"] = list(tail)[-self.settings.log_tail:]
        self.status = st

    async def _sync_session_dir(self) -> None:
        fn = getattr(self.runner, "session_dir", None)
        setter = getattr(self.live, "set_session_dir", None)
        if fn is None or setter is None:
            return
        try:
            sd = await call(fn)
        except Exception:
            return
        if sd != self._session_dir:
            self._session_dir = sd
            log.info("session folder: %s", sd)
            with contextlib.suppress(Exception):
                await call(setter, sd)

    async def _status_pump(self) -> None:
        period = 1.0 / self.settings.status_hz
        while True:
            started = time.monotonic()
            await self._refresh_status()
            await self._sync_session_dir()
            if self._events_poll:
                await self._poll_events_list()
            self._broadcast_status()
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    async def _telemetry_pump(self) -> None:
        fn = getattr(self.live, "telemetry", None)
        if fn is None:
            log.warning("live has no telemetry(): no telemetry messages")
            return
        period = 1.0 / self.settings.telemetry_hz
        failing = False
        while True:
            started = time.monotonic()
            if self.pages:
                try:
                    t = await call(fn)
                    failing = False
                except Exception:
                    if not failing:
                        log.exception("live.telemetry failed")
                    failing, t = True, None
                if isinstance(t, dict) and t:
                    text = dumps({**t, "type": "telemetry"})
                    for page in list(self.pages.values()):
                        page.push_telemetry(text)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    def _on_pushed_frame(self, channel: Any, jpeg: bytes) -> None:
        ch = channel_id(channel)
        if ch is not None and isinstance(jpeg, (bytes, bytearray, memoryview)):
            with self._pushed_lock:
                self._pushed[ch] = bytes(jpeg)

    async def _frame_pump(self) -> None:
        fn = getattr(self.live, "latest_frames", None)
        if fn is None and not hasattr(self.live, "subscribe"):
            log.warning("live has neither latest_frames() nor subscribe(): no frames")
            return
        period = 1.0 / self.settings.frame_fps
        failing = False
        while True:
            started = time.monotonic()
            if self.pages:
                frames: dict = {}
                if fn is not None:
                    try:
                        frames = await call(fn) or {}
                        failing = False
                    except Exception:
                        if not failing:
                            log.exception("live.latest_frames failed")
                        failing = True
                else:
                    with self._pushed_lock:
                        frames, self._pushed = self._pushed, {}
                if isinstance(frames, dict):
                    for key, value in frames.items():
                        self._offer_frame(key, value)
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    def _offer_frame(self, key: Any, value: Any) -> None:
        ch = channel_id(key)
        if ch is None or value is None:
            return
        if isinstance(value, (tuple, list)) and len(value) == 2:
            seq, data = value
            mark: Any = ("seq", seq)
        else:
            data, mark = value, None
        copied = isinstance(data, (bytearray, memoryview))
        if copied:
            data = bytes(data)
        if not isinstance(data, bytes) or not data:
            return
        prev = self._frame_seen.get(ch)
        if mark is None:
            # same object = the same frame. Equal bytes are NOT skipped: a static scene renders byte-identical JPEGs
            # and must keep arriving (else the page marks it stale). A bytearray / memoryview is copied here, so
            # only for those the content is compared.
            if prev is not None and prev[0] == "bytes" and (prev[1] is data or (copied and prev[1] == data)):
                return
            mark = ("bytes", data)
        elif prev == mark:
            return
        self._frame_seen[ch] = mark
        self.latest[ch] = data
        for page in list(self.pages.values()):
            page.push_frame(ch, data)

    # events: one pump for whatever runner.events() returns
    _events_poll = False

    async def _event_pump(self) -> None:
        fn = getattr(self.runner, "events", None)
        if fn is None:
            log.warning("runner has no events(): no event messages")
            return
        source = await call(fn)
        if isinstance(source, (list, tuple)):
            for ev in source:
                self._broadcast_event(ev)
            self._events_poll = True                      # the status pump calls events() again
            return
        if isinstance(source, asyncio.Queue):
            while True:
                self._broadcast_event(await source.get())
        elif hasattr(source, "__aiter__"):
            async for ev in source:
                self._broadcast_event(ev)
        elif hasattr(source, "get") and hasattr(source, "put"):         # queue.Queue (or alike)
            def get():
                try:
                    return True, source.get(timeout=0.5)
                except queue_mod.Empty:
                    return False, None
            while True:
                got, ev = await asyncio.to_thread(get)
                if got:
                    self._broadcast_event(ev)
        elif hasattr(source, "__next__") or hasattr(source, "__iter__"):
            it = iter(source)
            done = object()
            while True:
                ev = await asyncio.to_thread(next, it, done)
                if ev is done:
                    log.info("runner.events() ended")
                    return
                self._broadcast_event(ev)
        else:
            log.warning("runner.events() returned %s: no event messages", type(source).__name__)

    async def _poll_events_list(self) -> None:
        try:
            evs = await call(self.runner.events)
        except Exception:
            return
        for ev in evs or ():
            self._broadcast_event(ev)


# ---------------------------------------------------------------------------------------------------------- app

def run(settings: Settings, stub: Iterable[str] = ()) -> int:
    """Run until SIGINT / SIGTERM; the run is stopped (cleanly, then killed) before the server ends. A second
    Ctrl-C kills the run at once."""
    import signal

    async def main() -> int:
        runner, live, catalog = load_parts(settings, stub)
        server = UiServer(settings, runner, live, catalog)
        await server.start()
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        hits = {"n": 0}

        def on_signal(signum: int) -> None:
            hits["n"] += 1
            log.info("%s: stopping%s", signal.Signals(signum).name, " (again: killing the run)" if hits["n"] > 1
                     else "")
            if hits["n"] > 1:
                loop.create_task(server.stop_run_for_exit(force=True))
            stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal, sig)
        print(f"sim_deploy.ui: http://{HTTP_HOST}:{server.port}"
              f"{'  (stub: ' + ', '.join(sorted(stub)) + ')' if stub else ''}", flush=True)
        await stop.wait()
        await server.shutdown()
        log.info("stopped")
        return 0

    return asyncio.run(main())
