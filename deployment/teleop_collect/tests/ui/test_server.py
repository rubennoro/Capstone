"""The ui server against fake U1 / U2 servers (STRUCTURE.md 9.1, 9.2, 12 row E5): the bind, the page, the WebSocket
protocol, the acks, the previews, the deadman, teleop not answering, the ready rule, a contract break."""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import zmq
from deploy_contracts import Ack, encode
from ui_helpers import (FakeFrames, FakeTeleop, UiThread, idle_status, running_ui, ui_settings, wait_until)

from station_helpers import free_port
from teleop_collect.contracts import (CreateTask, DeleteEpisode, DeleteTask, GripperCommand, KeysUpdate, SelectTask,
                                      SetDataRoot, StartEpisode, StopArm, StopEpisode)
from teleop_collect.ui.bridge import FatalError
from teleop_collect.ui.scripted_page import ScriptedPage
from teleop_collect.ui.server import build_command


@pytest.fixture
def teleop():
    t = FakeTeleop()
    t.start()
    yield t
    t.stop()


@pytest.fixture
def frames():
    f = FakeFrames()
    f.start()
    yield f
    f.stop()


@pytest.fixture
def ui(teleop, frames, tmp_path):
    with running_ui(ui_settings(teleop.endpoint, frames.endpoint, tmp_path)) as u:
        yield u


@pytest.fixture
def page(ui):
    p = ScriptedPage(ui.url)
    p.connect()
    yield p
    p.close()


def get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ------------------------------------------------------------------------------------------------------- bind

def _listening(port: int) -> list[str]:
    """Local addresses (hex, as /proc/net/tcp*) listening on this port."""
    out = []
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(path, encoding="ascii").read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            addr, st = parts[1], parts[3]
            if st == "0A" and int(addr.rsplit(":", 1)[1], 16) == port:
                out.append(addr.rsplit(":", 1)[0])
    return out


def test_binds_127_0_0_1_only(teleop, frames, tmp_path):
    port = free_port()
    with running_ui(ui_settings(teleop.endpoint, frames.endpoint, tmp_path, http_port=port)) as u:
        assert u.app.port == port
        assert _listening(port) == ["0100007F"]            # 127.0.0.1, nothing on 0.0.0.0 or IPv6
        # another local address of this machine does not reach it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            assert s.connect_ex(("127.0.0.2", port)) != 0


def test_port_in_use_is_fatal(teleop, frames, tmp_path):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with pytest.raises(FatalError, match="cannot bind the page on 127.0.0.1"):
            UiThread(ui_settings(teleop.endpoint, frames.endpoint, tmp_path, http_port=port)).start()


# ------------------------------------------------------------------------------------------------------- HTTP

def test_serves_the_page_and_its_files(ui):
    status, body = get(ui.url + "/")
    assert status == 200 and b"<script src=\"/static/app.js\">" in body
    for name in ("app.js", "style.css", "index.html"):
        assert get(f"{ui.url}/static/{name}")[0] == 200
    assert get(ui.url + "/nothing")[0] == 404


def test_foreign_host_and_origin_refused(ui):
    assert get(ui.url + "/", {"Host": "evil.example:80"})[0] == 403
    assert get(ui.url + "/", {"Origin": "http://evil.example"})[0] == 403
    assert get(ui.url + "/", {"Origin": ui.url})[0] == 200
    assert get(ui.url + "/", {"Host": f"localhost:{ui.app.port}"})[0] == 200
    assert get(ui.url + "/", {"Origin": f"http://localhost:{ui.app.port}"})[0] == 200


@pytest.mark.parametrize("origin", ["null", "file://", "https://127.0.0.1:{port}", "http://127.0.0.1:{other}",
                                    "http://127.0.0.1", "http://localhost.evil.example:{port}",
                                    "http://127.0.0.1:{port}/x"])
def test_other_origins_refused_page_and_websocket(ui, teleop, origin):
    """A sandboxed iframe or a file:// page sends Origin: null; it must not get the page nor open the WebSocket."""
    import asyncio

    import aiohttp
    origin = origin.format(port=ui.app.port, other=ui.app.port + 1)
    assert get(ui.url + "/", {"Origin": origin})[0] == 403
    assert get(ui.url + "/static/app.js", {"Origin": origin})[0] == 403

    async def try_ws() -> int:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(ui.url + "/ws", origin=origin, timeout=3):
                    return 101
            except aiohttp.WSServerHandshakeError as e:
                return e.status

    teleop.clear()
    assert asyncio.run(try_ws()) == 403
    assert ui.app.server.pages == {}
    assert [type(r).__name__ for r in teleop.requests if type(r).__name__ != "GetStatus"] == []


def test_websocket_with_local_origin_or_none_allowed(ui):
    import asyncio

    import aiohttp

    async def try_ws(origin) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ui.url + "/ws", origin=origin, timeout=3) as ws:
                return not ws.closed

    assert asyncio.run(try_ws(f"http://127.0.0.1:{ui.app.port}"))
    assert asyncio.run(try_ws(None))


# ------------------------------------------------------------------------------------------------- ready rule

def test_ready_only_after_teleop_answered(frames, tmp_path):
    teleop = FakeTeleop()                                    # not started yet: nobody answers U1
    ready = tmp_path / "ready" / "ui.ready"
    with running_ui(ui_settings(teleop.endpoint, frames.endpoint, tmp_path), ready_file=ready) as u:
        time.sleep(0.8)
        assert u.app.bound.is_set() and not ready.exists()
        with ScriptedPage(u.url) as p:
            s = p.wait_for(lambda s: s["teleop_answering"] is False, 2.0, "teleop not answering")
            assert s["teleop_error"] == "teleop not answering" and "phase" not in s
            teleop.start()
            try:
                wait_until(ready.exists, 3.0, "the ready file")
                assert ready.read_text() == "ready\n"
                p.wait_for(lambda s: s["teleop_answering"] and s["phase"] == "idle", 2.0, "teleop answering")
            finally:
                teleop.stop()


def test_teleop_gone_never_ends_the_ui(ui, page, teleop):
    page.wait_for(lambda s: s["teleop_answering"], 2.0, "teleop answering")
    teleop.stop()
    s = page.wait_for(lambda s: not s["teleop_answering"], 3.0, "teleop not answering")
    assert s["teleop_error"] == "teleop not answering"
    assert s["phase"] == "idle"                               # the last status is still shown
    ack = page.command("stop_arm")
    assert ack["ok"] is False and ack["reason"] == "teleop not answering"
    assert ui.thread.is_alive() and ui.error is None
    teleop.start()
    page.wait_for(lambda s: s["teleop_answering"], 3.0, "teleop back")


# ------------------------------------------------------------------------------------------------ the protocol

def test_hello_and_status_at_status_hz(page):
    hello = page.hello
    assert hello["keys_resend_ms"] == 50.0 and hello["keys_stale_ms"] == 150.0
    s = page.wait_for(lambda s: s.get("phase") == "idle", 2.0, "a status")
    for key in ("ready", "task", "tasks", "episodes", "start_allowed", "start_refusal", "tcp_m", "cameras",
                "messages", "final_line", "teleop_answering", "frames_answering", "preview_health"):
        assert key in s, key
    assert s["tcp_m"] == pytest.approx([0.452, 0.003, 0.301])
    assert s["episodes"][0] == {"name": "episode_000", "outcome": "success", "duration_s": 24.1, "frames_main": 723}
    n0 = page._statuses
    time.sleep(1.0)
    assert 10 <= page._statuses - n0 <= 26                    # 20 Hz in these tests


def test_messages_accumulate_by_id(page, teleop):
    teleop.add_message("first")
    page.wait_for(lambda s: [m["text"] for m in s["messages"]] == ["first"], 2.0, "first message")
    teleop.add_message("second", "warn")
    s = page.wait_for(lambda s: len(s["messages"]) == 2, 2.0, "second message")
    assert [(m["id"], m["level"], m["text"]) for m in s["messages"]] == [(1, "info", "first"), (2, "warn", "second")]
    # the ui asks only for the messages after the newest it has
    from teleop_collect.contracts import GetStatus
    wait_until(lambda: teleop.of(GetStatus)[-1].after_message == 2, 1.0, "after_message 2")


def test_status_fields_follow_teleop(page, teleop):
    teleop.status = idle_status(phase="recording", recording_episode="episode_004", episode_s=3.5,
                                start_allowed=False, start_refusal="recording", at_limit=True)
    s = page.wait_for(lambda s: s["phase"] == "recording", 2.0, "recording")
    assert (s["recording_episode"], s["episode_s"], s["at_limit"]) == ("episode_004", 3.5, True)


# ----------------------------------------------------------------------------------------------------- commands

@pytest.mark.parametrize("name,args,cls,fields", [
    ("start_episode", {}, StartEpisode, {}),
    ("stop_arm", {}, StopArm, {}),
    ("select_task", {"name": "pick_block"}, SelectTask, {"name": "pick_block"}),
    ("create_task", {"name": "place_cup", "description": "the cup on the plate"}, CreateTask,
     {"name": "place_cup", "description": "the cup on the plate"}),
    ("stop_episode", {"outcome": "fail"}, StopEpisode, {"outcome": "fail"}),
    ("delete_episode", {"episode": "episode_002"}, DeleteEpisode, {"episode": "episode_002"}),
])
def test_each_command_reaches_teleop(page, teleop, name, args, cls, fields):
    ack = page.command(name, **args)
    assert ack == {"type": "ack", "id": ack["id"], "ok": True, "reason": ""}
    sent = teleop.of(cls)
    assert len(sent) == 1
    assert {k: getattr(sent[0], k) for k in fields} == fields


def test_refusal_reason_comes_back(page, teleop):
    teleop.refuse[StartEpisode] = "not at the start pose: h, then p on the PC"
    ack = page.command("start_episode")
    assert (ack["ok"], ack["reason"]) == (False, "not at the start pose: h, then p on the PC")


def test_acks_carry_their_ids(page):
    a, b = page.command("stop_arm"), page.command("stop_arm")
    assert b["id"] == a["id"] + 1


@pytest.mark.parametrize("name,args,reason", [
    ("fly", {}, "unknown command 'fly'"),
    ("delete_episode", {"episode": "../task"}, "does not match"),
    ("stop_episode", {"outcome": "maybe"}, "outcome"),
    ("select_task", {}, "missing argument 'name'"),
    ("stop_arm", {"now": True}, "unknown argument"),
    ("home", {}, "unknown command 'home'"),                             # no laptop home (the PC's h and p)
])
def test_bad_commands_refused_without_reaching_teleop(page, teleop, name, args, reason):
    teleop.clear()
    ack = page.command(name, **args)
    assert ack["ok"] is False and reason in ack["reason"]
    time.sleep(0.1)
    assert not [r for r in teleop.requests if type(r).__name__ != "GetStatus"]


def test_command_without_integer_id(page):
    page.send_raw({"type": "command", "name": "stop_arm", "id": "x"})
    wait_until(lambda: "x" in page._acks, 2.0, "the ack")
    assert page._acks["x"]["ok"] is False


def test_build_command_covers_every_page_command():
    assert isinstance(build_command("stop_arm", {}), StopArm)
    assert build_command("create_task", {"name": "a"}).description == ""


def test_build_command_delete_task_and_set_data_root():
    assert build_command("delete_task", {"name": "pick_block"}) == DeleteTask(name="pick_block")
    assert build_command("set_data_root", {"path": "~/CapstoneTeleop"}) == SetDataRoot(path="~/CapstoneTeleop")
    for name, args in (("delete_task", {}), ("set_data_root", {}), ("delete_task", {"name": "a", "episode": "b"})):
        with pytest.raises(ValueError):
            build_command(name, args)


def test_stop_arm_does_not_wait_behind_a_slow_command(teleop, frames, tmp_path):
    """stop_arm has its own lane: a slow button in flight does not hold it up."""
    with running_ui(ui_settings(teleop.endpoint, frames.endpoint, tmp_path, rpc_timeout_s=2.0)) as u:
        with ScriptedPage(u.url) as p:
            p.wait_for(lambda s: s["teleop_answering"], 2.0, "teleop")
            ids = []
            t = threading.Thread(target=lambda: ids.append(p.command("create_task", name="x", description="")))
            teleop.delay_s = 0.3
            t.start()
            time.sleep(0.05)
            ack = p.command("stop_arm")
            t.join()
            assert ack["ok"]
            teleop.delay_s = 0.0


# ---------------------------------------------------------------------------------------------------- previews

def test_previews_header_then_jpeg_and_only_new_frames(page, frames):
    wait_until(lambda: len(page.previews()) == 2, 3.0, "both previews")
    first = page.previews()
    wait_until(lambda: page.previews()["main"] > first["main"] + 3, 3.0, "newer previews")
    n = page.previews()["main"]
    assert page.preview_jpeg("main") == frames.jpeg("main", n)
    assert page.preview_jpeg("ultrawide") == frames.jpeg("ultrawide", page.previews()["ultrawide"])
    # every request after the first says which frame the ui already has
    afters = [r.main_after for r in frames.requests]
    assert afters[0] == -1 and afters[-1] > 100
    assert page.errors() == []
    s = page.status()
    assert s["frames_answering"] and s["preview_health"]["main"]["status"] == "ok"


def test_no_preview_requests_without_a_page(ui, frames):
    time.sleep(0.4)
    assert frames.requests == []


def test_frames_gone_shows_and_keeps_running(page, frames, ui):
    page.wait_for(lambda s: s["frames_answering"], 2.0, "frames answering")
    frames.stop()
    s = page.wait_for(lambda s: not s["frames_answering"], 3.0, "frames not answering")
    assert s["frames_error"] == "frames not answering" and ui.thread.is_alive()


# -------------------------------------------------------------------------------------------- keys and deadman

def test_keys_forwarded(page, teleop):
    page.keys(["x+", "tilt_left"], slow=True)
    wait_until(lambda: teleop.last_keys() is not None and teleop.last_keys().held == ("x+", "tilt_left"), 2.0,
               "the keys")
    k = teleop.last_keys()
    assert (k.slow, k.focus) == (True, True)


def test_hold_resends_then_releases(page, teleop):
    page.hold(["z+"], 0.5)
    wait_until(lambda: teleop.last_keys().held == (), 2.0, "the release")
    held = [k for k in teleop.of(KeysUpdate) if k.held == ("z+",)]
    assert 5 <= len(held) <= 14                               # every 50 ms for 0.5 s
    assert teleop.last_keys().focus is True


def test_deadman_on_silence(page, teleop):
    page.keys(["x+"])
    wait_until(lambda: teleop.last_keys() and teleop.last_keys().held == ("x+",), 2.0, "x+")
    t0 = time.monotonic()
    wait_until(lambda: teleop.last_keys().held == (), 1.0, "the deadman")
    dt = time.monotonic() - t0
    k = teleop.last_keys()
    assert (k.slow, k.focus) == (False, False)
    assert 0.1 <= dt <= 0.5                                   # keys_stale_ms 150


def test_deadman_on_close_is_immediate(teleop, frames, tmp_path):
    with running_ui(ui_settings(teleop.endpoint, frames.endpoint, tmp_path, keys_stale_ms=5000.0)) as u:
        p = ScriptedPage(u.url)
        p.connect()
        p.keys(["y-"])
        wait_until(lambda: teleop.last_keys() and teleop.last_keys().held == ("y-",), 2.0, "y-")
        t0 = time.monotonic()
        p.close()
        wait_until(lambda: teleop.last_keys().held == (), 1.0, "the release on close")
        assert time.monotonic() - t0 < 0.5
        assert teleop.last_keys().focus is False


def test_blur_releases_at_once(page, teleop):
    page.keys(["x-"])
    wait_until(lambda: teleop.last_keys() and teleop.last_keys().held == ("x-",), 2.0, "x-")
    page.blur()
    wait_until(lambda: teleop.last_keys().held == () and not teleop.last_keys().focus, 0.3, "the blur")


def test_bad_keys_count_as_released(page, teleop):
    page.keys(["x+"])
    wait_until(lambda: teleop.last_keys() and teleop.last_keys().held == ("x+",), 2.0, "x+")
    page.send_raw({"type": "keys", "held": ["W"], "slow": False, "focus": True})
    wait_until(lambda: teleop.last_keys().held == (), 1.0, "the release")
    wait_until(lambda: page.errors(), 1.0, "an error message")
    assert "keys" in page.errors()[0]


def test_unknown_message_type_answered_with_error(page):
    page.send_raw({"type": "dance"})
    page.send_raw([1, 2])
    wait_until(lambda: len(page.errors()) == 2, 1.0, "two errors")


def test_last_page_with_focus_drives_the_keys(ui, teleop):
    with ScriptedPage(ui.url) as a, ScriptedPage(ui.url) as b:
        a.keys(["x+"])
        wait_until(lambda: teleop.last_keys() and teleop.last_keys().held == ("x+",), 2.0, "a's x+")
        b.keys([])                                            # b took the focus
        wait_until(lambda: teleop.last_keys().held == (), 1.0, "b drives")
        a.blur()                                              # a is not driving: its blur changes nothing
        b.keys(["z-"])
        wait_until(lambda: teleop.last_keys().held == ("z-",), 1.0, "b's z-")
        time.sleep(0.05)
        assert ui.app.server.arbiter.active is not None
        b.blur()
        wait_until(lambda: teleop.last_keys().held == () and not teleop.last_keys().focus, 1.0, "b's blur")


def test_gripper_forwarded(page, teleop):
    page.gripper("close")
    page.gripper("open")
    wait_until(lambda: len(teleop.of(GripperCommand)) == 2, 2.0, "two gripper commands")
    assert [g.command for g in teleop.of(GripperCommand)] == ["close", "open"]
    page.send_raw({"type": "gripper", "command": "toggle"})
    wait_until(lambda: page.errors(), 1.0, "the error")


# --------------------------------------------------------------------------------------------- stop and fatal

def test_stop_releases_keys_and_closes_pages(teleop, frames, tmp_path):
    u = UiThread(ui_settings(teleop.endpoint, frames.endpoint, tmp_path, keys_stale_ms=5000.0)).start()
    p = ScriptedPage(u.url)
    p.connect()
    p.keys(["x+"])
    wait_until(lambda: teleop.last_keys() and teleop.last_keys().held == ("x+",), 2.0, "x+")
    u.stop()
    assert teleop.last_keys().held == ()
    wait_until(lambda: not p.connected, 2.0, "the page closed")
    p.close()
    assert u.error is None


def test_contract_break_from_teleop_is_fatal(frames, tmp_path):
    """teleop answering get_status with something else (a broken contract) ends the ui with a FATAL line."""
    endpoint = f"tcp://127.0.0.1:{free_port()}"
    stop = threading.Event()

    def bad_teleop() -> None:
        sock = zmq.Context.instance().socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(endpoint)
        try:
            while not stop.is_set():
                if sock.poll(10):
                    sock.recv()
                    sock.send(encode(Ack(ok=True, reason="")))
        finally:
            sock.close()

    t = threading.Thread(target=bad_teleop, daemon=True)
    t.start()
    try:
        u = UiThread(ui_settings(endpoint, frames.endpoint, tmp_path))
        u.thread.start()
        u.thread.join(5.0)
        assert not u.thread.is_alive()
        assert isinstance(u.error, FatalError)
        assert "teleop sent a message that breaks its contract" in u.error.what
    finally:
        stop.set()
        t.join(2.0)


def test_status_json_is_plain(page):
    s = page.wait_for(lambda s: "phase" in s, 2.0, "a status")
    json.dumps(s, allow_nan=False)
