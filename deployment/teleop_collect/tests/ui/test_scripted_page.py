"""ScriptedPage against the real ui server and fake U1/U2 (STRUCTURE.md 9.4): the round trips E7 relies on."""
from __future__ import annotations

import time

import pytest
from ui_helpers import FakeFrames, FakeTeleop, idle_status, running_ui, ui_settings, wait_until

from teleop_collect.contracts import CreateTask, GripperCommand, KeysUpdate, SelectTask, StopEpisode
from teleop_collect.ui.scripted_page import ScriptedPage


@pytest.fixture
def setup(tmp_path):
    teleop, frames = FakeTeleop(), FakeFrames()
    teleop.start()
    frames.start()
    try:
        with running_ui(ui_settings(teleop.endpoint, frames.endpoint, tmp_path)) as ui:
            yield ui, teleop, frames
    finally:
        teleop.stop()
        frames.stop()


def test_a_whole_episode_round_trip(setup):
    ui, teleop, frames = setup
    with ScriptedPage(ui.url, timeout_s=3.0) as page:
        assert page.connected
        s = page.wait_for(lambda s: s["phase"] == "idle" and s["start_allowed"], 2.0, "idle")
        assert s["teleop_answering"] is True
        assert page.command("create_task", name="place_cup", description="cup on plate")["ok"]
        assert page.command("select_task", name="place_cup")["ok"]
        assert teleop.of(CreateTask)[0].description == "cup on plate"
        assert teleop.of(SelectTask)[0].name == "place_cup"
        assert page.command("home")["ok"]
        assert page.command("start_episode")["ok"]
        teleop.status = idle_status(phase="recording", recording_episode="episode_000", episode_s=0.5,
                                    start_allowed=False, start_refusal="recording")
        page.wait_for(lambda s: s["phase"] == "recording", 2.0, "recording")
        page.hold(["x+"], 0.3)
        page.gripper("close")
        page.gripper("open")
        assert page.command("stop_episode", outcome="success")["ok"]
        assert teleop.of(StopEpisode)[0].outcome == "success"
        wait_until(lambda: len(teleop.of(GripperCommand)) == 2, 2.0, "the gripper")
        keys = teleop.of(KeysUpdate)
        assert any(k.held == ("x+",) for k in keys) and keys[-1].held == ()
        wait_until(lambda: set(page.previews()) == {"main", "ultrawide"}, 2.0, "previews")
        assert all(n > 100 for n in page.previews().values())
        assert page.errors() == []


def test_status_before_connect_raises():
    page = ScriptedPage("http://127.0.0.1:1")
    with pytest.raises(RuntimeError):
        page.status()
    assert not page.connected
    page.close()                                             # harmless when never connected


def test_connect_fails_cleanly_without_a_server():
    from station_helpers import free_port
    page = ScriptedPage(f"http://127.0.0.1:{free_port()}", timeout_s=1.0)
    with pytest.raises(Exception):
        page.connect()
    assert not page.connected


def test_wait_for_times_out_with_what(setup):
    ui, _, _ = setup
    with ScriptedPage(ui.url) as page:
        with pytest.raises(TimeoutError, match="no a miracle within 0.2 s"):
            page.wait_for(lambda s: s["phase"] == "miracle", 0.2, "a miracle")
        # a missing key counts as false, not an error
        with pytest.raises(TimeoutError):
            page.wait_for(lambda s: s["no_such_key"], 0.2, "nothing")


def test_hold_rejects_letters(setup):
    ui, _, _ = setup
    with ScriptedPage(ui.url) as page:
        with pytest.raises(ValueError, match="unknown key"):
            page.hold(["W"], 0.1)
        with pytest.raises(ValueError):
            page.gripper("toggle")


def test_blur_and_close(setup):
    ui, teleop, _ = setup
    page = ScriptedPage(ui.url)
    page.connect()
    page.keys(["tilt_fwd"], slow=True)
    wait_until(lambda: teleop.last_keys() and teleop.last_keys().held == ("tilt_fwd",), 2.0, "tilt_fwd")
    page.blur()
    wait_until(lambda: teleop.last_keys().focus is False, 1.0, "the blur")
    page.close()
    assert not page.connected


def test_sees_the_server_go_away(setup, tmp_path):
    teleop, frames = setup[1], setup[2]
    with running_ui(ui_settings(teleop.endpoint, frames.endpoint, tmp_path / "b")) as other:
        page = ScriptedPage(other.url)
        page.connect()
    wait_until(lambda: not page.connected, 2.0, "the close")
    with pytest.raises(ConnectionError):
        page.command("home")
    page.close()


def test_two_pages_at_once(setup):
    ui, _, _ = setup
    with ScriptedPage(ui.url) as a, ScriptedPage(ui.url) as b:
        a.wait_for(lambda s: "phase" in s, 2.0, "a status")
        b.wait_for(lambda s: "phase" in s, 2.0, "b status")
        assert a.command("home")["ok"] and b.command("home")["ok"]
        time.sleep(0.1)
        assert len(ui.app.server.pages) == 2
