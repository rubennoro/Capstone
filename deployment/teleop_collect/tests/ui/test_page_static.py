"""The page, read as text (no browser in the tests; STRUCTURE.md 9.3, 12 row E5): the key map of 4.1 in app.js, the
focus rules, offline (no external URL), the files referenced by index.html exist."""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from teleop_collect.contracts import GRIPPER_COMMANDS, MOTION_KEYS
from teleop_collect.ui.server import COMMANDS, STATIC_DIR

APP = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")

#: STRUCTURE.md 4.1, by KeyboardEvent.code.
EXPECTED_MOTION = {
    "KeyW": "x+", "KeyS": "x-", "KeyA": "y+", "KeyD": "y-", "KeyR": "z+", "KeyF": "z-",
    "KeyU": "twist+", "KeyO": "twist-", "KeyI": "tilt_fwd", "KeyK": "tilt_back", "KeyJ": "tilt_left",
    "KeyL": "tilt_right",
}
EXPECTED_GRIPPER = {"KeyN": "close", "KeyM": "open"}


def js_map(name: str) -> dict[str, str]:
    m = re.search(rf"const {name} = Object\.freeze\(\{{(.*?)\}}\);", APP, re.S)
    assert m, f"{name} not found in app.js"
    return dict(re.findall(r"(\w+):\s*\"([^\"]+)\"", m.group(1)))


def test_motion_key_map_is_structure_4_1():
    assert js_map("MOTION_KEY_MAP") == EXPECTED_MOTION
    assert sorted(EXPECTED_MOTION.values()) == sorted(MOTION_KEYS)


def test_gripper_slow_and_stop_keys():
    assert js_map("GRIPPER_KEY_MAP") == EXPECTED_GRIPPER
    assert set(EXPECTED_GRIPPER.values()) == set(GRIPPER_COMMANDS)
    assert 'const SLOW_KEYS = Object.freeze(["ShiftLeft", "ShiftRight"]);' in APP
    assert 'const STOP_ARM_KEY = "Escape";' in APP
    assert 'command("stop_arm")' in APP


def test_motion_key_order_is_teleops():
    m = re.search(r"const MOTION_KEY_ORDER = \[(.*?)\];", APP, re.S)
    assert m and re.findall(r"\"([^\"]+)\"", m.group(1)) == list(MOTION_KEYS)


def test_key_handling_rules():
    assert "e.repeat" in APP                                   # auto-repeat ignored
    assert 'addEventListener("keydown"' in APP and 'addEventListener("keyup"' in APP
    assert 'addEventListener("blur", loseKeys)' in APP
    assert 'addEventListener("visibilitychange"' in APP
    assert 'held: [], slow: false, focus: false' in APP           # what losing the keys sends
    assert 'state.status.phase === "recording"' in APP             # motion keys only while recording
    assert "keys_resend_ms" in APP


def test_every_button_command_is_known_to_the_server():
    sent = set(re.findall(r'command\("(\w+)"', APP))
    assert sent == set(COMMANDS)


def test_offline_no_external_urls():
    for text in (APP, HTML, CSS):
        assert not re.search(r"https?://", text.replace("ws://${location.host}", ""))
        assert "@import" not in text
    for ref in re.findall(r'(?:src|href)="([^"]+)"', HTML):
        assert ref.startswith("/static/")
        assert (STATIC_DIR / ref[len("/static/"):]).is_file(), ref


def test_page_shows_what_it_must():
    for needle in ("REC ", "start_refusal", "at workspace limit", "teleop not answering", "disconnected",
                   "Save: success", "Save: fail", "Discard", "Stop arm", "Start episode", "Home", "new task",
                   "confirm("):
        assert needle in APP + HTML, needle


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    r = subprocess.run(["node", "--check", str(STATIC_DIR / "app.js")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
