"""The page, read as text (no browser in the tests; STRUCTURE.md 9.3, 12 row E5): the key map of 4.1 in app.js, the
focus rules, offline (no external URL, the font bundled), the files referenced by index.html exist."""
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
THEME_JS = (STATIC_DIR / "theme.js").read_text(encoding="utf-8")

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
    for text in (APP, HTML, CSS, THEME_JS):
        assert not re.search(r"https?://", text.replace("ws://${location.host}", ""))   # also no SVG xmlns
        assert "@import" not in text
    for ref in re.findall(r'(?:src|href)="([^"]+)"', HTML):
        assert ref.startswith("/static/")
        assert (STATIC_DIR / ref[len("/static/"):]).is_file(), ref
    for ref in re.findall(r'url\("?([^")]+)"?\)', CSS):
        assert ref.startswith("/static/")
        assert (STATIC_DIR / ref[len("/static/"):]).is_file(), ref


def test_font_is_bundled_with_its_license():
    font = STATIC_DIR / "fonts" / "ibm-plex-sans-latin-wght-normal.woff2"
    assert font.read_bytes()[:4] == b"wOF2"
    license_text = (STATIC_DIR / "fonts" / "LICENSE.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License" in license_text and "IBM Corp" in license_text
    assert '@font-face' in CSS and 'url("/static/fonts/ibm-plex-sans-latin-wght-normal.woff2")' in CSS
    assert '--font: "IBM Plex Sans", -apple-system' in CSS
    for text in (APP, HTML, CSS):                                   # Inter is gone
        assert not re.search(r"\bInter\b", text) and "inter-latin" not in text
    assert sorted(p.name for p in (STATIC_DIR / "fonts").iterdir()) == ["LICENSE.txt",
                                                                       "ibm-plex-sans-latin-wght-normal.woff2"]


def test_dark_by_default_with_a_theme_toggle():
    assert '<html lang="en" data-theme="dark">' in HTML
    assert HTML.index('src="/static/theme.js"') < HTML.index("</head>")       # set before the first paint
    assert 'let theme = "dark";' in THEME_JS and "try {" in THEME_JS           # storage blocked: dark
    assert ':root, :root[data-theme="dark"] {' in CSS and ':root[data-theme="light"] {' in CSS
    assert "prefers-color-scheme" not in CSS                                 # not the OS setting
    assert '<button id="theme-toggle" type="button"' in HTML                 # a button: blurred on mouseup
    assert '"Light theme"' in APP and '"Dark theme"' in APP and "localStorage.setItem(THEME_KEY" in APP
    assert "toggleTheme" not in APP[APP.index("function onKeyDown"):APP.index("function stopArm")]


def test_typing_in_a_field_does_not_drive():
    """Every text field of the page is an INPUT (task form, storage folder, delete-task confirmation): isTyping
    covers it, so keys typed there never reach the arm (Esc still stops it)."""
    assert "target instanceof HTMLInputElement" in APP and "target instanceof HTMLSelectElement" in APP
    assert "if (isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;" in APP
    assert "contenteditable" not in HTML.lower() + APP.lower()
    for field in ("new-task-name", "new-task-description", "storage-path", "delete-task-input"):
        assert re.search(rf'<input id="{field}"', HTML), field


def test_page_shows_what_it_must():
    for needle in ("REC ", "start_refusal", "at workspace limit", "teleop not answering", "disconnected",
                   "Save: success", "Save: fail", "Discard", "Stop arm", "Start episode", "new task",
                   "state.confirmEpisode", "at the start pose ✓ ready to start",
                   "at the start-up pose ✓ ready to start (the PC checks it at Start)", "start_pose_confirmed"):
        assert needle in APP + HTML, needle
    assert "confirm(" not in APP                                   # inline confirmations, no browser dialog


def test_storage_folder_and_delete_task():
    assert 'command("set_data_root", { path })' in APP and "data_root" in APP
    assert 'command("delete_task", { name })' in APP
    # Delete task is enabled only when the typed name is exactly the task's
    assert '$("delete-task-input").value !== task' in APP
    assert "Type the task name to confirm" in APP


def test_review_rules():
    # Discard asks first; the save choice stays open (keys off) meanwhile
    assert "Yes, discard" in HTML and '$("btn-discard").onclick = () => { state.confirmDiscard = true; render(); };' in APP
    assert '$("btn-discard-yes").onclick = stopAs("discard");' in APP
    # a focused field = keys shown off; forms close when recording starts and on Stop arm / Esc
    assert "!isTyping(document.activeElement)" in APP and "Typing in a field" in APP
    assert 'addEventListener("focusin"' in APP and 'addEventListener("focusout"' in APP
    assert re.search(r'command\("stop_arm"\);\s*closeForms\(\);', APP)
    assert "if (recording && !state.wasRecording) closeForms();" in APP
    # the storage path is shown as it is (no ~ rewriting), only cut in the middle
    assert "/home" not in APP and "Users" not in APP
    # successes are not echoed: only refusals, cleared after a while
    assert "REFUSAL_SHOWN_MS" in APP and "✓`, false" not in APP


def test_no_home_button():
    """No laptop home: homing (h) and the start pose (p) are the PC's keyboard keys."""
    assert "btn-home" not in APP + HTML and ">Home<" not in HTML and '"homing"' not in APP
    assert "home" not in COMMANDS


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    for name in ("app.js", "theme.js"):
        r = subprocess.run(["node", "--check", str(STATIC_DIR / name)], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
