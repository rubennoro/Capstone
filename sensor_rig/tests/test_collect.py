"""T2: ``sensor-rig collect`` (session flow on mock backends, SimClock), the new live-abort rules, the push button."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from sensor_rig.drivers.base import CameraBackend, Frame, SimClock
from sensor_rig.drivers.mock_imu import MockImu
from sensor_rig.raw_episode import RawEpisode
from sensor_rig.record.button import ButtonEdgeDecoder, FakeButton, button_config
from sensor_rig.record.collect import (
    CollectOptions, CollectSession, InputSource, check_calibration, parse_key, signal_guard,
)
from sensor_rig.record.monitor import LiveAbortConfig, evaluate_live_abort
from sensor_rig.record.recorder import Recorder

SPEED = 4.0
S = 1_000_000_000


class ControlledCamera(CameraBackend):
    """Small frames at ``fps`` on the (sim) clock; ``stall`` pauses delivery, ``die`` makes the producer raise."""

    def __init__(self, name, w, h, fps, clock):
        super().__init__(name, w, h, fps, clock=clock)
        self.stall = threading.Event()
        self.die = threading.Event()
        self.starts = 0
        self._img = np.full((h, w, 3), 90, np.uint8)

    def start(self):
        self.starts += 1
        super().start()

    def _run(self):
        period = int(S / self.fps)
        t = self.clock.now_ns()
        seq = 0
        while not self._stopping.is_set():
            t += period
            self.clock.sleep_until(t)
            if self.die.is_set():
                raise IOError(f"{self.name}: simulated CSI failure")
            if self.stall.is_set():
                continue
            self._push(Frame(self._img, t, seq, self.clock.now_ns()))
            seq += 1


class CountingRecorder(Recorder):
    opens = 0

    def open(self):
        type(self).opens += 1
        return super().open()


class ScriptedInput(InputSource):
    """Steps: (delay_sim_s, cmd) where cmd is a key command, ("signal", SIG), or ("call", fn). The delay counts from
    the moment the session asks for input again after the previous step (so slow stops don't eat it)."""

    def __init__(self, clock, steps):
        self.clock = clock
        self.steps = list(steps)
        self._t0 = None

    def get(self, timeout_s):
        if not self.steps:
            time.sleep(timeout_s)
            return "quit"
        delay, cmd = self.steps[0]
        now = self.clock.now_ns()
        if self._t0 is None:
            self._t0 = now
        if now - self._t0 < delay * S:
            time.sleep(min(timeout_s, 0.02))
            return None
        self.steps.pop(0)
        self._t0 = None
        if isinstance(cmd, tuple) and cmd[0] == "signal":
            os.kill(os.getpid(), cmd[1])
            time.sleep(0.5)            # the handler raises in this (main) thread
            return None
        if isinstance(cmd, tuple) and cmd[0] == "call":
            cmd[1]()
            return None
        return cmd

    def put(self, cmd):
        self.steps.insert(0, (0, cmd))


@pytest.fixture
def small_rig(test_rig):
    cams = dict(test_rig.cameras)
    cams["main"] = dataclasses.replace(cams["main"], resolution=(96, 72), fps=30.0)
    cams["ultrawide"] = dataclasses.replace(cams["ultrawide"], resolution=(96, 72), fps=30.0)
    return test_rig.replace(cameras=cams)


class Harness:
    def __init__(self, rig, root, steps, live_abort=None, opts=None):
        self.clock = SimClock(speed=SPEED)
        self.cams = {n: ControlledCamera(n, *rig.cameras[n].resolution, rig.cameras[n].fps, self.clock)
                     for n in ("main", "ultrawide")}
        self.imu = MockImu.static(rate_hz=1000, clock=self.clock)
        la = live_abort or LiveAbortConfig(all_cameras_fps=True, max_frame_gap_s=0.25)
        CountingRecorder.opens = 0
        self.rec = CountingRecorder(rig, root, cameras=self.cams, imu=self.imu, clock=self.clock, warmup_s=1.2,
                                    monitor_period_s=0.5, keep_aborted=True, live_abort=la)
        self.lines: list[str] = []
        self.inp = ScriptedInput(self.clock, steps)
        o = opts or CollectOptions(session="t2sess", task_label="block_raise")
        o.tag_check = False          # flat test images have no finger tags (the tag check has its own tests)
        self.sess = CollectSession(rig, root, o, self.rec, self.inp, out=self.lines.append)
        self.root = Path(root)

    def run(self):
        with signal_guard():
            rc = self.sess.run()
        self.summary = json.loads(self.sess.summary_path.read_text())
        return rc

    def text(self):
        return "\n".join(self.lines)


def visible(root):
    return sorted(p.name for p in root.glob("20*/*") if not p.name.startswith("."))


def hidden(root):
    return sorted(p.name for p in root.glob("20*/.*"))


def assert_clean(root):
    assert not [p for p in root.glob("20*/.*.recording")], "a .recording folder was left behind"


CAL = [(0.2, "toggle"), (3.5, "toggle")]


def demo(seconds=2.0):
    return [(0.2, "toggle"), (seconds, "toggle")]


# ------------------------------------------------------------------------------------------------ session flow

def test_cal_then_three_demos(small_rig, tmp_path):
    h = Harness(small_rig, tmp_path, CAL + demo() + demo() + demo() + [(0.1, "quit")])
    assert h.run() == 0
    names = visible(tmp_path)
    cals = [n for n in names if "grippercalibration" in n]
    demos = [n for n in names if "_demonstration_" in n]
    assert len(cals) == 1 and len(demos) == 3, h.text()
    assert all(n.endswith("_t2sess_demonstration_left") for n in demos)
    for n in names:
        ep = RawEpisode(tmp_path / n[:10] / n)
        assert ep.meta["status"] == "complete" and ep.meta["side"] == "left" and ep.meta["gripper_id"] == "default"
        if ep.kind == "demonstration":
            assert ep.meta["gripper_calibration_run_name"] == cals[0]
            assert ep.meta["task_label"] == "block_raise"
    state = json.loads((tmp_path / ".sensor_rig_state.json").read_text())
    assert state == {"t2sess|left|default": cals[0]}
    s = h.summary
    assert s["counts"]["demos_kept"] == 3 and s["counts"]["calibrations_kept"] == 1 and s["end_reason"] == "quit"
    assert [e["status"] for e in s["episodes"]] == ["kept"] * 4
    assert CountingRecorder.opens == 1 and all(c.starts == 1 for c in h.cams.values())   # sensors opened once
    assert "CAL KEPT" in h.text() and "DEMO KEPT" in h.text() and "demos kept 3" in h.text()
    assert_clean(tmp_path)


def test_first_episode_is_always_calibration(small_rig, tmp_path):
    # the operator presses Enter for a "demo" first: collect records the calibration instead; no demo before it
    h = Harness(small_rig, tmp_path, demo(1.0) + [(0.1, "quit")])
    h.run()
    eps = h.summary["episodes"]
    assert [e["kind"] for e in eps] == ["gripper_calibration"]
    assert not [n for n in visible(tmp_path) if "_demonstration_" in n]


def test_failed_calibration_blocks_demos(small_rig, tmp_path):
    steps = [(0.2, "toggle"), (1.0, "toggle")] + demo(1.0) + CAL + demo() + [(0.1, "quit")]
    h = Harness(small_rig, tmp_path, steps)
    h.run()
    kinds = [(e["kind"], e["status"]) for e in h.summary["episodes"]]
    # short cal rejected -> the next Enter records a calibration again (rejected too: 1 s) -> good cal -> demo
    assert kinds == [("gripper_calibration", "rejected"), ("gripper_calibration", "rejected"),
                     ("gripper_calibration", "kept"), ("demonstration", "kept")], h.text()
    assert len([n for n in hidden(tmp_path) if n.endswith(".rejected")]) == 2
    good = [e["name"] for e in h.summary["episodes"] if e["status"] == "kept" and e["kind"] != "demonstration"][0]
    d = [e for e in h.summary["episodes"] if e["kind"] == "demonstration"][0]
    assert d["calibration_run"] == good
    assert "too short" in h.summary["episodes"][0]["reason"]
    assert json.loads((tmp_path / ".sensor_rig_state.json").read_text())["t2sess|left|default"] == good


def test_discard_keeps_folder_hidden(small_rig, tmp_path):
    h = Harness(small_rig, tmp_path, CAL + demo() + [(0.1, "discard"), (0.1, "quit")])
    h.run()
    demo_ep = [e for e in h.summary["episodes"] if e["kind"] == "demonstration"][0]
    assert demo_ep["status"] == "discarded"
    hid = tmp_path / demo_ep["name"][:10] / f".{demo_ep['name']}.discarded"
    assert hid.is_dir() and RawEpisode(hid).meta["status"] == "complete"     # never deleted
    assert not [n for n in visible(tmp_path) if "_demonstration_" in n]


def test_discarding_the_calibration_requires_a_new_one(small_rig, tmp_path):
    h = Harness(small_rig, tmp_path, CAL + [(0.1, "discard")] + demo(1.0) + [(0.1, "quit")])
    h.run()
    kinds = [(e["kind"], e["status"]) for e in h.summary["episodes"]]
    assert kinds[0] == ("gripper_calibration", "discarded") and kinds[1][0] == "gripper_calibration"
    assert "t2sess|left|default" not in json.loads((tmp_path / ".sensor_rig_state.json").read_text())


def test_abort_mid_episode(small_rig, tmp_path):
    h = Harness(small_rig, tmp_path, CAL + [(0.2, "toggle"), (1.0, "abort"), (0.1, "quit")])
    assert h.run() == 0
    ab = h.summary["episodes"][-1]
    assert ab["status"] == "aborted" and ab["reason"] == "aborted by operator"
    assert Path(ab["path"]).name == f".{ab['name']}.aborted" and Path(ab["path"]).is_dir()
    assert_clean(tmp_path)


def test_recalibrate_key(small_rig, tmp_path):
    h = Harness(small_rig, tmp_path, CAL + demo() + [(0.1, "calibrate")] + CAL + demo() + [(0.1, "quit")])
    h.run()
    eps = h.summary["episodes"]
    assert [e["kind"] for e in eps] == ["gripper_calibration", "demonstration", "gripper_calibration",
                                        "demonstration"]
    assert eps[1]["calibration_run"] == eps[0]["name"] and eps[3]["calibration_run"] == eps[2]["name"]


@pytest.mark.parametrize("sig,reason", [(signal.SIGTERM, "SIGTERM"), (signal.SIGINT, "SIGINT"),
                                        (signal.SIGHUP, "SIGHUP")])
def test_signal_mid_episode(small_rig, tmp_path, sig, reason):
    h = Harness(small_rig, tmp_path, CAL + [(0.2, "toggle"), (1.0, ("signal", sig))])
    rc = h.run()
    assert rc != 0 and h.summary["end_reason"] == reason
    last = h.summary["episodes"][-1]
    assert last["status"] == "aborted" and reason in last["reason"]
    assert_clean(tmp_path)
    assert len(hidden(tmp_path)) == 1 and hidden(tmp_path)[0].endswith(".aborted")
    assert not any(c.running for c in h.cams.values())          # sensors closed
    assert h.summary["counts"]["calibrations_kept"] == 1


def test_backend_dies_mid_episode(small_rig, tmp_path):
    h = Harness(small_rig, tmp_path, CAL + [(0.2, "toggle"), (1.0, ("call", lambda: h.cams["main"].die.set())),
                                            (2.0, "toggle")])
    rc = h.run()
    assert rc == 1 and h.summary["end_reason"].startswith("sensor_failure")
    last = h.summary["episodes"][-1]
    assert last["status"] == "aborted" and "main_error" in last["reason"] and "simulated CSI failure" in last["reason"]
    assert "SENSOR FAILURE" in h.text()
    assert_clean(tmp_path)


def test_disk_low_aborts(small_rig, tmp_path):
    la = LiveAbortConfig(min_disk_free_gb=1e9, all_cameras_fps=True, max_frame_gap_s=0.25)
    h = Harness(small_rig, tmp_path, [(0.2, "toggle"), (3.0, "toggle"), (0.1, "quit")], live_abort=la)
    h.run()
    ep = h.summary["episodes"][0]
    assert ep["status"] == "aborted" and ep["reason"].startswith("disk_low"), h.text()


def test_uw_stall_triggers_new_rule(small_rig, tmp_path):
    h = Harness(small_rig, tmp_path, CAL + [(0.2, "toggle"), (1.5, ("call", lambda: h.cams["ultrawide"].stall.set())),
                                            (2.5, "quit")])
    h.run()
    ep = h.summary["episodes"][1]
    assert ep["status"] == "aborted", h.text()
    assert ep["reason"].startswith(("frame_gap: ultrawide", "ultrawide_fps_low"))


def test_reuse_calibration(small_rig, tmp_path):
    Harness(small_rig, tmp_path, CAL + [(0.1, "quit")]).run()
    opts = CollectOptions(session="t2sess", reuse_calibration=True)
    h = Harness(small_rig, tmp_path, demo() + [(0.1, "quit")], opts=opts)
    h.run()
    eps = h.summary["episodes"]
    assert [e["kind"] for e in eps] == ["demonstration"] and "reusing calibration" in h.text()


def test_keys_and_calibration_check():
    assert parse_key("\n") == "toggle" and parse_key(" D ") == "discard" and parse_key("x") == "unknown:x"
    v = {"duration_s": 5.0, "imu_rate": 1000.0, "imu_rate_nominal": 1000.0,
         "cameras": {"main": {"fps": 30.0, "fps_nominal": 30.0, "drops": 0, "frames": 150},
                     "ultrawide": {"fps": 30.0, "fps_nominal": 30.0, "drops": 0, "frames": 150}}}
    assert check_calibration(v) == []
    assert any("ultrawide fps" in p for p in check_calibration({**v, "cameras": {**v["cameras"], "ultrawide": {
        "fps": 20.0, "fps_nominal": 30.0, "drops": 0, "frames": 100}}}))
    assert any("dropped" in p for p in check_calibration({**v, "cameras": {**v["cameras"], "main": {
        "fps": 30.0, "fps_nominal": 30.0, "drops": 2, "frames": 148}}}))
    assert any("IMU" in p for p in check_calibration({**v, "imu_rate": 800.0}))


# ------------------------------------------------------------------------------------------------ monitor rules

def _snap(uw_times, now, main_fps=30.0):
    main = (now - np.round(np.arange(int(3 * main_fps))[::-1] * S / main_fps)).astype(np.int64)
    return {"elapsed_s": 3.0, "now_ns": now, "disk_free_gb": 100.0,
            "cameras": {"main": {"fps_nominal": 30.0, "is_main": True, "frame_times": main,
                                 "queue_full_times": np.zeros(0, np.int64)},
                        "ultrawide": {"fps_nominal": 30.0, "is_main": False, "frame_times": uw_times,
                                      "queue_full_times": np.zeros(0, np.int64)}},
            "imu": None}


def test_monitor_uw_rules_opt_in():
    now = 20 * S
    stalled = (now - S - np.round(np.arange(60)[::-1] * S / 30)).astype(np.int64)     # nothing in the last 1 s
    old = LiveAbortConfig()
    assert evaluate_live_abort(_snap(stalled, now), old) is None                         # old behaviour unchanged
    assert evaluate_live_abort(_snap(stalled, now), LiveAbortConfig(all_cameras_fps=True)).startswith(
        "ultrawide_fps_low")
    assert evaluate_live_abort(_snap(stalled, now), LiveAbortConfig(max_frame_gap_s=0.25)).startswith(
        "frame_gap: ultrawide")
    # a single 0.4 s hole in an otherwise healthy stream: fps rule passes, gap rule fires
    t = (now - np.round(np.arange(90)[::-1] * S / 30)).astype(np.int64)
    t = np.concatenate([t[:60], t[72:]])
    assert evaluate_live_abort(_snap(t, now), LiveAbortConfig(all_cameras_fps=True)) is None
    assert evaluate_live_abort(_snap(t, now), LiveAbortConfig(max_frame_gap_s=0.25)).startswith("frame_gap")
    healthy = (now - np.round(np.arange(90)[::-1] * S / 30)).astype(np.int64)
    assert evaluate_live_abort(_snap(healthy, now), LiveAbortConfig(all_cameras_fps=True, max_frame_gap_s=0.25)) is None
    with pytest.raises(ValueError):
        LiveAbortConfig.from_dict({"max_frame_gap": 1})


# ------------------------------------------------------------------------------------------------ button

def test_button_decoder():
    d = ButtonEdgeDecoder(long_press_s=1.0, debounce_s=0.03)
    seq = [(0.00, True), (0.01, False), (0.02, True), (0.06, True), (0.30, True), (0.31, False), (0.35, False)]
    ev = [d.update(p, t) for t, p in seq]
    assert [e for e in ev if e] == ["toggle"]                         # bounce ignored, one short press
    d = ButtonEdgeDecoder(long_press_s=1.0, debounce_s=0.03)
    ev = [d.update(p, t) for t, p in [(0, True), (0.05, True), (1.2, True), (1.5, True), (1.6, False), (1.7, False)]]
    assert [e for e in ev if e] == ["abort"]                          # long press: abort once, no toggle on release


def test_fake_button_thread():
    events = []
    b = FakeButton(events.append, long_press_s=0.4)
    b.start()
    try:
        b.press(0.1)
        time.sleep(0.1)
        b.press(0.6)
        time.sleep(0.1)
    finally:
        b.stop()
    assert events == ["toggle", "abort"]


def test_button_config(small_rig):
    assert button_config(small_rig, None) is None
    assert button_config(small_rig, "gpio:12") == {"pin": 12, "active_low": True, "long_press_s": 1.5}
    with pytest.raises(ValueError):
        button_config(small_rig, "12")
    r = small_rig.replace(capture={**small_rig.capture, "button": {"enabled": True, "pin": 7, "long_press_s": 2}})
    assert button_config(r, None) == {"pin": 7, "active_low": True, "long_press_s": 2.0}


# ------------------------------------------------------------------------------------------------ real CLI process

def test_cli_collect_mock_subprocess_sigterm(tmp_path):
    """The real `sensor-rig collect --mock` (from the repo root): calibration by keyboard, then SIGTERM mid-demo."""
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo / "sensor_rig"), PYTHONUNBUFFERED="1")
    rig = tmp_path / "rig_small.yaml"
    text = (repo / "sensor_rig/sensor_rig/configs/rig_default.yaml").read_text()
    rig.write_text(text.replace("resolution: [1640, 1232]", "resolution: [320, 240]")
                   .replace("819.5, 615.5", "159.5, 119.5"))
    p = subprocess.Popen([sys.executable, "-m", "sensor_rig", "collect", "--mock", "--rig", str(rig), "--out",
                          str(tmp_path / "raw"), "--session", "rigclitest", "--warmup", "1"], cwd=str(tmp_path),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    lines = []
    threading.Thread(target=lambda: [lines.append(ln) for ln in p.stdout], daemon=True).start()

    def wait_for(s, t=30):
        t0 = time.time()
        while time.time() - t0 < t:
            if any(s in ln for ln in lines):
                return
            time.sleep(0.05)
        raise AssertionError(f"{s!r} not seen:\n{''.join(lines)}")

    wait_for("-> Enter: record the GRIPPER CALIBRATION")
    p.stdin.write("\n"); p.stdin.flush()
    wait_for("REC CALIBRATION")
    time.sleep(3.5)
    p.stdin.write("\n"); p.stdin.flush()
    wait_for("CAL KEPT")
    p.stdin.write("\n"); p.stdin.flush()
    wait_for("REC demo")
    time.sleep(1.0)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(30)
    out = "".join(lines)
    assert rc == 128 + signal.SIGTERM, out
    s = json.loads(next((tmp_path / "raw/.sensor_rig/sessions").glob("rigclitest_*.json")).read_text())
    assert s["end_reason"] == "SIGTERM" and [e["status"] for e in s["episodes"]] == ["kept", "aborted"]
    assert not list((tmp_path / "raw").glob("*/.*.recording"))
