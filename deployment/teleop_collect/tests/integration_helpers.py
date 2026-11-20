"""Helpers for the integration tests (tests/integration, STRUCTURE.md 11): `teleop_collect run` as a subprocess, the
laptop's processes started one by one, an in-process fake PC and fake Pi for the failure rows, the real franka_control
with its keyboard on a pipe, the page's button flow, and the checks of an episode folder on disk.

The start pose (2026-11-20, final-deployment's start-pose work, 927a5ba): there is no laptop home. The arm is put at
the start pose with the PC's keys h, then p (FrankaControl.press; the fake PC's operator stand-in under --mock), and
the page's Start waits for it (start() waits until Start is allowed).

Everything binds 127.0.0.1 only. Every station is a temporary one (station_helpers.write_station) whose session_root
and data_root are under the test's tmp_path; the C2/C3 ports are link.yaml's (5620/5630), so a test that needs them
is skipped when they are busy.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import av
import numpy as np
import pytest
import yaml
from deploy_contracts import Ack, LinkSettings, RpcClient, Start
from station_helpers import RECORDED_EPISODE, free_port, write_station

from teleop_collect.contracts import load_config, paths
from teleop_collect.contracts.station import Station
from teleop_collect.ui.scripted_page import ScriptedPage

EPISODE = RECORDED_EPISODE   # station_helpers: the --mock default recording
READY_PREFIX = "teleop_collect: ready: open "
#: Every file of PLAN 6.1 in a saved episode folder (COMPLETE is checked to be the newest).
EPISODE_FILES = ("meta.yaml", "main.mkv", "ultrawide.mkv", "packets.csv", "main_frames.csv", "ultrawide_frames.csv",
                 "robot_states.csv", "commands.csv", "keys.csv", "clock_pi.csv", "clock_pc.csv", "events.jsonl",
                 "COMPLETE")


# ------------------------------------------------------------------------------------------------ skips

def need_episode() -> None:
    if not (EPISODE / "meta.json").is_file():
        pytest.skip(f"no recorded Pi episode at {EPISODE}")


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def need_pc_ports_free() -> None:
    link = LinkSettings.load()
    for port in (link.state_port, link.commands_port):
        if port_busy(port):
            pytest.skip(f"127.0.0.1:{port} (link.yaml) is busy")


def mock_station(tmp_path: Path, **sections) -> Path:
    """A temporary station for --mock runs: free ports, roots under tmp_path, the fake Pi playing EPISODE."""
    frames = {"mock_episode": str(EPISODE), **sections.pop("frames", {})}
    return write_station(tmp_path, sections={"frames": frames, **sections})


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def wait_until(cond: Callable[[], bool], timeout_s: float, what: str, poll_s: float = 0.02) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no {what} within {timeout_s:g} s")
        time.sleep(poll_s)


def descendants(pid: int) -> list[int]:
    """Every process below pid (from /proc/<pid>/task/*/children)."""
    found: list[int] = []
    todo = [pid]
    while todo:
        p = todo.pop()
        try:
            tasks = os.listdir(f"/proc/{p}/task")
        except FileNotFoundError:
            continue
        for t in tasks:
            try:
                kids = Path(f"/proc/{p}/task/{t}/children").read_text().split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            for k in kids:
                found.append(int(k))
                todo.append(int(k))
    return found


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


# ------------------------------------------------------------------------------------------------ the launcher

class LauncherRun:
    """`teleop_collect run --mock --no-browser --station S` as a subprocess, its output collected on a thread."""

    def __init__(self, station: Path, args: Sequence[str] = ("--mock",)) -> None:
        self.station = Path(station)
        self.cmd = [sys.executable, "-m", "teleop_collect", "run", *args, "--no-browser", "--station",
                    str(self.station)]
        self.lines: list[str] = []
        self.url: str | None = None
        self.children: list[int] = []
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def start(self, ready_timeout_s: float = 150.0) -> str:
        """Start it and wait for `teleop_collect: ready: open <url>`; returns the url."""
        self._proc = subprocess.Popen(self.cmd, cwd=paths.folder(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, env=clean_env())
        self._reader = threading.Thread(target=self._read, name="launcher-output", daemon=True)
        self._reader.start()
        deadline = time.monotonic() + ready_timeout_s
        while self.url is None:
            if self._proc.poll() is not None:
                self._reader.join(5)
                raise AssertionError(f"teleop_collect run exited {self._proc.returncode} before ready:\n"
                                     f"{self.output[-4000:]}")
            if time.monotonic() > deadline:
                raise AssertionError(f"teleop_collect run not ready within {ready_timeout_s:g} s:\n"
                                     f"{self.output[-4000:]}")
            time.sleep(0.05)
        self.children = descendants(self._proc.pid)
        return self.url

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            with self._lock:
                self.lines.append(line)
            if line.startswith(READY_PREFIX):
                self.url = line[len(READY_PREFIX):].strip()

    @property
    def output(self) -> str:
        with self._lock:
            return "".join(self.lines)

    @property
    def pid(self) -> int:
        assert self._proc is not None
        return self._proc.pid

    def poll(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    def wait(self, timeout_s: float = 60.0) -> int:
        assert self._proc is not None
        code = self._proc.wait(timeout_s)
        if self._reader is not None:
            self._reader.join(5)
        return code

    def interrupt(self, timeout_s: float = 60.0) -> int:
        """Ctrl-C (SIGINT to the launcher only, as the terminal's Ctrl-C reaches only the launcher's group)."""
        assert self._proc is not None
        self._proc.send_signal(signal.SIGINT)
        return self.wait(timeout_s)

    def kill(self) -> None:
        """Teardown: SIGINT, then SIGKILL the launcher and every process it started."""
        if self._proc is None:
            return
        kids = descendants(self._proc.pid) or self.children
        if self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        for pid in kids:
            if alive(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    def __enter__(self) -> LauncherRun:
        return self

    def __exit__(self, *exc) -> None:
        self.kill()


def session_dir(tmp_path: Path) -> Path:
    found = sorted((Path(tmp_path) / "sessions").iterdir())
    assert len(found) == 1, found
    return found[0]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------------ processes one by one

class InProcessFakePc:
    """The fake PC in the test's own process on link.yaml's ports (127.0.0.1:5620/5630), so a test can call its
    hooks (restart, inject_fault, keyboard_stop, ...) while the laptop's processes run against it."""

    def __init__(self, tmp_path: Path, **overrides) -> None:
        from teleop_collect.fake_pc.fake_pc import FakePc
        from teleop_collect.fake_pc.settings import FakePcSettings
        config = load_config("fake_pc", None)
        config.update(overrides)
        out = Path(tmp_path) / "inproc_fake_pc"
        out.mkdir(parents=True, exist_ok=True)
        self.fake = FakePc(FakePcSettings.from_config(config, out), session_dir=out)

    def __enter__(self):
        self.fake.start()
        assert self.fake.ready.wait(10.0), "the in-process fake PC published no C2"
        return self.fake

    def __exit__(self, *exc) -> None:
        self.fake.close()


class InProcessFakePi:
    """A FakePiSender in the test's process playing EPISODE on 127.0.0.1:<port> (the station's pi.port), for a
    frames process started without --mock. pause()/resume() make a camera go down."""

    def __init__(self, port: int) -> None:
        from teleop_collect.frames.fake_pi_sender import FakePiSender
        from teleop_collect.frames.pi_recording import PiRecording
        from teleop_collect.frames.rig_file import camera_sizes, load_live_rig
        m = load_config("frames", None)["mock"]
        rig = load_live_rig(paths.deployment_dir() / "contracts" / "rigs" / "rig_pi_v3_640.yaml")
        recording = PiRecording.load(EPISODE, max_seconds=m["max_seconds"], jpeg_quality=m["jpeg_quality"],
                                     sizes=camera_sizes(rig))
        self.sender = FakePiSender(recording, f"tcp://127.0.0.1:{port}",
                                   pi_clock_offset_ns=int(m["pi_clock_offset_s"] * 1e9), drift_ppm=m["drift_ppm"],
                                   send_hwm=m["send_hwm"])

    def __enter__(self):
        self.sender.start()
        return self.sender

    def __exit__(self, *exc) -> None:
        self.sender.stop()


class LaptopProcesses:
    """frames, teleop and ui started one by one with the launcher's own ManagedProcess classes (same commands, logs,
    ready files and stop), for rows that need frames without --mock (an in-process fake Pi) and teleop with --mock
    (C2/C3 on 127.0.0.1). Stopped in reverse order."""

    def __init__(self, station_path: Path, frames_mock: bool = False, teleop_mock: bool = True) -> None:
        from teleop_collect.launcher.processes import FramesProcess, TeleopProcess, UiProcess
        from teleop_collect.launcher.session import Session
        station = Station.load(station_path, mock=True)
        self.session = Session.create(station.session_root, True, station_path)
        self.processes = [FramesProcess(station, frames_mock, self.session),
                          TeleopProcess(station, teleop_mock, self.session),
                          UiProcess(station, True, self.session)]
        self.station = station
        self.started: list = []
        self.url = station.http_url()

    def __enter__(self) -> LaptopProcesses:
        timeouts = {"frames": 30.0, "teleop": 150.0, "ui": 15.0}
        try:
            for p in self.processes:
                p.start()
                self.started.append(p)
                p.wait_ready(timeouts[p.NAME], 0.05)
        except BaseException:
            self.stop()
            raise
        return self

    def get(self, name: str):
        return next(p for p in self.processes if p.NAME == name)

    def stop(self) -> dict[str, int | None]:
        codes = {}
        for p in reversed(self.started):
            codes[p.NAME] = p.stop(5.0)
        self.started = []
        return codes

    def __exit__(self, *exc) -> None:
        self.stop()


FRANKA_CONTROL = (Path("~/Capstone-final-deployment/deployment/control_deploy/franka_control/build-nolibfranka/"
                      "franka_control").expanduser())
#: The cross test's homing.q_start_rad: control_deploy's C++ test value (test_fake_arm_run.cpp, KeyP...), i.e.
#: franka_control/config.yaml q_home_rad with joint 6 + 0.45 rad (a 2.8 s joint move from home; inside the Panda's
#: joint limits). The fake arm's TCP ends every joint move at fake_arm.home_O_T_TCP ((0.45, 0, 0.45) m, pointing
#: down), so the start pose's TCP is inside the box (floor z -0.005 @ 927a5ba) and tilted 0 deg from the cone axis.
Q_START_RAD = [0.0, -0.7853982, 0.0, -2.3561945, 0.0, 2.0207963, 0.7853982]
#: The PC's own C3 start refusal away from its start pose (supervisor.cpp onStart, once q_start_rad is set).
PC_NOT_AT_START_POSE_TEXT = "refused: not at the start pose: press h, then p on the PC"
#: The binary's log line at the end of each key's arm move (supervisor.cpp tick).
MOVE_DONE = {"h": "homing finished", "p": "start pose reached"}


class FrankaControl:
    """The real libfranka-less franka_control (rnoro/final-deployment, already built there; only run, never built)
    with `--mock --local --keyboard`: C2/C3 on 127.0.0.1:5620/5630, its keyboard (h, p, g, s) a pipe on stdin (the
    KeyboardReader reads the fd when it is not a terminal too). Its station sets `franka_control: homing: q_start_rad`
    (the start pose: p works and C3 start is refused away from it); station, session and log are under tmp_path and
    its cwd is tmp_path, so nothing is written into ~/Capstone-final-deployment."""

    def __init__(self, tmp_path: Path, q_start_rad: Sequence[float] | None = Q_START_RAD) -> None:
        if not (FRANKA_CONTROL.is_file() and os.access(FRANKA_CONTROL, os.X_OK)):
            pytest.skip(f"no libfranka-less franka_control at {FRANKA_CONTROL}")
        self.dir = Path(tmp_path) / "fc"
        self.dir.mkdir()
        self.station = self.dir / "fc_station.yaml"
        station: dict = {"session_root": str(self.dir / "sessions")}
        if q_start_rad is not None:     # merged key by key into franka_control/config.yaml (config_loader mergeInto)
            station["franka_control"] = {"homing": {"q_start_rad": [float(v) for v in q_start_rad]}}
        self.station.write_text(yaml.safe_dump(station))
        self.ready = self.dir / "fc.ready"
        self.log_path = self.dir / "franka_control.log"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> FrankaControl:
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen([str(FRANKA_CONTROL), "--station", str(self.station), "--mock", "--local",
                                      "--keyboard", "--ready-file", str(self.ready)], stdin=subprocess.PIPE,
                                     stdout=self._log, stderr=subprocess.STDOUT, cwd=self.dir)
        deadline = time.monotonic() + 30.0
        while not self.ready.exists():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.__exit__()
                raise AssertionError(f"franka_control not ready (exit {self.proc.returncode}):\n{self.log()[-3000:]}")
            time.sleep(0.05)
        return self

    def log(self) -> str:
        return self.log_path.read_text() if self.log_path.exists() else ""

    def key(self, c: str) -> None:
        """One key of the PC keyboard (h, p, g or s), as typed in its terminal."""
        assert self.proc is not None and self.proc.stdin is not None and len(c) == 1
        self.proc.stdin.write(c.encode())
        self.proc.stdin.flush()

    def wait_log(self, text: str, after: int, timeout_s: float) -> None:
        """Wait for `text` in the log past offset `after` (len(log()) before the action)."""
        wait_until(lambda: text in self.log()[after:], timeout_s, f"franka_control log line {text!r}")

    def press(self, c: str, timeout_s: float = 15.0) -> float:
        """The key h or p, then wait for the end of its arm move (MOVE_DONE in the log); returns the seconds taken."""
        after = len(self.log())
        t0 = time.monotonic()
        self.key(c)
        try:
            self.wait_log(MOVE_DONE[c], after, timeout_s)
        except TimeoutError as e:
            raise AssertionError(f"key {c}: {e}:\n{self.log()[after:][-3000:]}") from None
        return time.monotonic() - t0

    def start_direct(self) -> Ack:
        """A C3 start from a second client (a REQ next to teleop's; the PC's REP serves both), to see the PC's own
        answer. Used only where the PC must refuse it."""
        client = RpcClient("pc", LinkSettings.load().commands_endpoint(local=True))
        try:
            return client.request(Start(), Ack, 2.0)
        finally:
            client.close()

    def terminate(self, timeout_s: float = 10.0) -> int:
        assert self.proc is not None
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        return self.proc.wait(timeout_s)

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.proc is not None and self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self._log.close()


def final_deployment_status() -> str:
    """`git status --short` of ~/Capstone-final-deployment (read-only: must stay empty)."""
    return subprocess.run(["git", "-C", str(Path("~/Capstone-final-deployment").expanduser()), "status", "--short"],
                          capture_output=True, text=True, check=True).stdout


# ------------------------------------------------------------------------------------------------ the page's flow

def newest_message_id(page: ScriptedPage) -> int:
    msgs = page.status().get("messages") or []
    return max((m["id"] for m in msgs), default=0)


def wait_message(page: ScriptedPage, text: str, after_id: int, timeout_s: float) -> dict:
    """Wait for a page message containing `text` with an id above after_id; returns that status."""
    return page.wait_for(lambda s: any(m["id"] > after_id and text in m["text"] for m in s["messages"]), timeout_s,
                         f"page message {text!r}")


def open_page(url: str) -> ScriptedPage:
    page = ScriptedPage(url, timeout_s=10.0)
    page.connect()
    page.wait_for(lambda s: s["ready"] and s["phase"] == "idle", 30.0, "teleop ready and idle")
    return page


def start(page: ScriptedPage, timeout_s: float = 20.0) -> str:
    """Start episode once it is allowed (the arm back at the start pose: the PC's h, then p); returns the episode
    name."""
    page.wait_start_allowed(timeout_s)
    ack = page.command("start_episode")
    assert ack["ok"], ack
    s = page.wait_for(lambda s: s["phase"] == "recording" and s["recording_episode"], 10.0, "recording")
    return s["recording_episode"]


def wait_stand_in(page: ScriptedPage) -> dict:
    """After an episode's stop under the fake PC's operator stand-in (config/fake_pc.yaml auto_h_p_after_stop_s, 1 s):
    its h, then p (2 s of homing), then Start allowed again at the confirmed start pose. Also after an episode that
    did not move the arm, so the next start never meets the stand-in's h. Returns that status."""
    page.wait_for(lambda s: s["arm_mode"] == "homing", 5.0, "the stand-in's h")
    s = page.wait_start_allowed(10.0)
    assert s["at_start_pose"] and s["start_pose_confirmed"], s
    return s


def stop(page: ScriptedPage, outcome: str) -> dict:
    ack = page.command("stop_episode", outcome=outcome)
    assert ack["ok"], ack
    return page.wait_for(lambda s: s["phase"] == "idle" and s["recording_episode"] == "", 15.0, "idle after stop")


def tcp(page: ScriptedPage) -> list[float]:
    return list(page.status()["tcp_m"])


# ------------------------------------------------------------------------------------------------ files on disk

def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def demux_jpegs(path: Path) -> list[bytes]:
    """Every video packet of an .mkv, as bytes (MJPEG: one JPEG each)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "mjpeg", stream.codec_context.name
        return [bytes(p) for p in container.demux(stream) if p.size > 0]


def check_episode(folder: Path, *, task: str, outcome: str) -> dict:
    """The checks of a saved episode (PLAN 6): every file, COMPLETE the newest, both videos against their CSVs,
    plausible rates, meta.yaml with F_T_TCP. Returns meta.yaml."""
    folder = Path(folder)
    missing = [n for n in EPISODE_FILES if not (folder / n).is_file()]
    assert not missing, f"{folder}: missing {missing}"
    # COMPLETE last, two ways. (1) No file was modified after COMPLETE: the run's filesystems (ext4, tmpfs) keep ns
    # mtimes, and every writer is closed (and meta.yaml fsynced) before COMPLETE is created. (2) Every file is whole
    # and closed: each CSV ends in a newline with every row as wide as its header, every events.jsonl line parses
    # and the last one is the stop, meta.yaml parses to a full map. A file still open or cut off fails (2) even
    # when the mtimes tie.
    complete = (folder / "COMPLETE").stat().st_mtime_ns
    newer = [n for n in EPISODE_FILES if (folder / n).stat().st_mtime_ns > complete]
    assert not newer, f"written after COMPLETE: {newer}"
    assert (folder / "COMPLETE").stat().st_size == 0
    for name in EPISODE_FILES:
        if name.endswith(".csv"):
            text = (folder / name).read_text()
            assert text.endswith("\n"), f"{name}: last line cut off"
            rows = list(csv.reader(text.splitlines()))
            assert len(rows) >= 2, f"{name}: no rows"
            widths = {len(r) for r in rows}
            assert widths == {len(rows[0])}, f"{name}: rows of widths {sorted(widths)}, header {len(rows[0])}"
    events_text = (folder / "events.jsonl").read_text()
    assert events_text.endswith("\n")
    assert all(isinstance(json.loads(line), dict) for line in events_text.splitlines())

    meta = yaml.safe_load((folder / "meta.yaml").read_text())
    assert isinstance(meta, dict) and {"counts", "frames", "pc_hello", "clock", "teleop"} <= set(meta), sorted(meta)
    assert meta["task"] == task and meta["outcome"] == outcome and meta["episode"] == folder.name
    duration = float(meta["duration_s"])
    assert duration > 0.5
    F_T_TCP = meta["pc_hello"]["F_T_TCP"]
    frames_yaml = yaml.safe_load((paths.deployment_dir() / "contracts" / "frames.yaml").read_text())
    assert np.allclose(np.array(F_T_TCP, dtype=float), np.array(frames_yaml["F_T_TCP"], dtype=float), atol=1e-12)

    for camera in ("main", "ultrawide"):
        jpegs = demux_jpegs(folder / f"{camera}.mkv")
        rows = read_csv(folder / f"{camera}_frames.csv")
        assert len(jpegs) == len(rows) == meta["counts"][f"frames_{camera}"], (camera, len(jpegs), len(rows))
        assert 0.7 * 30 * duration <= len(rows) <= 1.2 * 30 * duration + 2, (camera, len(rows), duration)
        for i, (jpeg, row) in enumerate(zip(jpegs, rows)):
            assert int(row["video_index"]) == i, (camera, i, row["video_index"])
            assert len(jpeg) == int(row["bytes"]), (camera, i, len(jpeg), row["bytes"])
            assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9", (camera, i)
        nos = [int(r["frame_no"]) for r in rows]
        assert nos == sorted(nos) and len(set(nos)) == len(nos), camera   # a reused ultrawide is never written twice
    packets = read_csv(folder / "packets.csv")
    assert len(packets) == meta["counts"]["packets"] >= len(read_csv(folder / "main_frames.csv"))

    states = read_csv(folder / "robot_states.csv")
    assert 0.8 * 100 * duration <= len(states) <= 1.2 * 100 * duration + 5, (len(states), duration)
    seqs = [int(r["seq"]) for r in states]
    assert seqs == sorted(seqs)
    commands = read_csv(folder / "commands.csv")
    assert commands and commands[0]["request"] == "c3.start/v1" and commands[-1]["request"] == "c3.stop/v1"
    assert read_csv(folder / "clock_pi.csv") and read_csv(folder / "clock_pc.csv")
    events = [e["kind"] for e in jsonl(folder / "events.jsonl")]
    assert events[0] == "start" and events[-1] == "stop"
    return meta


def hold_speed(folder: Path, key_column: str = "O_T_TCP_target_03") -> dict[str, float]:
    """The commanded speed while the first motion key was held, from a saved episode: the slope of the C2 target
    (`key_column`, default x) against the PC time, fitted over the hold without its first and last 0.2 s, and the
    target's travel from the press to 0.5 s after the release."""
    keys = read_csv(folder / "keys.csv")
    press = next(int(r["t_ns"]) for r in keys if r["held"])
    release = next(int(r["t_ns"]) for r in keys if int(r["t_ns"]) > press and not r["held"])
    states = [r for r in read_csv(folder / "robot_states.csv") if r["t_ns"]]
    t = np.array([int(r["t_ns"]) for r in states], dtype=np.float64)
    t_pc = np.array([int(r["t_pc_ns"]) for r in states], dtype=np.float64)
    x = np.array([float(r[key_column]) for r in states])
    mid = (t > press + 0.2e9) & (t < release - 0.2e9)
    slope = float(np.polyfit(t_pc[mid] / 1e9, x[mid], 1)[0])
    before = x[t <= press][-1]
    after = x[t <= release + 0.5e9][-1]
    return {"slope_m_s": slope, "travel_m": float(after - before), "held_s": (release - press) / 1e9}


def task_dir(tmp_path: Path, task: str) -> Path:
    return Path(tmp_path) / "data_mock" / task          # every integration run is --mock: mock_data_root


def episode_names(tmp_path: Path, task: str) -> list[str]:
    return sorted(p.name for p in task_dir(tmp_path, task).glob("episode_*"))


def next_episode(tmp_path: Path, task: str) -> int:
    return int(yaml.safe_load((task_dir(tmp_path, task) / "task.yaml").read_text())["next_episode"])


__all__ = [name for name in dir() if not name.startswith("_")] + ["free_port"]
