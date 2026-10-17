"""``sensor-rig collect``: the one command for a recording session on the Jetson (T2).

    sensor-rig collect --rig rig.yaml --out /mnt/nvme/raw --session rig1020 [--side left] [--task-label block_raise]
                       [--gripper-id default] [--mock | --mock-replay DIR ...] [--button gpio:PIN]
                       [--reuse-calibration] [--force]

Flow: quick ``doctor`` (refuses on FAIL unless ``--force``) -> sensors opened ONCE for the whole session -> the gripper
calibration episode (required, like the iPhone app; checked before demos are allowed) -> demo loop.

Keys (type the letter, then Enter): Enter start/stop, ``d`` discard the last kept episode, ``a`` abort the running
episode, ``c`` record a new calibration next, ``s`` status, ``q`` quit. An optional push button (short press = Enter,
long press = abort) can be enabled in rig.yaml ``capture.button`` or with ``--button gpio:PIN``.

Nothing is ever deleted: aborted episodes are kept as ``.<NAME>.aborted``, discarded ones are renamed to
``.<NAME>.discarded`` and calibrations that fail the check to ``.<NAME>.rejected``. Every hidden folder is skipped by
pose/export (and by iPhUMI, which only sees exports). SIGINT / SIGTERM / SIGHUP abort the running episode (kept
hidden), close the sensors and still write the session summary ``<out>/.sensor_rig/sessions/<session>_<UTC>.json``.

The raw format, ``Recorder``, ``RawEpisode`` and ``SessionState`` are used as they are; calibration and demos always
share the session, side and gripper ID, and each demo names its calibration explicitly.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CAL_MIN_DURATION_S = 3.0
CAL_MIN_FPS_RATIO = 0.9
CAL_MIN_IMU_RATIO = 0.95
RECORD_MB_S = 30.0
COLLECT_LIVE_ABORT_DEFAULTS = {"all_cameras_fps": True, "max_frame_gap_s": 0.25}
KEYS = {"": "toggle", "d": "discard", "a": "abort", "c": "calibrate", "s": "status", "q": "quit"}
TMUX_TIP = ("tip: run collect inside tmux (tmux new -s rig; reattach with tmux attach -t rig) so an SSH drop does not "
            "end the session")


class SessionInterrupt(BaseException):
    """Raised in the main thread by the SIGTERM / SIGHUP handler (SIGINT raises KeyboardInterrupt)."""

    def __init__(self, signame: str):
        super().__init__(signame)
        self.signame = signame


@contextmanager
def signal_guard():
    """SIGTERM / SIGHUP -> SessionInterrupt in the main thread (SIGINT already raises KeyboardInterrupt)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise SessionInterrupt(signal.Signals(signum).name)

    old = {}
    for s in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if s is not None:
            old[s] = signal.signal(s, handler)
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


# --------------------------------------------------------------------------------------------------------------------
# Input sources
# --------------------------------------------------------------------------------------------------------------------
class InputSource:
    """Delivers commands ('toggle', 'discard', 'abort', 'calibrate', 'status', 'quit', or 'unknown:<text>')."""

    def get(self, timeout_s: float) -> str | None:
        raise NotImplementedError

    def put(self, cmd: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def parse_key(line: str) -> str:
    k = line.strip().lower()
    return KEYS.get(k, f"unknown:{k}")


class QueueInput(InputSource):
    def __init__(self):
        self.q: queue.Queue[str] = queue.Queue()

    def get(self, timeout_s: float) -> str | None:
        try:
            return self.q.get(timeout=max(0.0, timeout_s))
        except queue.Empty:
            return None

    def put(self, cmd: str) -> None:
        self.q.put(cmd)


class StdinInput(QueueInput):
    """Line-based keyboard input from a reader thread; EOF (stdin closed) means quit."""

    def __init__(self, stream=None):
        super().__init__()
        self.stream = stream if stream is not None else sys.stdin
        threading.Thread(target=self._loop, name="stdin", daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                line = self.stream.readline()
            except (OSError, ValueError):
                line = ""
            if line == "":
                self.put("quit")
                return
            self.put(parse_key(line))


# --------------------------------------------------------------------------------------------------------------------
# Episode verdicts
# --------------------------------------------------------------------------------------------------------------------
def dir_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e6


def episode_verdict(path: Path) -> dict:
    """Numbers from a finalized episode's meta.json (and a RawEpisode read-back)."""
    from sensor_rig.raw_episode import RawEpisode

    ep = RawEpisode(path)
    m = ep.meta
    s = m.get("sensors", {})
    d = {"duration_s": None, "cameras": {}, "imu_rate": None, "imu_rate_nominal": None, "mb": dir_size_mb(path)}
    if m.get("record_start_mono_ns") and m.get("record_stop_mono_ns"):
        d["duration_s"] = (m["record_stop_mono_ns"] - m["record_start_mono_ns"]) / 1e9
    for cam in ("main", "ultrawide"):
        c = s.get(cam, {})
        frames = len(ep.camera_times(cam))
        d["cameras"][cam] = {"frames": frames, "fps": c.get("fps_measured"), "fps_nominal": c.get("fps_nominal"),
                             "drops": int(sum((c.get("drops") or {}).values()))}
    imu = s.get("imu0", {})
    d["imu_rate"] = imu.get("rate_measured")
    d["imu_rate_nominal"] = imu.get("odr_hz")
    return d


def check_calibration(v: dict) -> list[str]:
    """Problems with a calibration episode (empty = accepted)."""
    out = []
    if v["duration_s"] is None or v["duration_s"] < CAL_MIN_DURATION_S:
        out.append(f"too short ({v['duration_s'] or 0:.1f} s < {CAL_MIN_DURATION_S:g} s): open and close the fingers "
                   "fully 2-3 times")
    for cam, c in v["cameras"].items():
        nom = c["fps_nominal"]
        if c["fps"] is None or (nom and c["fps"] < CAL_MIN_FPS_RATIO * nom):
            out.append(f"{cam} fps {c['fps'] or 0:.1f} < {CAL_MIN_FPS_RATIO} x {nom}")
        if c["drops"]:
            out.append(f"{cam} dropped {c['drops']} frames")
    nom = v["imu_rate_nominal"]
    if v["imu_rate"] is None or (nom and v["imu_rate"] < CAL_MIN_IMU_RATIO * nom):
        out.append(f"IMU rate {v['imu_rate'] or 0:.0f} < {CAL_MIN_IMU_RATIO} x {nom}")
    return out


def format_verdict(name: str, kind: str, v: dict | None, status: str, reason: str | None = None) -> str:
    label = "CAL" if kind == "gripper_calibration" else "DEMO"
    if v is None:
        return f"{label} {status}: {name}" + (f" ({reason})" if reason else "")
    cams = "  ".join(f"{c} {x['frames']} fr {x['fps'] or 0:.1f} fps drops {x['drops']}" for c, x in v["cameras"].items())
    imu = f"imu {v['imu_rate'] or 0:.0f} Hz"
    line = f"{label} {status}: {v['duration_s'] or 0:.1f} s  {cams}  {imu}  {v['mb']:.0f} MB  {name}"
    return line + (f"  ({reason})" if reason else "")


# --------------------------------------------------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class CollectOptions:
    session: str
    side: str = "left"
    gripper_id: str = "default"
    task_label: str | None = None
    note: str = ""
    reuse_calibration: bool = False
    status_period_s: float = 1.0
    tag_check: bool = True            # ArUco check of the calibration episode (off for synthetic --mock images)


@dataclass
class EpisodeRecord:
    name: str
    kind: str
    status: str                       # kept | aborted | discarded | rejected
    path: str | None = None
    reason: str | None = None
    calibration_run: str | None = None
    verdict: dict | None = None
    problems: list = field(default_factory=list)


class CollectSession:
    def __init__(self, rig, raw_root, opts: CollectOptions, recorder, inp: InputSource,
                 out: Callable[[str], None] | None = None, doctor_summary: dict | None = None,
                 rig_path: str | None = None, replay=None):
        from sensor_rig.timefmt import sanitize_session

        self.rig = rig
        self.raw_root = Path(raw_root)
        self.opts = opts
        self.session = sanitize_session(opts.session)
        self.rec = recorder
        self.inp = inp
        self._out = out or print
        self.doctor_summary = doctor_summary
        self.rig_path = rig_path
        self.episodes: list[EpisodeRecord] = []
        self.calibration: str | None = None        # accepted calibration in use
        self.cal_history: list[str] = []           # accepted calibrations of this session, newest last
        self.want_calibration = False
        self.active: tuple[str, str] | None = None  # (name, kind) of the running episode
        self.end_reason = "quit"
        self.summary_path: Path | None = None
        self.started_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.exit_code = 0
        self.replay = replay                       # drivers.replay_playlist.ReplayPlaylist (collect --mock-replay)

    # ---- output that never raises (SSH hangup closes the terminal) ----
    def out(self, msg: str) -> None:
        try:
            self._out(msg)
        except (OSError, ValueError):
            self._out = lambda m: None

    # ---- helpers ----
    def _free_minutes(self) -> str:
        probe = self.raw_root
        while not probe.exists() and probe != probe.parent:   # the raw root is created by Recorder.open()
            probe = probe.parent
        try:
            du = shutil.disk_usage(probe)
        except OSError:
            return "disk: unknown"
        return f"disk: {du.free / 1e9:.1f} GB free (~{du.free / 1e6 / RECORD_MB_S / 60:.0f} min of recording)"

    def _counts(self) -> str:
        demos = [e for e in self.episodes if e.kind == "demonstration"]
        kept = sum(e.status == "kept" for e in demos)
        ab = sum(e.status == "aborted" for e in demos)
        dis = sum(e.status == "discarded" for e in demos)
        return f"demos kept {kept}, aborted {ab}, discarded {dis}; calibration: {self.calibration or 'NONE'}"

    def _prompt(self) -> None:
        if self.calibration is None or self.want_calibration:
            self.out("-> Enter: record the GRIPPER CALIBRATION episode (hold the gripper still, open and close the "
                     "fingers fully 2-3 times, both finger tags visible; >= 3 s), Enter again to stop.")
        else:
            n = sum(e.kind == "demonstration" and e.status == "kept" for e in self.episodes) + 1
            self.out(f"-> Enter: start demo #{n}   (c recalibrate, d discard last, s status, q quit)")

    def _rename_hidden(self, path: Path, suffix: str) -> Path:
        dst = path.parent / f".{path.name}.{suffix}"
        if dst.exists():
            dst = path.parent / f".{path.name}.{suffix}.{int(time.time())}"
        os.replace(path, dst)
        return dst

    def _restore_calibration_state(self, dropped: str) -> None:
        state = self.rec.session_state
        if dropped in self.cal_history:
            self.cal_history.remove(dropped)
        prev = self.cal_history[-1] if self.cal_history else None
        if prev:
            state.set_calibration(self.session, self.opts.side, self.opts.gripper_id, prev)
        else:
            state.clear_calibration(self.session, self.opts.side, self.opts.gripper_id, dropped)
        if self.calibration == dropped:
            self.calibration = prev

    def _find_episode(self, name: str) -> Path | None:
        p = self.raw_root / name[:10] / name
        if (p / "meta.json").is_file():
            return p
        hits = [m.parent for m in self.raw_root.glob(f"*/{name}/meta.json")]
        return hits[0] if hits else None

    # ---- actions ----
    def _start(self) -> None:
        from sensor_rig.record.recorder import RecorderError

        kind = "gripper_calibration" if (self.calibration is None or self.want_calibration) else "demonstration"
        extra = None
        if self.replay is not None:
            extra = {"sim": {**self.replay.advance(kind), "mode": "collect_mock_replay"}}
        try:
            name = self.rec.start_episode(kind, session=self.session, side=self.opts.side,
                                          gripper_id=self.opts.gripper_id,
                                          calibration_run=self.calibration if kind == "demonstration" else None,
                                          task_label=self.opts.task_label, note=self.opts.note, extra_meta=extra)
        except RecorderError as e:
            if self.replay is not None:
                self.replay.end_pass()
            self.out(f"cannot start: {e}")
            return
        if extra:
            self.out(f"   replaying {extra['sim']['replay_source']} (offset {extra['sim']['replay_offset_ns']} ns)")
        self.active = (name, kind)
        self._last_status = time.monotonic()
        self.out(f"REC {'CALIBRATION' if kind == 'gripper_calibration' else 'demo'} {name}  (Enter stop, a abort)")

    def _stop(self) -> None:
        from sensor_rig.record.recorder import EpisodeAborted, RecorderError

        name, kind = self.active
        self.active = None
        if self.replay is not None:
            self.replay.end_pass()
        try:
            path = self.rec.stop_episode()
        except EpisodeAborted as e:
            self._aborted(name, kind, str(e))
            return
        except RecorderError as e:
            self._aborted(name, kind, str(e))
            return
        v = episode_verdict(path)
        rec = EpisodeRecord(name, kind, "kept", str(path), verdict=v,
                            calibration_run=self.calibration if kind == "demonstration" else None)
        if kind == "gripper_calibration":
            problems = check_calibration(v)
            if not problems and self.opts.tag_check:
                problems, note = self._tag_check(path)
                if note:
                    rec.reason = note
            if problems:
                hidden = self._rename_hidden(path, "rejected")
                rec.status, rec.path, rec.problems = "rejected", str(hidden), problems
                rec.reason = "; ".join(problems)
                self.rec.session_state.clear_calibration(self.session, self.opts.side, self.opts.gripper_id, name)
                if self.cal_history:   # the previous accepted calibration stays in the state file, but is not used
                    self.rec.session_state.set_calibration(self.session, self.opts.side, self.opts.gripper_id,
                                                           self.cal_history[-1])
                self.calibration = None
                self.want_calibration = True
                self.episodes.append(rec)
                self.out(format_verdict(name, kind, v, "REJECTED", rec.reason))
                self.out("   demos are blocked until a calibration passes; record it again (moved to "
                         f"{hidden.name}).")
                return
            self.calibration = name
            self.cal_history.append(name)
            self.want_calibration = False
        self.episodes.append(rec)
        self.out(format_verdict(name, kind, v, "KEPT", rec.reason))
        self.out(f"   {self._counts()}")

    def _tag_check(self, path: Path) -> tuple[list[str], str | None]:
        """(problems, note): reject if no ultrawide frame shows both finger tags; note a low fraction."""
        from sensor_rig.record.tagcheck import WARN_FRACTION, calibration_tag_check

        try:
            r = calibration_tag_check(path, self.rig)
        except Exception as e:  # noqa: BLE001 - e.g. cv2 without aruco: don't block, but say so
            return [], f"tag check skipped: {type(e).__name__}: {e}"
        if not r["ok"]:
            return [f"no ultrawide frame shows both finger tags ({r['frames']} frames checked, ids seen "
                    f"{r['ids_seen'] or 'none'}): check the tags are in view and lit"], None
        note = f"tags: both visible in {r['both']}/{r['frames']} frames"
        if r["warn"]:
            note += f" (< {WARN_FRACTION:.0%}: iPhUMI may fail to identify the gripper; consider recalibrating)"
        return [], note

    def _aborted(self, name: str, kind: str, reason: str) -> None:
        reason = (reason or "").replace("episode was aborted: ", "")
        path = self.raw_root / name[:10] / f".{name}.aborted"
        self.episodes.append(EpisodeRecord(name, kind, "aborted", str(path) if path.exists() else None, reason))
        self.out(format_verdict(name, kind, None, "ABORTED", reason))
        self.out(f"   {self._counts()}")

    def _discard(self) -> None:
        if self.active:
            self.out("recording: stop (Enter) or abort (a) first")
            return
        last = next((e for e in reversed(self.episodes) if e.status == "kept"), None)
        if last is None:
            self.out("nothing to discard")
            return
        hidden = self._rename_hidden(Path(last.path), "discarded")
        last.status, last.path = "discarded", str(hidden)
        if last.kind == "gripper_calibration":
            self._restore_calibration_state(last.name)
            self.out(f"discarded calibration {last.name} -> {hidden.name}; calibration now: "
                     f"{self.calibration or 'NONE (record a new one)'}")
        else:
            self.out(f"discarded {last.name} -> {hidden.name}")
        self.out(f"   {self._counts()}")

    def _check_monitor_abort(self) -> None:
        """The Monitor (or a dying backend) aborted the running episode behind our back; or the replayed source of a
        --mock-replay episode is finished (stop it before the source loops)."""
        if self.active and not self.rec.recording:
            name, kind = self.active
            self.active = None
            if self.replay is not None:
                self.replay.end_pass()
            self._aborted(name, kind, self.rec.last_abort_reason or "aborted")
            self._prompt()
        elif self.active and self.replay is not None and self.replay.finished:
            self.out("   replayed source finished: stopping the episode")
            self._stop()
            self._prompt()

    def _reuse(self) -> None:
        name = self.rec.session_state.get_calibration(self.session, self.opts.side, self.opts.gripper_id)
        path = self._find_episode(name) if name else None
        if path is None:
            self.out(f"--reuse-calibration: no calibration on disk for session={self.session} side={self.opts.side} "
                     f"gripper={self.opts.gripper_id}; record one now.")
            return
        problems = check_calibration(episode_verdict(path))
        if problems:
            self.out(f"--reuse-calibration: {name} fails the check ({'; '.join(problems)}); record a new one.")
            return
        self.calibration = name
        self.cal_history.append(name)
        self.out(f"reusing calibration {name}")

    # ---- main loop ----
    def run(self) -> int:
        from sensor_rig.record.recorder import RecorderError

        self.out(f"collect: session {self.session}, side {self.opts.side}, gripper {self.opts.gripper_id}, "
                 f"task {self.opts.task_label or '-'} -> {self.raw_root}")
        self.out(self._free_minutes())
        self.out(TMUX_TIP)
        self.out("keys: Enter start/stop | a abort | d discard last | c recalibrate | s status | q quit")
        try:
            self.rec.open()
        except (RecorderError, OSError, ValueError) as e:
            self.out(f"cannot open the sensors: {e}")
            self.end_reason = f"open_failed: {e}"
            self.exit_code = 1
            self.write_summary()
            return 1
        try:
            if self.opts.reuse_calibration:
                self._reuse()
            self._prompt()
            self._loop()
        except KeyboardInterrupt:
            self.end_reason = "SIGINT"
            self.exit_code = 130
        except SessionInterrupt as e:
            self.end_reason = e.signame
            self.exit_code = 128 + int(getattr(signal, e.signame, 15))
        finally:
            self._shutdown()
        return self.exit_code

    def _loop(self) -> None:
        self._last_status = time.monotonic()
        while True:
            cmd = self.inp.get(timeout_s=0.1)
            self._check_monitor_abort()
            if self.rec.pump_errors and not self.rec.recording:
                self.out(f"SENSOR FAILURE: {self.rec.pump_errors}; ending the session (check cables, then "
                         "`sensor-rig doctor`).")
                self.end_reason = f"sensor_failure: {self.rec.pump_errors}"
                self.exit_code = 1
                return
            now = time.monotonic()
            if self.active and now - self._last_status >= self.opts.status_period_s:
                self._last_status = now
                self.out("   " + self.rec.status_line())
            if cmd is None:
                continue
            if cmd == "toggle":
                if self.active:
                    self._stop()
                else:
                    self._start()
                if not self.active:
                    self._prompt()
            elif cmd == "abort":
                if self.active:
                    name, kind = self.active
                    self.active = None
                    self.rec.abort_episode("aborted by operator")
                    if self.replay is not None:
                        self.replay.end_pass()
                    self._aborted(name, kind, "aborted by operator")
                    self._prompt()
                else:
                    self.out("nothing is recording")
            elif cmd == "discard":
                self._discard()
                self._prompt()
            elif cmd == "calibrate":
                if self.active:
                    self.out("recording: stop first")
                else:
                    self.want_calibration = True
                    self._prompt()
            elif cmd == "status":
                self.out("   " + self.rec.status_line())
                self.out(f"   {self._counts()}; {self._free_minutes()}")
            elif cmd == "quit":
                if self.active:
                    name, kind = self.active
                    self.active = None
                    self.rec.abort_episode("quit while recording")
                    self._aborted(name, kind, "quit while recording")
                return
            else:
                self.out(f"unknown key {cmd.split(':', 1)[-1]!r}; keys: Enter, a, d, c, s, q")

    def _shutdown(self) -> None:
        try:
            if self.active:
                name, kind = self.active
                self.active = None
                if self.rec.recording:
                    self.rec.abort_episode(f"session ended ({self.end_reason})")
                self._aborted(name, kind, f"session ended ({self.end_reason})")
        finally:
            try:
                self.rec.close()
            finally:
                self.inp.close()
                self.write_summary()
                self.out(f"session {self.session} ended ({self.end_reason}): {self._counts()}")
                if self.summary_path:
                    self.out(f"summary: {self.summary_path}")

    def write_summary(self) -> Path:
        d = self.raw_root / ".sensor_rig" / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.session}_{self.started_utc}.json"
        demos = [e for e in self.episodes if e.kind == "demonstration"]
        doc = {
            "format": "sensor_rig.collect_session.v1",
            "session": self.session, "side": self.opts.side, "gripper_id": self.opts.gripper_id,
            "task_label": self.opts.task_label, "raw_root": str(self.raw_root), "rig": self.rig_path,
            "rig_name": getattr(self.rig, "rig_name", None),
            "started_utc": self.started_utc, "ended_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "end_reason": self.end_reason, "exit_code": self.exit_code, "doctor": self.doctor_summary,
            "calibration_in_use": self.calibration,
            "counts": {"demos_kept": sum(e.status == "kept" for e in demos),
                       "demos_aborted": sum(e.status == "aborted" for e in demos),
                       "demos_discarded": sum(e.status == "discarded" for e in demos),
                       "calibrations_kept": sum(e.kind == "gripper_calibration" and e.status == "kept"
                                                for e in self.episodes),
                       "calibrations_rejected": sum(e.status == "rejected" for e in self.episodes)},
            "episodes": [e.__dict__ for e in self.episodes],
        }
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1, default=str)
            f.write("\n")
        os.replace(tmp, path)
        self.summary_path = path
        return path


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def register(subparsers) -> None:
    p = subparsers.add_parser("collect", help="the one command for a recording session (doctor -> calibration -> "
                                              "demos), sensors open once", description=__doc__.splitlines()[0])
    p.add_argument("--rig", default=None, help="rig.yaml (default: configs/rig_default.yaml, or the replayed "
                                               "episode's rig.yaml with --mock-replay)")
    p.add_argument("--out", required=True, help="raw root: episodes go to <out>/<YYYY-MM-DD>/<NAME>/")
    p.add_argument("--session", required=True, help="session name (rig-only names, e.g. rig1020; never an "
                                                    "iPhone session name)")
    p.add_argument("--side", default="left", choices=["left", "right"],
                   help="gripper side (default left, like the iPhone sessions; the single-arm pipeline maps it)")
    p.add_argument("--task-label", default=None)
    p.add_argument("--gripper-id", default="default")
    p.add_argument("--note", default="")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="synthetic mock cameras + static IMU (laptop)")
    g.add_argument("--mock-replay", default=None, nargs="+", metavar="DIR",
                   help="replay raw (e.g. simulated) episodes, one source per recorded episode (calibration sources "
                        "for calibrations, demo sources for demos); each episode stops when its source ends")
    p.add_argument("--button", default=None, metavar="gpio:PIN", help="enable the push button on this BOARD pin")
    p.add_argument("--reuse-calibration", action="store_true",
                   help="use the last accepted calibration of this session/side/gripper instead of recording one")
    p.add_argument("--force", action="store_true", help="start even if doctor reports a FAIL")
    p.add_argument("--warmup", type=float, default=2.0)
    p.set_defaults(func=cmd_collect)


def make_backends(rig, args):
    """(cameras, imu, playlist) for --mock / --mock-replay, or (None, None, None) for the rig's real backends."""
    from sensor_rig.drivers.base import make_camera, make_imu

    if args.mock:
        cams = {c: make_camera(rig, c, backend="mock") for c in ("main", "ultrawide")}
        return cams, make_imu(rig, backend="mock"), None
    if args.mock_replay:
        from sensor_rig.drivers.replay_playlist import PlaylistCamera, PlaylistImu, ReplayPlaylist

        pl = ReplayPlaylist(args.mock_replay)
        return {c: PlaylistCamera(pl, c) for c in ("main", "ultrawide")}, PlaylistImu(pl), pl
    return None, None, None


def cmd_collect(args) -> int:
    from sensor_rig.doctor import DEFAULT_RIG, DoctorOptions, run_doctor
    from sensor_rig.record.button import GpioButton, button_config
    from sensor_rig.record.monitor import LiveAbortConfig
    from sensor_rig.record.recorder import Recorder
    from sensor_rig.rig_config import load_rig

    from sensor_rig.naming import session_name_problems
    from sensor_rig.timefmt import sanitize_session

    try:
        probs = session_name_problems(sanitize_session(args.session))
    except ValueError as e:
        probs = [str(e)]
    if probs:
        print("collect: " + "; ".join(probs), file=sys.stderr)
        return 2
    if args.rig:
        rig_path = args.rig
    elif args.mock_replay:
        from sensor_rig.raw_episode import find_raw_episodes

        eps = [e for d in args.mock_replay for e in find_raw_episodes(Path(d))]
        if not eps:
            print(f"collect: no raw episodes in {args.mock_replay}", file=sys.stderr)
            return 2
        rig_path = str(eps[0].path / "rig.yaml")
    else:
        rig_path = str(DEFAULT_RIG)
    mock = bool(args.mock or args.mock_replay)
    from sensor_rig.rig_config import load_rig_arg

    load_rig_arg(rig_path)          # missing / invalid rig file -> RigFileError -> exit 2
    rep = run_doctor(rig_path, DoctorOptions(mock=mock, quick=True, sensors=False, out=args.out))
    print(rep.table())
    if not rep.ok:
        if not args.force:
            print("doctor found FAILs: fix them (see the -> lines) or pass --force to record anyway", file=sys.stderr)
            return 1
        print("\n" + "!" * 78 + "\n!!  --force: RECORDING DESPITE DOCTOR FAILURES. CHECK EVERY EPISODE.\n" + "!" * 78)
    rig = load_rig(rig_path)
    cap = rig.capture if isinstance(rig.capture, dict) else {}
    live = dict(cap.get("live_abort") or {})
    for k, v in COLLECT_LIVE_ABORT_DEFAULTS.items():
        live.setdefault(k, v)
    try:
        cams, imu, playlist = make_backends(rig, args)
        bcfg = button_config(rig, args.button)
    except (ValueError, OSError) as e:
        print(f"collect: {e}", file=sys.stderr)
        return 2
    rec = Recorder(rig, args.out, cameras=cams, imu=imu, warmup_s=args.warmup, keep_aborted=True,
                   live_abort=LiveAbortConfig.from_dict(live), log=lambda m: None, status_lines=False)
    inp = StdinInput()
    button = None
    if bcfg:
        def on_button(ev):
            inp.put("toggle" if ev == "toggle" else "abort")
        button = GpioButton(on_button, bcfg["pin"], bcfg["active_low"], bcfg["long_press_s"])
        try:
            button.start()
            print(f"button on pin {bcfg['pin']}: short press = start/stop, long press = abort")
        except RuntimeError as e:
            print(f"collect: {e}", file=sys.stderr)
            return 2
    opts = CollectOptions(session=args.session, side=args.side, gripper_id=args.gripper_id,
                          task_label=args.task_label, note=args.note, reuse_calibration=args.reuse_calibration,
                          tag_check=not args.mock)
    if args.mock:
        print("--mock: synthetic images have no finger tags, so the calibration tag check is off")
    doc = {"ok": rep.ok, "forced": bool(args.force and not rep.ok),
           "fails": [r.name for r in rep.results if r.status == "FAIL"],
           "warns": [r.name for r in rep.results if r.status == "WARN"]}
    sess = CollectSession(rig, args.out, opts, rec, inp, doctor_summary=doc, rig_path=rig_path, replay=playlist)
    try:
        with signal_guard():
            return sess.run()
    finally:
        if button is not None:
            button.stop()
        _quiet_stdout_if_gone()


def _quiet_stdout_if_gone() -> None:
    """After an SSH hangup the terminal is gone: flushing stdout at exit would fail and turn the exit code into 120.
    Point stdout/stderr at /dev/null instead, so the session's own exit code (e.g. 129 for SIGHUP) survives."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stream.fileno())
                os.close(devnull)
            except (OSError, ValueError):
                pass
