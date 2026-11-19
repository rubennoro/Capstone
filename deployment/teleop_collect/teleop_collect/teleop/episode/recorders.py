"""The per-episode recorders of teleop (PLAN 6.1, 6.2; STRUCTURE.md 8.2).

`EpisodeRecording` is opened by Start and closed by Stop / Abort. It owns five files in the episode folder (frames
writes its own files next to them):

| file               | recorder          | written by                                        |
|--------------------|-------------------|---------------------------------------------------|
| robot_states.csv   | `StateRecorder`   | the pc-listener thread, every C2 (thread-safe)    |
| commands.csv       | `CommandRecorder` | the main loop, every C3 request with its reply    |
| keys.csv           | `KeyRecorder`     | the main loop, every key state change             |
| clock_pc.csv       | `PcClockRecorder` | the main loop, every hello sample and the fit     |
| events.jsonl       | `EventRecorder`   | the main loop: start, stop, home, hints, problems |

Values: integers as written, floats in their shortest exact form (repr), booleans 1 / 0, missing values empty. Every
row is flushed as it is written (a few hundred small writes a second at most), so at most the row being written is
lost in a crash; the episode is then incomplete anyway (no COMPLETE) and deleted at the next start-up. Writes after
close are ignored. Files are created exclusively: an existing file is never overwritten.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import threading
from pathlib import Path
from typing import IO, Any, Iterable, Mapping

import numpy as np

from deploy_contracts import Ack, ArmState, ErrorReply, SetWaypoints, SetWaypointsReply

from teleop_collect.teleop.episode.store import EpisodeDir
from teleop_collect.teleop.motion.keys import HeldKeys
from teleop_collect.teleop.motion.twist import TWIST_COLUMNS, Twist

log = logging.getLogger(__name__)

STATES_FILE = "robot_states.csv"
COMMANDS_FILE = "commands.csv"
KEYS_FILE = "keys.csv"
CLOCK_PC_FILE = "clock_pc.csv"
EVENTS_FILE = "events.jsonl"
FILES = (STATES_FILE, COMMANDS_FILE, KEYS_FILE, CLOCK_PC_FILE, EVENTS_FILE)


def matrix_columns(prefix: str) -> list[str]:
    """`<prefix>_00` ... `<prefix>_33`, row-major."""
    return [f"{prefix}_{r}{c}" for r in range(4) for c in range(4)]


STATE_COLUMNS = (
    "t_pc_ns", "t_ns", "arrival_ns", "seq", "mode",
    *matrix_columns("O_T_TCP"), *matrix_columns("O_T_TCP_target"),
    *(f"q_{i}" for i in range(7)), *(f"dq_{i}" for i in range(7)),
    "gripper_width_m", "gate_accepted", "gate_shortened", "gate_pulled_back", "gate_rejected", "loop_late", "error",
)
KEY_COLUMNS = ("t_ns", "held", "slow", "focus")
CLOCK_PC_COLUMNS = ("kind", "t_send_ns", "t_recv_ns", "t_pc_ns", "t_ns", "offset_ns", "drift_ppm", "samples", "used",
                    "rtt_min_ms")
REPLY_COLUMNS = ("accepted", "shortened", "skipped", "rejected", "rejected_reasons", "ack_ok", "ack_reason")


def command_columns(waypoints: int) -> tuple[str, ...]:
    """commands.csv: the request, K waypoint slots (wp<k>_t_pc_ns, wp<k>_00..33, wp<k>_width_m), twist, reply."""
    wps: list[str] = []
    for k in range(waypoints):
        wps += [f"wp{k}_t_pc_ns", *matrix_columns(f"wp{k}"), f"wp{k}_width_m"]
    return ("t_send_ns", "t_reply_ns", "request", "chunk_id", "n", *wps, *TWIST_COLUMNS, "slow", *REPLY_COLUMNS)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def _floats(a: np.ndarray) -> list[str]:
    return [repr(v) for v in np.asarray(a, dtype=np.float64).reshape(-1).tolist()]


class _CsvFile:
    """One CSV file with a header, created exclusively, flushed after every row."""

    def __init__(self, path: Path, columns: Iterable[str]) -> None:
        self.path = path
        self.columns = tuple(columns)
        self._f: IO[str] | None = open(path, "x", newline="", encoding="utf-8")
        self._w = csv.writer(self._f, lineterminator="\n")
        self._w.writerow(self.columns)
        self._f.flush()

    @property
    def closed(self) -> bool:
        return self._f is None

    def row(self, values: list[str]) -> None:
        if self._f is None:
            return
        if len(values) != len(self.columns):
            raise ValueError(f"{self.path.name}: {len(values)} values for {len(self.columns)} columns")
        self._w.writerow(values)
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            f, self._f = self._f, None
            f.close()


class StateRecorder:
    """robot_states.csv: every C2 message while recording. Thread-safe: written from the pc-listener thread while the
    main thread may close. A write error never raises into the listener: it is kept in `error` and stops the file."""

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._csv = _CsvFile(path, STATE_COLUMNS)
        self._error = ""
        self._messages = 0
        self._seq_gaps = 0
        self._seq_missing = 0
        self._last_seq: int | None = None
        self._first: ArmState | None = None
        self._last: ArmState | None = None

    def write(self, state: ArmState, arrival_ns: int, t_ns: int | None) -> None:
        with self._lock:
            if self._csv.closed or self._error:
                return
            try:
                g = state.gate
                self._csv.row([
                    _fmt(state.t_ns), _fmt(t_ns), _fmt(arrival_ns), _fmt(state.seq), state.mode,
                    *_floats(state.O_T_TCP), *_floats(state.O_T_TCP_target), *_floats(state.q), *_floats(state.dq),
                    _fmt(state.gripper_width_m), _fmt(g.accepted), _fmt(g.shortened), _fmt(g.pulled_back),
                    _fmt(g.rejected), _fmt(state.loop_late), state.error,
                ])
            except Exception as e:                       # noqa: BLE001: kept for the main loop, never raised here
                self._error = f"{self._csv.path}: {type(e).__name__}: {e}"
                log.error("state recorder stopped: %s", self._error)
                return
            self._messages += 1
            if self._last_seq is not None and state.seq > self._last_seq + 1:
                self._seq_gaps += 1
                self._seq_missing += state.seq - self._last_seq - 1
            self._last_seq = state.seq
            if self._first is None:
                self._first = state
            self._last = state

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def counts(self) -> dict[str, int]:
        with self._lock:
            first, last = self._first, self._last
            out = {"c2_messages": self._messages, "c2_seq_gaps": self._seq_gaps, "c2_seq_missing": self._seq_missing}
        if first is not None and last is not None:
            out["pulled_back"] = int(last.gate.pulled_back - first.gate.pulled_back)
            out["gate_shortened"] = int(last.gate.shortened - first.gate.shortened)
            out["loop_late"] = int(last.loop_late - first.loop_late)
        else:
            out.update(pulled_back=0, gate_shortened=0, loop_late=0)
        return out

    def close(self) -> None:
        with self._lock:
            self._csv.close()


class CommandRecorder:
    """commands.csv: every C3 request teleop sent while recording, with its reply. Main thread."""

    def __init__(self, path: Path, waypoints: int) -> None:
        if isinstance(waypoints, bool) or not isinstance(waypoints, int) or waypoints < 1:
            raise ValueError(f"waypoints {waypoints!r} must be an int >= 1")
        self.waypoints = waypoints
        self._csv = _CsvFile(path, command_columns(waypoints))
        self._c = dict(requests=0, sets_sent=0, sets_all_skipped=0, waypoints_sent=0, waypoints_skipped=0,
                       waypoints_shortened=0, waypoints_rejected=0, no_reply=0)

    def write(self, t_send_ns: int, t_reply_ns: int | None, request: str, msg: SetWaypoints | None,
              reply: SetWaypointsReply | Ack | None, twist: Twist | None, slow: bool) -> None:
        if self._csv.closed:
            return
        wp_cells: list[str] = []
        n = 0
        chunk_id = None
        if msg is not None:
            n = len(msg.t_ns)
            if n > self.waypoints:
                raise ValueError(f"commands.csv has {self.waypoints} waypoint slots, the set has {n}")
            chunk_id = msg.chunk_id
            for k in range(self.waypoints):
                if k < n:
                    wp_cells += [_fmt(msg.t_ns[k]), *_floats(msg.O_T_TCP[k]), _fmt(msg.gripper_width_m[k])]
                else:
                    wp_cells += [""] * 18
        else:
            wp_cells = [""] * (18 * self.waypoints)
        twist_cells = [_fmt(v) for v in twist.values()] if twist is not None else [""] * len(TWIST_COLUMNS)
        reply_cells = [""] * len(REPLY_COLUMNS)
        if isinstance(reply, SetWaypointsReply):
            reasons = "; ".join(f"{r.index}: {r.reason}" for r in reply.rejected)
            reply_cells = [_fmt(reply.accepted), _fmt(reply.shortened), _fmt(reply.skipped),
                           _fmt(len(reply.rejected)), reasons, "", ""]
        elif isinstance(reply, Ack):
            reply_cells[5:] = [_fmt(reply.ok), reply.reason]
        elif isinstance(reply, ErrorReply):
            reply_cells[5:] = [_fmt(False), f"error: {reply.error}"]
        self._csv.row([_fmt(t_send_ns), _fmt(t_reply_ns), request, _fmt(chunk_id), _fmt(n if msg else None),
                       *wp_cells, *twist_cells, _fmt(bool(slow)), *reply_cells])
        c = self._c
        c["requests"] += 1
        if reply is None:
            c["no_reply"] += 1
        if msg is not None:
            c["sets_sent"] += 1
            c["waypoints_sent"] += n
            if isinstance(reply, SetWaypointsReply):
                c["waypoints_skipped"] += reply.skipped
                c["waypoints_shortened"] += reply.shortened
                c["waypoints_rejected"] += len(reply.rejected)
                if n > 0 and reply.skipped >= n:
                    c["sets_all_skipped"] += 1

    def counts(self) -> dict[str, int]:
        return dict(self._c)

    def close(self) -> None:
        self._csv.close()


class KeyRecorder:
    """keys.csv: every key state change from the ui (`held` = the key names, space-separated). Main thread."""

    def __init__(self, path: Path) -> None:
        self._csv = _CsvFile(path, KEY_COLUMNS)
        self._changes = 0

    def write(self, t_ns: int, keys: HeldKeys, focus: bool) -> None:
        if self._csv.closed:
            return
        self._csv.row([_fmt(t_ns), " ".join(keys.names()), _fmt(keys.slow), _fmt(bool(focus))])
        self._changes += 1

    def counts(self) -> dict[str, int]:
        return {"key_changes": self._changes}

    def close(self) -> None:
        self._csv.close()


class PcClockRecorder:
    """clock_pc.csv: every hello sample (kind `hello`: send, receive, t_pc_ns) and the PC clock fit (kind `fit`, about
    once a second): the raw samples, so the PC -> laptop map can be refitted offline. Main thread."""

    def __init__(self, path: Path) -> None:
        self._csv = _CsvFile(path, CLOCK_PC_COLUMNS)
        self._hellos = 0

    def hello(self, t_send_ns: int, t_recv_ns: int, t_pc_ns: int) -> None:
        if self._csv.closed:
            return
        self._csv.row(["hello", _fmt(t_send_ns), _fmt(t_recv_ns), _fmt(t_pc_ns), "", "", "", "", "", ""])
        self._hellos += 1

    def fit(self, t_ns: int, offset_ns: int | None, drift_ppm: float | None, samples: int, used: int,
            rtt_min_ms: float | None) -> None:
        if self._csv.closed:
            return
        self._csv.row(["fit", "", "", "", _fmt(t_ns), _fmt(offset_ns), _fmt(drift_ppm), _fmt(samples), _fmt(used),
                       _fmt(rtt_min_ms)])

    def counts(self) -> dict[str, int]:
        return {"hellos": self._hellos}

    def close(self) -> None:
        self._csv.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


class EventRecorder:
    """events.jsonl: one JSON object per line, {"t_ns", "kind", "detail"}, laptop time. Main thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f: IO[str] | None = open(path, "x", encoding="utf-8")

    def write(self, t_ns: int, kind: str, detail: str | Mapping[str, Any] | None = None) -> None:
        if self._f is None:
            return
        detail_out = dict(detail) if isinstance(detail, Mapping) else detail
        if isinstance(detail_out, float) and not math.isfinite(detail_out):
            detail_out = str(detail_out)
        line = json.dumps({"t_ns": int(t_ns), "kind": str(kind), "detail": detail_out}, default=_json_default,
                          ensure_ascii=False, allow_nan=False)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            f, self._f = self._f, None
            f.close()


class EpisodeRecording:
    """Opened by Start, closed by Stop / Abort: owns teleop's five files of one episode.

    `waypoints` is K (MotionSettings.waypoints): commands.csv has that many waypoint slots.
    """

    def __init__(self, episode: EpisodeDir, waypoints: int = 2) -> None:
        self.episode = episode
        d = Path(episode.path)
        opened: list[Any] = []
        try:
            self.states = StateRecorder(d / STATES_FILE)
            opened.append(self.states)
            self.commands = CommandRecorder(d / COMMANDS_FILE, waypoints)
            opened.append(self.commands)
            self.keys = KeyRecorder(d / KEYS_FILE)
            opened.append(self.keys)
            self.pc_clock = PcClockRecorder(d / CLOCK_PC_FILE)
            opened.append(self.pc_clock)
            self.events = EventRecorder(d / EVENTS_FILE)
            opened.append(self.events)
        except BaseException:
            for r in opened:
                r.close()
            raise
        self._closed = False

    @property
    def error(self) -> str:
        """"" or the state recorder's write error (the listener thread never raises it): RECORDING_FAILED."""
        return self.states.error

    def counts(self) -> dict[str, int]:
        """c2_messages, c2_seq_gaps, c2_seq_missing, pulled_back, gate_shortened, loop_late (deltas over the
        episode), requests, sets_sent, sets_all_skipped, waypoints_sent / _skipped / _shortened / _rejected, no_reply,
        key_changes, hellos."""
        out: dict[str, int] = {}
        for r in (self.states, self.commands, self.keys, self.pc_clock):
            out.update(r.counts())
        return out

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close every file. Idempotent; writes after it are ignored."""
        if self._closed:
            return
        self._closed = True
        problems = []
        for r in (self.states, self.commands, self.keys, self.pc_clock, self.events):
            try:
                r.close()
            except OSError as e:
                problems.append(f"{type(r).__name__}: {e}")
        if problems:
            raise OSError("closing the episode files: " + "; ".join(problems))
