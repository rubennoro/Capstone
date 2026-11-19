# adapted from rnoro/final-deployment:deployment/policy_deploy/policy_deploy/manager/pc_listener.py @ 2ffe641
# changes: the recorder hook of teleop_collect STRUCTURE.md 6.1: set_recorder(StateRecorder | None), swapped under a
# lock, called per C2 message with (state, arrival_ns, t_ns mapped to the laptop or None); a recorder write error is
# kept apart (recorder_error(), the main loop aborts the episode with RECORDING_FAILED) and drops the recorder, the
# thread goes on; logger name teleop.pc_listener; imports. The source's own header follows.
# adapted from rnoro/deployment:deployment/message_comms/message_comms/state_listener.py @ 91b178b
# changes: renamed PcListener, process name "PC"; each message's laptop arrival is stamped; PcClock.add_state when t_ns
# advanced; arrival and seq to LinkStats; history.add raises PcRestarted on seq going back.
"""PcListener: the C2 thread. Every C2 message into StateHistory, the PC clock map, LinkStats and, while an episode
records, the state recorder (STRUCTURE.md 6.1).

The first exception of the link (a broken message, seq going back) is stored and the thread ends; the main thread
concludes on it. A recorder's exception is stored separately and only that recorder is dropped.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from deploy_contracts import ArmState, Subscriber

from teleop_collect.contracts import NotSynced
from teleop_collect.teleop.link_stats import LinkStats
from teleop_collect.teleop.pc_clock import PcClock
from teleop_collect.teleop.state_history import StateHistory

log = logging.getLogger("teleop.pc_listener")


class StateSink(Protocol):
    """E4's StateRecorder (STRUCTURE.md 8.2)."""

    def write(self, state: ArmState, arrival_ns: int, t_ns: int | None) -> None: ...


class PcListener:
    def __init__(self, endpoint: str, history: StateHistory, pc_clock: PcClock, stats: LinkStats, recv_hwm: int,
                 poll_s: float, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._history = history
        self._pc_clock = pc_clock
        self._stats = stats
        self._poll_s = poll_s
        self._clock = clock
        self.endpoint = endpoint
        self._sub = Subscriber("PC", endpoint, ArmState, recv_hwm)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pc-listener", daemon=True)
        self._error: BaseException | None = None
        self._recorder_lock = threading.Lock()
        self._recorder: StateSink | None = None
        self._recorder_error: BaseException | None = None
        self._received = 0
        self._last_arrival_ns: int | None = None
        self._last_t_ns: int | None = None
        self._closed = False

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Sets the stop event, joins the thread (at most about poll_s), closes the socket."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2 * self._poll_s + 0.5)
        if not self._closed and not self._thread.is_alive():
            self._closed = True
            self._sub.close()

    def set_recorder(self, recorder: StateSink | None) -> None:
        """Every C2 message from now on also goes to recorder.write (None: none). Clears an old recorder error.
        Returns once no write to the previous recorder is in progress."""
        with self._recorder_lock:
            self._recorder = recorder
            self._recorder_error = None

    def recorder_error(self) -> BaseException | None:
        with self._recorder_lock:
            return self._recorder_error

    def error(self) -> BaseException | None:
        return self._error

    def received(self) -> int:
        return self._received

    def last_arrival_ns(self) -> int | None:
        """Laptop time the newest C2 message arrived (liveness: STATE_LOST)."""
        return self._last_arrival_ns

    def _record(self, state: ArmState, arrival: int) -> None:
        with self._recorder_lock:
            recorder = self._recorder
            if recorder is None:
                return
            try:
                t_ns = self._pc_clock.snapshot().to_laptop_ns(int(state.t_ns), arrival)
            except NotSynced:
                t_ns = None
            try:
                recorder.write(state, arrival, t_ns)
            except Exception as exc:          # the episode is aborted by the main loop; the link goes on
                log.error("state recorder failed: %s", exc)
                self._recorder_error = exc
                self._recorder = None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                state = self._sub.receive(timeout_s=self._poll_s)
                if state is None:
                    continue
                arrival = self._clock()
                self._history.add(state)
                if self._last_t_ns is None or state.t_ns > self._last_t_ns:
                    self._pc_clock.add_state(int(state.t_ns), arrival)
                    self._last_t_ns = int(state.t_ns)
                self._stats.c2(arrival, int(state.seq))
                self._received += 1
                self._last_arrival_ns = arrival
                self._record(state, arrival)
        except BaseException as exc:      # reported to the main thread, which concludes
            log.error("C2 listener stopped: %s", exc)
            self._error = exc
