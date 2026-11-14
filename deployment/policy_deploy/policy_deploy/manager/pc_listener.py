# adapted from rnoro/deployment:deployment/message_comms/message_comms/state_listener.py @ 91b178b
# changes: renamed PcListener, process name "PC"; each message's laptop arrival is stamped; PcClock.add_state when t_ns
# advanced; arrival and seq to LinkStats; history.add raises PcRestarted on seq going back.
"""PcListener: the C2 thread. Every C2 message into StateHistory, the PC clock map and LinkStats (STRUCTURE.md 8.3).

The first exception is stored and the thread ends; the main thread concludes on it.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from deploy_contracts import ArmState, Subscriber

from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.pc_clock import PcClock
from policy_deploy.manager.state_history import StateHistory

log = logging.getLogger("manager.pc_listener")


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

    def error(self) -> BaseException | None:
        return self._error

    def received(self) -> int:
        return self._received

    def last_arrival_ns(self) -> int | None:
        """Laptop time the newest C2 message arrived (liveness: STATE_LOST)."""
        return self._last_arrival_ns

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
        except BaseException as exc:      # reported to the main thread, which concludes
            log.error("C2 listener stopped: %s", exc)
            self._error = exc
