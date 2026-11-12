"""StateListener: the C2 thread. Receives every state message into StateHistory (STRUCTURE.md 4.3)."""
from __future__ import annotations

import logging
import threading

from contracts import ArmState, Subscriber
from message_comms.state_history import StateHistory

log = logging.getLogger("message_comms.state_listener")


class StateListener:
    def __init__(self, endpoint: str, history: StateHistory, recv_hwm: int, poll_s: float) -> None:
        self._history = history
        self._poll_s = poll_s
        self._sub = Subscriber("franka_control", endpoint, ArmState, recv_hwm)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="state-listener", daemon=True)
        self._error: BaseException | None = None
        self._received = 0
        self._closed = False

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Sets the stop event, joins the thread (at most about poll_s), closes the socket."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2 * self._poll_s)
        if not self._closed and not self._thread.is_alive():
            self._closed = True
            self._sub.close()

    def error(self) -> BaseException | None:
        return self._error

    def received(self) -> int:
        return self._received

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                state = self._sub.receive(timeout_s=self._poll_s)
                if state is not None:
                    self._history.add(state)
                    self._received += 1
        except BaseException as exc:      # reported to the main thread, which concludes
            log.error("state listener stopped: %s", exc)
            self._error = exc
