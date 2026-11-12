"""ClockSync: the time_sync exchanges with the Pi, thread "clock-sync" (STRUCTURE.md §5.7, §7)."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

from contracts import ContractError, RemoteError, RpcClient, TimeSyncReply, TimeSyncRequest, TransportTimeout

from process_frames.clock_map import ClockMap, ClockSample
from process_frames.errors import FatalError

if TYPE_CHECKING:
    from process_frames.settings import ClockSettings

log = logging.getLogger("process_frames.clock_sync")

#: How often run() re-reads the clock while waiting for the next exchange (so tests can drive time).
_TICK_S = 0.01
#: At most one "time_sync timed out" log line this often.
_TIMEOUT_LOG_EVERY_NS = 10_000_000_000


class ClockSync:
    def __init__(self, endpoint: str, clock_map: ClockMap, settings: "ClockSettings",
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.endpoint = endpoint
        self._map = clock_map
        self._settings = settings
        self._clock = clock
        self._client: RpcClient | None = None      # opened on the first sync_once, in the calling thread
        self._last_timeout_log: int | None = None
        self.exchanges = 0
        self.timeouts = 0

    def sync_once(self) -> bool:
        """One exchange. False on a timeout (counted, not an error); FatalError if the Pi broke C0."""
        if self._client is None:
            self._client = RpcClient(name="pi", endpoint=self.endpoint)
        t_send = self._clock()
        try:
            reply = self._client.request(TimeSyncRequest(), TimeSyncReply, self._settings.timeout_s)
        except TransportTimeout:
            self.timeouts += 1
            now = self._clock()
            if self._last_timeout_log is None or now - self._last_timeout_log >= _TIMEOUT_LOG_EVERY_NS:
                self._last_timeout_log = now
                log.warning("no time_sync reply from %s within %.3f s (%d timeouts so far)", self.endpoint,
                            self._settings.timeout_s, self.timeouts)
            return False
        except ContractError as e:
            raise FatalError(f"C0 time_sync reply breaks its contract: {e}", "check the Pi sender version") from e
        except RemoteError as e:
            raise FatalError(f"the Pi refused time_sync: {e}", "check the Pi sender version") from e
        t_recv = self._clock()
        self.exchanges += 1
        self._map.add_sample(ClockSample(t_send_ns=t_send, t_pi_ns=int(reply.t_pi_ns), t_recv_ns=t_recv))
        return True

    def run(self, stop: threading.Event) -> None:
        """startup_exchanges back to back, then one every interval_s, until ``stop``."""
        interval_ns = int(self._settings.interval_s * 1e9)
        try:
            for _ in range(self._settings.startup_exchanges):
                if stop.is_set():
                    return
                self.sync_once()
            next_ns = self._clock() + interval_ns
            while not stop.is_set():
                now = self._clock()
                if now >= next_ns:
                    self.sync_once()
                    next_ns += interval_ns
                    if next_ns <= now:
                        next_ns = now + interval_ns
                else:
                    stop.wait(min((next_ns - now) / 1e9, _TICK_S))
        finally:
            self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
