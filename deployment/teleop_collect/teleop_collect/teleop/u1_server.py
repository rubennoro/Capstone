"""U1Server: the REP socket the ui talks to (STRUCTURE.md 4.4, 6.6), and MessageLog, the page's messages.

Served from teleop's main loop only: `serve(n)` answers at most n waiting requests without blocking (serve_once(0)),
every request going to one `dispatch` callable (TeleopLoop.dispatch), which decides by the request's class and the
phase. A request that breaks the U1 contract is answered error/v1 by deploy_contracts' RpcServer and never reaches
dispatch.
"""
from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from deploy_contracts import Message, RpcServer

from teleop_collect.contracts import (STATUS_MAX_MESSAGES, CreateTask, DeleteEpisode, GetStatus, GripperCommand,
                                      HomeRequest, KeysUpdate, SelectTask, StartEpisode, StopArm, StopEpisode,
                                      UiMessage)

#: Every U1 request (STRUCTURE.md 4.4).
U1_REQUESTS: tuple[type[Message], ...] = (GetStatus, KeysUpdate, GripperCommand, SelectTask, CreateTask, HomeRequest,
                                          StartEpisode, StopEpisode, DeleteEpisode, StopArm)


class U1Server:
    def __init__(self, endpoint: str, dispatch: Callable[[Message], Message]) -> None:
        """Binds endpoint (station.u1_endpoint(), or tcp://127.0.0.1:* in tests)."""
        self._rpc = RpcServer(endpoint, {cls: dispatch for cls in U1_REQUESTS})
        self.endpoint = self._rpc.endpoint

    def serve(self, max_requests: int) -> int:
        """Answer up to max_requests waiting requests, never waiting; returns how many were answered."""
        n = 0
        while n < max_requests and self._rpc.serve_once(0.0):
            n += 1
        return n

    def close(self) -> None:
        self._rpc.close()


class MessageLog:
    """The page's messages: ids count up from 1, the newest `kept` stay in memory. One lock (the status is built on
    the main thread, but a test may read it from another)."""

    def __init__(self, kept: int) -> None:
        self._lock = threading.Lock()
        self._messages: deque[UiMessage] = deque(maxlen=kept)
        self._next_id = 1

    def add(self, level: str, text: str) -> UiMessage:
        with self._lock:
            msg = UiMessage(id=self._next_id, level=level, text=" ".join(str(text).split()))
            self._next_id += 1
            self._messages.append(msg)
            return msg

    def after(self, after_id: int, limit: int = STATUS_MAX_MESSAGES) -> list[UiMessage]:
        """The messages with id > after_id, oldest first, at most limit (the ui asks again for the rest)."""
        with self._lock:
            return [m for m in self._messages if m.id > after_id][:limit]

    def texts(self) -> list[str]:
        with self._lock:
            return [m.text for m in self._messages]
