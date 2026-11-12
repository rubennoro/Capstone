"""Request/reply over ZeroMQ: RpcClient and RpcServer (STRUCTURE.md section 6).

A request is sent once and never resent. On a timeout or a cancel the client closes its socket with linger 0 and
opens a new one, so a late reply is thrown away ("lazy pirate"). Servers bind, clients connect. Each helper is used
by one thread.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Mapping, TypeVar

import zmq

from contracts.codec import decode, encode
from contracts.common import ErrorReply
from contracts.errors import ContractError, RemoteError, TransportCancelled, TransportTimeout
from contracts.message import Message

M = TypeVar("M", bound=Message)

log = logging.getLogger("contracts.rpc")


def _ms(seconds: float) -> int:
    return max(0, int(math.ceil(seconds * 1000.0)))


class RpcClient:
    """REQ client of one server. `name` is the server's process name, used in error messages."""

    def __init__(self, name: str, endpoint: str) -> None:
        self.name = name
        self.endpoint = endpoint
        self._ctx = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._open()

    def _open(self) -> None:
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.endpoint)
        self._socket = sock

    def _reset(self) -> None:
        """Drop the socket (and any reply still on its way) and open a fresh one."""
        if self._socket is not None:
            self._socket.close(linger=0)
        self._open()

    def request(self, msg: Message, expect: type[M] | tuple[type[Message], ...], timeout_s: float,
                poll_s: float | None = None, cancel: Callable[[], bool] | None = None) -> M:
        """Send msg once and wait for one reply of a type in `expect`.

        Raises TransportTimeout (no reply within timeout_s), TransportCancelled (cancel() turned true; checked every
        poll_s), RemoteError (the server sent ErrorReply) or ContractError (the reply broke its contract or is not
        one of `expect`).
        """
        if (poll_s is None) != (cancel is None):
            raise ValueError("give both poll_s and cancel, or neither")
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if poll_s is not None and not poll_s > 0:
            raise ValueError(f"poll_s must be > 0, got {poll_s}")
        if self._socket is None:
            raise RuntimeError(f"RpcClient for {self.name} is closed")
        sock = self._socket
        sock.send(encode(msg))
        start = time.monotonic()
        deadline = start + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            wait = remaining if poll_s is None else min(poll_s, remaining)
            if wait > 0 and sock.poll(_ms(wait), zmq.POLLIN):
                data = sock.recv()
                reply = decode(data, expect)
                if isinstance(reply, ErrorReply):
                    raise RemoteError(self.name, reply)
                return reply  # type: ignore[return-value]
            if cancel is not None and cancel():
                self._reset()
                raise TransportCancelled(self.name, self.endpoint, msg.SCHEMA, time.monotonic() - start)
            if time.monotonic() >= deadline:
                self._reset()
                raise TransportTimeout(self.name, self.endpoint, msg.SCHEMA, timeout_s)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


class RpcServer:
    """REP server: decodes each request, calls its handler by class, sends the handler's reply.

    It always replies: a request that breaks its contract or has no handler gets ErrorReply naming the field; a
    handler that raises gets ErrorReply with the exception text (and the exception is logged). It keeps running.
    """

    def __init__(self, endpoint: str, handlers: Mapping[type[Message], Callable[[Message], Message]]) -> None:
        self._handlers = dict(handlers)
        self._socket: zmq.Socket | None = zmq.Context.instance().socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self.endpoint: str = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def _answer(self, data: bytes) -> Message:
        try:
            request = decode(data)
        except ContractError as e:
            return ErrorReply(request="" if e.schema == "?" else e.schema, error=str(e), field=e.field)
        handler = self._handlers.get(type(request))
        if handler is None:
            known = ", ".join(c.SCHEMA for c in self._handlers)
            err = ContractError(request.SCHEMA, "", "unexpected_message",
                                f"this server does not handle it (it handles {known})")
            return ErrorReply(request=request.SCHEMA, error=str(err), field="")
        try:
            reply = handler(request)
        except Exception as e:  # the server keeps serving; the client decides to conclude
            log.exception("handler for %s raised", request.SCHEMA)
            return ErrorReply(request=request.SCHEMA, error=f"{request.SCHEMA}: handler failed: "
                              f"{type(e).__name__}: {e}", field="")
        if not isinstance(reply, Message):
            log.error("handler for %s returned %s, not a Message", request.SCHEMA, type(reply).__name__)
            return ErrorReply(request=request.SCHEMA, error=f"{request.SCHEMA}: handler failed: returned "
                              f"{type(reply).__name__}, not a Message", field="")
        return reply

    def serve_once(self, timeout_s: float) -> bool:
        """Handle at most one request, waiting up to timeout_s. False if none came."""
        if self._socket is None:
            raise RuntimeError("RpcServer is closed")
        if not self._socket.poll(_ms(timeout_s), zmq.POLLIN):
            return False
        data = self._socket.recv()
        self._socket.send(encode(self._answer(data)))
        return True

    def serve_forever(self, stop: threading.Event, poll_s: float) -> None:
        """Serve until stop is set, checking it every poll_s."""
        while not stop.is_set():
            self.serve_once(poll_s)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
