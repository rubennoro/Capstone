"""robotmq request helper that never resends (reviewer F1).

robotmq's RMQClient.request_with_data/peek_data/pop_data default to automatic_resend=True: on timeout they RESEND the
request (up to 800 times) instead of failing. For inference that means a slow call is computed twice and the reply
belongs to stale observations; for a dead server the client hangs for up to 800 x timeout. Every franka_deploy RPC
goes through `request()` (automatic_resend=False): a timeout raises RpcTimeout and the caller must hold the arm.
"""

from __future__ import annotations

import time
from typing import Any

import robotmq


class RpcError(RuntimeError):
    """The server replied with an error (upstream servers reply with a string starting with 'ERROR')."""


class RpcTimeout(TimeoutError):
    """No reply within timeout_s. The request was sent exactly once."""


def request(client: "robotmq.RMQClient", topic: str, payload: Any, timeout_s: float) -> Any:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    data = robotmq.serialize(payload)
    t0 = time.monotonic()
    try:
        raw = client.request_with_data(topic=topic, data=data, timeout_s=float(timeout_s), automatic_resend=False)
    except RuntimeError as e:
        if "No reply from server" in str(e):
            raise RpcTimeout(f"{topic}: no reply within {timeout_s:.3f} s "
                             f"(waited {time.monotonic() - t0:.3f} s, not resent)") from e
        raise RpcError(f"{topic}: {e}") from e
    reply = robotmq.deserialize(raw)
    if isinstance(reply, str) and reply.startswith("ERROR"):
        raise RpcError(f"{topic}: {reply}")
    return reply


def peek(client: "robotmq.RMQClient", topic: str, n: int, timeout_s: float):
    """peek_data without resend. Returns (list[bytes], timestamps)."""
    try:
        return client.peek_data(topic=topic, n=n, timeout_s=float(timeout_s), automatic_resend=False)
    except RuntimeError as e:
        if "No reply from server" in str(e):
            raise RpcTimeout(f"peek {topic}: no reply within {timeout_s:.3f} s") from e
        raise RpcError(f"peek {topic}: {e}") from e


def pop(client: "robotmq.RMQClient", topic: str, n: int, timeout_s: float):
    """pop_data without resend. Returns (list[bytes], timestamps)."""
    try:
        return client.pop_data(topic=topic, n=n, timeout_s=float(timeout_s), automatic_resend=False)
    except RuntimeError as e:
        if "No reply from server" in str(e):
            raise RpcTimeout(f"pop {topic}: no reply within {timeout_s:.3f} s") from e
        raise RpcError(f"pop {topic}: {e}") from e


class NoResendClient:
    """Drop-in for robotmq.RMQClient inside upstream classes: every call is sent once, with a bounded timeout.

    Upstream code passes its own timeout_s (often none -> robotmq default 1 s with automatic_resend=True, i.e. up to
    800 resends). Here the timeout comes from `timeouts[topic]` (or `default_timeout_s`), `automatic_resend` is always
    False, a missing reply raises RpcTimeout and a robotmq error raises RpcError. `on_request` (optional) is called
    with (topic, request_bytes, reply_bytes, seconds) after each successful request_with_data (used for taps).
    """

    def __init__(self, client_name: str, server_endpoint: str, default_timeout_s: float = 0.5,
                 timeouts: dict | None = None, log_level=None):
        if log_level is None:
            self._client = robotmq.RMQClient(client_name, server_endpoint)
        else:
            self._client = robotmq.RMQClient(client_name, server_endpoint, log_level)
        self.name = client_name
        self.server_endpoint = server_endpoint
        self.default_timeout_s = float(default_timeout_s)
        self.timeouts = dict(timeouts or {})
        self.on_request = None

    def _t(self, topic):
        return float(self.timeouts.get(topic, self.default_timeout_s))

    def _call(self, what, topic, fn):
        t0 = time.monotonic()
        try:
            return fn()
        except RuntimeError as e:
            waited = time.monotonic() - t0
            if "No reply from server" in str(e):
                err = RpcTimeout(f"{what} {topic} on {self.server_endpoint}: no reply within {self._t(topic):.3f} s "
                                 f"(raised after {waited:.3f} s)")
            else:
                err = RpcError(f"{what} {topic} on {self.server_endpoint}: {e}")
            err.t_call_start = t0
            err.waited_s = waited
            raise err from e

    def request_with_data(self, topic, data, timeout_s=None, automatic_resend=None):
        t0 = time.monotonic()
        reply = self._call("request", topic, lambda: self._client.request_with_data(
            topic=topic, data=data, timeout_s=self._t(topic), automatic_resend=False))
        if self.on_request is not None:
            self.on_request(topic, data, reply, time.monotonic() - t0)
        return reply

    def peek_data(self, topic, n, timeout_s=None, automatic_resend=None):
        return self._call("peek", topic, lambda: self._client.peek_data(
            topic=topic, n=n, timeout_s=self._t(topic), automatic_resend=False))

    def pop_data(self, topic, n, timeout_s=None, automatic_resend=None):
        return self._call("pop", topic, lambda: self._client.pop_data(
            topic=topic, n=n, timeout_s=self._t(topic), automatic_resend=False))

    def put_data(self, topic, data, timeout_s=None, automatic_resend=None):
        return self._call("put", topic, lambda: self._client.put_data(
            topic=topic, data=data, timeout_s=self._t(topic), automatic_resend=False))

    def get_topic_status(self, topic, timeout_s=None):
        return self._call("status", topic, lambda: self._client.get_topic_status(topic=topic, timeout_s=self._t(topic)))

    def get_last_retrieved_data(self):
        return self._client.get_last_retrieved_data()

    def get_timestamp(self):
        return self._client.get_timestamp()
