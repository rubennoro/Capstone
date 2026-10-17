"""Bridge link: ZeroMQ REQ/REP (pyzmq; releases the GIL, unlike robotmq) with robotmq's numpy-aware serializer.

Client: every request is sent ONCE with an explicit timeout; on timeout the socket is closed (linger 0) and
re-created ("lazy pirate"), and RpcTimeout is raised. The server replies {"ok": bool, "data"| "error"}.
"""

from __future__ import annotations

import time

import zmq
from robotmq import deserialize, serialize

from franka_deploy.rmq_rpc import RpcError, RpcTimeout


class BridgeClient:
    def __init__(self, endpoint: str, timeout_s: float = 0.005, timeouts: dict | None = None):
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self.timeouts = dict(timeouts or {})
        self.ctx = zmq.Context.instance()
        self._sock = None
        self._connect()

    def _connect(self):
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self.ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    def request(self, topic: str, data=None, timeout_s: float | None = None):
        t = float(timeout_s if timeout_s is not None else self.timeouts.get(topic, self.timeout_s))
        t0 = time.monotonic()
        self._sock.send(serialize({"topic": topic, "data": data}))
        if not self._sock.poll(int(max(1, t * 1000)), zmq.POLLIN):
            self._connect()
            err = RpcTimeout(f"bridge {topic} on {self.endpoint}: no reply within {t:.3f} s")
            err.waited_s = time.monotonic() - t0
            raise err
        rep = deserialize(self._sock.recv())
        if not rep.get("ok"):
            raise RpcError(f"bridge {topic}: {rep.get('error')}")
        return rep.get("data")

    def close(self):
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None


class BridgeServerSocket:
    def __init__(self, endpoint: str):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(endpoint)

    def poll(self, timeout_ms: int):
        if self.sock.poll(timeout_ms, zmq.POLLIN):
            msg = deserialize(self.sock.recv())
            return msg["topic"], msg.get("data")
        return None, None

    def reply(self, ok: bool, data=None, error: str | None = None):
        self.sock.send(serialize({"ok": ok, "data": data, "error": error}))

    def close(self):
        self.sock.close(linger=0)
