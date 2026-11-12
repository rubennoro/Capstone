"""RawServer: a server on a free port that answers REQ clients with whatever bytes the test gives (or never answers).

A ROUTER socket, so "never answer" is simply not replying: REQ clients drop their socket on a timeout anyway.
"""
from __future__ import annotations

import threading

import zmq

from contracts import encode


class RawServer:
    def __init__(self, answer) -> None:
        """answer(request_bytes) -> reply bytes, a Message, or None (never answer that request)."""
        self._answer = answer
        self._sock = zmq.Context.instance().socket(zmq.ROUTER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind("tcp://127.0.0.1:*")
        self.endpoint = self._sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self.received: list[bytes] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._sock.poll(10, zmq.POLLIN):
                continue
            identity, empty, data = self._sock.recv_multipart()
            self.received.append(data)
            reply = self._answer(data)
            if reply is not None:
                self._sock.send_multipart([identity, empty, reply if isinstance(reply, bytes) else encode(reply)])

    def close(self) -> None:
        self._stop.set()
        self._thread.join(1.0)
        self._sock.close(linger=0)
