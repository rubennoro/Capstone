"""Publish/subscribe over ZeroMQ: Publisher and Subscriber for single messages, PacketPublisher and PacketSubscriber
for C0's multipart packets (STRUCTURE.md section 6).

Publishers bind, subscribers connect. Subscribers get every message (no "keep only the newest" option).
"""
from __future__ import annotations

import math
from typing import Generic, TypeVar

import zmq

from contracts.c0_frames import Packet
from contracts.codec import decode, encode
from contracts.errors import ContractError
from contracts.message import Message

M = TypeVar("M", bound=Message)


class Publisher:
    """PUB socket. publish() never blocks: a message is dropped when send_hwm messages are already queued."""

    def __init__(self, endpoint: str, send_hwm: int) -> None:
        self._socket: zmq.Socket | None = zmq.Context.instance().socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, send_hwm)
        self._socket.bind(endpoint)
        self.endpoint: str = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def publish(self, msg: Message) -> None:
        if self._socket is None:
            raise RuntimeError("Publisher is closed")
        try:
            self._socket.send(encode(msg), zmq.NOBLOCK)
        except zmq.Again:
            pass  # queue full: dropped, as documented

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


class Subscriber(Generic[M]):
    """SUB socket for one message type. `name` is the publisher's process name, for messages."""

    def __init__(self, name: str, endpoint: str, expect: type[M], recv_hwm: int) -> None:
        self.name = name
        self.endpoint = endpoint
        self.expect = expect
        self._socket: zmq.Socket | None = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, recv_hwm)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)

    def receive(self, timeout_s: float) -> M | None:
        """The next message, or None if none came within timeout_s. ContractError on a broken or other message."""
        if self._socket is None:
            raise RuntimeError(f"Subscriber to {self.name} is closed")
        if not self._socket.poll(max(0, int(math.ceil(timeout_s * 1000.0))), zmq.POLLIN):
            return None
        msg = decode(self._socket.recv(), self.expect)
        if not isinstance(msg, self.expect):  # decode lets ErrorReply through; a subscriber never expects one
            raise ContractError(msg.SCHEMA, "", "unexpected_message",
                                f"expected {self.expect.SCHEMA}, got {msg.SCHEMA}")
        return msg

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


class PacketPublisher:
    """PUB socket for C0 packets, one ZeroMQ multipart message each. publish() never blocks: a packet is dropped
    whole when send_hwm messages are already queued (the subscriber sees a packet_seq gap)."""

    def __init__(self, endpoint: str, send_hwm: int) -> None:
        self._socket: zmq.Socket | None = zmq.Context.instance().socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, send_hwm)
        self._socket.bind(endpoint)
        self.endpoint: str = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def publish(self, packet: Packet) -> None:
        if self._socket is None:
            raise RuntimeError("PacketPublisher is closed")
        try:
            self._socket.send_multipart(packet.to_parts(), zmq.NOBLOCK)
        except zmq.Again:
            pass  # queue full: dropped whole, as documented

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


class PacketSubscriber:
    """SUB socket for C0 packets. `name` is the publisher's process name, for messages. ZeroMQ delivers a multipart
    message whole or not at all, so a packet is never half-received."""

    def __init__(self, name: str, endpoint: str, recv_hwm: int) -> None:
        self.name = name
        self.endpoint = endpoint
        self._socket: zmq.Socket | None = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, recv_hwm)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)

    def receive(self, timeout_s: float) -> Packet | None:
        """The next packet, or None if none came within timeout_s. ContractError if it breaks C0."""
        if self._socket is None:
            raise RuntimeError(f"PacketSubscriber to {self.name} is closed")
        if not self._socket.poll(max(0, int(math.ceil(timeout_s * 1000.0))), zmq.POLLIN):
            return None
        return Packet.from_parts(self._socket.recv_multipart())

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
