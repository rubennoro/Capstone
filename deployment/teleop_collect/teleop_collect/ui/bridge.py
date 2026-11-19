"""The ui's two REQ clients, U1 (teleop) and U2 (frames), driven from the event loop (STRUCTURE.md 9.1).

Every request runs on a worker thread (asyncio run_in_executor), so the event loop never blocks. Each kind of
traffic has its own "lane": a one-thread executor with its own REQ socket (ZeroMQ sockets are not thread-safe, and a
REQ socket carries one request at a time). Lanes:

- status:  u1.get_status at status_hz;
- keys:    u1.keys, in order (one lane, so a release can never overtake the press before it);
- command: u1.gripper and the buttons;
- stop:    u1.stop_arm alone, so Stop arm / Esc never waits behind another button;
- preview: u2.get_preview at preview_hz.

Errors pass through unchanged (deploy_contracts): TransportTimeout (not answering), RemoteError (an error/v1 reply)
and ContractError (the reply broke its contract). The server decides what each means.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

from deploy_contracts import Ack, Message, RpcClient

from teleop_collect.contracts import (GetPreview, GetStatus, GripperCommand, KeysUpdate, Preview, StopArm,
                                      TeleopStatus)

T = TypeVar("T")

LANES = ("status", "keys", "command", "stop", "preview")


class FatalError(Exception):
    """The ui cannot go on: one line, what failed and what to check (the FATAL line)."""

    def __init__(self, what: str, check: str = "") -> None:
        self.what = what
        self.check = check
        super().__init__(f"{what}: {check}" if check else what)


class _PerThreadClient:
    """One RpcClient per calling thread (each lane is one thread), all closed together."""

    def __init__(self, process: str, endpoint: str, timeout_s: float) -> None:
        self.process = process
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._local = threading.local()
        self._all: list[RpcClient] = []
        self._lock = threading.Lock()
        self._closed = False

    def _client(self) -> RpcClient:
        client = getattr(self._local, "client", None)
        if client is None:
            with self._lock:
                if self._closed:
                    raise RuntimeError(f"the {self.process} client is closed")
                client = RpcClient(self.process, self.endpoint)
                self._all.append(client)
            self._local.client = client
        return client

    def request(self, msg: Message, expect: type[T]) -> T:
        return self._client().request(msg, expect, self.timeout_s)  # type: ignore[arg-type]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            clients, self._all = self._all, []
        for client in clients:
            client.close()


class U1Client:
    """ui -> teleop (STRUCTURE.md 4.4). Thread-safe: one REQ socket per calling thread."""

    def __init__(self, endpoint: str, timeout_s: float) -> None:
        self._rpc = _PerThreadClient("teleop", endpoint, timeout_s)

    def get_status(self, after_message: int) -> TeleopStatus:
        return self._rpc.request(GetStatus(after_message=int(after_message)), TeleopStatus)

    def keys(self, held: Sequence[str], slow: bool, focus: bool) -> Ack:
        return self._rpc.request(KeysUpdate(held=tuple(held), slow=bool(slow), focus=bool(focus)), Ack)

    def gripper(self, command: str) -> Ack:
        return self._rpc.request(GripperCommand(command=command), Ack)

    def send(self, msg: Message) -> Ack:
        """Any U1 request answered with ack/v1 (the buttons)."""
        return self._rpc.request(msg, Ack)

    def close(self) -> None:
        self._rpc.close()


class U2Client:
    """ui -> frames (STRUCTURE.md 4.3). Thread-safe: one REQ socket per calling thread."""

    def __init__(self, endpoint: str, timeout_s: float) -> None:
        self._rpc = _PerThreadClient("frames", endpoint, timeout_s)

    def get_preview(self, main_after: int, ultrawide_after: int) -> Preview:
        return self._rpc.request(GetPreview(main_after=int(main_after), ultrawide_after=int(ultrawide_after)),
                                 Preview)

    def close(self) -> None:
        self._rpc.close()


class Bridge:
    """The two clients behind awaitable calls, one lane (worker thread) per kind of traffic."""

    def __init__(self, u1_endpoint: str, preview_endpoint: str, timeout_s: float) -> None:
        self.u1 = U1Client(u1_endpoint, timeout_s)
        self.u2 = U2Client(preview_endpoint, timeout_s)
        self._lanes = {lane: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ui-{lane}") for lane in LANES}

    async def _run(self, lane: str, fn: Callable[..., T], *args) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._lanes[lane], fn, *args)

    async def get_status(self, after_message: int) -> TeleopStatus:
        return await self._run("status", self.u1.get_status, after_message)

    async def keys(self, held: Sequence[str], slow: bool, focus: bool) -> Ack:
        return await self._run("keys", self.u1.keys, tuple(held), slow, focus)

    async def gripper(self, command: str) -> Ack:
        return await self._run("command", self.u1.gripper, command)

    async def command(self, msg: Message) -> Ack:
        lane = "stop" if isinstance(msg, StopArm) else "command"
        return await self._run(lane, self.u1.send, msg)

    async def get_preview(self, main_after: int, ultrawide_after: int) -> Preview:
        return await self._run("preview", self.u2.get_preview, main_after, ultrawide_after)

    def close(self) -> None:
        """Wait for every lane's request in flight (each ends within rpc_timeout_s), then close the sockets."""
        for executor in self._lanes.values():
            executor.shutdown(wait=True, cancel_futures=True)
        self.u1.close()
        self.u2.close()
