"""FrankaClient: C3 to franka_control (STRUCTURE.md 4.2)."""
from __future__ import annotations

from collections.abc import Callable

from contracts import (Ack, ArmState, ContractError, GetState, Heartbeat, Home, RpcClient, SetWaypoints,
                       SetWaypointsReply, Start, Stop, TransportCancelled)
from message_comms.errors import Refused, RequestCancelled
from message_comms.settings import Timeouts

PROCESS = "franka_control"


class FrankaClient:
    def __init__(self, endpoint: str, timeouts: Timeouts, poll_s: float) -> None:
        """poll_s: the slice home() waits in between interrupted() checks (loop_ms)."""
        self._rpc = RpcClient(PROCESS, endpoint)
        self._timeouts = timeouts
        self._poll_s = poll_s

    def heartbeat(self) -> None:
        self._ack(self._rpc.request(Heartbeat(), Ack, self._timeouts.commands_s), Heartbeat.SCHEMA)

    def start(self) -> Ack:
        """A refusal is returned, not raised (STRUCTURE.md 5.3)."""
        return self._rpc.request(Start(), Ack, self._timeouts.commands_s)

    def stop(self) -> None:
        self._ack(self._rpc.request(Stop(), Ack, self._timeouts.commands_s), Stop.SCHEMA)

    def home(self, interrupted: Callable[[], bool]) -> None:
        """Waits up to home_s in poll_s slices; interrupted() true -> socket dropped, RequestCancelled."""
        try:
            ack = self._rpc.request(Home(), Ack, self._timeouts.home_s, poll_s=self._poll_s, cancel=interrupted)
        except TransportCancelled as exc:
            raise RequestCancelled(PROCESS, Home.SCHEMA) from exc
        self._ack(ack, Home.SCHEMA)

    def set_waypoints(self, msg: SetWaypoints) -> SetWaypointsReply:
        reply = self._rpc.request(msg, SetWaypointsReply, self._timeouts.commands_s)
        schema = SetWaypointsReply.SCHEMA
        if reply.chunk_id != msg.chunk_id:
            raise ContractError(schema, "chunk_id", "value",
                                f"echo {reply.chunk_id}, request had {msg.chunk_id}")
        sent = len(msg.t_ns)
        counted = reply.accepted + reply.skipped + len(reply.rejected)
        if counted != sent:
            raise ContractError(schema, "accepted", "value",
                                f"accepted {reply.accepted} + skipped {reply.skipped} + rejected "
                                f"{len(reply.rejected)} = {counted}, but {sent} waypoints were sent")
        return reply

    def get_state(self) -> ArmState:
        return self._rpc.request(GetState(), ArmState, self._timeouts.commands_s)

    def close(self) -> None:
        self._rpc.close()

    @staticmethod
    def _ack(ack: Ack, request: str) -> None:
        if not ack.ok:
            raise Refused(PROCESS, request, ack.reason)
