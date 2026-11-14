# adapted from rnoro/deployment:deployment/message_comms/message_comms/franka_client.py @ 91b178b
# changes: renamed PcClient, process name "PC"; adds hello() (stamped round trip, no poll_s/cancel); timeouts from
# link.yaml (c3_timeout_s) and manager.timeouts.home_s; home() returns its Ack (the loop decides by it, incl. the
# `stopped:` prefix); every request's round trip goes to LinkStats.
"""PcClient: every C3 request, from the manager's main thread only (STRUCTURE.md 8.1; TOP 3.7, rule H1)."""
from __future__ import annotations

import time
from collections.abc import Callable

from deploy_contracts import (Ack, ArmState, ContractError, GetState, Heartbeat, Hello, HelloReply, Home, RpcClient,
                              SetWaypoints, SetWaypointsReply, Start, Stop, TransportCancelled, home_was_stopped)

from policy_deploy.manager.errors import Refused, RequestCancelled
from policy_deploy.manager.link_stats import LinkStats
from policy_deploy.manager.pc_clock import HelloSample

PROCESS = "PC"
HOME_REFUSED_HOMING = "refused: mode is homing"


class PcClient:
    def __init__(self, endpoint: str, c3_timeout_s: float, home_timeout_s: float, poll_s: float,
                 stats: LinkStats | None = None, clock: Callable[[], int] = time.monotonic_ns) -> None:
        """poll_s: the slice home() waits in between interrupted() checks (the heartbeat period)."""
        self._rpc = RpcClient(PROCESS, endpoint)
        self.endpoint = endpoint
        self._timeout_s = c3_timeout_s
        self._home_timeout_s = home_timeout_s
        self._poll_s = poll_s
        self._stats = stats
        self._clock = clock
        self.last_rtt_ns: int | None = None

    def _request(self, msg, expect):
        t0 = self._clock()
        reply = self._rpc.request(msg, expect, self._timeout_s)
        self.last_rtt_ns = self._clock() - t0
        if self._stats is not None:
            self._stats.rtt(msg.SCHEMA, self.last_rtt_ns)
        return reply

    def hello(self) -> tuple[HelloReply, HelloSample]:
        """One hello: t_send just before the request, t_recv just after the reply. A plain request without poll_s or
        cancel: a sliced poll would add delay on the receive side only, which the estimator would read as offset."""
        t_send = self._clock()
        reply = self._rpc.request(Hello(), HelloReply, self._timeout_s)
        t_recv = self._clock()
        self.last_rtt_ns = t_recv - t_send
        if self._stats is not None:
            self._stats.rtt(Hello.SCHEMA, self.last_rtt_ns)
        return reply, HelloSample(t_send_ns=t_send, t_pc_ns=int(reply.t_pc_ns), t_recv_ns=t_recv)

    def heartbeat(self) -> None:
        self._ack(self._request(Heartbeat(), Ack), Heartbeat.SCHEMA)

    def get_state(self) -> ArmState:
        """Mode checks only: a get_state reply is never a C2 sample (TOP 3.6 obligation 3)."""
        return self._request(GetState(), ArmState)

    def start(self) -> Ack:
        """A refusal is returned, not raised (STRUCTURE.md 6.4)."""
        return self._request(Start(), Ack)

    def stop(self) -> None:
        self._ack(self._request(Stop(), Ack), Stop.SCHEMA)

    def home(self, interrupted: Callable[[], bool]) -> Ack:
        """Waits up to home_s in poll_s slices; interrupted() true -> socket dropped, RequestCancelled. The Ack is
        returned whatever it says: ok, stopped (home_stopped), refused because the PC is homing, or failed."""
        try:
            return self._rpc.request(Home(), Ack, self._home_timeout_s, poll_s=self._poll_s, cancel=interrupted)
        except TransportCancelled as exc:
            raise RequestCancelled(PROCESS, Home.SCHEMA) from exc

    @staticmethod
    def home_stopped(ack: Ack) -> bool:
        """The one place that reads the contract prefix: ok false and the reason starts with "stopped:"."""
        return home_was_stopped(ack) or (not ack.ok and ack.reason.startswith("stopped:"))

    @staticmethod
    def home_refused_homing(ack: Ack) -> bool:
        """The PC's own h/g started in between: a hint, not a failure."""
        return not ack.ok and ack.reason.startswith(HOME_REFUSED_HOMING)

    def set_waypoints(self, msg: SetWaypoints) -> SetWaypointsReply:
        reply = self._request(msg, SetWaypointsReply)
        schema = SetWaypointsReply.SCHEMA
        if reply.chunk_id != msg.chunk_id:
            raise ContractError(schema, "chunk_id", "value", f"echo {reply.chunk_id}, request had {msg.chunk_id}")
        sent = len(msg.t_ns)
        counted = reply.accepted + reply.skipped + len(reply.rejected)
        if counted != sent:
            raise ContractError(schema, "accepted", "value",
                                f"accepted {reply.accepted} + skipped {reply.skipped} + rejected "
                                f"{len(reply.rejected)} = {counted}, but {sent} waypoints were sent")
        return reply

    def close(self) -> None:
        self._rpc.close()

    @staticmethod
    def _ack(ack: Ack, request: str) -> None:
        if not ack.ok:
            raise Refused(PROCESS, request, ack.reason)
